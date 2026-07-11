"""Adaptive ingest concurrency: mode resolution, the pure control law, the
resizable gate, and the controller loop.

The control law (:func:`decide`) is exercised with plain numbers -- no clock, no
asyncio, no GPU -- since it is a pure function; the gate and controller get small
asyncio tests with injected signals so they run without hardware.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from lilbee.data.ingest.adaptive import (
    AGGRESSIVE,
    CONSERVATIVE,
    AdaptiveController,
    ConcurrencyMode,
    ControllerState,
    ResizableGate,
    Signals,
    decide,
    enumerate_fleet_devices,
    make_signal_sampler,
    profile_for,
    resolve_mode,
)

MIN, MAX = 1, 100


def _signals(throughput: float, **over: float | None) -> Signals:
    """A tick with every safety signal comfortable, overridable per test."""
    base: dict[str, float | None] = {
        "gpu_util_pct": 50.0,
        "gpu_temp_c": 60.0,
        "cpu_pct": 50.0,
        "ram_free_frac": 0.5,
    }
    base.update(over)
    return Signals(throughput=throughput, **base)  # type: ignore[arg-type]


def _state(
    permits: int,
    ewma: float | None = 100.0,
    direction: int = 1,
    cool_down: int = 0,
    w_min: float | None = None,
):
    return ControllerState(
        permits=permits, ewma_tput=ewma, direction=direction, cool_down=cool_down, w_min=w_min
    )


# --- mode resolution -------------------------------------------------------


def test_mode_defaults_to_static(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LILBEE_INGEST_CONCURRENCY", raising=False)
    assert resolve_mode() is ConcurrencyMode.STATIC


@pytest.mark.parametrize(
    ("value", "mode"),
    [
        ("static", ConcurrencyMode.STATIC),
        ("adaptive-conservative", ConcurrencyMode.ADAPTIVE_CONSERVATIVE),
        ("ADAPTIVE-AGGRESSIVE", ConcurrencyMode.ADAPTIVE_AGGRESSIVE),
    ],
)
def test_mode_parses_known_values(monkeypatch: pytest.MonkeyPatch, value: str, mode) -> None:
    monkeypatch.setenv("LILBEE_INGEST_CONCURRENCY", value)
    assert resolve_mode() is mode


def test_unknown_mode_falls_back_to_static(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LILBEE_INGEST_CONCURRENCY", "turbo")
    assert resolve_mode() is ConcurrencyMode.STATIC


def test_profile_for_maps_modes() -> None:
    assert profile_for(ConcurrencyMode.STATIC) is None
    assert profile_for(ConcurrencyMode.ADAPTIVE_CONSERVATIVE) is CONSERVATIVE
    assert profile_for(ConcurrencyMode.ADAPTIVE_AGGRESSIVE) is AGGRESSIVE
    assert CONSERVATIVE.step(100) == 1
    assert AGGRESSIVE.step(100) == 3  # 1 + isqrt(100)//4


# --- pure control law ------------------------------------------------------


def test_first_tick_probes_upward() -> None:
    out = decide(CONSERVATIVE, _state(10, ewma=None), _signals(100), MIN, MAX)
    assert out.permits == 11
    assert out.ewma_tput == 100
    assert out.direction == 1


def test_clear_throughput_gain_climbs_in_direction() -> None:
    out = decide(CONSERVATIVE, _state(10, ewma=100, direction=1), _signals(200), MIN, MAX)
    assert out.permits == 11  # new_ewma 130 > band -> +1


def test_clear_throughput_loss_reverses_direction() -> None:
    out = decide(CONSERVATIVE, _state(10, ewma=100, direction=1), _signals(0), MIN, MAX)
    assert out.direction == -1
    assert out.permits == 9


def test_inside_dead_band_holds() -> None:
    out = decide(CONSERVATIVE, _state(10, ewma=100, direction=1), _signals(101), MIN, MAX)
    assert out.permits == 10


@pytest.mark.parametrize("danger", ["gpu_temp_c", "cpu_pct", "ram_free_frac"])
def test_critical_signal_forces_backoff_and_cooldown(danger: str) -> None:
    crit = {"gpu_temp_c": 90.0, "cpu_pct": 98.0, "ram_free_frac": 0.05}[danger]
    out = decide(CONSERVATIVE, _state(10, ewma=100), _signals(500, **{danger: crit}), MIN, MAX)
    assert out.permits == 5  # halved
    assert out.cool_down == CONSERVATIVE.cool_down_intervals


@pytest.mark.parametrize(
    ("field", "value"),
    [("cpu_pct", 92.0), ("ram_free_frac", 0.15), ("gpu_temp_c", 82.0), ("gpu_util_pct", 98.0)],
)
def test_soft_pressure_vetoes_increase(field: str, value: float) -> None:
    # Throughput is rising, so without the veto it would climb; the veto holds it.
    out = decide(
        CONSERVATIVE, _state(10, ewma=100, direction=1), _signals(500, **{field: value}), MIN, MAX
    )
    assert out.permits == 10


def test_saturated_and_falling_steps_back() -> None:
    # GPU saturated AND throughput slipping -> USL retrograde single step down.
    out = decide(
        CONSERVATIVE, _state(10, ewma=100, direction=1), _signals(0, gpu_util_pct=99.0), MIN, MAX
    )
    assert out.permits == 9
    assert out.direction == -1


def test_latency_gradient_vetoes_increase() -> None:
    # An established low baseline (w_min) plus an inflated current residence time
    # (permits high, throughput low) vetoes the climb before throughput rolls over.
    out = decide(CONSERVATIVE, _state(20, ewma=100, direction=1, w_min=0.1), _signals(10), MIN, MAX)
    assert out.permits == 20  # held: w_est 20/73 ~ 0.27 > 0.1 * 1.5


def test_residence_baseline_is_tracked() -> None:
    out = decide(CONSERVATIVE, _state(10, ewma=100, w_min=None), _signals(100), MIN, MAX)
    assert out.w_min == pytest.approx(0.1)  # 10 / 100


def test_cooldown_suppresses_increase_and_decrements() -> None:
    out = decide(
        CONSERVATIVE, _state(10, ewma=100, direction=1, cool_down=2), _signals(500), MIN, MAX
    )
    assert out.permits == 10
    assert out.cool_down == 1


def test_clamped_to_max() -> None:
    out = decide(CONSERVATIVE, _state(MAX, ewma=100, direction=1), _signals(500), MIN, MAX)
    assert out.permits == MAX


def test_clamped_to_min_on_backoff() -> None:
    out = decide(CONSERVATIVE, _state(1, ewma=100), _signals(500, cpu_pct=99.0), MIN, MAX)
    assert out.permits == MIN


# --- resizable gate --------------------------------------------------------


async def test_gate_blocks_at_limit_and_release_admits() -> None:
    gate = ResizableGate(1)
    await gate.acquire()
    admitted = asyncio.Event()

    async def second() -> None:
        await gate.acquire()
        admitted.set()

    task = asyncio.create_task(second())
    await asyncio.sleep(0.01)
    assert not admitted.is_set()
    await gate.release()
    await asyncio.wait_for(admitted.wait(), 1)
    await task


async def test_gate_grow_wakes_a_waiter() -> None:
    gate = ResizableGate(1)
    await gate.acquire()
    got = asyncio.Event()

    async def waiter() -> None:
        await gate.acquire()
        got.set()

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0.01)
    assert not got.is_set()
    await gate.set_limit(2)
    await asyncio.wait_for(got.wait(), 1)
    await task


async def test_gate_shrink_lowers_ceiling() -> None:
    gate = ResizableGate(2)
    await gate.acquire()
    await gate.acquire()  # active == limit == 2
    await gate.set_limit(1)  # over the new ceiling; must drain to below it
    blocked = asyncio.Event()

    async def waiter() -> None:
        await gate.acquire()
        blocked.set()

    task = asyncio.create_task(waiter())
    await gate.release()  # active 2 -> 1, still not < limit(1)
    await asyncio.sleep(0.01)
    assert not blocked.is_set()
    await gate.release()  # active 1 -> 0 < 1
    await asyncio.wait_for(blocked.wait(), 1)
    await task


# --- controller loop -------------------------------------------------------


async def test_controller_climbs_on_rising_throughput() -> None:
    gate = ResizableGate(10)
    done = {"c": 0}
    tick = {"n": 0}

    def completed() -> int:
        return done["c"]

    def sample(throughput: float) -> Signals:
        return _signals(throughput)

    async def fake_sleep(_: float) -> None:
        tick["n"] += 1
        done["c"] += 100 * tick["n"]  # rising per-interval deltas
        if tick["n"] >= 6:
            raise asyncio.CancelledError

    ctrl = AdaptiveController(
        gate, CONSERVATIVE, sample, completed, permit_min=1, permit_max=50, sleep=fake_sleep
    )
    with contextlib.suppress(asyncio.CancelledError):
        await ctrl.run()
    assert gate.limit > 10


async def test_controller_survives_a_sampler_failure() -> None:
    gate = ResizableGate(10)
    tick = {"n": 0}

    def completed() -> int:
        return 0

    def sample(throughput: float) -> Signals:
        raise RuntimeError("probe blew up")

    async def fake_sleep(_: float) -> None:
        tick["n"] += 1
        if tick["n"] >= 3:
            raise asyncio.CancelledError

    ctrl = AdaptiveController(
        gate, CONSERVATIVE, sample, completed, permit_min=1, permit_max=50, sleep=fake_sleep
    )
    with contextlib.suppress(asyncio.CancelledError):
        await ctrl.run()
    assert gate.limit == 10  # tick failures swallowed; the tuner never crashed


async def test_controller_holds_and_skips_resize_when_flat() -> None:
    gate = ResizableGate(10)
    done = {"c": 0}
    tick = {"n": 0}

    def completed() -> int:
        return done["c"]

    def sample(throughput: float) -> Signals:
        return _signals(throughput)

    async def fake_sleep(_: float) -> None:
        tick["n"] += 1
        done["c"] += 100  # constant delta -> flat throughput after the first probe
        if tick["n"] >= 4:
            raise asyncio.CancelledError

    ctrl = AdaptiveController(
        gate, CONSERVATIVE, sample, completed, permit_min=1, permit_max=50, sleep=fake_sleep
    )
    with contextlib.suppress(asyncio.CancelledError):
        await ctrl.run()
    assert gate.limit == 11  # one probe up, then flat -> holds (no further resize)


async def test_controller_backs_off_on_emergency() -> None:
    gate = ResizableGate(20)
    tick = {"n": 0}

    def completed() -> int:
        return 0

    def sample(throughput: float) -> Signals:
        return _signals(throughput, gpu_temp_c=90.0)  # over TEMP_CRIT

    async def fake_sleep(_: float) -> None:
        tick["n"] += 1
        if tick["n"] >= 2:  # let one decide() run, then stop
            raise asyncio.CancelledError

    ctrl = AdaptiveController(
        gate, CONSERVATIVE, sample, completed, permit_min=1, permit_max=50, sleep=fake_sleep
    )
    with contextlib.suppress(asyncio.CancelledError):
        await ctrl.run()
    assert gate.limit < 20


async def test_gate_context_manager_admits_and_releases() -> None:
    gate = ResizableGate(1)
    async with gate:
        blocked = asyncio.Event()

        async def waiter() -> None:
            async with gate:
                blocked.set()

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.01)
        assert not blocked.is_set()  # held inside the outer `async with`
    await asyncio.wait_for(blocked.wait(), 1)  # released on exit
    await task


# --- real signal sampler ---------------------------------------------------


def test_sampler_reports_mean_util_and_max_temp(monkeypatch: pytest.MonkeyPatch) -> None:
    from lilbee.providers.fleet import gpu_stats

    stats = {
        0: gpu_stats.GpuStat(0, utilization_pct=40, free_bytes=1, total_bytes=2, temperature_c=70),
        1: gpu_stats.GpuStat(1, utilization_pct=80, free_bytes=1, total_bytes=2, temperature_c=85),
    }
    monkeypatch.setattr(gpu_stats, "probe_gpu_stats", lambda _devices: stats)
    signals = make_signal_sampler([object()])(throughput=12.0)
    assert signals.throughput == 12.0
    assert signals.gpu_util_pct == 60.0  # mean(40, 80)
    assert signals.gpu_temp_c == 85.0  # max(70, 85)
    assert 0.0 <= signals.ram_free_frac <= 1.0


def test_sampler_without_gpu_telemetry_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    from lilbee.providers.fleet import gpu_stats

    monkeypatch.setattr(gpu_stats, "probe_gpu_stats", lambda _devices: {})
    signals = make_signal_sampler([])(throughput=0.0)
    assert signals.gpu_util_pct is None
    assert signals.gpu_temp_c is None


# --- device enumeration ----------------------------------------------------


def test_enumerate_devices_returns_probe_result(monkeypatch: pytest.MonkeyPatch) -> None:
    from lilbee.providers.fleet import binary, planning

    sentinel = [object()]
    monkeypatch.setattr(binary, "resolve_llama_server", lambda: "llama-server")
    monkeypatch.setattr(planning, "resolve_devices", lambda _binary: sentinel)
    assert enumerate_fleet_devices() == sentinel


def test_enumerate_devices_degrades_to_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    from lilbee.providers.fleet import binary

    def boom() -> str:
        raise RuntimeError("no engine")

    monkeypatch.setattr(binary, "resolve_llama_server", boom)
    assert enumerate_fleet_devices() == []
