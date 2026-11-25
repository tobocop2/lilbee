"""The device probe is remembered through one cache and never runs on the event loop."""

from __future__ import annotations

import dataclasses
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from lilbee.data import offload
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.devices import EngineDllMissingError, EngineIllegalInstructionError
from lilbee.providers.fleet.replicas import gpu_device_count
from tests.test_engine_call_sites import SPY_DEVICE, SpyEngine

PROBE_TIMEOUT_S = 60.0


class Clock:
    """A settable stand-in for the monotonic clock the read cache stamps with."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class WedgedEngine(SpyEngine):
    """An engine whose probe spends the whole timeout, then fails like a wedged driver."""

    def __init__(self, binary: Path, clock: Clock) -> None:
        super().__init__(binary)
        self._clock = clock

    def discover_devices(self, binary: Path):
        self.probed.append(binary)
        self._clock.now += PROBE_TIMEOUT_S
        raise ProviderError("device probe did not respond", provider="spy")


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    fake = Clock()
    monkeypatch.setattr(planning, "time", SimpleNamespace(monotonic=fake, sleep=time.sleep))
    return fake


@pytest.fixture
def wedged(tmp_path: Path, clock: Clock):
    binary = tmp_path / "wedged-engine"
    binary.write_bytes(b"v1")
    planning.clear_read_device_cache()
    yield WedgedEngine(binary, clock)
    planning.clear_read_device_cache()


@pytest.fixture
def healthy(tmp_path: Path, clock: Clock):
    binary = tmp_path / "healthy-engine"
    binary.write_bytes(b"v1")
    planning.clear_read_device_cache()
    yield SpyEngine(binary, devices=(SPY_DEVICE, dataclasses.replace(SPY_DEVICE, index=4)))
    planning.clear_read_device_cache()


def test_a_timed_out_probe_is_waited_on_once_in_the_failure_window(wedged: WedgedEngine) -> None:
    with pytest.raises(ProviderError, match="did not respond"):
        gpu_device_count(wedged)
    for _ in range(3):
        with pytest.raises(ProviderError, match="did not respond"):
            gpu_device_count(wedged)
    assert len(wedged.probed) == 1


def test_the_failure_window_runs_from_the_end_of_the_probe(
    wedged: WedgedEngine, clock: Clock
) -> None:
    with pytest.raises(ProviderError):
        gpu_device_count(wedged)
    clock.now += planning._DEVICE_PROBE_FAILURE_TTL_S - 1
    with pytest.raises(ProviderError):
        gpu_device_count(wedged)
    assert len(wedged.probed) == 1
    clock.now += 2
    with pytest.raises(ProviderError):
        gpu_device_count(wedged)
    assert len(wedged.probed) == 2


def test_replacing_the_engine_binary_in_place_probes_again(wedged: WedgedEngine) -> None:
    with pytest.raises(ProviderError):
        gpu_device_count(wedged)
    wedged.binary().write_bytes(b"v2, upgraded in place")
    with pytest.raises(ProviderError):
        gpu_device_count(wedged)
    assert len(wedged.probed) == 2


def test_a_kept_reading_replaces_a_cached_failure(wedged: WedgedEngine) -> None:
    with pytest.raises(ProviderError):
        gpu_device_count(wedged)
    probed = len(wedged.probed)
    reading = planning.DeviceReading([], planning.EngineBackend.UNKNOWN)
    planning._read_device_cache.remember(wedged.binary(), reading)
    assert planning._read_device_cache.get(wedged.binary(), wedged) is reading
    assert len(wedged.probed) == probed


def test_replacing_the_engine_binary_in_place_counts_the_cards_again(healthy: SpyEngine) -> None:
    assert gpu_device_count(healthy) == 2
    healthy.binary().write_bytes(b"v2, upgraded in place")
    healthy._devices.append(dataclasses.replace(SPY_DEVICE, index=7))
    assert gpu_device_count(healthy) == 3


@pytest.mark.parametrize(
    "error",
    [EngineDllMissingError, EngineIllegalInstructionError],
    ids=["dll-missing", "illegal-instruction"],
)
def test_an_engine_that_cannot_load_is_probed_again_on_the_next_call(
    healthy: SpyEngine, error: type[ProviderError]
) -> None:
    class Unloadable(SpyEngine):
        def discover_devices(self, binary: Path):
            self.probed.append(binary)
            raise error("engine cannot load", provider="spy")

    engine = Unloadable(healthy.binary())
    assert gpu_device_count(engine) == 1
    assert gpu_device_count(engine) == 1
    assert len(engine.probed) == 2


def test_an_unloadable_replacement_does_not_leave_the_old_reading_answering(
    healthy: SpyEngine,
) -> None:
    class Switch(SpyEngine):
        broken = False

        def discover_devices(self, binary: Path):
            if self.broken:
                self.probed.append(binary)
                raise EngineDllMissingError("engine cannot load", provider="spy")
            return super().discover_devices(binary)

    cards = (SPY_DEVICE, dataclasses.replace(SPY_DEVICE, index=4))
    engine = Switch(healthy.binary(), devices=cards)
    assert gpu_device_count(engine) == 2
    engine.binary().write_bytes(b"v2, runtime missing")
    engine.broken = True
    assert gpu_device_count(engine) == 1
    engine.broken = False
    assert gpu_device_count(engine) == 2
    assert len(engine.probed) == 3


def test_clearing_the_read_cache_ends_the_remembered_failure(wedged: WedgedEngine) -> None:
    with pytest.raises(ProviderError):
        gpu_device_count(wedged)
    planning.clear_read_device_cache()
    with pytest.raises(ProviderError):
        gpu_device_count(wedged)
    assert len(wedged.probed) == 2


def test_a_count_is_read_once_inside_the_success_window(healthy: SpyEngine, clock: Clock) -> None:
    assert gpu_device_count(healthy) == 2
    clock.now += planning._DEVICE_PROBE_TTL_S / 2
    assert gpu_device_count(healthy) == 2
    assert len(healthy.probed) == 1


def test_a_count_is_read_again_after_the_success_window(healthy: SpyEngine, clock: Clock) -> None:
    assert gpu_device_count(healthy) == 2
    clock.now += planning._DEVICE_PROBE_TTL_S + 1
    assert gpu_device_count(healthy) == 2
    assert len(healthy.probed) == 2


def test_the_success_window_is_two_seconds_on_both_sides_of_its_edge(
    healthy: SpyEngine, clock: Clock
) -> None:
    assert gpu_device_count(healthy) == 2
    stamp = clock.now
    clock.now = stamp + 1.99
    assert gpu_device_count(healthy) == 2
    assert len(healthy.probed) == 1
    clock.now = stamp + 2.0
    assert gpu_device_count(healthy) == 2
    assert len(healthy.probed) == 2


def test_the_failure_window_is_sixty_seconds_on_both_sides_of_its_edge(
    wedged: WedgedEngine, clock: Clock
) -> None:
    with pytest.raises(ProviderError):
        gpu_device_count(wedged)
    stamp = clock.now
    clock.now = stamp + 30
    with pytest.raises(ProviderError):
        gpu_device_count(wedged)
    clock.now = stamp + 59.99
    with pytest.raises(ProviderError):
        gpu_device_count(wedged)
    assert len(wedged.probed) == 1
    clock.now = stamp + 60.0
    with pytest.raises(ProviderError):
        gpu_device_count(wedged)
    assert len(wedged.probed) == 2


def test_a_good_probe_after_an_expired_failure_ends_the_failure(
    healthy: SpyEngine, clock: Clock
) -> None:
    class Recovering(SpyEngine):
        failing = True

        def discover_devices(self, binary: Path):
            if self.failing:
                self.probed.append(binary)
                raise ProviderError("device probe did not respond", provider="spy")
            return super().discover_devices(binary)

    cards = (SPY_DEVICE, dataclasses.replace(SPY_DEVICE, index=4))
    engine = Recovering(healthy.binary(), devices=cards)
    with pytest.raises(ProviderError, match="did not respond"):
        gpu_device_count(engine)
    engine.failing = False
    clock.now += 61
    assert gpu_device_count(engine) == 2
    clock.now += 1
    assert gpu_device_count(engine) == 2
    assert len(engine.probed) == 2


def test_the_count_and_the_placement_view_share_one_probe(healthy: SpyEngine) -> None:
    assert gpu_device_count(healthy) == 2
    assert len(planning.probed_devices(healthy)) == 2
    assert len(healthy.probed) == 1


def test_the_launch_probe_reads_fresh_after_a_remembered_count(healthy: SpyEngine) -> None:
    assert gpu_device_count(healthy) == 2
    planning._probe_engine_devices(healthy)
    assert len(healthy.probed) == 2


async def test_the_ingest_pool_is_sized_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop_thread = threading.current_thread()
    sized_on: list[threading.Thread] = []

    def sizing() -> int:
        sized_on.append(threading.current_thread())
        return 3

    monkeypatch.setattr(offload, "max_workers", sizing)

    @offload.owns_ingest_pool
    async def run() -> int:
        pool = offload._run_pool.get()
        assert pool is not None
        return pool._max_workers

    assert await run() == 3
    assert len(sized_on) == 1
    assert sized_on[0] is not loop_thread
