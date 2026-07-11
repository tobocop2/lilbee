"""Adaptive ingest concurrency: hill-climb to the hardware's throughput knee.

The extraction-admission limit (how many documents are in their compute phase at
once) can run in one of three modes, chosen by ``LILBEE_INGEST_CONCURRENCY``:

- ``static`` -- a fixed limit (the pipeline's ``_max_concurrent()``); the proven
  default and the guaranteed-safe fallback.
- ``adaptive-conservative`` / ``adaptive-aggressive`` -- a background controller
  resizes the limit every few seconds, climbing while throughput still improves
  and backing off the instant a safety signal (CPU, free RAM, GPU temperature)
  says the box is under pressure.

The control law is a safety-gated AIMD hill-climb on smoothed throughput. The
only thing that pushes the limit up is throughput still improving, so the fixed
point is *this box's* real operating knee rather than a hardcoded utilization
target (BBR / Kleinrock). GPU utilization is used only as a saturation veto, never
as a setpoint. Multiplicative decrease on any danger signal is AIMD's proven
fast-safe retreat; EWMA smoothing plus an asymmetric dead band and a one-step slew
limit keep it from oscillating (CLR ThreadPool hill-climbing, TCP Vegas).

``decide`` is a pure function -- the whole control law with no clock, asyncio, or
hardware -- so the policy is unit-tested with plain numbers. ``ResizableGate`` and
``AdaptiveController`` are the async machinery around it.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lilbee.providers.fleet.gpu_stats import DeviceLike

log = logging.getLogger(__name__)

_MODE_ENV = "LILBEE_INGEST_CONCURRENCY"


class ConcurrencyMode(StrEnum):
    """How the extraction-admission limit is chosen for a sync run."""

    STATIC = "static"
    ADAPTIVE_CONSERVATIVE = "adaptive-conservative"
    ADAPTIVE_AGGRESSIVE = "adaptive-aggressive"


def resolve_mode() -> ConcurrencyMode:
    """The concurrency mode from ``LILBEE_INGEST_CONCURRENCY``; ``static`` by default.

    An unset or unrecognized value falls back to ``static`` (a warning is logged for
    an unrecognized one), so a typo can never silently enable adaptive control.
    """
    raw = os.environ.get(_MODE_ENV, "").strip().lower()
    if not raw:
        return ConcurrencyMode.STATIC
    try:
        return ConcurrencyMode(raw)
    except ValueError:
        log.warning(
            "Ignoring %s=%r: expected one of %s; using %s.",
            _MODE_ENV,
            raw,
            [m.value for m in ConcurrencyMode],
            ConcurrencyMode.STATIC.value,
        )
        return ConcurrencyMode.STATIC


@dataclass(frozen=True)
class SafetyLimits:
    """Hard guardrails, identical across adaptive profiles -- safety is never relaxed.

    Crossing a ``_crit`` line forces an immediate multiplicative backoff; a ``_warn``
    / ``_soft`` line only vetoes further increases. Temperature and RAM are read raw
    (never smoothed -- a fire alarm should not be averaged away).
    """

    gpu_sat_pct: float = 97.0
    cpu_soft_pct: float = 90.0
    cpu_crit_pct: float = 97.0
    ram_soft_free: float = 0.20
    ram_min_free: float = 0.10
    temp_warn_c: float = 80.0
    temp_crit_c: float = 85.0
    decrease_factor: float = 0.5


@dataclass(frozen=True)
class ConcurrencyProfile:
    """Climb-speed parameters for one adaptive profile (safety limits are shared)."""

    name: str
    interval_s: float
    ewma_gamma: float
    deadband_frac: float
    cool_down_intervals: int
    sqrt_step: bool
    latency_veto_ratio: float  # veto increases once residence time inflates past baseline x this
    safety: SafetyLimits = SafetyLimits()

    def step(self, permits: int) -> int:
        """Additive step size at the current limit; larger early when ``sqrt_step``."""
        if self.sqrt_step:
            return 1 + math.isqrt(max(0, permits)) // 4
        return 1


CONSERVATIVE = ConcurrencyProfile(
    name="conservative",
    interval_s=5.0,
    ewma_gamma=0.3,
    deadband_frac=0.05,
    cool_down_intervals=3,
    sqrt_step=False,
    latency_veto_ratio=1.5,
)
AGGRESSIVE = ConcurrencyProfile(
    name="aggressive",
    interval_s=2.0,
    ewma_gamma=0.5,
    deadband_frac=0.03,
    cool_down_intervals=2,
    sqrt_step=True,
    latency_veto_ratio=2.0,
)


def profile_for(mode: ConcurrencyMode) -> ConcurrencyProfile | None:
    """The profile for an adaptive mode, or None for ``static``."""
    return {
        ConcurrencyMode.ADAPTIVE_CONSERVATIVE: CONSERVATIVE,
        ConcurrencyMode.ADAPTIVE_AGGRESSIVE: AGGRESSIVE,
    }.get(mode)


@dataclass(frozen=True)
class Signals:
    """One tick's measured state. ``gpu_*`` are None when no GPU telemetry is available."""

    throughput: float  # OCR pages completed since the previous tick (work done, not doc count)
    gpu_util_pct: float | None
    gpu_temp_c: float | None
    cpu_pct: float
    ram_free_frac: float


@dataclass(frozen=True)
class ControllerState:
    """The controller's carried state between ticks."""

    permits: int
    ewma_tput: float | None  # smoothed throughput from the previous tick; None at start
    direction: int  # +1 climbing, -1 backing off -- the last hill-climb step's sign
    cool_down: int  # intervals remaining during which increases are suppressed
    w_min: float | None = None  # smallest observed residence-time estimate (Little's Law baseline)


def _is_critical(signals: Signals, s: SafetyLimits) -> bool:
    """A signal that demands an immediate hard backoff (thermal / OOM / CPU meltdown)."""
    temp = signals.gpu_temp_c
    return (
        (temp is not None and temp >= s.temp_crit_c)
        or signals.ram_free_frac <= s.ram_min_free
        or signals.cpu_pct >= s.cpu_crit_pct
    )


def _increase_vetoed(
    profile: ConcurrencyProfile,
    signals: Signals,
    *,
    cool_down: int,
    gpu_saturated: bool,
    w_est: float | None,
    w_min: float | None,
) -> bool:
    """Whether soft pressure, saturation, cool-down, or inflating latency blocks a climb."""
    s = profile.safety
    temp = signals.gpu_temp_c
    latency_inflated = (
        w_est is not None and w_min is not None and w_est > w_min * profile.latency_veto_ratio
    )
    return (
        cool_down > 0
        or signals.cpu_pct >= s.cpu_soft_pct
        or signals.ram_free_frac < s.ram_soft_free
        or (temp is not None and temp >= s.temp_warn_c)
        or gpu_saturated
        or latency_inflated
    )


def _hill_climb(
    profile: ConcurrencyProfile,
    permits: int,
    direction: int,
    delta: float | None,
    new_ewma: float,
    clamp: Callable[[int], int],
) -> tuple[int, int]:
    """One dead-banded hill-climb step; returns the next (permits, direction).

    No baseline yet -> a gentle probe up; a clear gain -> step in the same direction;
    a clear loss -> reverse; inside the dead band -> hold.
    """
    if delta is None:
        return clamp(permits + profile.step(permits)), 1
    band = profile.deadband_frac * (new_ewma if new_ewma > 0 else 1.0)
    if delta > band:
        return clamp(permits + direction * profile.step(permits)), direction
    if delta < -band:
        direction = -direction
        return clamp(permits + direction * profile.step(permits)), direction
    return permits, direction


def decide(
    profile: ConcurrencyProfile,
    state: ControllerState,
    signals: Signals,
    permit_min: int,
    permit_max: int,
) -> ControllerState:
    """Pure control law: fold one tick of signals into the next permit target.

    Priority: (1) any critical safety signal forces a multiplicative decrease and a
    cool-down; (2) a soft-pressure, GPU-saturation, or latency-gradient signal vetoes
    increases (with a single additive decrease when GPUs are saturated *and* throughput
    is falling -- the Universal Scalability Law's retrograde region); (3) otherwise
    hill-climb toward the knee. Always clamped to ``[permit_min, permit_max]``.

    The latency veto is the leading knee indicator (TCP Vegas / Netflix Gradient2):
    residence time ``W`` is estimated by Little's Law as ``permits / throughput``; once
    ``W`` inflates past its observed minimum (the unloaded baseline) by the profile's
    ratio, work is queueing at the bottleneck and increases stop -- before throughput
    visibly rolls over.
    """
    if state.ewma_tput is None:
        new_ewma: float = signals.throughput
        delta: float | None = None
    else:
        gamma = profile.ewma_gamma
        new_ewma = gamma * signals.throughput + (1.0 - gamma) * state.ewma_tput
        delta = new_ewma - state.ewma_tput

    permits = state.permits
    cool_down = max(0, state.cool_down - 1)

    # Little's Law residence-time estimate and its running minimum (the latency baseline).
    w_est = permits / new_ewma if new_ewma > 0 else None
    w_min = state.w_min if w_est is None else min(state.w_min or w_est, w_est)

    def clamp(p: int) -> int:
        return max(permit_min, min(permit_max, p))

    def out(new_permits: int, new_direction: int, new_cool_down: int) -> ControllerState:
        return ControllerState(new_permits, new_ewma, new_direction, new_cool_down, w_min)

    if _is_critical(signals, profile.safety):
        backoff = clamp(int(permits * profile.safety.decrease_factor))
        return out(backoff, -1, profile.cool_down_intervals)

    util = signals.gpu_util_pct
    gpu_saturated = util is not None and util >= profile.safety.gpu_sat_pct
    if gpu_saturated and delta is not None and delta < 0:
        return out(clamp(permits - profile.step(permits)), -1, cool_down)  # USL retrograde

    if _increase_vetoed(
        profile, signals, cool_down=cool_down, gpu_saturated=gpu_saturated, w_est=w_est, w_min=w_min
    ):
        return out(permits, state.direction, cool_down)

    permits, direction = _hill_climb(profile, permits, state.direction, delta, new_ewma, clamp)
    return out(permits, direction, cool_down)


class ResizableGate:
    """An async admission gate whose permit ceiling can change while it is in use.

    Same ``async with`` shape as ``asyncio.Semaphore``, plus ``set_limit``: growing
    wakes blocked acquirers, shrinking lowers the ceiling and lets the surplus drain
    as active holders release. The limit never drops below one, so a shrink can never
    deadlock a run.
    """

    def __init__(self, limit: int) -> None:
        self._limit = max(1, limit)
        self._active = 0
        self._cond = asyncio.Condition()

    @property
    def limit(self) -> int:
        return self._limit

    async def acquire(self) -> None:
        async with self._cond:
            await self._cond.wait_for(lambda: self._active < self._limit)
            self._active += 1

    async def release(self) -> None:
        async with self._cond:
            self._active -= 1
            self._cond.notify_all()

    async def __aenter__(self) -> ResizableGate:
        await self.acquire()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.release()

    async def set_limit(self, new_limit: int) -> None:
        async with self._cond:
            self._limit = max(1, new_limit)
            self._cond.notify_all()


class AdaptiveController:
    """Drives a :class:`ResizableGate`'s limit from live signals until cancelled.

    ``sample(throughput)`` returns the current :class:`Signals`; ``completed()`` is a
    monotonic count of finished documents, from which per-interval throughput is
    derived. Both are injected so the controller runs in tests with no clock or GPU.
    """

    def __init__(
        self,
        gate: ResizableGate,
        profile: ConcurrencyProfile,
        sample: Callable[[float], Signals],
        completed: Callable[[], int],
        *,
        permit_min: int,
        permit_max: int,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._gate = gate
        self._profile = profile
        self._sample = sample
        self._completed = completed
        self._permit_min = permit_min
        self._permit_max = permit_max
        self._sleep = sleep

    async def run(self) -> None:
        """Sample-decide-resize loop. Runs until the task is cancelled.

        A tuning tick is best-effort: a transient sampling/probe failure is logged and
        skipped, never propagated, so the controller can never crash the ingest it is
        only advising. Cancellation (``CancelledError``) still ends the loop cleanly.
        """
        state = ControllerState(self._gate.limit, None, 1, 0)
        last_completed = self._completed()
        while True:
            await self._sleep(self._profile.interval_s)
            try:
                now_completed = self._completed()
                throughput = float(max(0, now_completed - last_completed))
                last_completed = now_completed
                state = decide(
                    self._profile,
                    state,
                    self._sample(throughput),
                    self._permit_min,
                    self._permit_max,
                )
                if state.permits != self._gate.limit:
                    await self._gate.set_limit(state.permits)
                    log.debug("adaptive ingest: limit -> %d", state.permits)
            except Exception:
                log.debug("adaptive ingest: tuning tick failed; skipping", exc_info=True)


def enumerate_fleet_devices() -> Sequence[DeviceLike]:
    """The GPU devices to read telemetry from, or empty when none can be enumerated.

    Any failure (no engine binary, probe error) degrades to an empty list, which the
    caller treats as "no fleet to feed" and falls back to the static limit.
    """
    try:
        from lilbee.providers.fleet.binary import resolve_llama_server
        from lilbee.providers.fleet.planning import resolve_devices

        return resolve_devices(resolve_llama_server())
    except Exception:
        log.debug("adaptive ingest: device enumeration failed; using static limit", exc_info=True)
        return []


def make_signal_sampler(devices: Sequence[DeviceLike]) -> Callable[[float], Signals]:
    """Build a sampler that reads mean GPU util, max GPU temp, CPU %, and free RAM.

    GPU util/temp are None when no device reports them (the controller then relies on
    the CPU and RAM guards alone); throughput is supplied by the controller.
    """
    import psutil

    from lilbee.providers.fleet.gpu_stats import probe_gpu_stats

    def sample(throughput: float) -> Signals:
        stats = probe_gpu_stats(devices)
        utils = [g.utilization_pct for g in stats.values() if g.utilization_pct is not None]
        temps = [g.temperature_c for g in stats.values() if g.temperature_c is not None]
        vm = psutil.virtual_memory()
        return Signals(
            throughput=throughput,
            gpu_util_pct=(sum(utils) / len(utils)) if utils else None,
            gpu_temp_c=float(max(temps)) if temps else None,
            cpu_pct=psutil.cpu_percent(interval=None),
            ram_free_frac=vm.available / vm.total,  # psutil total is always positive
        )

    return sample
