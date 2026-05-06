"""Background ticker that drives WorkerPool.reap_idle and ping_role on a fixed cadence."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from concurrent.futures import Future
from dataclasses import dataclass

from lilbee.providers.worker.pool import PoolRuntime, WorkerPool

log = logging.getLogger(__name__)

# 30s is the smallest interval that meaningfully reduces wall-clock idle
# accumulation without burning bg-loop wakeups when nothing is live (the
# tick is a cheap no-op when no role channels are alive). Tunable via
# monkeypatch in tests; not exposed as a user setting because the per-role
# max_idle_s knob already covers the meaningful axis.
_TICK_INTERVAL_S = 30.0


@dataclass
class HealthTickerHandle:
    """Holds the ticker's background future for later cancellation."""

    future: Future[None] | None = None


async def _tick_once(pool: WorkerPool, runtime: PoolRuntime) -> None:
    """Run one reap + ping pass against every currently-live role.

    Bridges the bg-loop (where this coroutine runs) to the PoolRuntime
    loop (which owns the per-role asyncio.Lock instances). We never block
    the bg-loop while waiting for the pool's response. ``asyncio.CancelledError``
    is a ``BaseException`` (not an ``Exception``) so it escapes the
    ``except Exception`` guards naturally and unwinds the outer
    ``_ticker_loop`` cleanly.
    """
    live_roles = tuple(role for role in pool.registered_roles if pool.accessor(role).is_alive)
    if not live_roles:
        return
    try:
        await asyncio.wrap_future(runtime.submit(pool.reap_idle()))
    except Exception:
        log.debug("Pool reap_idle failed", exc_info=True)
    for role in live_roles:
        try:
            await asyncio.wrap_future(runtime.submit(pool.ping_role(role)))
        except Exception:
            # Pool's restart-on-crash policy already records the crash; the
            # next real call respawns the role lazily.
            log.debug("Health ping failed for role=%s", role, exc_info=True)


async def _ticker_loop(pool: WorkerPool, runtime: PoolRuntime, interval_s: float) -> None:
    """Sleep then tick forever. Cancellation is the normal exit path."""
    try:
        while True:
            await asyncio.sleep(interval_s)
            await _tick_once(pool, runtime)
    except asyncio.CancelledError:
        return


def start_health_ticker(
    pool: WorkerPool,
    runtime: PoolRuntime,
    bg_loop: asyncio.AbstractEventLoop,
    *,
    interval_s: float = _TICK_INTERVAL_S,
) -> HealthTickerHandle:
    """Schedule the ticker on *bg_loop* and return a handle for cancellation."""
    handle = HealthTickerHandle()
    handle.future = asyncio.run_coroutine_threadsafe(
        _ticker_loop(pool, runtime, interval_s), bg_loop
    )
    return handle


def stop_health_ticker(handle: HealthTickerHandle, *, timeout: float = 5.0) -> None:
    """Cancel the ticker and wait briefly for it to wind down. Idempotent.

    CancelledError is the expected exit; any other exception is treated
    as best-effort cleanup because the bg-loop owns the task lifetime
    past this point.
    """
    future = handle.future
    if future is None:
        return
    handle.future = None
    future.cancel()
    with contextlib.suppress(TimeoutError, asyncio.CancelledError, Exception):
        future.result(timeout=timeout)


__all__ = [
    "_TICK_INTERVAL_S",
    "HealthTickerHandle",
    "start_health_ticker",
    "stop_health_ticker",
]
