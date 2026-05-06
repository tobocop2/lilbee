"""Unit tests for the Services-owned worker-pool health ticker.

Cadence is monkeypatched to a fast value so the assertions land
deterministically without any real-time waits. The ticker bridges
``lilbee-bg-loop`` to the :class:`PoolRuntime` loop; both loops are real
so the test exercises the cross-loop dispatch path.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from lilbee.providers.worker import health_ticker
from lilbee.providers.worker.health_ticker import (
    HealthTickerHandle,
    _tick_once,
    start_health_ticker,
    stop_health_ticker,
)
from lilbee.providers.worker.pool import PoolRuntime
from lilbee.runtime import asyncio_loop


class _FakeAccessor:
    """Stand-in for the real RoleAccessor with a controllable is_alive."""

    def __init__(self, alive: bool) -> None:
        self.alive = alive

    @property
    def is_alive(self) -> bool:
        return self.alive


class _FakePool:
    """Minimal pool stand-in for ticker tests.

    Records every reap/ping call. ``is_alive`` per-role is injected at
    construction so the test can pin which roles count as 'live' without
    spawning real subprocesses.
    """

    def __init__(self, alive_roles: dict[str, bool]) -> None:
        self._alive = dict(alive_roles)
        self.reap_calls = 0
        self.ping_calls: list[str] = []
        self._lock = threading.Lock()

    @property
    def registered_roles(self) -> tuple[str, ...]:
        return tuple(self._alive)

    def accessor(self, role: str) -> _FakeAccessor:
        return _FakeAccessor(self._alive.get(role, False))

    async def reap_idle(self) -> tuple[str, ...]:
        with self._lock:
            self.reap_calls += 1
        return ()

    async def ping_role(self, role: str, *, timeout: float | None = None) -> None:
        with self._lock:
            self.ping_calls.append(role)


@pytest.fixture()
def real_runtime():
    runtime = PoolRuntime()
    runtime.start()
    yield runtime
    runtime.shutdown(timeout=5.0)


@pytest.fixture()
def bg_loop():
    """Use the project's real bg-loop so the test exercises the actual wiring."""
    loop = asyncio_loop.get_loop()
    yield loop


def test_tick_once_skips_when_no_roles_live(real_runtime) -> None:
    pool = _FakePool({"embed": False, "chat": False})

    async def _drive() -> None:
        await _tick_once(pool, real_runtime)

    asyncio.run(_drive())
    assert pool.reap_calls == 0
    assert pool.ping_calls == []


def test_tick_once_reaps_and_pings_each_live_role(real_runtime) -> None:
    pool = _FakePool({"embed": True, "chat": True, "rerank": False})

    async def _drive() -> None:
        await _tick_once(pool, real_runtime)

    asyncio.run(_drive())
    assert pool.reap_calls == 1
    assert sorted(pool.ping_calls) == ["chat", "embed"]


def test_tick_once_swallows_ping_failures(real_runtime, caplog) -> None:
    """A failing ping must not stop the rest of the tick from completing."""

    class _PartiallyDeadPool(_FakePool):
        async def ping_role(self, role: str, *, timeout: float | None = None) -> None:
            if role == "embed":
                raise RuntimeError("ping blew up")
            await super().ping_role(role, timeout=timeout)

    pool = _PartiallyDeadPool({"embed": True, "chat": True})

    async def _drive() -> None:
        await _tick_once(pool, real_runtime)

    with caplog.at_level("DEBUG", logger="lilbee.providers.worker.health_ticker"):
        asyncio.run(_drive())
    assert pool.reap_calls == 1
    assert "chat" in pool.ping_calls
    assert "embed" not in pool.ping_calls


def test_tick_once_swallows_reap_failures(real_runtime, caplog) -> None:
    """A failing reap must not stop the ping passes."""

    class _BadReapPool(_FakePool):
        async def reap_idle(self) -> tuple[str, ...]:
            raise RuntimeError("reap blew up")

    pool = _BadReapPool({"embed": True})

    async def _drive() -> None:
        await _tick_once(pool, real_runtime)

    with caplog.at_level("DEBUG", logger="lilbee.providers.worker.health_ticker"):
        asyncio.run(_drive())
    assert pool.ping_calls == ["embed"]


def test_start_and_stop_round_trip_with_real_loops(monkeypatch, real_runtime, bg_loop) -> None:
    """End-to-end: ticker schedules ticks on bg-loop, dispatches to runtime.

    Asserts only ``reap_calls`` because reap fires synchronously inside the
    same tick as the ping; checking both with separate ``>=`` thresholds
    introduces a racy snapshot whenever the ticker happens to fire between
    the two reads.
    """
    pool = _FakePool({"embed": True})
    handle = start_health_ticker(pool, real_runtime, bg_loop, interval_s=0.05)
    try:
        # Wait until the ticker has fired at least twice; bound the wait so
        # a stuck ticker fails the test instead of hanging it.
        deadline = time.monotonic() + 5.0
        while pool.reap_calls < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pool.reap_calls >= 2
        # ping_calls must contain at least one "embed" entry (a tick that
        # incremented reap_calls also issued a ping for the live role).
        assert "embed" in pool.ping_calls
    finally:
        stop_health_ticker(handle)
    assert handle.future is None


def test_stop_is_idempotent() -> None:
    handle = HealthTickerHandle()
    stop_health_ticker(handle)
    stop_health_ticker(handle)


def test_default_tick_interval_is_thirty_seconds() -> None:
    """Document the cadence; bumping requires touching this assertion."""
    assert health_ticker._TICK_INTERVAL_S == 30.0


def test_ticker_loop_exits_cleanly_on_cancel(real_runtime, bg_loop) -> None:
    """Cancellation raised inside _ticker_loop returns instead of escaping."""
    pool = _FakePool({"embed": False})
    handle = start_health_ticker(pool, real_runtime, bg_loop, interval_s=10.0)
    # Sleep is the await point; cancel should unwind cleanly.
    stop_health_ticker(handle, timeout=2.0)
    assert handle.future is None


def test_tick_once_propagates_cancellation_during_reap(real_runtime) -> None:
    """A CancelledError raised inside reap_idle must escape, not be swallowed."""

    class _ReapHangsForeverPool(_FakePool):
        async def reap_idle(self) -> tuple[str, ...]:
            await asyncio.sleep(60.0)
            return ()  # pragma: no cover -- never reached; cancellation lands first

    pool = _ReapHangsForeverPool({"embed": True})

    async def _drive() -> None:
        task = asyncio.create_task(_tick_once(pool, real_runtime))
        await asyncio.sleep(0.05)  # let the reap call enter the wait
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_drive())


def test_tick_once_propagates_cancellation_during_ping(real_runtime) -> None:
    """A CancelledError raised inside ping_role must escape, not be swallowed."""

    class _PingHangsForeverPool(_FakePool):
        async def ping_role(self, role: str, *, timeout: float | None = None) -> None:
            await asyncio.sleep(60.0)  # pragma: no cover -- cancellation lands first

    pool = _PingHangsForeverPool({"embed": True})

    async def _drive() -> None:
        task = asyncio.create_task(_tick_once(pool, real_runtime))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_drive())
