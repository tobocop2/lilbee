"""Unit tests for lilbee.asyncio_loop."""

from __future__ import annotations

import asyncio
import threading

import pytest

from lilbee import asyncio_loop


@pytest.fixture(autouse=True)
def _reset_loop():
    """Ensure each test starts and ends with no background loop running."""
    asyncio_loop.shutdown()
    yield
    asyncio_loop.shutdown()


def test_run_returns_coroutine_result() -> None:
    async def echo(value: int) -> int:
        return value * 2

    assert asyncio_loop.run(echo(7)) == 14


def test_run_propagates_exceptions() -> None:
    async def boom() -> None:
        raise RuntimeError("expected")

    with pytest.raises(RuntimeError, match="expected"):
        asyncio_loop.run(boom())


def test_run_propagates_asyncio_cancelled_error() -> None:
    """run_coroutine_threadsafe wraps asyncio.CancelledError as
    concurrent.futures.CancelledError; asyncio_loop.run must unwrap it so
    callers can still write `except asyncio.CancelledError:`. The original
    message isn't preserved through the asyncio scheduler, but the exception
    class is what callers match on.
    """

    async def cancel_self() -> None:
        raise asyncio.CancelledError("stop")

    with pytest.raises(asyncio.CancelledError):
        asyncio_loop.run(cancel_self())


def test_run_awaits_subtasks_to_completion() -> None:
    marker: list[str] = []

    async def subtask() -> None:
        await asyncio.sleep(0)
        marker.append("ran")

    async def outer() -> None:
        await asyncio.gather(subtask(), subtask())

    asyncio_loop.run(outer())
    assert marker == ["ran", "ran"]


def test_get_loop_reuses_loop_across_calls() -> None:
    loop_a = asyncio_loop.get_loop()
    loop_b = asyncio_loop.get_loop()
    assert loop_a is loop_b
    assert loop_a.is_running()


def test_loop_runs_on_dedicated_daemon_thread() -> None:
    asyncio_loop.get_loop()
    thread = next(t for t in threading.enumerate() if t.name == "lilbee-bg-loop")
    assert thread.daemon is True
    assert thread.is_alive()


def test_shutdown_is_idempotent() -> None:
    asyncio_loop.get_loop()
    asyncio_loop.shutdown()
    asyncio_loop.shutdown()  # second call is a no-op


def test_shutdown_without_start_is_noop() -> None:
    asyncio_loop.shutdown()


def test_atexit_register_called_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated get_loop/shutdown cycles must not pollute the atexit queue.

    Each get_loop that starts a fresh loop would otherwise register another
    shutdown callback. Because shutdown is idempotent the extra calls are
    harmless at exit time, but the accumulation is still a leak.
    """
    import lilbee.asyncio_loop as mod

    # Reset the register-once flag so this test sees a clean register path.
    mod._atexit_registered = False

    registrations: list[object] = []
    monkeypatch.setattr(mod.atexit, "register", lambda fn, *a, **kw: registrations.append(fn))

    mod.get_loop()
    mod.shutdown()
    mod.get_loop()
    mod.shutdown()
    mod.get_loop()
    mod.shutdown()

    assert registrations == [mod.shutdown]


def test_get_loop_restarts_after_shutdown() -> None:
    first = asyncio_loop.get_loop()
    asyncio_loop.shutdown()
    assert first.is_closed()

    second = asyncio_loop.get_loop()
    assert second is not first
    assert second.is_running()


def test_shutdown_cancels_pending_tasks() -> None:
    started = threading.Event()
    was_cancelled = threading.Event()

    async def long_runner() -> None:
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            was_cancelled.set()
            raise

    loop = asyncio_loop.get_loop()
    asyncio.run_coroutine_threadsafe(long_runner(), loop)
    assert started.wait(timeout=2.0)

    asyncio_loop.shutdown()
    assert was_cancelled.is_set()


def test_run_from_multiple_threads() -> None:
    async def identity(n: int) -> int:
        await asyncio.sleep(0)
        return n

    results: dict[int, int] = {}

    def worker(n: int) -> None:
        results[n] = asyncio_loop.run(identity(n))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert results == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}


def test_shutdown_drain_swallows_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Best-effort drain must still stop the loop even if the drain coroutine fails."""
    import concurrent.futures

    loop = asyncio_loop.get_loop()

    def boom(coro: object, _loop: object) -> concurrent.futures.Future[None]:
        # Close the coroutine so it doesn't surface as "never awaited"
        coro.close()  # type: ignore[attr-defined]
        fut: concurrent.futures.Future[None] = concurrent.futures.Future()
        fut.set_exception(RuntimeError("drain failed"))
        return fut

    monkeypatch.setattr("lilbee.asyncio_loop.asyncio.run_coroutine_threadsafe", boom)
    asyncio_loop.shutdown()
    assert loop.is_closed()
