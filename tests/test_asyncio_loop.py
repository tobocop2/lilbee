"""Unit tests for lilbee.runtime.asyncio_loop."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import selectors
import signal
import socket
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from lilbee.runtime import asyncio_loop
from lilbee.runtime.lock import ResetRefusedError, _no_sync_running, sync_running
from tests.conftest import read_takes_hooked, reset_held_on_a_thread


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
    """`except asyncio.CancelledError:` at call sites must still match."""

    async def cancel_self() -> None:
        raise asyncio.CancelledError("stop")

    with pytest.raises(asyncio.CancelledError):
        asyncio_loop.run(cancel_self())


def test_run_without_join_leaves_a_stuck_thread_and_no_current_loop() -> None:
    """An interrupt out of the run must not wait on a worker thread that never returns."""
    release, returned = threading.Event(), threading.Event()
    loops: list[asyncio.AbstractEventLoop] = []

    def held() -> None:
        release.wait()
        returned.set()

    async def interrupted() -> None:
        loops.append(asyncio.get_running_loop())
        asyncio.get_running_loop().run_in_executor(None, held)
        raise KeyboardInterrupt

    try:
        with pytest.raises(KeyboardInterrupt):
            asyncio_loop.run_without_join(interrupted())
        assert not returned.is_set()
        assert loops[0].is_closed()
        with pytest.raises(RuntimeError, match="no current event loop"):
            asyncio.get_event_loop_policy().get_event_loop()
    finally:
        release.set()


def test_run_without_join_returns_the_result() -> None:
    async def echo(value: int) -> int:
        return value * 2

    assert asyncio_loop.run_without_join(echo(7)) == 14


class _PressBeforeTheSleep(selectors.DefaultSelector):
    """A selector that takes a Ctrl-C after the loop's last step and before its sleep."""

    def __init__(self) -> None:
        super().__init__()
        self.after_press: asyncio.Future[None] | None = None
        self.woken: list[bool] = []

    def select(self, timeout: float | None = None):
        if timeout is not None:
            return super().select(timeout)
        signal.raise_signal(signal.SIGINT)
        # What is ready now is what ends the sleep.
        ready = super().select(0)
        self.woken.append(bool(ready))
        assert self.after_press is not None
        self.after_press.set_result(None)
        return ready


posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="The proactor loop sets its own wakeup fd"
)


@pytest.fixture
def presses():
    """The signals a handler took that only records, as the first Ctrl-C of a command does."""
    seen: list[int] = []
    previous = signal.signal(signal.SIGINT, lambda signum, _frame: seen.append(signum))
    yield seen
    signal.signal(signal.SIGINT, previous)


def _wakeup_fd() -> int:
    """The wakeup descriptor now set, left in place."""
    fd = signal.set_wakeup_fd(-1)
    signal.set_wakeup_fd(fd)
    return fd


@posix_only
def test_a_signal_just_before_the_loop_sleeps_wakes_it(monkeypatch, presses) -> None:
    """A signal interrupts only a sleep that has started; an earlier one must leave a wakeup."""
    selector = _PressBeforeTheSleep()
    monkeypatch.setattr(asyncio, "new_event_loop", lambda: asyncio.SelectorEventLoop(selector))

    async def sleeps() -> None:
        selector.after_press = asyncio.get_running_loop().create_future()
        await selector.after_press

    asyncio_loop.run_without_join(sleeps())

    assert presses == [signal.SIGINT]
    assert selector.woken == [True]


@posix_only
def test_the_wakeup_is_read_so_the_loop_sleeps_again(monkeypatch, presses) -> None:
    selector = selectors.DefaultSelector()
    monkeypatch.setattr(asyncio, "new_event_loop", lambda: asyncio.SelectorEventLoop(selector))

    async def press() -> list[bool]:
        signal.raise_signal(signal.SIGINT)
        woken = [bool(selector.select(0))]
        # One turn finds the wakeup, the next one reads it.
        for _ in range(2):
            await asyncio.sleep(0)
        return [*woken, bool(selector.select(0))]

    assert asyncio_loop.run_without_join(press()) == [True, False]


@posix_only
@pytest.mark.parametrize("error", [None, KeyboardInterrupt])
def test_run_without_join_gives_the_wakeup_fd_back(error) -> None:
    held: list[int] = []
    reader, writer = socket.socketpair()

    async def hold() -> None:
        held.append(_wakeup_fd())
        if error is not None:
            raise error

    with reader, writer:
        writer.setblocking(False)
        before = signal.set_wakeup_fd(writer.fileno())
        try:
            with contextlib.suppress(KeyboardInterrupt):
                asyncio_loop.run_without_join(hold())
        finally:
            after = signal.set_wakeup_fd(before)
        assert held[0] not in (-1, writer.fileno())
        assert after == writer.fileno()


def test_run_without_join_off_the_main_thread_sets_no_wakeup_fd() -> None:
    """Only the main thread can set one."""

    async def echo() -> int:
        return 14

    before = _wakeup_fd()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(asyncio_loop.run_without_join, echo()).result() == 14
    assert _wakeup_fd() == before


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
    """Repeated get_loop/shutdown cycles must not stack atexit callbacks."""
    import lilbee.runtime.asyncio_loop as mod

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


def test_shutdown_does_not_wait_for_the_reset_a_sync_waits_for(tmp_path: Path) -> None:
    waiting = threading.Event()

    @contextmanager
    def _acquire(_lock, _timeout):
        waiting.set()
        yield

    async def _sync() -> None:
        async with sync_running(tmp_path):
            pass

    with (
        reset_held_on_a_thread(tmp_path),
        read_takes_hooked(_acquire),
    ):
        sync = asyncio.run_coroutine_threadsafe(_sync(), asyncio_loop.get_loop())
        assert waiting.wait(5)
        asyncio_loop.shutdown(drain_budget_s=2.0)
        sync_ended = sync.cancelled()
        with (
            pytest.raises(ResetRefusedError, match="is running on this library"),
            _no_sync_running(tmp_path),
        ):
            pass

    assert sync_ended, "the shutdown gave up on a sync that waited for the reset"


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

    monkeypatch.setattr("lilbee.runtime.asyncio_loop.asyncio.run_coroutine_threadsafe", boom)
    asyncio_loop.shutdown()
    assert loop.is_closed()
