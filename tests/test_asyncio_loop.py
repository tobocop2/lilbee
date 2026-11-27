"""Unit tests for lilbee.runtime.asyncio_loop."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import contextvars
import gc
import inspect
import os
import selectors
import signal
import socket
import sys
import threading
import time
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from lilbee.app import services as services_mod
from lilbee.runtime import asyncio_loop
from lilbee.runtime.cancellation import SignalFlag, sigint_cancel
from lilbee.runtime.lock import ResetRefusedError, _no_sync_running, sync_running
from tests._signals import press_ctrl_c_in_a_run
from tests._windows_loop import windows_proactor_loop
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


def test_run_without_join_cancels_the_tasks_a_run_leaves_so_their_cleanup_runs() -> None:
    """A run that ends early leaves the cleanup of its other tasks to the cancel, as asyncio.run."""
    cleaned: list[str] = []

    async def held() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.append("held")

    async def interrupted() -> None:
        asyncio.get_running_loop().create_task(held())
        await asyncio.sleep(0)
        raise SystemExit(143)

    with pytest.raises(SystemExit):
        asyncio_loop.run_without_join(interrupted())
    assert cleaned == ["held"]


def test_run_main_returns_the_result() -> None:
    async def echo(value: int) -> int:
        return value * 2

    assert asyncio_loop.run_main(echo(7)) == 14


def test_no_exit_is_held_while_the_main_task_runs() -> None:
    """A signal before the main task is done leaves no stop queued, so its exit raises at once."""

    async def asks() -> bool:
        return asyncio_loop.hold_exit(SystemExit(143))

    assert asyncio_loop.run_main(asks()) is False


def _asks_once_done(kept: list[bool], status: int) -> asyncio.Task[object]:
    """The running task, asking for an exit in a done callback queued after the run's stop."""
    task = asyncio.current_task()
    assert task is not None
    task.add_done_callback(lambda _: kept.append(asyncio_loop.hold_exit(SystemExit(status))))
    return task


def test_an_exit_asked_for_after_the_main_task_finished_leaves_when_the_run_returns() -> None:
    """The run's stop is still queued, so the exit waits; in the close it raises at once."""
    in_run: list[bool] = []
    in_close: list[bool] = []

    async def left() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            in_close.append(asyncio_loop.hold_exit(SystemExit(129)))

    async def finishes() -> str:
        _asks_once_done(in_run, 143)
        asyncio.get_running_loop().create_task(left())
        await asyncio.sleep(0)
        return "done"

    with pytest.raises(SystemExit) as raised:
        asyncio_loop.run_main(finishes())
    assert raised.value.code == 143
    assert in_run == [True]
    assert in_close == [False]


@contextmanager
def _counted_loops() -> Iterator[list[asyncio.AbstractEventLoop]]:
    """Record every loop the library builds through asyncio's one factory."""
    built: list[asyncio.AbstractEventLoop] = []
    real = asyncio.events.new_event_loop

    def counted() -> asyncio.AbstractEventLoop:
        built.append(real())
        return built[-1]

    asyncio.events.new_event_loop = counted
    asyncio.new_event_loop = counted
    try:
        yield built
    finally:
        asyncio.events.new_event_loop = real
        asyncio.new_event_loop = real


_NESTED_RUNS = [
    pytest.param(asyncio_loop.run_main, id="run_main"),
    pytest.param(asyncio_loop.run_without_join, id="run_without_join"),
]


@pytest.mark.parametrize("nested_run", _NESTED_RUNS)
def test_a_nested_run_is_refused_before_a_loop_is_built(nested_run) -> None:
    async def inner() -> int:
        return 1

    refused: list[object] = []

    async def outer() -> None:
        coro = inner()
        with (
            _counted_loops() as built,
            pytest.raises(RuntimeError, match="inside a running event loop"),
        ):
            nested_run(coro)
        refused.extend([len(built), inspect.getcoroutinestate(coro)])

    asyncio_loop.run_main(outer())
    assert refused == [0, inspect.CORO_CLOSED]


def test_a_nested_run_of_a_future_like_awaitable_is_refused_untouched() -> None:
    class Waits:
        def __await__(self):
            yield from ()

    async def outer() -> None:
        with pytest.raises(RuntimeError, match="inside a running event loop"):
            asyncio_loop.run_without_join(Waits())

    asyncio_loop.run_main(outer())


@pytest.mark.parametrize("nested_run", _NESTED_RUNS)
def test_a_nested_refusal_keeps_the_outer_run_and_its_held_exit(nested_run) -> None:
    """The refusal leaves the outer run's main task recorded, so its exit is still held."""
    outer_task: list[asyncio.Task[object] | None] = []

    async def inner() -> None:
        raise AssertionError("the refused run must not start")

    async def outer() -> int:
        coro = inner()
        with pytest.raises(RuntimeError):
            nested_run(coro)
        outer_task.append(asyncio_loop._main_run.task)
        asyncio.get_running_loop().call_soon(_hold_or_raise)
        await asyncio.sleep(0)
        return 7

    with pytest.raises(SystemExit) as raised:
        asyncio_loop.run_main(outer())
    assert raised.value.code == 143
    assert outer_task[0] is not None


def test_repeated_nested_refusals_leave_the_outer_run_and_a_later_run_clean() -> None:
    async def inner() -> int:
        return 1

    async def outer() -> int:
        for nested_run in (asyncio_loop.run_main, asyncio_loop.run_without_join) * 2:
            with pytest.raises(RuntimeError, match="inside a running event loop"):
                nested_run(inner())
        return 7

    assert asyncio_loop.run_main(outer()) == 7
    assert asyncio_loop.run_without_join(inner()) == 1
    assert asyncio_loop.run_main(inner()) == 1


def _hold_or_raise() -> None:
    """The signal handler's contract: raise the exit unless the run's end keeps it."""
    stop = SystemExit(143)
    if not asyncio_loop.hold_exit(stop):
        raise stop


def test_run_without_join_raises_an_exit_held_after_its_main_task_is_done() -> None:
    """A handler after the main task ends keeps its exit for the run's end, as run_main does."""

    async def finishes() -> str:
        asyncio.get_running_loop().call_soon(_hold_or_raise)
        return "done"

    with pytest.raises(SystemExit) as raised:
        asyncio_loop.run_without_join(finishes())
    assert raised.value.code == 143
    assert asyncio_loop._main_run.task is None
    assert asyncio_loop._main_run.held is None


def test_run_without_join_off_the_main_thread_records_no_main_task() -> None:
    """A signal runs on the main thread only, so a run on another thread holds no exit."""
    seen: list[object] = []

    async def finishes() -> str:
        seen.append(asyncio_loop._main_run.task)
        return "done"

    worker = threading.Thread(target=lambda: asyncio_loop.run_without_join(finishes()))
    worker.start()
    worker.join(timeout=10)
    assert seen == [None]


class _Woken(Exception):
    """Raised by the test's signal handler, to end the run it interrupts."""


async def _awaits_the_worker(worker: Callable[[], None]) -> None:
    """The main task waits for the worker, so the loop sleeps in its select."""
    await asyncio.to_thread(worker)


async def _leaves_the_worker_to_the_close(worker: Callable[[], None]) -> None:
    """The main task ends at once, so the runner's close joins the running worker."""
    asyncio.get_running_loop().run_in_executor(None, worker)


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no pthread_kill")
@pytest.mark.parametrize(
    "main_task",
    [
        pytest.param(_awaits_the_worker, id="while_the_loop_sleeps"),
        pytest.param(_leaves_the_worker_to_the_close, id="during_the_close_of_the_runner"),
    ],
)
def test_run_main_wakes_for_a_signal_that_another_thread_took(main_task) -> None:
    """The kernel can give a process signal to a worker thread; the wait must still end.

    Python runs the handler on the main thread only, and a sleep in the selector does
    not end for a signal that another thread took. The wakeup descriptor ends it, and
    it outlives the runner's close, which joins a worker that still runs: without it a
    second Ctrl-C or SIGTERM waits for the worker's end.
    """
    sent: list[float] = []
    handled: list[float] = []

    def handler(signum: int, frame: object) -> None:
        handled.append(time.monotonic())
        raise _Woken

    def worker() -> None:
        time.sleep(0.3)  # the loop sleeps, or the close joins this thread, by now
        sent.append(time.monotonic())
        # pthread_kill sends to this thread, as the kernel can for a process signal.
        signal.pthread_kill(threading.get_ident(), signal.SIGUSR1)
        time.sleep(2)  # the worker outlives the signal, so nothing else wakes the loop

    previous = signal.signal(signal.SIGUSR1, handler)
    try:
        with pytest.raises(_Woken):
            asyncio_loop.run_main(main_task(worker))
    finally:
        signal.signal(signal.SIGUSR1, previous)
    assert handled[0] - sent[0] < 1.0


def test_run_main_closes_its_loop_when_the_wakeup_cannot_start(monkeypatch) -> None:
    """A failed wakeup setup leaves no open loop behind."""
    loops: list[asyncio.AbstractEventLoop] = []

    def no_wakeup(loop: asyncio.AbstractEventLoop):
        loops.append(loop)
        raise OSError("too many open files")

    async def never_runs() -> None:
        raise AssertionError("the main task must not start")

    monkeypatch.setattr(asyncio_loop, "_signals_wake", no_wakeup)
    main = never_runs()
    with pytest.raises(OSError, match="too many open files"):
        asyncio_loop.run_main(main)
    main.close()

    assert len(loops) == 1
    assert loops[0].is_closed()


def test_run_main_off_the_main_thread_holds_no_exit() -> None:
    """A signal runs on the main thread only, so a run on another thread is a plain asyncio.run."""
    kept: list[bool] = []
    results: list[str] = []

    async def finishes() -> str:
        _asks_once_done(kept, 143)
        return "done"

    worker = threading.Thread(target=lambda: results.append(asyncio_loop.run_main(finishes())))
    worker.start()
    worker.join(timeout=10)
    assert kept == [False]
    assert results == ["done"]


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


@posix_only
def test_an_empty_wakeup_socket_ends_the_read_without_an_error(monkeypatch) -> None:
    """A ready socket can be empty when its reader runs; the read must not raise."""
    readers: list[tuple[Any, tuple[Any, ...]]] = []
    real_add_reader = asyncio.SelectorEventLoop.add_reader

    def add_reader(self, fd, callback, *args):
        readers.append((callback, args))
        return real_add_reader(self, fd, callback, *args)

    monkeypatch.setattr(asyncio.SelectorEventLoop, "add_reader", add_reader)

    async def read_the_empty_socket() -> None:
        callback, args = readers[-1]
        callback(*args)

    asyncio_loop.run_without_join(read_the_empty_socket())
    assert len(readers) == 1


@contextmanager
def _sigint_handler(handler: Any) -> Iterator[None]:
    previous = signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def _join_the_run_thread() -> None:
    for thread in threading.enumerate():
        if thread.name == "lilbee-run":
            thread.join(timeout=5)


@posix_only
def test_run_interruptible_runs_on_a_thread_that_blocks_sigint() -> None:
    flag = SignalFlag()

    async def where() -> tuple[threading.Thread, set[int], asyncio.AbstractEventLoop]:
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        return threading.current_thread(), mask, asyncio.get_running_loop()

    before = _wakeup_fd()
    thread, mask, loop = asyncio_loop.run_interruptible(where(), flag)
    assert thread is not threading.main_thread()
    assert signal.SIGINT in mask
    assert signal.SIGINT not in signal.pthread_sigmask(signal.SIG_BLOCK, set())
    assert loop.is_closed()
    assert flag.winding_down
    assert _wakeup_fd() == before


def test_a_run_takes_the_main_threads_context() -> None:
    marker: contextvars.ContextVar[str] = contextvars.ContextVar("marker")
    marker.set("main")

    async def read() -> str:
        return marker.get()

    assert asyncio_loop.run_interruptible(read(), SignalFlag()) == "main"


def test_the_thread_of_a_run_has_ended_when_the_run_returns(monkeypatch) -> None:
    """No scope counts the thread of an ended run as a thread the run left behind."""
    real_set_event_loop = asyncio.set_event_loop

    def slow_unset(loop: asyncio.AbstractEventLoop | None) -> None:
        # The run's thread lingers after it reports the end, so only a join waits for it.
        if loop is None and threading.current_thread() is not threading.main_thread():
            time.sleep(0.3)
        real_set_event_loop(loop)

    monkeypatch.setattr(asyncio, "set_event_loop", slow_unset)

    async def where() -> threading.Thread:
        return threading.current_thread()

    thread = asyncio_loop.run_interruptible(where(), SignalFlag())
    assert not thread.is_alive()


def test_run_interruptible_cancels_and_awaits_the_tasks_a_run_leaves() -> None:
    cleaned: list[str] = []
    tasks: list[asyncio.Task[None]] = []

    async def left() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.append("left")

    async def leaves_a_task() -> str:
        tasks.append(asyncio.get_running_loop().create_task(left()))
        await asyncio.sleep(0)
        return "done"

    assert asyncio_loop.run_interruptible(leaves_a_task(), SignalFlag()) == "done"
    assert cleaned == ["left"]


@posix_only
def test_a_ctrl_c_that_stops_the_run_cancels_its_task() -> None:
    flag = SignalFlag()

    async def waits_forever() -> None:
        # SIGINT is blocked here, so the main thread's wait takes it.
        os.kill(os.getpid(), signal.SIGINT)
        await asyncio.Event().wait()

    def press(_signum: int, _frame: object) -> None:
        flag.set()
        flag.stop_run()

    with _sigint_handler(press), pytest.raises(asyncio.CancelledError):
        asyncio_loop.run_interruptible(waits_forever(), flag)
    assert flag.is_set()
    assert flag.winding_down


def test_a_raising_handler_ends_the_wait_for_a_run_stuck_in_a_sync_step() -> None:
    """The main thread never waits on the stuck loop; the run's tasks get the cancel.

    The loop stays in the sync step until the main thread has queued the cancel, and
    the task then waits on a sleep, so the cancel always finds it pending.
    """
    release = threading.Event()
    cancelled = threading.Event()
    loops: list[asyncio.AbstractEventLoop] = []

    def interrupt(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    async def stuck() -> None:
        loops.append(asyncio.get_running_loop())
        press_ctrl_c_in_a_run()
        release.wait()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    flag = SignalFlag()
    try:
        with _sigint_handler(interrupt), pytest.raises(KeyboardInterrupt):
            asyncio_loop.run_interruptible(stuck(), flag)
        assert not release.is_set()
        assert flag.winding_down
    finally:
        release.set()
    assert cancelled.wait(5)
    _join_the_run_thread()
    loops[0].close()


# On Windows the production proactor loop and the test-wide selector loop; elsewhere this
# platform's loop, and the same run while the platform reads as Windows.
_RUN_PLATFORMS = ["proactor", "selector"] if sys.platform == "win32" else ["native", "win32-name"]


@pytest.fixture(params=_RUN_PLATFORMS)
def run_platform(request, monkeypatch) -> Iterator[None]:
    """Run the test's command loop as each platform runs it."""
    if request.param == "proactor":
        with windows_proactor_loop():
            yield
        return
    if request.param == "win32-name":
        monkeypatch.setattr(sys, "platform", "win32")
    yield


@pytest.fixture
def quits(monkeypatch) -> list[str]:
    """Record the force quits of the Ctrl-C scope instead of ending the test process."""
    seen: list[str] = []
    monkeypatch.setattr(services_mod, "force_quit", lambda: seen.append("scope"))
    monkeypatch.setattr(services_mod, "force_quit_from_handler", lambda: seen.append("handler"))
    return seen


def test_a_ctrl_c_in_a_catch_all_step_stops_the_run_at_its_next_await(run_platform, quits) -> None:
    """The first press cancels the run's task, so a library that swallows it cannot keep the run."""
    swallowed: list[BaseException] = []

    async def library_step(cancel: SignalFlag) -> str:
        try:
            press_ctrl_c_in_a_run(cancel)
        except BaseException as exc:  # a library's catch-all
            swallowed.append(exc)
        await asyncio.sleep(10)
        return "slept out"

    started = time.monotonic()
    with sigint_cancel() as cancel, pytest.raises(asyncio.CancelledError):
        asyncio_loop.run_interruptible(library_step(cancel), cancel)
    assert swallowed == []
    assert time.monotonic() - started < 5
    assert quits == []


def test_a_second_ctrl_c_ends_a_run_stuck_in_a_catch_all_step(run_platform, quits) -> None:
    """The second press raises on the main thread's wait, never inside the stuck step."""
    swallowed: list[BaseException] = []
    release = threading.Event()
    loops: list[asyncio.AbstractEventLoop] = []

    async def stuck(cancel: SignalFlag) -> str:
        loops.append(asyncio.get_running_loop())
        pressed = False
        deadline = time.monotonic() + 10
        while not release.is_set() and time.monotonic() < deadline:
            try:  # a library step that swallows every exception and never awaits
                if not pressed:
                    pressed = True
                    press_ctrl_c_in_a_run(cancel)
                    press_ctrl_c_in_a_run()
                time.sleep(0.01)
            except BaseException as exc:
                swallowed.append(exc)
        return "stuck to the end"

    try:
        with pytest.raises(KeyboardInterrupt), sigint_cancel() as cancel:
            asyncio_loop.run_interruptible(stuck(cancel), cancel)
        assert swallowed == []
        assert quits == ["scope"]
    finally:
        release.set()
        _join_the_run_thread()
        # The run is left to the process exit, so its loop is still open here.
        if not loops[0].is_closed():
            loops[0].close()


@contextmanager
def _two_presses_as_the_start_wakes() -> Iterator[list[str]]:
    """Press Ctrl-C twice where Windows runs the presses of a run's start: in its lock wait.

    The start of a thread waits on a condition until the new thread says it started.
    Here the new thread says so only once the main thread waits, and the presses run
    as the woken wait enters its reacquire of the condition's lock. On Windows a press
    that a thread of the run makes runs there when the main thread wakes into it.
    """
    main = threading.get_ident()
    before = {thread.ident for thread in threading.enumerate()}
    waiting = threading.Event()
    pressed: list[str] = []
    real_set = threading.Event.set
    real_release_save = threading.Condition._release_save
    real_acquire_restore = threading.Condition._acquire_restore

    def set_once_the_main_thread_waits(self: threading.Event) -> None:
        if threading.get_ident() not in before and not pressed:
            waiting.wait(5)
        real_set(self)

    def release_save(self: threading.Condition) -> None:
        real_release_save(self)
        if threading.get_ident() == main:
            real_set(waiting)

    def acquire_restore(self: threading.Condition, state: object) -> None:
        if threading.get_ident() == main and waiting.is_set() and not pressed:
            pressed.append("twice")
            signal.raise_signal(signal.SIGINT)
            signal.raise_signal(signal.SIGINT)
        real_acquire_restore(self, state)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(threading.Event, "set", set_once_the_main_thread_waits)
        patch.setattr(threading.Condition, "_release_save", release_save)
        patch.setattr(threading.Condition, "_acquire_restore", acquire_restore)
        yield pressed


def test_two_ctrl_cs_in_the_start_of_a_run_interrupt_it_from_its_wait(
    run_platform, quits, monkeypatch
) -> None:
    """A raise in the start's lock wait would leave its lock released: the wait raises instead."""
    loops: list[asyncio.AbstractEventLoop] = []
    real_new_event_loop = asyncio.new_event_loop

    def new_event_loop() -> asyncio.AbstractEventLoop:
        loops.append(real_new_event_loop())
        return loops[-1]

    async def waits_forever() -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(asyncio, "new_event_loop", new_event_loop)
    try:
        with (
            pytest.raises(KeyboardInterrupt) as raised,
            sigint_cancel() as cancel,
            _two_presses_as_the_start_wakes() as pressed,
        ):
            asyncio_loop.run_interruptible(waits_forever(), cancel)
        assert pressed == ["twice"]
        assert raised.traceback[-1].name == "_wait_for"
        assert cancel.is_set()
        assert cancel.interrupted
        assert quits == ["scope"]
    finally:
        _join_the_run_thread()
        for loop in loops:
            if not loop.is_closed():
                loop.close()


def test_run_interruptible_off_the_main_thread_runs_on_that_thread() -> None:
    async def where() -> threading.Thread:
        return threading.current_thread()

    def run() -> tuple[threading.Thread, threading.Thread]:
        return asyncio_loop.run_interruptible(where(), SignalFlag()), threading.current_thread()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        ran_on, caller = pool.submit(run).result()
    assert ran_on is caller


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


class _WakeupTakingLoop(asyncio.SelectorEventLoop):
    """A loop that, made on the main thread, takes the signal wakeup, as the proactor loop does.

    The Windows proactor loop sets its self-pipe as the wakeup descriptor when it is
    made on the main thread, and clears it only when it is closed on the main thread.
    """

    def __init__(self) -> None:
        super().__init__()
        self._wakeup_reader, self._wakeup_writer = socket.socketpair()
        self._wakeup_writer.setblocking(False)
        if threading.current_thread() is threading.main_thread():
            signal.set_wakeup_fd(self._wakeup_writer.fileno())

    def close(self) -> None:
        if threading.current_thread() is threading.main_thread():
            signal.set_wakeup_fd(-1)
        self._wakeup_reader.close()
        self._wakeup_writer.close()
        super().close()


class _WakeupTakingPolicy(asyncio.DefaultEventLoopPolicy):
    def new_event_loop(self) -> asyncio.AbstractEventLoop:
        return _WakeupTakingLoop()


@pytest.fixture
def wakeup_taking_loops() -> Iterator[None]:
    """Make every new loop take the signal wakeup on the main thread; start with no wakeup set."""
    previous_policy = asyncio.get_event_loop_policy()
    previous_fd = signal.set_wakeup_fd(-1)
    asyncio.set_event_loop_policy(_WakeupTakingPolicy())
    try:
        yield
    finally:
        asyncio_loop.shutdown()
        asyncio.set_event_loop_policy(previous_policy)
        signal.set_wakeup_fd(previous_fd)


def test_the_background_loop_leaves_the_main_threads_signal_wakeup_alone(
    wakeup_taking_loops: None,
) -> None:
    # A loop closed off the main thread would leave its closed self-pipe as the wakeup.
    asyncio_loop.get_loop()
    assert signal.set_wakeup_fd(-1) == -1
    asyncio_loop.shutdown()
    assert signal.set_wakeup_fd(-1) == -1


class _NoLoopPolicy(asyncio.DefaultEventLoopPolicy):
    def new_event_loop(self) -> asyncio.AbstractEventLoop:
        raise OSError("no loop")


def test_a_background_loop_that_cannot_be_made_raises_from_get_loop() -> None:
    previous = asyncio.get_event_loop_policy()
    asyncio.set_event_loop_policy(_NoLoopPolicy())
    try:
        with pytest.raises(OSError, match="no loop"):
            asyncio_loop.get_loop()
    finally:
        asyncio.set_event_loop_policy(previous)
    assert asyncio_loop.get_loop().is_running()


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


def _block_the_loop_thread() -> tuple[asyncio.AbstractEventLoop, threading.Thread, threading.Event]:
    """Start a coroutine that holds the loop thread without yielding until the event is set."""
    started = threading.Event()
    release = threading.Event()

    async def blocker() -> None:
        started.set()
        release.wait(30)

    loop = asyncio_loop.get_loop()
    thread = next(t for t in threading.enumerate() if t.name == "lilbee-bg-loop")
    asyncio.run_coroutine_threadsafe(blocker(), loop)
    assert started.wait(timeout=2.0)
    return loop, thread, release


@contextmanager
def _no_runtime_warning() -> Iterator[None]:
    """Fail when the block, or a garbage collection after it, warns, as "never awaited" does."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
        gc.collect()
    assert [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)] == []


def test_shutdown_with_a_stuck_loop_thread_stays_within_its_budget() -> None:
    with _no_runtime_warning():
        loop, thread, release = _block_the_loop_thread()
        try:
            started = time.monotonic()
            asyncio_loop.shutdown(drain_budget_s=0.5)
            elapsed = time.monotonic() - started
            assert elapsed < 1.5, "the drain and the join each took a budget"
            assert thread.is_alive()
            assert thread.daemon
            assert not loop.is_closed()
        finally:
            release.set()
        thread.join(5.0)
        assert not thread.is_alive()
        assert loop.is_closed(), "the loop thread closes the loop it stopped"


def test_get_loop_after_a_shutdown_that_timed_out_gives_a_working_loop() -> None:
    with _no_runtime_warning():
        old, thread, release = _block_the_loop_thread()
        try:
            asyncio_loop.shutdown(drain_budget_s=0.3)
            new = asyncio_loop.get_loop()
            assert new is not old

            async def answer() -> int:
                return 7

            assert asyncio_loop.run(answer()) == 7
        finally:
            release.set()
            thread.join(5.0)


def test_a_task_that_swallows_its_cancel_does_not_keep_the_old_loop_running() -> None:
    started = threading.Event()
    swallowing = threading.Event()
    stop = threading.Event()

    async def swallower() -> None:
        started.set()
        while not stop.is_set():
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                swallowing.set()

    with _no_runtime_warning():
        loop = asyncio_loop.get_loop()
        thread = next(t for t in threading.enumerate() if t.name == "lilbee-bg-loop")
        asyncio.run_coroutine_threadsafe(swallower(), loop)
        try:
            assert started.wait(timeout=2.0)
            asyncio_loop.shutdown(drain_budget_s=0.3)
            assert swallowing.wait(timeout=2.0), "the drain never cancelled the task"
            thread.join(5.0)
            assert not thread.is_alive(), "the old loop runs on beside the new one"
            assert loop.is_closed()
        finally:
            stop.set()
            thread.join(5.0)


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
