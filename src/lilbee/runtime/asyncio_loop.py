"""Process-lifetime background asyncio loop for TUI workers.

One loop on a daemon thread. Every @work(thread=True) worker and every provider
backend call (chat, stream, embed, rerank) runs its coroutines here, from the
TUI, the CLI and the server alike.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import contextlib
import select
import signal
import socket
import sys
import threading
from collections.abc import Awaitable, Coroutine, Iterator
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from lilbee.runtime.cancellation import SignalFlag

T = TypeVar("T")

# concurrent.futures raises this exact RuntimeError message when submitting to
# a shutdown executor (Python 3.11+). There is no dedicated exception class to
# catch, so callers have to string-match the message. The constant + helper
# below give every caller one place to depend on instead of duplicating the
# string + isinstance check inline.
_EXECUTOR_SHUTDOWN_MSG = "cannot schedule new futures after shutdown"
# Seconds the pending tasks get to unwind when the loop shuts down.
_DRAIN_BUDGET_S = 10.0
# Python writes one byte for each signal; what a read leaves wakes the loop again.
_WAKEUP_READ_BYTES = 4096


def is_executor_shutdown(exc: BaseException) -> bool:
    """True if ``exc`` is the concurrent.futures shutdown-race ``RuntimeError``."""
    return isinstance(exc, RuntimeError) and _EXECUTOR_SHUTDOWN_MSG in str(exc)


_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_lock = threading.Lock()
_atexit_registered = False


def get_loop() -> asyncio.AbstractEventLoop:
    """Return the background loop, starting it on a daemon thread if needed."""
    global _loop, _thread, _atexit_registered
    with _lock:
        if _loop is not None and not _loop.is_closed():
            return _loop
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=loop.run_forever,
            name="lilbee-bg-loop",
            daemon=True,
        )
        thread.start()
        _loop = loop
        _thread = thread
        if not _atexit_registered:
            # Register once per process; shutdown is idempotent, so restarting
            # the loop later doesn't need a second registration.
            atexit.register(shutdown)
            _atexit_registered = True
        return loop


def run(coro: Coroutine[Any, Any, T]) -> T:
    """Submit *coro* to the background loop from any thread; block for result.

    Exceptions raised inside *coro* propagate unchanged, including
    asyncio.CancelledError.
    """
    loop = get_loop()
    try:
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
    except concurrent.futures.CancelledError as exc:
        # run_coroutine_threadsafe re-raises cancellation as the concurrent
        # flavour; rewrap so `except asyncio.CancelledError` still matches.
        raise asyncio.CancelledError(*exc.args) from None


@contextlib.contextmanager
def _signals_wake(loop: asyncio.AbstractEventLoop) -> Iterator[None]:
    """Make each signal wake *loop*, which sleeps until a descriptor is ready.

    A signal interrupts only a sleep that has started. One that lands just before
    leaves its Python handler unrun until the loop wakes for something else. Python
    takes a wakeup descriptor on the main thread only, and the proactor loop of
    Windows sets its own.
    """
    if sys.platform == "win32" or threading.current_thread() is not threading.main_thread():
        yield
    else:  # pragma: posix only - the proactor loop of Windows has no add_reader
        reader, writer = socket.socketpair()
        with reader, writer:
            reader.setblocking(False)
            writer.setblocking(False)
            loop.add_reader(reader, _read_wakeups, reader)
            previous = signal.set_wakeup_fd(writer.fileno())
            try:
                yield
            finally:
                signal.set_wakeup_fd(previous)
                loop.remove_reader(reader)


def _read_wakeups(reader: socket.socket) -> None:
    """Read the bytes the wakeups left in *reader*; an empty socket has nothing left to read.

    A ready socket can be empty by the time its reader runs, so an empty read is the
    end of the read, not an error.
    """
    with contextlib.suppress(BlockingIOError):
        reader.recv(_WAKEUP_READ_BYTES)


def _send_wakeup(writer: socket.socket) -> None:
    """Wake a wait on *writer*'s pair; a full or closed socket has a wakeup or no wait."""
    with contextlib.suppress(OSError):
        writer.send(b"\0")


def _wait_for(done: concurrent.futures.Future[Any]) -> None:
    """Wait on this thread until *done* ends, while its signal handlers run at once.

    The wait sleeps on a socket that both the end of *done* and Python's signal
    wakeup write to. A lock wait does not wake for a signal that another thread
    took, and macOS can give a process signal to any thread. A handler that raises
    ends the wait with its exception.
    """
    reader, writer = socket.socketpair()
    with reader, writer:
        reader.setblocking(False)
        writer.setblocking(False)
        done.add_done_callback(lambda _done: _send_wakeup(writer))
        previous = signal.set_wakeup_fd(writer.fileno())
        try:
            while not done.done():
                select.select([reader], [], [])
                _read_wakeups(reader)
        finally:
            signal.set_wakeup_fd(previous)


def _run_with_sigint_blocked(
    loop: asyncio.AbstractEventLoop,
    task: asyncio.Future[T],
    done: concurrent.futures.Future[T],
) -> None:
    """Run *task* on *loop* on this thread, where SIGINT is blocked, and put its end in *done*.

    The block holds for every thread this one starts, so no Ctrl-C handler runs on
    them and no signal lands inside the loop's own code.
    """
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    asyncio.set_event_loop(loop)
    try:
        done.set_result(loop.run_until_complete(task))
    except BaseException as exc:
        done.set_exception(exc)
    finally:
        asyncio.set_event_loop(None)


def _cancel_all_tasks(loop: asyncio.AbstractEventLoop) -> None:
    for task in asyncio.all_tasks(loop):
        task.cancel()


def run_interruptible(coro: Awaitable[T], cancel: SignalFlag) -> T:
    """Run *coro* on its own loop on a worker thread; this thread only waits for the run.

    Python runs a signal handler only on the main thread, so a Ctrl-C never lands
    in the loop's code: a raise there could drop a task's wakeup. A stop of the run
    through *cancel* hands the cancel of the run's task to the loop. A handler that raises during
    the run ends only the wait. The run's tasks then get a cancel, and the run is
    left to the process exit, so a run stuck in a sync step or a native call does
    not keep the command. Off the main thread, and on Windows, *coro* runs on this
    thread as ``run_without_join`` runs it.
    """
    if sys.platform == "win32" or threading.current_thread() is not threading.main_thread():
        return run_without_join(coro)
    loop = asyncio.new_event_loop()
    # The task takes this thread's context, as a run on this thread would.
    task = asyncio.ensure_future(coro, loop=loop)
    done: concurrent.futures.Future[T] = concurrent.futures.Future()
    cancel.bind_run(lambda: loop.call_soon_threadsafe(task.cancel))
    runner = threading.Thread(
        target=_run_with_sigint_blocked,
        args=(loop, task, done),
        name="lilbee-run",
        daemon=True,
    )
    try:
        runner.start()
        _wait_for(done)
        # The run has ended, so the thread only returns: the scope sees no thread of it.
        runner.join()
    except BaseException:
        loop.call_soon_threadsafe(_cancel_all_tasks, loop)
        raise
    finally:
        cancel.unbind_run()
    loop.close()
    return done.result()


def run_without_join(coro: Awaitable[T]) -> T:
    """Run *coro* on its own loop on this thread, then close the loop.

    ``asyncio.run`` joins the threads of ``asyncio.to_thread`` as it closes, so a
    Ctrl-C would wait there for a stuck one. A signal wakes the loop, so its handler
    runs at once. Tasks the run leaves are cancelled and awaited, as ``asyncio.run``
    does, so their own cleanup runs (the TUI's driver gives the terminal back).
    """
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        with _signals_wake(loop):
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.run_until_complete(_drain(loop))
    finally:
        loop.close()
        # A closed loop left current poisons a later get_event_loop on this thread.
        asyncio.set_event_loop(None)


class _MainRun:
    """The ``run_main`` on the main thread: its main task, and the exit held for the run's end."""

    def __init__(self) -> None:
        self.task: asyncio.Task[Any] | None = None
        self.held: SystemExit | None = None

    def run(self, main: Coroutine[Any, Any, T]) -> T:
        if threading.current_thread() is not threading.main_thread():
            return asyncio.run(main)
        with asyncio.Runner() as runner:
            try:
                return runner.run(self._as_main(main))
            finally:
                # Cleared first: from here on a signal raises its exit at once, so
                # one during the runner's close still cuts its join of the workers.
                self.task = None
                held, self.held = self.held, None
                if held is not None:
                    raise held

    async def _as_main(self, main: Coroutine[Any, Any, T]) -> T:
        self.task = asyncio.current_task()
        return await main

    def hold(self, stop: SystemExit) -> bool:
        if self.task is None or not self.task.done():
            return False
        self.held = stop
        return True


_main_run = _MainRun()


def run_main(main: Coroutine[Any, Any, T]) -> T:
    """Run *main* as ``asyncio.run`` does; an exit that ``hold_exit`` kept is raised after it.

    A finished *main* queues the stop of its run. An exit raised by a signal handler
    before that stop runs leaves it queued, the shutdown of the runner ends on it, and
    its RuntimeError replaces the exit. So on the main thread, the handler asks
    ``hold_exit`` first, and the exit leaves once the run of *main* has returned.
    """
    return _main_run.run(main)


def hold_exit(stop: SystemExit) -> bool:
    """Keep *stop* for the end of the run when ``run_main``'s main task is done; True when kept."""
    return _main_run.hold(stop)


def shutdown(drain_budget_s: float = _DRAIN_BUDGET_S) -> None:
    """Cancel pending tasks and wait *drain_budget_s* for them, then stop the loop. Idempotent."""
    global _loop, _thread
    with _lock:
        loop, _loop = _loop, None
        thread, _thread = _thread, None
    if loop is None or loop.is_closed():
        return
    # Best-effort drain; always stop the loop even if drain raised.
    with contextlib.suppress(Exception):
        asyncio.run_coroutine_threadsafe(_drain(loop), loop).result(timeout=drain_budget_s)
    loop.call_soon_threadsafe(loop.stop)
    if thread is not None:
        thread.join(timeout=10.0)
    loop.close()


async def _drain(loop: asyncio.AbstractEventLoop) -> None:
    pending = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    # Give scheduled close callbacks a chance to run before we stop the loop.
    await asyncio.sleep(0.05)
