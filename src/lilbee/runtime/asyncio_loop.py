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
import signal
import socket
import sys
import threading
from collections.abc import Awaitable, Coroutine, Iterator
from typing import Any, TypeVar

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
            loop.add_reader(reader, reader.recv, _WAKEUP_READ_BYTES)
            previous = signal.set_wakeup_fd(writer.fileno())
            try:
                yield
            finally:
                signal.set_wakeup_fd(previous)
                loop.remove_reader(reader)


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
