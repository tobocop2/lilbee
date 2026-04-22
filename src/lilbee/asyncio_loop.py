"""Process-lifetime background asyncio loop for TUI workers.

Windows ProactorEventLoop leaks subprocess transports when loops are opened
and closed per call. A loop kept alive for the life of the TUI app lets
transport close callbacks run on a live loop, avoiding "I/O operation on
closed pipe" during interpreter shutdown.

Scope: TUI @work(thread=True) workers. Not for CLI one-shots (use
asyncio.run()) or the server (owns its own loop). Integration test fixtures
use run_until_complete on a session-scoped loop instead.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import contextlib
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_lock = threading.Lock()


def get_loop() -> asyncio.AbstractEventLoop:
    """Return the background loop, starting it on a daemon thread if needed."""
    global _loop, _thread
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
        atexit.register(shutdown)
        return loop


def run(coro: Coroutine[Any, Any, T]) -> T:
    """Submit *coro* to the background loop from any thread; block for result.

    Drop-in replacement for asyncio.run() in TUI worker contexts. Exceptions
    raised inside *coro* propagate unchanged, including asyncio.CancelledError
    (run_coroutine_threadsafe would otherwise wrap it as
    concurrent.futures.CancelledError, breaking any `except asyncio.CancelledError`
    handler at the call site).
    """
    loop = get_loop()
    try:
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
    except concurrent.futures.CancelledError as exc:
        raise asyncio.CancelledError(*exc.args) from None


def shutdown() -> None:
    """Cancel pending tasks, stop the loop, join the thread. Idempotent."""
    global _loop, _thread
    with _lock:
        loop, _loop = _loop, None
        thread, _thread = _thread, None
    if loop is None or loop.is_closed():
        return
    # Best-effort drain; always stop the loop even if drain raised.
    with contextlib.suppress(Exception):
        asyncio.run_coroutine_threadsafe(_drain(loop), loop).result(timeout=10.0)
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
    # Let subprocess transport close callbacks flush before the loop stops.
    await asyncio.sleep(0.05)
