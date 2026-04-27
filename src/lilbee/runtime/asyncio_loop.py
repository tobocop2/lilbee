"""Process-lifetime background asyncio loop for TUI workers.

One loop on a daemon thread, used by every @work(thread=True) worker.
CLI one-shots and the server own their own loops — don't route them here.
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
    # Give scheduled close callbacks a chance to run before we stop the loop.
    await asyncio.sleep(0.05)
