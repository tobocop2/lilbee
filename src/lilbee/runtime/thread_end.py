"""Wait for work on a thread through a cancel of the caller."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Awaitable, Callable
from typing import TypeVar

import anyio
import anyio.lowlevel

from lilbee.runtime.cancellation import TaskCancelledError

_Result = TypeVar("_Result")


async def run_to_thread_end(
    start: Callable[[], Awaitable[_Result]], cancel: threading.Event
) -> _Result:
    """Await *start()*, which runs work on a thread, and return only when that thread has ended.

    A caller that is already cancelled starts nothing. A cancel that arrives during the
    work sets *cancel* at once and is held until the thread ends. The cancel is then
    raised if the work stopped on it with ``TaskCancelledError``. Otherwise the outcome
    of the work stands, and ``cancel.is_set()`` says that a cancel arrived.
    """
    await anyio.lowlevel.checkpoint_if_cancelled()
    running = asyncio.ensure_future(start())
    try:
        await asyncio.wait([running])
    except asyncio.CancelledError:
        cancel.set()
        # Shielded, because a cancelled anyio scope cancels its task again at every wait.
        with anyio.CancelScope(shield=True):
            while not running.done():
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.wait([running])
        if isinstance(running.exception(), TaskCancelledError):
            raise
    return running.result()
