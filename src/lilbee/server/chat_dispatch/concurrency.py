"""Process-wide chat-generation gate shared by every chat-generating route."""

from __future__ import annotations

import asyncio
from functools import lru_cache


class ChatBusyError(Exception):
    """Raised when the chat backend is already serving a request."""


@lru_cache(maxsize=1)
def chat_lock() -> asyncio.Lock:
    """Return the process-wide chat lock (created lazily on first call)."""
    return asyncio.Lock()


def acquire_or_raise_busy() -> None:
    """Raise :class:`ChatBusyError` if the lock is held; otherwise no-op.

    Route handlers run on a single event-loop thread, so the held-check
    and the subsequent ``await lock.acquire()`` are atomic from the
    loop's perspective.
    """
    if chat_lock().locked():
        raise ChatBusyError
