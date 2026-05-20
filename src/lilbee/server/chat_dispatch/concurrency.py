"""Process-wide chat-generation gate shared by every chat-generating route."""

from __future__ import annotations

import asyncio
from functools import lru_cache

# Default upper bound for waiting on the in-flight chat lock before
# surfacing a real "busy" response. Tuned to absorb the typical opencode
# retry storm (one user turn can take 30-60s on slow local models) while
# still bounding the worst case for a genuinely-stuck request.
DEFAULT_BUSY_WAIT_S = 60.0


class ChatBusyError(Exception):
    """Raised when the chat backend is already serving a request after a wait."""


@lru_cache(maxsize=1)
def chat_lock() -> asyncio.Lock:
    """Return the process-wide chat lock (created lazily on first call)."""
    return asyncio.Lock()


async def acquire_chat_lock_or_busy(timeout: float | None = None) -> None:
    """Acquire the chat lock, waiting up to *timeout* seconds.

    Raises :class:`ChatBusyError` only when the wait times out (the previous
    request is still streaming after *timeout*). Returns with the lock held;
    callers must ``chat_lock().release()`` in a ``finally``.

    Holding the request server-side avoids the 429 storm pattern: opencode's
    stream-timeout retries enqueue on the lock instead of bouncing back
    immediately, and a genuinely-stuck request still surfaces a real 429
    once *timeout* elapses. *timeout* defaults to :data:`DEFAULT_BUSY_WAIT_S`
    at call time so tests can monkeypatch the module constant.
    """
    effective_timeout = DEFAULT_BUSY_WAIT_S if timeout is None else timeout
    try:
        await asyncio.wait_for(chat_lock().acquire(), timeout=effective_timeout)
    except TimeoutError as exc:
        raise ChatBusyError from exc
