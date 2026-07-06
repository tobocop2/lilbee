"""Process-wide chat-generation admission gate shared by every chat route."""

from __future__ import annotations

import asyncio
from collections import deque
from functools import lru_cache

# Default upper bound for waiting on a free chat slot before surfacing a real
# "busy" response. Tuned to absorb the typical opencode retry storm (one user
# turn can take 30-60s on slow local models) while still bounding the worst case
# for a genuinely-stuck backend.
DEFAULT_BUSY_WAIT_S = 60.0


class ChatBusyError(Exception):
    """Raised when every chat slot is in use after waiting."""


class ChatGate:
    """Admit up to the backend's live slot capacity; the rest wait, then 429.

    Capacity is read at admission time, not fixed when the gate is created, so a
    model swap or provider change that changes the slot count takes effect at
    once, and a single in-process model (capacity 1) is never oversubscribed.
    The fleet reports its ``--parallel`` slot count, so its continuous-batching
    slots are actually used instead of one request at a time.
    """

    def __init__(self) -> None:
        self._in_flight = 0
        self._waiters: deque[asyncio.Future[None]] = deque()

    @property
    def in_flight(self) -> int:
        """Chat generations currently admitted."""
        return self._in_flight

    async def acquire(self, capacity: int, timeout: float) -> None:
        """Reserve a slot, waiting up to *timeout*; raise ChatBusyError if full."""
        ceiling = max(1, capacity)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while self._in_flight >= ceiling:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise ChatBusyError(_busy_message(timeout))
            waiter: asyncio.Future[None] = loop.create_future()
            self._waiters.append(waiter)
            try:
                # asyncio.timeout, not wait_for: 3.11's wait_for swallows a task
                # cancellation that races a completed waiter (fixed in 3.12).
                async with asyncio.timeout(remaining):
                    await waiter
            except TimeoutError as exc:
                self._renotify_if_woken(waiter)
                raise ChatBusyError(_busy_message(timeout)) from exc
            except asyncio.CancelledError:
                self._renotify_if_woken(waiter)
                raise
            finally:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
        self._in_flight += 1

    async def release(self) -> None:
        """Free an acquired slot and wake one waiter.

        Contains no awaits: the decrement and wake-up run synchronously on the
        event loop, so a cancellation delivered to the caller (for example a
        client disconnect tearing down a streaming response) can never abort
        the release halfway and leak the slot.
        """
        if self._in_flight > 0:
            self._in_flight -= 1
        self._wake_next()

    def _wake_next(self) -> None:
        """Wake the oldest still-pending waiter, if any."""
        while self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():
                waiter.set_result(None)
                return

    def _renotify_if_woken(self, waiter: asyncio.Future[None]) -> None:
        """Pass a wake-up consumed by a bailing-out waiter on to the next one."""
        if waiter.done() and not waiter.cancelled():
            self._wake_next()


def _busy_message(timeout: float) -> str:
    rendered = f"{timeout:.1f}s" if timeout < 1 else f"{timeout:.0f}s"
    return f"Chat backend busy: all slots in use after {rendered}. Retry shortly."


@lru_cache(maxsize=1)
def chat_gate() -> ChatGate:
    """Return the process-wide chat admission gate (created lazily)."""
    return ChatGate()


async def acquire_chat_slot_or_busy(capacity: int, timeout: float | None = None) -> None:
    """Reserve one of *capacity* chat slots, waiting up to *timeout*.

    Raises :class:`ChatBusyError` only when the wait times out. Returns with the
    slot held; callers must ``await release_chat_slot()`` in a ``finally``.
    *timeout* defaults to :data:`DEFAULT_BUSY_WAIT_S`.
    """
    effective_timeout = DEFAULT_BUSY_WAIT_S if timeout is None else timeout
    await chat_gate().acquire(capacity, effective_timeout)


async def release_chat_slot() -> None:
    """Release a chat slot reserved by :func:`acquire_chat_slot_or_busy`."""
    await chat_gate().release()


class ChatSlotGuard:
    """Releases one acquired chat slot at most once across multiple cleanup paths.

    A streaming route acquires its slot before the SSE generator runs; a client
    that disconnects before the generator's first iteration means the generator
    body (and its ``finally``) never executes. Every cleanup path (generator
    ``finally``, response after-send hook, explicit ``aclose``) releases through
    the same guard, so whichever fires first frees the slot and the rest no-op.
    """

    def __init__(self) -> None:
        self._released = False

    @property
    def released(self) -> bool:
        """True once the slot has been freed."""
        return self._released

    async def release(self) -> None:
        """Free the slot on first call; later calls are no-ops."""
        if self._released:
            return
        self._released = True
        # ChatGate.release never yields to the event loop, so no cancellation
        # can land between flipping the flag and the slot actually freeing.
        await release_chat_slot()
