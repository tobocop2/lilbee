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
    """Admit up to the backend's live slot capacity; the rest wait FIFO, then 429.

    Capacity is read at admission time, not fixed when the gate is created, so a
    model swap or provider change that changes the slot count takes effect at
    once, and a single in-process model (capacity 1) is never oversubscribed.
    The fleet reports its ``--parallel`` slot count, so its continuous-batching
    slots are actually used instead of one request at a time. The latest count
    any caller reports is remembered, so a capacity increase also admits the
    waiters that are already queued rather than leaving them to time out
    against a backend that now has room.

    Slots are *handed to* waiters rather than merely signalled. Waking a waiter
    reserves its slot in the same synchronous step, and a request arriving while
    a waiter is queued joins the back of the queue instead of testing the
    counter. Without both, a newcomer could take the slot in the window between
    the wake-up and the woken waiter resuming, and the waiter would rejoin at
    the tail: the requests that had waited longest would be overtaken
    repeatedly and would be the ones to time out.
    """

    def __init__(self) -> None:
        self._in_flight = 0
        # Last slot count reported by a caller; the gate has no other view of it.
        self._ceiling = 1
        self._waiters: deque[asyncio.Future[None]] = deque()

    @property
    def in_flight(self) -> int:
        """Chat generations currently admitted."""
        return self._in_flight

    async def acquire(self, capacity: int, timeout: float) -> None:
        """Reserve a slot, waiting up to *timeout*; raise ChatBusyError if full."""
        loop = asyncio.get_running_loop()
        self._observe_capacity(max(1, capacity))
        # Admit straight away only with room AND nobody ahead in the queue.
        if not self._waiters and self._in_flight < self._ceiling:
            self._in_flight += 1
            return
        if timeout <= 0:
            raise ChatBusyError(_busy_message(timeout))
        waiter: asyncio.Future[None] = loop.create_future()
        self._waiters.append(waiter)
        try:
            # asyncio.timeout, not wait_for: 3.11's wait_for swallows a task
            # cancellation that races a completed waiter (fixed in 3.12).
            async with asyncio.timeout(timeout):
                await waiter
            # Returning means a slot was reserved for us by whoever woke us;
            # there is nothing left to claim.
        except TimeoutError as exc:
            self._abandon(waiter)
            raise ChatBusyError(_busy_message(timeout)) from exc
        except asyncio.CancelledError:
            self._abandon(waiter)
            raise
        finally:
            if waiter in self._waiters:
                self._waiters.remove(waiter)

    async def release(self) -> None:
        """Free an acquired slot and admit whoever it makes room for.

        Contains no awaits: the decrement and wake-ups run synchronously on the
        event loop, so a cancellation delivered to the caller (for example a
        client disconnect tearing down a streaming response) can never abort
        the release halfway and leak the slot.
        """
        if self._in_flight > 0:
            self._in_flight -= 1
        self._admit_waiters()

    def _observe_capacity(self, ceiling: int) -> None:
        """Record the caller's live slot count and admit anyone it now fits."""
        self._ceiling = ceiling
        self._admit_waiters()

    def _admit_waiters(self) -> None:
        """Hand a reserved slot to each queued waiter that now fits."""
        while self._in_flight < self._ceiling:
            waiter = self._pop_live_waiter()
            if waiter is None:
                return
            # Reserve before waking: the slot is the waiter's from this moment,
            # so nothing entering acquire() in between can take it.
            self._in_flight += 1
            waiter.set_result(None)

    def _pop_live_waiter(self) -> asyncio.Future[None] | None:
        """Remove and return the oldest waiter still able to take a slot."""
        while self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():
                return waiter
        return None

    def _abandon(self, waiter: asyncio.Future[None]) -> None:
        """Give back a slot that was reserved for a waiter that then bailed out."""
        if waiter.done() and not waiter.cancelled():
            if self._in_flight > 0:
                self._in_flight -= 1
            self._admit_waiters()


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
