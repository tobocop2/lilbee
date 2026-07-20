"""SSE stream primitives shared by every streaming handler."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import deque
from collections.abc import AsyncGenerator, Callable
from typing import Any, NamedTuple

from pydantic import BaseModel

from lilbee.core.config import cfg
from lilbee.providers.base import ProviderErrorKind, filter_options
from lilbee.runtime.progress import (
    DetailedProgressCallback,
    EventType,
    ProgressEvent,
    SseErrorCode,
    SseEvent,
)
from lilbee.server.chat_completions_api.errors import CompletionsErrorCode

log = logging.getLogger(__name__)

# Machine-readable ``code`` on an SSE error event. Load-time failures use
# SseErrorCode; failed provider calls reuse ProviderErrorKind directly; the
# RAG chat stream reuses the wire-layer CompletionsErrorCode for typed
# dispatch errors (unknown model, no tool support, context overflow) so a
# single client-facing vocabulary covers both surfaces.
SseErrorCodeValue = SseErrorCode | ProviderErrorKind | CompletionsErrorCode


def sse_event(event: str, data: Any) -> str:
    """Format a single Server-Sent Event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def sse_error(
    message: str, *, code: SseErrorCodeValue | None = None, detail: str | None = None
) -> str:
    """Format an SSE error event with optional structured ``code`` / ``detail``."""
    payload: dict[str, Any] = {"message": message}
    if code is not None:
        payload["code"] = code
    if detail is not None:
        payload["detail"] = detail
    return sse_event(SseEvent.ERROR, payload)


# Phrases that describe an allocation failure. "llama_context" alone used to be
# here, but that is the prefix llama.cpp stamps on every context-subsystem
# diagnostic, so n_ctx-over-training-context and KV-cache rejections were all
# reported as "model too large", pointing at a smaller model when the fix is a
# config change.
_OOM_MARKERS = ("failed to load", "free ram", "try a smaller model", "failed to allocate")
_NOT_INSTALLED_MARKERS = ("is not installed", "is not available", "pull it first")


def classify_load_error(message: str) -> tuple[SseErrorCode | None, str]:
    """Return ``(code, user_message)`` for an SSE error event.

    Maps the llama.cpp out-of-memory diagnostic and the "configured model
    isn't installed" failure to stable codes. Anything else returns a generic
    code-less message.
    """
    lowered = message.lower()
    if any(marker in lowered for marker in _OOM_MARKERS):
        return SseErrorCode.MODEL_TOO_LARGE, "Model too large for available RAM"
    if any(marker in lowered for marker in _NOT_INSTALLED_MARKERS):
        return (
            SseErrorCode.MODEL_NOT_INSTALLED,
            "Active model isn't installed. Pull it from the catalog.",
        )
    return None, "Internal error"


def sse_done(data: dict[str, Any]) -> str:
    """Format an SSE done event."""
    return sse_event(SseEvent.DONE, data)


def _resolve_generation_options(options: dict[str, Any] | None) -> dict[str, Any] | None:
    """Merge HTTP-supplied options with config, allowlisting sampling keys only.

    ``filter_options`` is the validation boundary for untrusted callers: it
    drops anything outside the sampling allowlist (e.g. injected ``api_base`` /
    ``api_key``) before the values reach a provider.
    """
    return cfg.generation_options(**filter_options(options)) if options else None


# Cap on buffered SSE events per stream. A bulk sync emits per-file and
# per-chunk progress far faster than a slow client reads; past this bound the
# stream sheds the oldest progress event rather than buffering millions.
SSE_QUEUE_MAX_EVENTS = 1000

# Poll ceiling once the producer task has finished, for the case where the
# queue drains empty without the sentinel ever arriving.
_FINISHED_PRODUCER_POLL_S = 1.0

# Progress-class event types: high-frequency, safe to coalesce under
# backpressure. Everything else (done, errors, crawl/setup lifecycle) must land.
_DROPPABLE_EVENT_TYPES: frozenset[EventType | SseEvent] = frozenset(
    {
        EventType.FILE_START,
        EventType.FILE_DONE,
        EventType.BATCH_PROGRESS,
        EventType.EMBED,
        EventType.EXTRACT,
        EventType.CRAWL_PAGE,
        EventType.SETUP_PROGRESS,
        SseEvent.PROGRESS,
    }
)


class _QueuedEvent(NamedTuple):
    """A queued SSE payload tagged with whether backpressure may shed it."""

    payload: str | None
    droppable: bool


class SseEventQueue(asyncio.Queue[str | None]):
    """Bounded SSE queue: progress events shed under backpressure, the rest land.

    ``put_nowait`` (lifecycle events, tokens, the ``None`` sentinel) always
    enqueues, evicting the oldest progress event first when at capacity.
    ``put_event_nowait`` enqueues progress-protocol events, dropping the oldest
    progress event (or the incoming one when the head is not progress) at
    capacity. ``join()`` semantics are not supported.
    """

    _queue: deque[_QueuedEvent]

    def __init__(self, max_events: int = SSE_QUEUE_MAX_EVENTS) -> None:
        super().__init__()
        self._max_events = max_events
        self._put_droppable = False
        self.dropped_events = 0
        # Set once the queue is full of undroppable events, i.e. the consumer
        # has stopped reading. See put_nowait.
        self.stalled = False

    def _put(self, item: str | None) -> None:
        self._queue.append(_QueuedEvent(item, self._put_droppable))

    def _get(self) -> str | None:
        return self._queue.popleft().payload

    def _evict_oldest_droppable(self) -> bool:
        """Shed the oldest progress event anywhere in the queue; True when one went.

        Scans rather than checking only the head. A stream that interleaves
        tokens with progress puts a non-droppable event at the head almost
        immediately, and head-only eviction then reported "nothing to shed"
        while the queue still held progress events it was allowed to drop.
        """
        for index, event in enumerate(self._queue):
            if event.droppable:
                del self._queue[index]
                self.dropped_events += 1
                return True
        return False

    def put_nowait(self, item: str | None) -> None:
        """Enqueue an always-delivered event, evicting old progress when full."""
        if self.qsize() >= self._max_events and not self._evict_oldest_droppable():
            # Nothing left that may be shed: the queue is full of events this
            # class is required to deliver (chat and RAG tokens go through
            # here), and the consumer has not taken one in _max_events. That
            # is a stalled or already-gone client, so record it. Growing the
            # queue for the rest of the generation is the alternative, and it
            # is unbounded by construction.
            self.stalled = True
        self._put_droppable = False
        super().put_nowait(item)

    def put_event_nowait(self, payload: str, event_type: EventType | SseEvent) -> None:
        """Enqueue a progress-protocol event, shedding progress when full."""
        if event_type not in _DROPPABLE_EVENT_TYPES:
            self.put_nowait(payload)
            return
        if self.qsize() >= self._max_events and not self._evict_oldest_droppable():
            self.dropped_events += 1
            return
        self._put_droppable = True
        try:
            super().put_nowait(payload)
        finally:
            self._put_droppable = False


class SseStream:
    """Context object for SSE streaming with cancellation support.
    Bundles the queue, cancel event, and progress callback that every SSE
    endpoint needs.  Call :meth:`drain` to yield events until the task
    completes or the client disconnects.
    """

    def __init__(self) -> None:
        self.queue: SseEventQueue = SseEventQueue()
        self.cancel = threading.Event()
        self.loop = asyncio.get_running_loop()
        self.callback: DetailedProgressCallback = self._build_callback()

    def put_threadsafe(self, item: str | None) -> None:
        """Enqueue an always-delivered event from a worker thread.

        ``asyncio.Queue.put_nowait`` is not thread-safe: it wakes a pending
        getter via ``Future.set_result``, which must run on the loop thread. A
        producer running under ``run_in_executor`` therefore hands the put back
        to the loop instead of mutating the queue directly.
        """
        self.loop.call_soon_threadsafe(self._put_and_check_stall, item)

    def _put_and_check_stall(self, item: str | None) -> None:
        """Enqueue on the loop thread, cancelling the producer if it has stalled.

        Chat and RAG tokens are in the always-deliver class, so a fast
        generation streaming to a client that has stopped reading fills the
        queue with events nothing is permitted to shed and it grows until the
        generation ends. A consumer that has not taken a single event in a
        full queue's worth is gone or as good as gone; cancelling is the same
        signal a detected disconnect sends, just reached a different way.
        """
        self.queue.put_nowait(item)
        if self.queue.stalled and not self.cancel.is_set():
            log.warning(
                "SSE consumer stalled with %d undroppable events queued; "
                "cancelling the producer.",
                self.queue.qsize(),
            )
            self.cancel.set()

    def _build_callback(self) -> DetailedProgressCallback:
        """Create a progress callback that serializes events into the queue.
        Safe to call from both the event-loop thread and worker threads.
        """
        loop = self.loop
        queue = self.queue

        def _callback(event_type: EventType, data: ProgressEvent) -> None:
            serialized = data.model_dump() if isinstance(data, BaseModel) else data
            payload = f"event: {event_type}\ndata: {json.dumps(serialized)}\n\n"
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is loop:
                queue.put_event_nowait(payload, event_type)
            else:
                loop.call_soon_threadsafe(queue.put_event_nowait, payload, event_type)

        return _callback

    async def _flush_pending(self) -> AsyncGenerator[str, None]:
        """Events left behind the sentinel by a producer that outran the consumer.

        A fast producer can enqueue its sentinel before its threadsafe progress
        callbacks run; one loop tick lets them land.
        """
        await asyncio.sleep(0)
        while not self.queue.empty():
            leftover = self.queue.get_nowait()
            if leftover is not None:
                yield leftover

    def terminal_frame(
        self,
        task: asyncio.Task[Any] | asyncio.Future[Any],
        payload: Callable[[Any], dict[str, Any]],
    ) -> str | None:
        """The final SSE frame for a finished producer, or None if there is none.

        Every streaming handler ended with the same five lines: check cancel,
        check the task finished and was not cancelled, turn an exception into
        sse_error, otherwise sse_done of the result. The copies had drifted --
        the crawler-setup one skipped the cancel check and so emitted a done
        frame to a client that had already disconnected -- which is the reason
        this lives in one place now.
        """
        if self.cancel.is_set() or not task.done() or task.cancelled():
            return None
        exc = task.exception()
        if exc is not None:
            return sse_error(str(exc))
        return sse_done(payload(task.result()))

    @staticmethod
    def _drain_waiters(
        getter: asyncio.Future[str | None],
        task: asyncio.Task[Any] | asyncio.Future[Any],
    ) -> set[asyncio.Future[Any]]:
        """Futures the drain loop waits on. A finished task is left out: it
        would resolve the wait instantly on every pass and spin the loop."""
        return {getter} if task.done() else {getter, task}

    @staticmethod
    def _drain_timeout(task: asyncio.Task[Any] | asyncio.Future[Any]) -> float | None:
        """How long one drain pass may sleep.

        While the producer runs, the only thing the timeout serves is the
        heartbeat, and the task is in the wait set for everything else, so a
        disabled heartbeat can wait indefinitely. Once the task has finished it
        is out of the wait set, so this bounds the one remaining case: a queue
        that empties without the sentinel ever arriving.
        """
        if task.done():
            return _FINISHED_PRODUCER_POLL_S
        interval = cfg.sse_heartbeat_interval
        return interval if interval > 0 else None

    async def drain(
        self, task: asyncio.Task[Any] | asyncio.Future[Any], label: str
    ) -> AsyncGenerator[str, None]:
        """Yield SSE strings until a sentinel arrives; cancel *task* on client disconnect.

        Emits a ``heartbeat`` event whenever the producer queue stays
        idle longer than ``cfg.sse_heartbeat_interval`` seconds so
        clients that enforce a stream-idle timeout don't abort.

        The pending ``queue.get`` survives across poll rounds (``asyncio.wait``,
        not ``wait_for``): cancelling a completed get on the timeout boundary
        would drop the event it already popped from the queue.

        The wait covers *task* as well, so a producer that dies without a
        sentinel wakes the loop directly. That is what lets the timeout be the
        seconds-scale heartbeat interval rather than a fixed 0.1s tick: the
        loop used to wake ten times a second per open stream purely to
        re-evaluate two conditions that neither need that resolution.
        """
        last_yielded = time.monotonic()
        getter: asyncio.Future[str | None] | None = None
        try:
            while True:
                if getter is None:
                    getter = asyncio.ensure_future(self.queue.get())
                done, _ = await asyncio.wait(
                    self._drain_waiters(getter, task),
                    timeout=self._drain_timeout(task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if getter not in done:
                    now = time.monotonic()
                    heartbeat_interval = cfg.sse_heartbeat_interval
                    if heartbeat_interval > 0 and now - last_yielded >= heartbeat_interval:
                        last_yielded = now
                        yield sse_event(SseEvent.HEARTBEAT, {"ts": time.time()})
                    # Fallback for producers that die without a sentinel.
                    if task.done() and self.queue.empty():
                        getter.cancel()
                        break
                    continue
                item = getter.result()
                getter = None
                if item is None:
                    async for leftover in self._flush_pending():
                        last_yielded = time.monotonic()
                        yield leftover
                    break
                last_yielded = time.monotonic()
                yield item
        except (asyncio.CancelledError, GeneratorExit):
            log.info("%s cancelled by client", label)
            self.cancel.set()
            task.cancel()
            if getter is not None:
                getter.cancel()
