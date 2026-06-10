"""SSE stream primitives shared by every streaming handler."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import deque
from collections.abc import AsyncGenerator
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


_OOM_MARKERS = ("failed to load", "free ram", "try a smaller model", "llama_context")
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

# Progress-class event types: high-frequency, safe to coalesce under
# backpressure. Everything else (done, errors, crawl/setup lifecycle) must land.
_DROPPABLE_EVENT_TYPES = frozenset(
    {
        EventType.FILE_START,
        EventType.FILE_DONE,
        EventType.BATCH_PROGRESS,
        EventType.EMBED,
        EventType.EXTRACT,
        EventType.CRAWL_PAGE,
        EventType.SETUP_PROGRESS,
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

    def _put(self, item: str | None) -> None:
        self._queue.append(_QueuedEvent(item, self._put_droppable))

    def _get(self) -> str | None:
        return self._queue.popleft().payload

    def _evict_oldest_droppable(self) -> bool:
        """Shed the queue head when it is a progress event; True when a slot freed."""
        if self._queue and self._queue[0].droppable:
            self._queue.popleft()
            self.dropped_events += 1
            return True
        return False

    def put_nowait(self, item: str | None) -> None:
        """Enqueue an always-delivered event, evicting old progress when full."""
        if self.qsize() >= self._max_events:
            self._evict_oldest_droppable()
        self._put_droppable = False
        super().put_nowait(item)

    def put_event_nowait(self, payload: str, event_type: EventType) -> None:
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
        """
        last_yielded = time.monotonic()
        getter: asyncio.Future[str | None] | None = None
        try:
            while True:
                if getter is None:
                    getter = asyncio.ensure_future(self.queue.get())
                done, _ = await asyncio.wait({getter}, timeout=0.1)
                if not done:
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
