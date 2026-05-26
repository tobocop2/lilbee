"""SSE stream primitives shared by every streaming handler."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import AsyncGenerator
from typing import Any

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

log = logging.getLogger(__name__)

# Machine-readable ``code`` on an SSE error event. Load-time failures use
# SseErrorCode; failed provider calls reuse ProviderErrorKind directly rather
# than mirroring it into a second enum.
SseErrorCodeValue = SseErrorCode | ProviderErrorKind


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
_NOT_INSTALLED_MARKERS = ("not found in registry", "is not available", "pull it first")


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


class SseStream:
    """Context object for SSE streaming with cancellation support.
    Bundles the queue, cancel event, and progress callback that every SSE
    endpoint needs.  Call :meth:`drain` to yield events until the task
    completes or the client disconnects.
    """

    def __init__(self) -> None:
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()
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
                queue.put_nowait(payload)
            else:
                loop.call_soon_threadsafe(queue.put_nowait, payload)

        return _callback

    async def drain(
        self, task: asyncio.Task[Any] | asyncio.Future[Any], label: str
    ) -> AsyncGenerator[str, None]:
        """Yield SSE strings until a sentinel arrives; cancel *task* on client disconnect.

        Emits a ``heartbeat`` event whenever the producer queue stays
        idle longer than ``cfg.sse_heartbeat_interval`` seconds so
        clients that enforce a stream-idle timeout don't abort.
        """
        last_yielded = time.monotonic()
        try:
            while True:
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                except TimeoutError:
                    now = time.monotonic()
                    heartbeat_interval = cfg.sse_heartbeat_interval
                    if heartbeat_interval > 0 and now - last_yielded >= heartbeat_interval:
                        last_yielded = now
                        yield sse_event(SseEvent.HEARTBEAT, {"ts": time.time()})
                    # Fallback for producers that die without a sentinel.
                    if task.done() and self.queue.empty():
                        break
                    continue
                if item is None:
                    break
                last_yielded = time.monotonic()
                yield item
        except (asyncio.CancelledError, GeneratorExit):
            log.info("%s cancelled by client", label)
            self.cancel.set()
            task.cancel()
