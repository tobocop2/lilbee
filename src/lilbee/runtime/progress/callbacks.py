"""Progress callback protocol, no-op default, and shared progress context."""

from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from lilbee.runtime.progress.types import (
    BatchProgressEvent,
    CrawlDoneEvent,
    CrawlPageEvent,
    CrawlStartEvent,
    EmbedEvent,
    EventType,
    ExtractEvent,
    FileDoneEvent,
    FileStartEvent,
    SetupDoneEvent,
    SetupProgressEvent,
    SetupStartEvent,
    SyncDoneEvent,
)

ProgressEvent = (
    FileStartEvent
    | FileDoneEvent
    | BatchProgressEvent
    | ExtractEvent
    | EmbedEvent
    | SyncDoneEvent
    | CrawlStartEvent
    | CrawlPageEvent
    | CrawlDoneEvent
    | SetupStartEvent
    | SetupProgressEvent
    | SetupDoneEvent
)

DetailedProgressCallback = Callable[[EventType, ProgressEvent], None]

# When set, vision updates the batch task's description instead of creating its own bar.
# Value is (Progress, batch_task_id).
shared_progress: ContextVar[tuple[Any, Any] | None] = ContextVar("shared_progress", default=None)


def noop_callback(event_type: EventType, data: ProgressEvent) -> None:
    """Default no-op callback — discards all events."""
