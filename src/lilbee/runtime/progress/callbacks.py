"""Progress callback protocol and no-op default."""

from collections.abc import Callable

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


def noop_callback(event_type: EventType, data: ProgressEvent) -> None:
    """Default no-op callback: discards all events."""
