"""Progress callback protocol and no-op default."""

from collections.abc import Callable

from lilbee.runtime.progress.types import (
    AnalyzeEvent,
    BatchProgressEvent,
    CrawlDoneEvent,
    CrawlPageEvent,
    CrawlPageFailedEvent,
    CrawlStartEvent,
    EmbedEvent,
    EventType,
    ExtractEvent,
    FileDoneEvent,
    FileStartEvent,
    LibraryUpgradeEvent,
    OcrStartEvent,
    SetupDoneEvent,
    SetupProgressEvent,
    SetupStartEvent,
    SyncDoneEvent,
    SyncWaitEvent,
    WikiPageEvent,
    WikiPhaseEvent,
)

ProgressEvent = (
    FileStartEvent
    | FileDoneEvent
    | BatchProgressEvent
    | ExtractEvent
    | OcrStartEvent
    | EmbedEvent
    | SyncDoneEvent
    | SyncWaitEvent
    | CrawlStartEvent
    | CrawlPageEvent
    | CrawlPageFailedEvent
    | CrawlDoneEvent
    | SetupStartEvent
    | SetupProgressEvent
    | SetupDoneEvent
    | WikiPhaseEvent
    | WikiPageEvent
    | AnalyzeEvent
    | LibraryUpgradeEvent
)

DetailedProgressCallback = Callable[[EventType, ProgressEvent], None]


def noop_callback(event_type: EventType, data: ProgressEvent) -> None:
    """Default no-op callback: discards all events."""
