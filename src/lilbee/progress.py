"""Granular progress callback protocol for streaming pipeline events."""

from collections.abc import Callable
from contextvars import ContextVar
from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class EventType(StrEnum):
    """Progress event types emitted during sync/ingest."""

    FILE_START = "file_start"
    FILE_DONE = "file_done"
    BATCH_PROGRESS = "batch_progress"
    DONE = "done"
    EMBED = "embed"
    EXTRACT = "extract"
    CRAWL_START = "crawl_start"
    CRAWL_PAGE = "crawl_page"
    CRAWL_DONE = "crawl_done"
    SETUP_START = "setup_start"
    SETUP_PROGRESS = "setup_progress"
    SETUP_DONE = "setup_done"


class SseEvent(StrEnum):
    """SSE event names used in the HTTP streaming protocol."""

    TOKEN = "token"  # noqa: S105 -- SSE event name, not a credential
    REASONING = "reasoning"
    SOURCES = "sources"
    ERROR = "error"
    DONE = "done"
    PROGRESS = "progress"
    HEARTBEAT = "heartbeat"


class FileStartEvent(BaseModel):
    """Emitted when a file begins ingestion."""

    file: str
    total_files: int
    current_file: int


class FileDoneEvent(BaseModel):
    """Emitted when a file finishes ingestion (success or error)."""

    file: str
    status: str
    chunks: int


class BatchProgressEvent(BaseModel):
    """Emitted after each file completes during batch ingestion."""

    file: str
    status: str
    current: int
    total: int


class ExtractEvent(BaseModel):
    """Emitted per page during vision OCR extraction."""

    file: str
    page: int
    total_pages: int


class EmbedEvent(BaseModel):
    """Emitted per batch during embedding."""

    file: str
    chunk: int
    total_chunks: int


class CrawlStartEvent(BaseModel):
    """Emitted when a crawl operation begins."""

    url: str
    depth: int


# Sentinel used in CrawlPageEvent.total when the crawl's final page count is
# not yet known (BFS streaming, page N emitted before N+1 is discovered).
# Consumers (plugin, TUI, CLI) treat total <= 0 as indeterminate progress.
CRAWL_TOTAL_UNKNOWN = -1


class CrawlPageEvent(BaseModel):
    """Emitted per page during crawling."""

    url: str
    current: int
    total: int


class CrawlDoneEvent(BaseModel):
    """Emitted when a crawl operation completes."""

    pages_crawled: int
    files_written: int


class SyncDoneEvent(BaseModel):
    """Emitted when the sync operation completes."""

    added: int
    updated: int
    removed: int
    failed: int


class SetupStartEvent(BaseModel):
    """Emitted when a setup/bootstrap operation begins."""

    component: str
    size_estimate_bytes: int | None = None


class SetupProgressEvent(BaseModel):
    """Emitted periodically during a setup/bootstrap operation."""

    component: str
    downloaded_bytes: int
    total_bytes: int | None = None
    detail: str = ""


class SetupDoneEvent(BaseModel):
    """Emitted when a setup/bootstrap operation completes."""

    component: str
    success: bool
    error: str | None = None


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
