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


DetailedProgressCallback = Callable[[EventType, dict[str, Any]], None]

# When set, vision updates the batch task's description instead of creating its own bar.
# Value is (Progress, batch_task_id).
shared_progress: ContextVar[tuple[Any, Any] | None] = ContextVar("shared_progress", default=None)


def noop_callback(event_type: EventType, data: dict[str, Any]) -> None:
    """Default no-op callback — discards all events."""


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


class SyncDoneEvent(BaseModel):
    """Emitted when the sync operation completes."""

    added: int
    updated: int
    removed: int
    failed: int
