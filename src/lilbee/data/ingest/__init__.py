"""Document sync engine. Keeps documents/ dir in sync with LanceDB."""

from __future__ import annotations

from lilbee.data.ingest.code import ingest_code_sync
from lilbee.data.ingest.discovery import classify_file, discover_files, file_hash
from lilbee.data.ingest.extract import (
    content_type_to_mode,
    extraction_config,
    ingest_document,
    ingest_markdown,
)
from lilbee.data.ingest.pipeline import detect_pending, ingest_batch, sync
from lilbee.data.ingest.types import ExtractMode, SyncResult

__all__ = [
    "ExtractMode",
    "SyncResult",
    "classify_file",
    "content_type_to_mode",
    "detect_pending",
    "discover_files",
    "extraction_config",
    "file_hash",
    "ingest_batch",
    "ingest_code_sync",
    "ingest_document",
    "ingest_markdown",
    "sync",
]
