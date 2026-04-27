"""Document sync engine — keeps documents/ dir in sync with LanceDB."""

from __future__ import annotations

from lilbee.data.ingest.code import ingest_code_sync
from lilbee.data.ingest.discovery import classify_file, discover_files, file_hash
from lilbee.data.ingest.extract import (
    _has_meaningful_text,
    _should_run_ocr,
    _vision_fallback,
    content_type_to_mode,
    extraction_config,
    ingest_document,
    ingest_markdown,
)
from lilbee.data.ingest.pipeline import (
    _apply_result,
    _incremental_wiki_update,
    _ingest_file,
    ingest_batch,
    sync,
)
from lilbee.data.ingest.types import ExtractMode, SyncResult, _IngestResult

__all__ = [
    "ExtractMode",
    "SyncResult",
    "_IngestResult",
    "_apply_result",
    "_has_meaningful_text",
    "_incremental_wiki_update",
    "_ingest_file",
    "_should_run_ocr",
    "_vision_fallback",
    "classify_file",
    "content_type_to_mode",
    "discover_files",
    "extraction_config",
    "file_hash",
    "ingest_batch",
    "ingest_code_sync",
    "ingest_document",
    "ingest_markdown",
    "sync",
]
