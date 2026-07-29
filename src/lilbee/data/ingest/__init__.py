"""Document sync engine. Keeps documents/ dir in sync with LanceDB."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from lilbee.data.ingest.types import ExtractMode, SyncResult

if TYPE_CHECKING:
    from lilbee.data.extract.document import (
        content_type_to_mode,
        extraction_config,
        ingest_document,
        ingest_markdown,
    )
    from lilbee.data.ingest.code import ingest_code_sync
    from lilbee.data.ingest.discovery import classify_file, discover_files, file_hash
    from lilbee.data.ingest.pipeline import detect_pending, ingest_stream, sync

# Resolved on first access (PEP 562). The pipeline/code/discovery modules import
# the extract package, whose document module imports ingest leaves (offload,
# title, types) in turn; loading both halves at package-import time is a cycle.
# Deferring these lets an ingest leaf import run without dragging in extract.
_LAZY = {
    "content_type_to_mode": "lilbee.data.extract.document",
    "extraction_config": "lilbee.data.extract.document",
    "ingest_document": "lilbee.data.extract.document",
    "ingest_markdown": "lilbee.data.extract.document",
    "ingest_code_sync": "lilbee.data.ingest.code",
    "classify_file": "lilbee.data.ingest.discovery",
    "discover_files": "lilbee.data.ingest.discovery",
    "file_hash": "lilbee.data.ingest.discovery",
    "detect_pending": "lilbee.data.ingest.pipeline",
    "ingest_stream": "lilbee.data.ingest.pipeline",
    "sync": "lilbee.data.ingest.pipeline",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


__all__ = [
    "ExtractMode",
    "SyncResult",
    "classify_file",
    "content_type_to_mode",
    "detect_pending",
    "discover_files",
    "extraction_config",
    "file_hash",
    "ingest_code_sync",
    "ingest_document",
    "ingest_markdown",
    "ingest_stream",
    "sync",
]
