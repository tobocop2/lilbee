"""Shared ingest types and constants."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple, TypedDict

from pydantic import BaseModel

from lilbee.data.store import (
    ChunkType,
    ConceptRecords,
    PageTextRecord,
    SourceStat,
    SourceStatBackfill,
)

# PDF and image content types route to paginated extraction; every other format
# routes to markdown extraction. content_type is derived per-file in
# discovery.classify_file (PDFs and images grouped; others keyed by extension).
PDF_CONTENT_TYPE = "pdf"
IMAGE_CONTENT_TYPE = "image"
MARKDOWN_OUTPUT = "markdown"
MARKDOWN_MIME = "text/markdown"


class FileToProcess(NamedTuple):
    """A file queued for ingestion with its metadata."""

    name: str
    path: Path
    content_type: str
    file_hash: str
    needs_cleanup: bool
    stat: SourceStat | None = None


class FileChangePlan(NamedTuple):
    """Outcome of diffing disk files against the tracked sources."""

    files_to_process: list[FileToProcess]
    added: dict[str, None]
    updated: dict[str, None]
    unchanged: int
    stat_backfills: list[SourceStatBackfill]


class OcrBackendName(StrEnum):
    """OCR backends lilbee selects in OcrConfig: xberg's tesseract or lilbee's vision plugin."""

    TESSERACT = "tesseract"
    LILBEE_VISION = "lilbee-vision"


class ExtractMode(StrEnum):
    """Extraction topology: paginated (PDFs/images) vs markdown output (text formats)."""

    MARKDOWN = "markdown"
    PAGINATED = "paginated"


class ChunkRecord(TypedDict):
    """A single store-ready chunk record matching store.CHUNKS_SCHEMA."""

    source: str
    content_type: str
    chunk_type: ChunkType
    page_start: int
    page_end: int
    line_start: int
    line_end: int
    chunk: str
    chunk_index: int
    vector: list[float]


class SyncResult(BaseModel):
    """Summary of a sync operation."""

    added: list[str] = []
    updated: list[str] = []
    removed: list[str] = []
    unchanged: int = 0
    failed: list[str] = []
    skipped: list[str] = []
    # Chunks whose text exceeded the embedder's char budget and were truncated
    # before embedding. Non-zero means some tail content did not reach the index.
    truncated: int = 0

    def __str__(self) -> str:
        lines = [
            f"Added: {len(self.added)}",
            f"Updated: {len(self.updated)}",
            f"Removed: {len(self.removed)}",
            f"Unchanged: {self.unchanged}",
            f"Skipped: {len(self.skipped)}",
            f"Failed: {len(self.failed)}",
            f"Truncated: {self.truncated}",
        ]
        for f in self.skipped:
            lines.append(f"  [yellow]{f}[/yellow]")
        for f in self.failed:
            lines.append(f"  [red]{f}[/red]")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SyncResult(added={len(self.added)}, updated={len(self.updated)}, "
            f"removed={len(self.removed)}, unchanged={self.unchanged}, "
            f"skipped={len(self.skipped)}, failed={len(self.failed)}, "
            f"truncated={self.truncated})"
        )

    def __rich__(self) -> str:
        return self.__str__()


@dataclass
class _IngestResult:
    """Outcome of a single file ingestion attempt.

    ``records`` carries the produced (extracted + embedded) chunks until the
    batched flush writes them; ``None`` on a failed file. ``needs_cleanup``
    travels with the records so the flush can delete the source's old chunks in
    the same transaction. ``page_texts`` carries the per-page text dataset rows
    and ``concept_records`` the file's concept-table rows, both written by the
    same flush.
    """

    name: str
    path: Path
    chunk_count: int
    error: Exception | None
    file_hash: str = ""
    records: list[ChunkRecord] | None = None
    needs_cleanup: bool = True
    page_texts: list[PageTextRecord] | None = None
    stat: SourceStat | None = None
    concept_records: ConceptRecords | None = None
