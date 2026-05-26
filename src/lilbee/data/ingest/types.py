"""Shared ingest types and constants."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple, TypedDict

from pydantic import BaseModel

from lilbee.data.store import ChunkType


class FileToProcess(NamedTuple):
    """A file queued for ingestion with its metadata."""

    name: str
    path: Path
    content_type: str
    file_hash: str
    needs_cleanup: bool


# Minimum total chars for extracted text to be considered meaningful.
# 50 chars ≈ 12 words: if a PDF yields less, it's almost certainly a scanned
# document with no embedded text layer. Text PDFs with even just a title page
# easily exceed this threshold; blank/scan-only PDFs yield 0 chars.
MIN_MEANINGFUL_CHARS = 50

PDF_CONTENT_TYPE = "pdf"
MARKDOWN_OUTPUT = "markdown"
TESSERACT_BACKEND = "tesseract"


class ExtractMode(StrEnum):
    """Extraction topology: pagination / OCR / output format."""

    MARKDOWN = "markdown"
    PAGINATED = "paginated"
    PAGINATED_OCR = "paginated_ocr"


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
    """Outcome of a single file ingestion attempt."""

    name: str
    path: Path
    chunk_count: int
    error: Exception | None
    file_hash: str = ""


# Extension → content_type string for document formats handled by kreuzberg
DOCUMENT_EXTENSION_MAP: dict[str, str] = {
    **{ext: "text" for ext in (".md", ".txt", ".html", ".rst", ".yaml", ".yml")},
    ".pdf": PDF_CONTENT_TYPE,
    **{ext: ext.lstrip(".") for ext in (".docx", ".xlsx", ".pptx")},
    ".epub": "epub",
    **{ext: "image" for ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp")},
    **{ext: "data" for ext in (".csv", ".tsv")},
    ".xml": "xml",
    **{ext: "json" for ext in (".json", ".jsonl")},
}
