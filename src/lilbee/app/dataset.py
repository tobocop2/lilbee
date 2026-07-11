"""Surface-neutral export/import use cases over the per-page text dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from lilbee.app.services import get_services
from lilbee.data.export import (
    DatasetFormat,
    build_page_dataset,
    decode_format,
    deserialize_dataset,
    import_dataset,
    load_page_dataset,
    resolve_format,
    serialize_dataset,
    write_dataset,
)
from lilbee.data.store import EmbeddingModelMismatchError, PageTextRecord
from lilbee.runtime.progress import DetailedProgressCallback, noop_callback

if TYPE_CHECKING:
    import pyarrow as pa


class DatasetError(Exception):
    """User-facing export/import failure surfaces render as-is."""


class ExportSummary(BaseModel):
    """Result of a path-based export."""

    command: str = "export"
    format: str
    output: str
    pages: int
    sources: int


class ImportSummary(BaseModel):
    """Result of an import."""

    command: str = "import"
    sources: list[str]
    pages: int
    chunks: int


@dataclass
class ExportPayload:
    """In-memory export for byte transport (HTTP download)."""

    data: bytes
    fmt: DatasetFormat
    pages: int
    sources: int


def require_format(value: str) -> DatasetFormat:
    """Decode an explicit *value* into a format; there is no path to infer from."""
    if not value:
        raise DatasetError("format is required (parquet or jsonl)")
    try:
        return decode_format(value)
    except ValueError as exc:
        raise DatasetError(str(exc)) from None


def _build_validated(source: str | None) -> pa.Table:
    """Build the dataset table for *source* (or all), validating the request."""
    store = get_services().store
    if source is not None and source not in {s["filename"] for s in store.get_sources()}:
        raise DatasetError(f"Source not found: {source}")
    table = build_page_dataset(store, source)
    if table.num_rows == 0:
        raise DatasetError("Nothing to export: the store has no indexed pages.")
    return table


def export_to_path(output: Path, fmt_value: str, source: str | None) -> ExportSummary:
    """Write the per-page dataset to *output*; format from *fmt_value* or suffix."""
    try:
        fmt = resolve_format(fmt_value, output)
    except ValueError as exc:
        raise DatasetError(str(exc)) from None
    table = _build_validated(source)
    write_dataset(table, output, fmt)
    return ExportSummary(
        format=str(fmt),
        output=str(output),
        pages=table.num_rows,
        sources=len(table.column("source").unique()),
    )


def export_to_bytes(fmt_value: str, source: str | None) -> ExportPayload:
    """Encode the per-page dataset to bytes; empty *fmt_value* defaults to parquet."""
    fmt = require_format(fmt_value) if fmt_value else DatasetFormat.PARQUET
    table = _build_validated(source)
    return ExportPayload(
        data=serialize_dataset(table, fmt),
        fmt=fmt,
        pages=table.num_rows,
        sources=len(table.column("source").unique()),
    )


async def _run_import(
    rows: list[PageTextRecord], on_progress: DetailedProgressCallback
) -> ImportSummary:
    """Re-embed *rows* into the store, mapping the mismatch error for surfaces."""
    if not rows:
        raise DatasetError("Dataset has no pages to import.")
    store = get_services().store
    try:
        result = await import_dataset(store, rows, on_progress=on_progress)
    except EmbeddingModelMismatchError as exc:
        raise DatasetError(str(exc)) from None
    return ImportSummary(sources=result.sources, pages=result.pages, chunks=result.chunks)


async def import_from_path(
    path: Path, fmt_value: str, on_progress: DetailedProgressCallback = noop_callback
) -> ImportSummary:
    """Load and import a dataset file; format from *fmt_value* or suffix."""
    try:
        fmt = resolve_format(fmt_value, path)
        rows = load_page_dataset(path, fmt)
    except ValueError as exc:
        raise DatasetError(str(exc)) from None
    return await _run_import(rows, on_progress)


async def import_from_bytes(
    data: bytes, fmt_value: str, on_progress: DetailedProgressCallback = noop_callback
) -> ImportSummary:
    """Decode and import dataset *data*; *fmt_value* is required (no filename)."""
    fmt = require_format(fmt_value)
    try:
        rows = deserialize_dataset(data, fmt)
    except ValueError as exc:
        raise DatasetError(str(exc)) from None
    return await _run_import(rows, on_progress)
