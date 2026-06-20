"""Per-page text dataset: build/write from a store, and import one back."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, cast

from lilbee.data.ingest.extract import chunk_and_embed_pages
from lilbee.data.store import ChunkWrite, PageTextRecord, SourceType
from lilbee.runtime.progress import DetailedProgressCallback, noop_callback

if TYPE_CHECKING:
    from lilbee.data.store import Store


class DatasetFormat(StrEnum):
    """On-disk format for the per-page text dataset."""

    PARQUET = "parquet"
    JSONL = "jsonl"


@dataclass
class ImportResult:
    """Summary of an `import_dataset` run."""

    sources: list[str]
    pages: int
    chunks: int


def decode_format(value: str) -> DatasetFormat:
    """Decode an explicit format string, raising a user-facing ``ValueError``."""
    try:
        return DatasetFormat(value)
    except ValueError:
        raise ValueError(f"Unsupported format: {value!r} (expected parquet or jsonl)") from None


def resolve_format(value: str, path: Path) -> DatasetFormat:
    """Pick a format from explicit *value*, else the *path* suffix.

    Raises ``ValueError`` with a user-facing message when neither yields a
    known format.
    """
    if value:
        return decode_format(value)
    suffix = path.suffix.lower().lstrip(".")
    try:
        return DatasetFormat(suffix)
    except ValueError:
        raise ValueError(
            f"Could not infer format from {path.name!r}; use a .parquet or .jsonl path"
        ) from None


def build_page_dataset(store: Store, source: str | None = None) -> list[PageTextRecord]:
    """Collect per-page text rows for every source (or just *source*).

    Sources captured at ingest are returned verbatim from the page-text table.
    Sources without captured text (older indexes, code) are reconstructed from
    the chunks table; that reconstruction concatenates chunk text per page, so
    chunk overlap may repeat a little text across page boundaries.
    """
    names = [source] if source is not None else sorted(s["filename"] for s in store.get_sources())
    captured = store.page_text_sources()
    rows: list[PageTextRecord] = []
    for name in names:
        if name in captured:
            rows.extend(store.get_page_texts(name))
        else:
            rows.extend(_reconstruct_from_chunks(store, name))
    rows.sort(key=lambda r: (r["source"], r["page"]))
    return rows


def _reconstruct_from_chunks(store: Store, source: str) -> list[PageTextRecord]:
    """Rebuild per-page rows for *source* by joining its chunks per page."""
    by_page: dict[int, list[tuple[int, str]]] = {}
    content_type = ""
    for chunk in store.get_chunks_by_source(source):
        content_type = chunk.content_type or content_type
        by_page.setdefault(chunk.page_start, []).append((chunk.chunk_index, chunk.chunk))
    rows: list[PageTextRecord] = []
    for page in sorted(by_page):
        ordered = [text for _, text in sorted(by_page[page])]
        rows.append(
            PageTextRecord(
                source=source, page=page, text="\n".join(ordered), content_type=content_type
            )
        )
    return rows


def _serialize_parquet(rows: list[PageTextRecord]) -> bytes:
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist([dict(r) for r in rows])
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


def _serialize_jsonl(rows: list[PageTextRecord]) -> bytes:
    return "".join(json.dumps(dict(row)) + "\n" for row in rows).encode("utf-8")


_SERIALIZERS = {DatasetFormat.PARQUET: _serialize_parquet, DatasetFormat.JSONL: _serialize_jsonl}


def serialize_dataset(rows: list[PageTextRecord], fmt: DatasetFormat) -> bytes:
    """Encode *rows* to dataset bytes in the given format."""
    return _SERIALIZERS[fmt](rows)


def write_dataset(rows: list[PageTextRecord], path: Path, fmt: DatasetFormat) -> None:
    """Write *rows* to *path* in the given format."""
    path.write_bytes(serialize_dataset(rows, fmt))


def _coerce_row(raw: dict) -> PageTextRecord:
    """Validate one raw dataset row into a `PageTextRecord`."""
    try:
        return PageTextRecord(
            source=str(raw["source"]),
            page=int(raw["page"]),
            text=str(raw["text"]),
            content_type=str(raw.get("content_type", "")),
        )
    except (KeyError, TypeError, ValueError):
        raise ValueError("Dataset row is missing required source/page/text fields") from None


def _deserialize_parquet(data: bytes) -> list[PageTextRecord]:
    import io

    import pyarrow.parquet as pq

    return [_coerce_row(row) for row in pq.read_table(io.BytesIO(data)).to_pylist()]


def _deserialize_jsonl(data: bytes) -> list[PageTextRecord]:
    rows: list[PageTextRecord] = []
    for line in data.decode("utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(_coerce_row(json.loads(stripped)))
    return rows


_DESERIALIZERS = {
    DatasetFormat.PARQUET: _deserialize_parquet,
    DatasetFormat.JSONL: _deserialize_jsonl,
}


def deserialize_dataset(data: bytes, fmt: DatasetFormat) -> list[PageTextRecord]:
    """Decode dataset bytes in the given format back into rows."""
    return _DESERIALIZERS[fmt](data)


def load_page_dataset(path: Path, fmt: DatasetFormat) -> list[PageTextRecord]:
    """Read a per-page text dataset back from disk."""
    if not path.exists():
        raise ValueError(f"Dataset not found: {path}")
    return deserialize_dataset(path.read_bytes(), fmt)


async def import_dataset(
    store: Store,
    rows: list[PageTextRecord],
    *,
    on_progress: DetailedProgressCallback = noop_callback,
) -> ImportResult:
    """Re-chunk and re-embed *rows* under the current embedder.

    Each source's pages are embedded and stored as detached ``IMPORTED``
    chunks plus their page texts. Raises ``EmbeddingModelMismatchError`` (before
    any write) when the store was built by a different embedder.
    """
    store.assert_embedding_compatible()
    by_source: dict[str, list[PageTextRecord]] = {}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row)

    imported: list[str] = []
    total_pages = 0
    total_chunks = 0
    for name, source_rows in by_source.items():
        source_rows.sort(key=lambda r: r["page"])
        content_type = source_rows[0]["content_type"] or "text"
        page_texts = [(r["page"], r["text"]) for r in source_rows]
        chunks = await chunk_and_embed_pages(page_texts, name, content_type, on_progress)
        # One locked transaction (cleanup + chunks + page texts + source row) so a
        # failure can't leave the source with its old rows deleted and no new ones;
        # the embedding-dim check inside runs before the cleanup delete.
        store.write_chunks_batch(
            [
                ChunkWrite(
                    source=name,
                    file_hash="",
                    records=cast(list[dict], chunks),
                    needs_cleanup=True,
                    page_texts=[dict(r) for r in source_rows],
                    source_type=SourceType.IMPORTED,
                )
            ]
        )
        imported.append(name)
        total_pages += len(source_rows)
        total_chunks += len(chunks)
    return ImportResult(sources=sorted(imported), pages=total_pages, chunks=total_chunks)
