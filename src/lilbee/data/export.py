"""Per-page text dataset: build/write from a store, and import one back."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import IO, TYPE_CHECKING, cast

from lilbee.data.extract.document import _title_scope, chunk_and_embed_pages
from lilbee.data.store import ChunkWrite, PageTextRecord, SourceMeta, SourceType
from lilbee.data.title import derive_title
from lilbee.runtime.progress import DetailedProgressCallback, noop_callback

if TYPE_CHECKING:
    import pyarrow as pa

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


def build_page_dataset(store: Store, source: str | None = None) -> pa.Table:
    """Collect per-page text rows as an Arrow table for every source (or *source*).

    Sources captured at ingest are read verbatim from the page-text table in a
    single columnar scan. Sources without captured text (older indexes, code) are
    reconstructed from the chunks table; that reconstruction concatenates chunk
    text per page, so chunk overlap may repeat a little text across page
    boundaries. The whole set stays in Arrow so the writers avoid per-row Python
    objects, and the per-source query loop that hung a large export is gone
    (bb-bqg).
    """
    import pyarrow as pa

    tracked = _with_wide_offsets(store.sources_arrow())
    if source is not None:
        table = _with_wide_offsets(store.page_texts_arrow(source))
        if table.num_rows == 0:
            table = _reconstructed_arrow(store, [source], table.schema)
    else:
        # One scan for every captured page instead of a filtered query per source:
        # the per-source loop was O(sources) and its fixed per-query overhead, not
        # data size, hung the export on a large store. The semi-join restricts it
        # to tracked sources, so an orphaned page-text row (source record gone)
        # stays out, matching the old get_sources()-scoped universe.
        keys = tracked.select(["source"])
        table = _with_wide_offsets(store.page_texts_arrow()).join(
            keys, keys="source", join_type="left semi"
        )
        # Tracked sources the scan found no page text for: older indexes and code,
        # rebuilt from their chunks. Normally empty, and an anti-join keeps the
        # comparison in Arrow rather than differencing two sets of every filename.
        missing = keys.join(table.select(["source"]), keys="source", join_type="left anti")
        extra = _reconstructed_arrow(
            store, sorted(missing.column("source").to_pylist()), table.schema
        )
        if extra.num_rows:
            table = pa.concat_tables([table, extra])
    # Denormalize each source's title/authors/created_at onto its page rows, so an
    # export/import cycle keeps them instead of falling back to the filename stem.
    # Left outer: a source with no metadata keeps its pages, with nulls.
    table = table.join(tracked, keys="source", join_type="left outer")
    return table.sort_by([("source", "ascending"), ("page", "ascending")])


def _with_wide_offsets(table: pa.Table) -> pa.Table:
    """*table* with its string columns retyped to 64-bit offsets.

    pyarrow's ``string`` addresses a column's data with int32 offsets, capping it
    at 2GB, and every step below this one (filter, concat, metadata append, sort)
    materializes one array per column. A corpus whose page text passes 2GB
    overflows there, so the export widens on the way out of the scan; the store's
    own schema is untouched. Both writers take the wider type, and parquet records
    it in its arrow metadata and reads it back as ``large_string``, so the rows an
    import decodes are ordinary strings either way.
    """
    import pyarrow as pa

    return table.cast(
        pa.schema(
            [
                field.with_type(pa.large_string()) if pa.types.is_string(field.type) else field
                for field in table.schema
            ]
        )
    )


def _reconstructed_arrow(store: Store, sources: list[str], schema: pa.Schema) -> pa.Table:
    """Chunk-reconstructed pages for *sources* as an Arrow table in *schema*."""
    import pyarrow as pa

    records = [dict(r) for name in sources for r in _reconstruct_from_chunks(store, name)]
    return pa.Table.from_pylist(records, schema=schema)


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


# Rows encoded per write. A page row is a few hundred bytes of text, so this
# keeps a batch in the low megabytes whatever the corpus size.
_WRITE_BATCH_ROWS = 10_000


def _write_parquet(table: pa.Table, sink: IO[bytes]) -> None:
    """Encode *table* into *sink* as parquet, one row group per batch.

    Build-vs-buy: pyarrow's own writer does the batching, and feeding a
    ParquetWriter row group by row group produces byte-identical output.
    """
    import pyarrow.parquet as pq

    pq.write_table(table, sink, row_group_size=_WRITE_BATCH_ROWS)


def _write_jsonl(table: pa.Table, sink: IO[bytes]) -> None:
    """Encode *table* into *sink* as jsonl, one write per batch.

    One JSON object per row, keyed by the table's columns, so jsonl stays in step
    with the schema (and with parquet) rather than a hardcoded field list. Only a
    batch is converted to Python objects at a time: converting the whole table
    cost about 19x the size of the file it produced (77GB peak for a 4.15GB
    export of an 8.8M-row corpus), because the row dicts, the joined string and
    its encoded bytes were all live at once.
    """
    for batch in table.to_batches(max_chunksize=_WRITE_BATCH_ROWS):
        sink.write("".join(json.dumps(row) + "\n" for row in batch.to_pylist()).encode("utf-8"))


_WRITERS = {DatasetFormat.PARQUET: _write_parquet, DatasetFormat.JSONL: _write_jsonl}


def serialize_dataset(table: pa.Table, fmt: DatasetFormat) -> bytes:
    """Encode the dataset *table* to bytes in the given format.

    For callers that must hand back one buffer (the HTTP download). A file
    export uses :func:`write_dataset`, which never holds the encoded dataset.
    """
    import io

    buffer = io.BytesIO()
    _WRITERS[fmt](table, buffer)
    return buffer.getvalue()


def write_dataset(table: pa.Table, path: Path, fmt: DatasetFormat) -> None:
    """Write the dataset *table* to *path* in the given format, a batch at a time."""
    with path.open("wb") as sink:
        _WRITERS[fmt](table, sink)


def _coerce_row(raw: dict) -> PageTextRecord:
    """Validate one raw dataset row into a `PageTextRecord`.

    The denormalized source metadata (title/authors/created_at) is carried
    through when present so a file export/import cycle preserves it.
    """
    try:
        row = PageTextRecord(
            source=str(raw["source"]),
            page=int(raw["page"]),
            text=str(raw["text"]),
            content_type=str(raw.get("content_type", "")),
        )
    except (KeyError, TypeError, ValueError):
        raise ValueError("Dataset row is missing required source/page/text fields") from None
    if raw.get("title") is not None:
        row["title"] = str(raw["title"])
    if raw.get("authors") is not None:
        row["authors"] = str(raw["authors"])
    if raw.get("created_at") is not None:
        row["created_at"] = str(raw["created_at"])
    return row


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


def _page_text_row(row: PageTextRecord) -> dict:
    """Project a dataset row down to the ``_page_texts`` columns.

    A dataset carries the source's metadata denormalized on every page row; the
    page-texts table has no such columns, so they are dropped before the write.
    """
    return {
        "source": row["source"],
        "page": row["page"],
        "text": row["text"],
        "content_type": row["content_type"],
    }


def _source_meta_from_rows(rows: list[PageTextRecord], name: str) -> SourceMeta:
    """Recover a source's extraction metadata from its dataset rows.

    The values are identical on every page row, so the first carries them. A
    dataset exported before the metadata columns existed has none, in which case
    the title falls back to the cleaned filename stem.
    """
    first: dict = dict(rows[0]) if rows else {}
    stored = first.get("title")
    title = stored.strip() if isinstance(stored, str) and stored.strip() else derive_title(name)
    return SourceMeta(
        title=title,
        authors=first.get("authors") or "",
        created_at=first.get("created_at") or "",
    )


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
        # Datasets exported with the metadata columns round-trip the extracted
        # title/authors/created_at; older ones carry none, so fall back to the
        # stem-derived title that keeps imported chunks visible to the title arm.
        meta = _source_meta_from_rows(source_rows, name)
        title = meta.title
        with _title_scope(title):
            chunks = await chunk_and_embed_pages(page_texts, name, content_type, on_progress)
        for chunk in chunks:
            chunk["title"] = title or None
        # One locked transaction (cleanup + chunks + page texts + source row) so a
        # failure can't leave the source with its old rows deleted and no new ones;
        # the embedding-dim check inside runs before the cleanup delete.
        await asyncio.to_thread(
            store.write_chunks_batch,
            [
                ChunkWrite(
                    source=name,
                    file_hash="",
                    records=cast(list[dict], chunks),
                    needs_cleanup=True,
                    page_texts=[_page_text_row(r) for r in source_rows],
                    source_type=SourceType.IMPORTED,
                    meta=meta,
                )
            ],
        )
        imported.append(name)
        total_pages += len(source_rows)
        total_chunks += len(chunks)
    return ImportResult(sources=sorted(imported), pages=total_pages, chunks=total_chunks)
