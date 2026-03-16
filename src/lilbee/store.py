"""LanceDB vector store operations."""

import logging
from datetime import UTC, datetime
from typing import Required

import lancedb
import pyarrow as pa
from typing_extensions import TypedDict

from lilbee.config import CHUNKS_TABLE, SOURCES_TABLE, cfg

log = logging.getLogger(__name__)

# Ephemeral runtime flag — resets on process start, correct since FTS index
# may not exist yet.  Set True after ensure_fts_index() succeeds.
_fts_index_ready = False


class SearchChunk(TypedDict, total=False):
    """A search result row from LanceDB — chunk fields plus score/distance.

    Core fields are Required (always present).  Score fields are optional
    because hybrid results carry ``_relevance_score`` while vector-only
    results carry ``_distance``.
    """

    source: Required[str]
    content_type: Required[str]
    page_start: Required[int]
    page_end: Required[int]
    line_start: Required[int]
    line_end: Required[int]
    chunk: Required[str]
    chunk_index: Required[int]
    vector: Required[list[float]]
    _distance: float
    _relevance_score: float


def _chunks_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("source", pa.utf8()),
            pa.field("content_type", pa.utf8()),
            pa.field("page_start", pa.int32()),
            pa.field("page_end", pa.int32()),
            pa.field("line_start", pa.int32()),
            pa.field("line_end", pa.int32()),
            pa.field("chunk", pa.utf8()),
            pa.field("chunk_index", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), cfg.embedding_dim)),
        ]
    )


def _sources_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("filename", pa.utf8()),
            pa.field("file_hash", pa.utf8()),
            pa.field("ingested_at", pa.utf8()),
            pa.field("chunk_count", pa.int32()),
        ]
    )


def get_db() -> lancedb.DBConnection:
    cfg.lancedb_dir.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(cfg.lancedb_dir))


def _table_names(db: lancedb.DBConnection) -> list[str]:
    """Get list of table names, handling the ListTablesResponse object."""
    result = db.list_tables()
    return result.tables if hasattr(result, "tables") else list(result)


def ensure_table(db: lancedb.DBConnection, name: str, schema: pa.Schema) -> lancedb.table.Table:
    if name in _table_names(db):
        return db.open_table(name)
    try:
        return db.create_table(name, schema=schema)
    except ValueError:
        return db.open_table(name)


def _open_table(name: str) -> lancedb.table.Table | None:
    """Open a table if it exists, otherwise return None."""
    db = get_db()
    if name not in _table_names(db):
        return None
    return db.open_table(name)


def safe_delete(table: lancedb.table.Table, predicate: str) -> None:
    """Delete rows matching predicate, logging on failure."""
    try:
        table.delete(predicate)
    except Exception:
        log.warning("Failed to delete rows matching: %s", predicate, exc_info=True)


def _escape_sql_string(value: str) -> str:
    """Escape single quotes for SQL predicates."""
    return value.replace("'", "''")


def ensure_fts_index() -> None:
    """Create or replace the FTS index on the chunks table.

    No-op when the table doesn't exist or is empty.  Sets _fts_index_ready
    on success so hybrid_search can be used.
    """
    global _fts_index_ready
    table = _open_table(CHUNKS_TABLE)
    if table is None:
        return
    try:
        table.create_fts_index("chunk", replace=True)
        _fts_index_ready = True
        log.debug("FTS index created/replaced on '%s'", CHUNKS_TABLE)
    except Exception:
        log.debug("FTS index creation failed (empty table?)", exc_info=True)


def add_chunks(records: list[dict]) -> int:
    """Add chunk records to the store. Returns count added."""
    global _fts_index_ready
    _fts_index_ready = False
    if not records:
        return 0
    for rec in records:
        vec = rec.get("vector", [])
        if len(vec) != cfg.embedding_dim:
            raise ValueError(
                f"Vector dimension mismatch: expected {cfg.embedding_dim}, got {len(vec)} "
                f"(source={rec.get('source', '?')})"
            )
    db = get_db()
    table = ensure_table(db, CHUNKS_TABLE, _chunks_schema())
    table.add(records)
    return len(records)


def _hybrid_search(
    table: lancedb.table.Table,
    query_text: str,
    query_vector: list[float],
    top_k: int,
) -> list[SearchChunk]:
    """Run hybrid (vector + FTS) search with RRF reranking."""
    from lancedb.rerankers import RRFReranker

    results: list[SearchChunk] = (
        table.search(query_type="hybrid")
        .vector(query_vector)
        .text(query_text)
        .rerank(RRFReranker())
        .limit(top_k)
        .to_list()
    )
    return results


def search(
    query_vector: list[float],
    top_k: int | None = None,
    max_distance: float | None = None,
    query_text: str | None = None,
) -> list[SearchChunk]:
    """Search for similar chunks — hybrid when FTS index is available, else vector-only.

    Results with distance > max_distance are filtered out (vector-only path).
    Pass max_distance=0 to disable filtering.
    """
    if top_k is None:
        top_k = cfg.top_k
    if max_distance is None:
        max_distance = cfg.max_distance
    table = _open_table(CHUNKS_TABLE)
    if table is None:
        return []

    if query_text and not _fts_index_ready:
        ensure_fts_index()

    if query_text and _fts_index_ready:
        try:
            return _hybrid_search(table, query_text, query_vector, top_k)
        except Exception:
            log.debug("Hybrid search failed, falling back to vector-only", exc_info=True)

    results: list[SearchChunk] = table.search(query_vector).metric("cosine").limit(top_k).to_list()
    if max_distance > 0:
        results = [r for r in results if r["_distance"] <= max_distance]
    return results


def get_chunks_by_source(source: str) -> list[SearchChunk]:
    """Return all chunks for a given source file."""
    table = _open_table(CHUNKS_TABLE)
    if table is None:
        return []
    escaped = _escape_sql_string(source)
    rows: list[SearchChunk] = table.search().where(f"source = '{escaped}'").to_list()
    return rows


def delete_by_source(source: str) -> None:
    """Delete all chunks from a given source file."""
    table = _open_table(CHUNKS_TABLE)
    if table is not None:
        safe_delete(table, f"source = '{_escape_sql_string(source)}'")


def get_sources() -> list[dict]:
    """Get all tracked source file records."""
    table = _open_table(SOURCES_TABLE)
    if table is None:
        return []
    result: list[dict] = table.to_arrow().to_pylist()
    return result


def upsert_source(filename: str, file_hash: str, chunk_count: int) -> None:
    """Add or update a source file tracking record."""
    db = get_db()
    table = ensure_table(db, SOURCES_TABLE, _sources_schema())
    safe_delete(table, f"filename = '{_escape_sql_string(filename)}'")
    table.add(
        [
            {
                "filename": filename,
                "file_hash": file_hash,
                "ingested_at": datetime.now(UTC).isoformat(),
                "chunk_count": chunk_count,
            }
        ]
    )


def delete_source(filename: str) -> None:
    """Remove a source file tracking record."""
    table = _open_table(SOURCES_TABLE)
    if table is not None:
        safe_delete(table, f"filename = '{_escape_sql_string(filename)}'")


def drop_all() -> None:
    """Drop all tables — used by rebuild."""
    global _fts_index_ready
    _fts_index_ready = False
    db = get_db()
    for name in _table_names(db):
        db.drop_table(name)
