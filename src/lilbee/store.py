"""LanceDB vector store operations."""

import logging
from datetime import UTC, datetime

import lancedb
import pyarrow as pa

from lilbee.config import CHUNKS_TABLE, EMBEDDING_DIM, LANCEDB_DIR, SOURCES_TABLE, TOP_K

log = logging.getLogger(__name__)

_CHUNKS_SCHEMA = pa.schema(
    [
        pa.field("source", pa.utf8()),
        pa.field("content_type", pa.utf8()),
        pa.field("page_start", pa.int32()),
        pa.field("page_end", pa.int32()),
        pa.field("line_start", pa.int32()),
        pa.field("line_end", pa.int32()),
        pa.field("chunk", pa.utf8()),
        pa.field("chunk_index", pa.int32()),
        pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
    ]
)

_SOURCES_SCHEMA = pa.schema(
    [
        pa.field("filename", pa.utf8()),
        pa.field("file_hash", pa.utf8()),
        pa.field("ingested_at", pa.utf8()),
        pa.field("chunk_count", pa.int32()),
    ]
)


def _get_db() -> lancedb.DBConnection:
    LANCEDB_DIR.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(LANCEDB_DIR))


def _table_names(db: lancedb.DBConnection) -> list[str]:
    """Get list of table names, handling the ListTablesResponse object."""
    result = db.list_tables()
    return result.tables if hasattr(result, "tables") else list(result)


def _ensure_table(db: lancedb.DBConnection, name: str, schema: pa.Schema) -> lancedb.table.Table:
    if name in _table_names(db):
        return db.open_table(name)
    return db.create_table(name, schema=schema)


def _open_table(name: str) -> lancedb.table.Table | None:
    """Open a table if it exists, otherwise return None."""
    db = _get_db()
    if name not in _table_names(db):
        return None
    return db.open_table(name)


def _safe_delete(table: lancedb.table.Table, predicate: str) -> None:
    """Delete rows matching predicate, logging on failure."""
    try:
        table.delete(predicate)
    except Exception:
        log.warning("Failed to delete rows matching: %s", predicate, exc_info=True)


def add_chunks(records: list[dict]) -> int:
    """Add chunk records to the store. Returns count added."""
    if not records:
        return 0
    db = _get_db()
    table = _ensure_table(db, CHUNKS_TABLE, _CHUNKS_SCHEMA)
    table.add(records)
    return len(records)


def search(query_vector: list[float], top_k: int = TOP_K) -> list[dict]:
    """Search for similar chunks by vector similarity."""
    table = _open_table(CHUNKS_TABLE)
    if table is None:
        return []
    result: list[dict] = table.search(query_vector).limit(top_k).to_list()
    return result


def delete_by_source(source: str) -> None:
    """Delete all chunks from a given source file."""
    table = _open_table(CHUNKS_TABLE)
    if table is not None:
        _safe_delete(table, f"source = '{source}'")


def get_sources() -> list[dict]:
    """Get all tracked source file records."""
    table = _open_table(SOURCES_TABLE)
    if table is None:
        return []
    result: list[dict] = table.to_arrow().to_pylist()
    return result


def upsert_source(filename: str, file_hash: str, chunk_count: int) -> None:
    """Add or update a source file tracking record."""
    db = _get_db()
    table = _ensure_table(db, SOURCES_TABLE, _SOURCES_SCHEMA)
    _safe_delete(table, f"filename = '{filename}'")
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
        _safe_delete(table, f"filename = '{filename}'")


def drop_all() -> None:
    """Drop all tables — used by rebuild."""
    db = _get_db()
    for name in _table_names(db):
        db.drop_table(name)
