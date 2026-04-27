"""LanceDB plumbing helpers: table introspection, safe deletes, SQL escaping, error text."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from lilbee.lock import write_lock

from .types import CHUNK_TYPE_RAW

if TYPE_CHECKING:
    import lancedb
    import lancedb.table
    import pyarrow as pa

log = logging.getLogger(__name__)


def install_lancedb_thread_error_suppressor() -> None:
    """Install a ``threading.excepthook`` that swallows lancedb shutdown noise.
    lancedb has no ``close()`` API and its internal event loop thread crashes
    during Python interpreter teardown. The exception is harmless (the process
    is exiting anyway) but pollutes CLI/TUI output. This is opt-in so importing
    ``lilbee.data.store`` has no hidden side effects; call it once from the CLI/TUI
    bootstrap.
    """
    original = threading.excepthook

    def _hook(args: threading.ExceptHookArgs) -> None:
        if args.thread and "LanceDB" in args.thread.name:
            return
        original(args)

    threading.excepthook = _hook


def _table_names(db: lancedb.DBConnection) -> list[str]:
    """Get list of table names, handling the ListTablesResponse object."""
    result = db.list_tables()
    try:
        return result.tables  # type: ignore[no-any-return, union-attr]
    except AttributeError:
        return list(result)  # type: ignore[arg-type]


def ensure_table(db: lancedb.DBConnection, name: str, schema: pa.Schema) -> lancedb.table.Table:
    if name in _table_names(db):
        return db.open_table(name)
    try:
        return db.create_table(name, schema=schema)
    except ValueError:
        return db.open_table(name)


def _safe_delete_unlocked(table: lancedb.table.Table, predicate: str) -> None:
    """Delete rows matching predicate, logging on failure. Caller must hold write lock."""
    try:
        table.delete(predicate)
    except Exception:
        log.warning("Failed to delete rows matching: %s", predicate, exc_info=True)


def safe_delete(table: lancedb.table.Table, predicate: str) -> None:
    """Delete rows matching predicate, logging on failure."""
    with write_lock():
        _safe_delete_unlocked(table, predicate)


def escape_sql_string(value: str) -> str:
    """Escape single quotes for SQL predicates."""
    return value.replace("\\", "\\\\").replace("'", "''")


def _chunk_type_predicate(chunk_type: str) -> str:
    """SQL predicate that matches ``chunk_type`` while tolerating NULL rows.

    Rows written before ``chunk_type`` was populated land as NULL. They
    are semantically raw, so a ``'raw'`` filter still includes them; a
    ``'wiki'`` filter excludes them.
    """
    escaped = escape_sql_string(chunk_type)
    if chunk_type == CHUNK_TYPE_RAW:
        return f"(chunk_type = '{escaped}' OR chunk_type IS NULL)"
    return f"chunk_type = '{escaped}'"


def _has_fts_index(table: lancedb.table.Table) -> bool:
    """Return True when an FTS index on the chunk column already exists."""
    try:
        for idx in table.list_indices():
            if idx.index_type == "FTS" and "chunk" in idx.columns:
                return True
    except Exception:
        return False
    return False


def _sources_search_filter(search: str | None) -> str | None:
    """Case-insensitive filename WHERE clause, or ``None`` for empty *search*."""
    if not search:
        return None
    escaped = escape_sql_string(search.lower())
    return f"LOWER(filename) LIKE '%{escaped}%'"


def _embedding_mismatch_message(
    persisted_model: str,
    persisted_dim: int,
    current_model: str,
    current_dim: int,
) -> str:
    return (
        f"The vector store was built with embedding model '{persisted_model}' "
        f"(dim {persisted_dim}), but lilbee is now configured to use "
        f"'{current_model}' (dim {current_dim}). Search and ingest are disabled "
        "until the store is rebuilt under the new model. "
        'Run `lilbee rebuild` or POST /api/sync with `{"force_rebuild": true}`.'
    )
