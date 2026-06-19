"""LanceDB plumbing helpers: table introspection, safe deletes, SQL escaping, error text."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from lilbee.catalog.refs import hf_repo_from_ref
from lilbee.runtime.lock import write_lock

from .types import LOCAL_OWNER, ChunkType

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


def local_owner_predicate() -> str:
    """SQL predicate selecting the local human's own memories."""
    return f"owner = '{LOCAL_OWNER}'"


def human_recall_predicate() -> str:
    """SQL predicate for the human: own memories plus any an agent has shared.

    The mirror of :func:`agent_recall_predicate`: ``shared=True`` on an agent
    memory means "expose to the human's TUI/CLI", so the human's view must
    include those rather than only ``owner = 'local'``.
    """
    return f"owner = '{LOCAL_OWNER}' OR (shared = true AND owner != '{LOCAL_OWNER}')"


def agent_recall_predicate(owner: str) -> str:
    """SQL predicate for an agent: its own memories plus the human's shared ones."""
    return f"owner = '{escape_sql_string(owner)}' OR (shared = true AND owner = '{LOCAL_OWNER}')"


def _chunk_type_predicate(chunk_type: ChunkType | str) -> str:
    """SQL predicate that matches ``chunk_type`` while tolerating NULL rows.

    Rows written before ``chunk_type`` was populated land as NULL. They
    are semantically raw, so a ``'raw'`` filter still includes them; a
    ``'wiki'`` filter excludes them.
    """
    escaped = escape_sql_string(chunk_type)
    if chunk_type == ChunkType.RAW:
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


def _has_vector_index(table: lancedb.table.Table) -> bool:
    """Return True when an ANN index on the vector column already exists.

    LanceDB reports IVF index types as ``IvfPq`` / ``IvfFlat`` etc., so the
    family match is case-insensitive.
    """
    try:
        for idx in table.list_indices():
            if "IVF" in idx.index_type.upper() and "vector" in idx.columns:
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


def refs_compatible(
    persisted_ref: str,
    current_ref: str,
    persisted_dim: int,
    current_dim: int,
) -> bool:
    """Return True when *persisted_ref* and *current_ref* describe the same embedder.

    Compatible iff dims match and either the raw refs are equal or the persisted
    ref is the legacy bare-repo form (``<org>/<repo>`` without a ``.gguf``
    filename) whose repo matches the current canonical full ref. The legacy
    asymmetry exists because pre-canonical lilbee versions persisted only the
    repo; the current code persists the full ``<org>/<repo>/<filename>.gguf``.
    Two different ``.gguf`` files in the same repo are not lumped together
    (different quantizations can produce subtly different vectors), so both-
    full-ref strict identity is preserved.
    """
    if persisted_dim != current_dim:
        return False
    if persisted_ref == current_ref:
        return True
    if persisted_ref.endswith(".gguf"):
        return False
    if not current_ref.endswith(".gguf"):
        return False
    return hf_repo_from_ref(current_ref) == persisted_ref
