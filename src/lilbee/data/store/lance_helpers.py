"""LanceDB plumbing helpers: table introspection, safe deletes, SQL escaping, error text."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from lilbee.catalog.refs import hf_repo_from_ref
from lilbee.runtime.lock import write_lock

from .types import LOCAL_OWNER, ChunkType

if TYPE_CHECKING:
    from pathlib import Path

    import pyarrow as pa
    from lancedb.db import LanceDBConnection
    from lancedb.index import IndexConfig
    from lancedb.table import LanceTable

log = logging.getLogger(__name__)

# The chunks table's body-text column: FTS-indexed and searched by the lexical
# arm, named here so the store's query and index code shares one spelling with
# its sibling _TITLE_COLUMN.
_CHUNK_COLUMN = "chunk"

# Index types as LanceDB's IndexConfig reports them.
_FTS_INDEX_TYPE = "FTS"
_SCALAR_INDEX_TYPES = frozenset({"bitmap", "btree"})


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


def table_names(db: LanceDBConnection) -> list[str]:
    """Every table name in *db*, following ``list_tables`` pages to the end."""
    names: list[str] = []
    page_token: str | None = None
    while True:
        page = db.list_tables(page_token=page_token)
        names.extend(page.tables)
        page_token = page.page_token
        if not page_token:
            return names


def ensure_table(db: LanceDBConnection, name: str, schema: pa.Schema) -> LanceTable:
    table: LanceTable
    if name in table_names(db):
        table = db.open_table(name)
        return table
    try:
        table = db.create_table(name, schema=schema)
    except ValueError:
        table = db.open_table(name)
    return table


def _safe_delete_unlocked(table: LanceTable, predicate: str) -> bool:
    """Delete rows matching predicate. Caller must hold write lock.

    Returns True when the delete succeeded, False when it raised (logged). The
    return lets delete-then-add callers avoid corrupting state on a swallowed
    failure (e.g. inserting a row whose stale predecessor was never removed).
    """
    try:
        table.delete(predicate)
        return True
    except Exception:
        log.warning("Failed to delete rows matching: %s", predicate, exc_info=True)
        return False


def safe_delete(table: LanceTable, predicate: str, lancedb_dir: Path | None = None) -> bool:
    """Delete rows matching predicate, logging on failure. Returns success.

    Pass the store's ``lancedb_dir`` so the write lock coordinates on that
    instance's data dir; ``None`` falls back to the global config dir.
    """
    with write_lock(lancedb_dir):
        return _safe_delete_unlocked(table, predicate)


def escape_sql_string(value: str) -> str:
    """Escape a value for a single-quoted SQL string literal in a LanceDB predicate.

    LanceDB's Datafusion engine follows standard SQL: the only escape inside a
    ``'...'`` literal is doubling the single quote. Backslash is an ordinary
    character, so escaping it (``\\`` -> ``\\\\``) corrupts the literal and makes a
    value containing a backslash (e.g. a Windows path) never match.
    """
    return value.replace("'", "''")


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

    ``'raw'`` means document content, so it matches extracted table rows as
    well as raw ones, and the NULL rows written before the column existed.
    Scoping a search to the user's documents must not silently drop their
    tables. A ``'wiki'`` filter matches only generated pages.
    """
    escaped = escape_sql_string(chunk_type)
    if chunk_type == ChunkType.RAW:
        return f"(chunk_type IN ('{ChunkType.RAW}', '{ChunkType.TABLE}') OR chunk_type IS NULL)"
    return f"chunk_type = '{escaped}'"


def _index_registry(table: LanceTable) -> list[IndexConfig] | None:
    """The indexes registered on *table*, or ``None`` when the registry is unreadable.

    lancedb opens each FTS index's files while it lists the indexes, so the
    listing raises when an FTS index has lost its files. ``None`` reports
    that state; an empty list means the table has no index. The listing does
    not open scalar or vector index files, so :func:`_dangling_indices` checks
    those.
    """
    try:
        return list(table.list_indices())
    except Exception:
        return None


def _has_fts_index(indices: list[IndexConfig], column: str = _CHUNK_COLUMN) -> bool:
    """True when *indices* hold an FTS index on *column*."""
    return any(idx.index_type == _FTS_INDEX_TYPE and column in idx.columns for idx in indices)


def _has_scalar_index(indices: list[IndexConfig], column: str) -> bool:
    """True when *indices* hold a scalar index on *column*.

    lilbee builds only scalar indexes on the columns it prefilters by, never an
    FTS or vector index, so any index touching *column* is the scalar one.
    """
    return any(column in idx.columns for idx in indices)


def _is_vector_index(idx: IndexConfig) -> bool:
    """True when *idx* is an ANN index on the vector column.

    LanceDB reports IVF index types as ``IvfPq`` / ``IvfFlat`` etc., so the
    family match is case-insensitive.
    """
    return "IVF" in idx.index_type.upper() and "vector" in idx.columns


def _has_vector_index(indices: list[IndexConfig]) -> bool:
    """True when *indices* hold an ANN index on the vector column."""
    return any(_is_vector_index(idx) for idx in indices)


def _dangling_indices(
    table: LanceTable, indices: list[IndexConfig], lancedb_dir: Path
) -> list[IndexConfig]:
    """The indexes in *indices* whose directory under ``_indices`` is gone.

    LanceDB keeps the registration after the index directory is removed, and
    every query through the index then fails on a missing file. The listing,
    ``optimize()`` and ``index_stats()`` do not notice for scalar and vector
    indexes; only the directory does.
    """
    indices_dir = lancedb_dir / f"{table.name}.lance" / "_indices"
    # An index without a uuid has no directory under _indices to check.
    return [
        idx
        for idx in indices
        if idx.index_uuid is not None and not (indices_dir / idx.index_uuid).is_dir()
    ]


def _scalar_index_dangling(
    table: LanceTable, indices: list[IndexConfig], lancedb_dir: Path
) -> list[str]:
    """Column names whose scalar (BITMAP/BTree) index is registered but its files are gone."""
    columns = [
        column
        for idx in _dangling_indices(table, indices, lancedb_dir)
        if idx.index_type.lower() in _SCALAR_INDEX_TYPES
        for column in idx.columns
    ]
    return list(dict.fromkeys(columns))


def _vector_index_dangling(
    table: LanceTable, indices: list[IndexConfig], lancedb_dir: Path
) -> bool:
    """True when the ANN index on the vector column is registered but its files are gone."""
    return any(_is_vector_index(idx) for idx in _dangling_indices(table, indices, lancedb_dir))


def _escape_like_wildcards(value: str) -> str:
    """Escape LIKE metacharacters so a search term matches literally.

    ``%`` and ``_`` are wildcards inside a LIKE pattern; without escaping, a
    search for ``a_b`` would also match ``axb``. Backslash is escaped first
    because it is the ESCAPE character the predicate declares.
    """
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _sources_search_filter(search: str | None, *, include_title: bool = False) -> str | None:
    """Case-insensitive filename (and optionally title) WHERE clause, or ``None``.

    *include_title* requires the caller to have checked the column exists
    (pre-title stores lack it).
    """
    if not search:
        return None
    escaped = escape_sql_string(_escape_like_wildcards(search.lower()))
    clause = f"LOWER(filename) LIKE '%{escaped}%' ESCAPE '\\'"
    if include_title:
        clause = f"({clause} OR LOWER(title) LIKE '%{escaped}%' ESCAPE '\\')"
    return clause


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
