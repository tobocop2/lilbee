"""The ``Store`` class: high-level LanceDB read/write API used across lilbee."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from lilbee.core.config import (
    CHUNKS_TABLE,
    CITATIONS_TABLE,
    META_TABLE,
    SOURCES_TABLE,
    Config,
)
from lilbee.core.security import validate_path_within
from lilbee.runtime.lock import write_lock

from .lance_helpers import (
    _chunk_type_predicate,
    _embedding_mismatch_message,
    _has_fts_index,
    _safe_delete_unlocked,
    _sources_search_filter,
    _table_names,
    ensure_table,
    escape_sql_string,
    refs_compatible,
)
from .ranking import mmr_rerank
from .schema import _citations_schema, _meta_schema, _sources_schema
from .types import (
    META_DELETE_ALL_PREDICATE,
    META_SCHEMA_VERSION,
    READ_CONSISTENCY_INTERVAL,
    CitationRecord,
    EmbeddingModelMismatchError,
    RemoveResult,
    SearchChunk,
    SourceRecord,
    StoreMeta,
)

if TYPE_CHECKING:
    import lancedb
    import lancedb.table

log = logging.getLogger(__name__)


def _hybrid_search(
    table: lancedb.table.Table,
    query_text: str,
    query_vector: list[float],
    top_k: int,
    chunk_type: str | None = None,
) -> list[SearchChunk]:
    """Run hybrid (vector + FTS) search with RRF reranking.

    When ``chunk_type`` is set, the predicate is pushed into the query so
    the limit applies *after* the type filter. Post-filtering would
    silently starve wiki-only queries whose matches live past the top-K
    hybrid window.
    """
    from lancedb.rerankers import RRFReranker

    query = (
        table.search(query_type="hybrid")
        .vector(query_vector)
        .text(query_text)
        .rerank(RRFReranker())
    )
    if chunk_type:
        query = query.where(_chunk_type_predicate(chunk_type))
    rows = query.limit(top_k).to_list()
    return [SearchChunk(**r) for r in rows]


_MAX_THRESHOLD = 1.0
_MAX_FILTER_ITERATIONS = 20  # safety cap to prevent runaway loops


def _get_distance(chunk: SearchChunk) -> float:
    """Extract distance as a sortable float (inf for None)."""
    return chunk.distance if chunk.distance is not None else float("inf")


def _count_within_threshold(sorted_results: list[SearchChunk], threshold: float) -> int:
    """Count results whose distance is within the given threshold."""
    for i, r in enumerate(sorted_results):
        if _get_distance(r) > threshold:
            return i
    return len(sorted_results)


class Store:
    """LanceDB vector store: wraps all DB operations with config-driven defaults."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._fts_ready: bool = False
        self._db: lancedb.DBConnection | None = None
        # Cache of {filename: ingested_at} rebuilt only when sources
        # mutate; callers (temporal filter) hit it per-query.
        self._source_ingested_cache: dict[str, str] | None = None

    def _invalidate_source_cache(self) -> None:
        """Drop the cached {filename: ingested_at} map."""
        self._source_ingested_cache = None

    def source_ingested_at_map(self) -> dict[str, str]:
        """Return {filename: ingested_at} for every source, cached until mutation.

        Best-effort: a reader racing a concurrent invalidation can store a
        pre-mutation snapshot. The only consumer (temporal query filter)
        treats a missing/stale entry as "do not filter," so staleness
        degrades ranking precision, never correctness.
        """
        if self._source_ingested_cache is not None:
            return self._source_ingested_cache
        mapping = {s["filename"]: s.get("ingested_at", "") for s in self.get_sources()}
        self._source_ingested_cache = mapping
        return mapping

    def _chunks_schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field("source", pa.utf8()),
                pa.field("content_type", pa.utf8()),
                pa.field("chunk_type", pa.utf8()),
                pa.field("page_start", pa.int32()),
                pa.field("page_end", pa.int32()),
                pa.field("line_start", pa.int32()),
                pa.field("line_end", pa.int32()),
                pa.field("chunk", pa.utf8()),
                pa.field("chunk_index", pa.int32()),
                pa.field("vector", pa.list_(pa.float32(), self._config.embedding_dim)),
            ]
        )

    def get_meta(self) -> StoreMeta | None:
        """Return the persisted store metadata row, or ``None`` if unset."""
        table = self.open_table(META_TABLE)
        if table is None:
            return None
        rows = table.search().limit(1).to_list()
        if not rows:
            return None
        row = rows[0]
        return StoreMeta(
            embedding_model=row["embedding_model"],
            embedding_dim=int(row["embedding_dim"]),
            schema_version=int(row["schema_version"]),
            updated_at=row["updated_at"],
        )

    def _write_meta_unlocked(self, *, embedding_model: str, embedding_dim: int) -> None:
        """Overwrite the single ``_meta`` row with the supplied identity.

        Caller must hold ``write_lock()``. Args are passed explicitly rather than
        re-read from ``self._config`` so the caller can snapshot cfg at a coherent
        instant and not race with a concurrent ``set_embedding_model``.
        """
        db = self.get_db()
        table = ensure_table(db, META_TABLE, _meta_schema())
        _safe_delete_unlocked(table, META_DELETE_ALL_PREDICATE)
        table.add(
            [
                {
                    "embedding_model": embedding_model,
                    "embedding_dim": embedding_dim,
                    "schema_version": META_SCHEMA_VERSION,
                    "updated_at": datetime.now(UTC).isoformat(),
                }
            ]
        )

    def _has_chunks(self) -> bool:
        """Return True when the chunks table exists and has at least one row."""
        chunks = self.open_table(CHUNKS_TABLE)
        return chunks is not None and chunks.count_rows() > 0

    def has_chunks(self) -> bool:
        """Public predicate: True iff the store currently holds at least one chunk."""
        return self._has_chunks()

    def initialize_meta_if_legacy(self) -> bool:
        """Pin a legacy store's identity to the current cfg if not already set.

        Returns ``True`` when a meta row was just written. No-op when meta already
        exists or no chunks are present. This is the path that converts a
        pre-upgrade store (chunks present, no ``_meta``) into a gated store. It
        snapshots cfg under the write lock to keep the recorded identity coherent
        with what the gate is comparing against.
        """
        if self.get_meta() is not None:
            return False
        if not self._has_chunks():
            return False
        embedding_model = self._config.embedding_model
        embedding_dim = self._config.embedding_dim
        with write_lock():
            # Re-check under the lock so two callers do not both warn-and-write.
            if self.get_meta() is not None:
                return False
            log.warning(
                "Legacy store has chunks but no _meta row. Initializing _meta from "
                "the current configuration (embedding_model=%s, embedding_dim=%d). "
                "If you changed embedding_model before upgrading, run `lilbee rebuild` "
                "to ensure the store is consistent.",
                embedding_model,
                embedding_dim,
            )
            self._write_meta_unlocked(embedding_model=embedding_model, embedding_dim=embedding_dim)
            return True

    def _ensure_embedding_compat(self) -> None:
        """Raise when the persisted embedding identity drifts from cfg.

        Pure check, no side effects. Migration of legacy stores (chunks present,
        no ``_meta``) is the caller's responsibility via ``initialize_meta_if_legacy``;
        rewriting a legacy bare-repo ``_meta`` row to the canonical full ref is
        the caller's responsibility via ``canonicalize_meta_if_legacy``. This
        method stays safe to call from inside an existing ``write_lock()`` (no
        recursive lock attempt). cfg fields are snapshotted at entry so the
        comparison is coherent even if another thread mutates them mid-call.
        """
        current_model = self._config.embedding_model
        current_dim = self._config.embedding_dim
        meta = self.get_meta()
        if meta is None:
            return
        if refs_compatible(
            meta["embedding_model"], current_model, meta["embedding_dim"], current_dim
        ):
            return
        raise EmbeddingModelMismatchError(
            _embedding_mismatch_message(
                persisted_model=meta["embedding_model"],
                persisted_dim=meta["embedding_dim"],
                current_model=current_model,
                current_dim=current_dim,
            )
        )

    def _needs_canonical_meta_rewrite(
        self, meta: StoreMeta | None, current_model: str, current_dim: int
    ) -> bool:
        """True iff *meta* is the legacy form and refs-compatible with current cfg."""
        if meta is None or meta["embedding_model"] == current_model:
            return False
        return refs_compatible(
            meta["embedding_model"], current_model, meta["embedding_dim"], current_dim
        )

    def canonicalize_meta_if_legacy(self) -> bool:
        """Rewrite a legacy bare-repo ``_meta`` row to the canonical full ref.

        Pre-canonical lilbee persisted only ``<org>/<repo>`` in
        ``_meta.embedding_model``. The current code persists the full
        ``<org>/<repo>/<filename>.gguf``. When the two refer to the same
        model under :func:`refs_compatible` but differ as raw strings, the
        meta row is rewritten so the legacy name never surfaces. Returns
        ``True`` on write; ``False`` when missing, already canonical, or
        incompatible (the gate handles incompatibility).
        """
        current_model = self._config.embedding_model
        current_dim = self._config.embedding_dim
        if not self._needs_canonical_meta_rewrite(self.get_meta(), current_model, current_dim):
            return False
        with write_lock():
            meta = self.get_meta()  # re-read under the lock for racing callers
            if not self._needs_canonical_meta_rewrite(meta, current_model, current_dim):
                return False
            assert meta is not None  # filtered above  # noqa: S101
            log.info(
                "Migrating legacy embedding ref in store meta: %r -> %r",
                meta["embedding_model"],
                current_model,
            )
            self._write_meta_unlocked(embedding_model=current_model, embedding_dim=current_dim)
            return True

    def get_db(self) -> lancedb.DBConnection:
        if self._db is None:
            import lancedb as _lancedb

            self._config.lancedb_dir.mkdir(parents=True, exist_ok=True)
            self._db = _lancedb.connect(
                str(self._config.lancedb_dir),
                read_consistency_interval=READ_CONSISTENCY_INTERVAL,
            )
        return self._db

    def open_table(self, name: str) -> lancedb.table.Table | None:
        """Open a table if it exists, otherwise return None."""
        db = self.get_db()
        if name not in _table_names(db):
            return None
        return db.open_table(name)

    def ensure_fts_index(self) -> None:
        """Create the chunks FTS index, or run ``optimize()`` once it exists.

        ``optimize()`` folds newly added rows into the FTS index and also
        runs LanceDB's default compaction + version pruning (default prune
        window: 7 days). Work scales with recent deltas rather than total
        chunk count, so large corpora no longer pay the full
        ``create_fts_index(replace=True)`` rebuild cost on every sync.
        """
        with write_lock():
            table = self.open_table(CHUNKS_TABLE)
            if table is None:
                return
            try:
                if _has_fts_index(table):
                    table.optimize()
                    log.debug("FTS index optimized on '%s'", CHUNKS_TABLE)
                else:
                    table.create_fts_index("chunk", replace=False)
                    log.debug("FTS index created on '%s'", CHUNKS_TABLE)
                self._fts_ready = True
            except Exception:
                log.debug("FTS index ensure failed (empty table?)", exc_info=True)

    def add_chunks(self, records: list[dict]) -> int:
        """Add chunk records to the store. Returns count added.

        Raises ``EmbeddingModelMismatchError`` if the persisted ``_meta`` row was
        written under a different embedding model than the current ``cfg``. On the
        first write to a fresh store, ``_meta`` is initialized from the current cfg.

        The gate runs inside the write lock and uses a single cfg snapshot so a
        concurrent ``set_embedding_model`` cannot slip a write in past a stale
        compatibility check.
        """
        with write_lock():
            embedding_model = self._config.embedding_model
            embedding_dim = self._config.embedding_dim
            self._ensure_embedding_compat()
            self._fts_ready = False
            if not records:
                return 0
            for rec in records:
                vec = rec.get("vector", [])
                if len(vec) != embedding_dim:
                    raise ValueError(
                        f"Vector dimension mismatch: expected {embedding_dim}, "
                        f"got {len(vec)} (source={rec.get('source', '?')})"
                    )
            db = self.get_db()
            table = ensure_table(db, CHUNKS_TABLE, self._chunks_schema())
            table.add(records)
            if self.get_meta() is None:
                self._write_meta_unlocked(
                    embedding_model=embedding_model, embedding_dim=embedding_dim
                )
            return len(records)

    def bm25_probe(self, query_text: str, top_k: int = 5) -> list[SearchChunk]:
        """Quick BM25-only search for confidence checking. Returns up to top_k results."""
        table = self.open_table(CHUNKS_TABLE)
        if table is None:
            return []
        if not self._fts_ready:
            self.ensure_fts_index()
        if not self._fts_ready:
            return []  # pragma: no cover
        try:
            rows = table.search(query_text, query_type="fts").limit(top_k).to_list()
            return [SearchChunk(**r) for r in rows]
        except Exception:
            log.debug("BM25 probe failed", exc_info=True)
            return []

    def search(
        self,
        query_vector: list[float],
        top_k: int | None = None,
        max_distance: float | None = None,
        query_text: str | None = None,
        chunk_type: str | None = None,
    ) -> list[SearchChunk]:
        """Search for similar chunks. Hybrid when FTS available, else vector-only.

        Results with distance > max_distance are filtered out (vector-only path).
        Pass max_distance=0 to disable filtering.
        When *chunk_type* is set, only chunks of that type ("raw" or "wiki") are returned.

        Raises ``EmbeddingModelMismatchError`` if the persisted ``_meta`` row was
        written under a different embedding model than the current ``cfg``.
        """
        if top_k is None:
            top_k = self._config.top_k
        if max_distance is None:
            max_distance = self._config.max_distance
        table = self.open_table(CHUNKS_TABLE)
        if table is None:
            return []
        self.initialize_meta_if_legacy()
        self.canonicalize_meta_if_legacy()
        self._ensure_embedding_compat()

        if query_text and not self._fts_ready:
            self.ensure_fts_index()

        if query_text and self._fts_ready:
            try:
                return _hybrid_search(table, query_text, query_vector, top_k, chunk_type)
            except Exception:
                log.debug("Hybrid search failed, falling back to vector-only", exc_info=True)

        candidate_k = top_k * self._config.candidate_multiplier
        query = table.search(query_vector).metric("cosine").limit(candidate_k)
        if chunk_type:
            query = query.where(_chunk_type_predicate(chunk_type))
        rows = query.to_list()
        log.debug(
            "Vector search: query=%r, candidates=%d, max_distance=%.2f",
            query_text or "vector-only",
            len(rows),
            max_distance,
        )
        if rows:
            distances = [r.get("distance", 0) for r in rows[:5]]
            log.debug("Top 5 distances: %s", distances)
        results = [SearchChunk(**r) for r in rows]
        return self._filter_and_rerank(results, query_vector, top_k, max_distance)

    def _filter_and_rerank(
        self,
        results: list[SearchChunk],
        query_vector: list[float],
        top_k: int,
        max_distance: float,
    ) -> list[SearchChunk]:
        """Apply the configured distance filter, then MMR-rerank down to top_k."""
        if max_distance > 0:
            before = len(results)
            if self._config.adaptive_threshold:
                results = self._adaptive_filter(results, top_k, max_distance)
                filter_name = "adaptive"
            else:
                results = self._fixed_filter(results, max_distance)
                filter_name = "fixed"
            log.debug(
                "After %s filter: %d/%d results, threshold=%.2f",
                filter_name,
                len(results),
                before,
                max_distance,
            )
        if len(results) > top_k:
            results = mmr_rerank(query_vector, results, top_k, self._config.mmr_lambda)
        return results

    def _adaptive_filter(
        self, results: list[SearchChunk], top_k: int, initial_threshold: float
    ) -> list[SearchChunk]:
        """Widen cosine distance threshold when too few results.
        Inspired by grantflow's (grantflow-ai/grantflow) adaptive retrieval
        pattern which widens thresholds on recursive retry. Step size and
        cap are configurable via ``self._config.adaptive_threshold_step``.

        Pre-sorts results by distance for a single-pass cutoff search.
        Step size is ``self._config.adaptive_threshold_step`` (default 0.2).
        """
        cap = max(initial_threshold, _MAX_THRESHOLD)
        step = self._config.adaptive_threshold_step

        sorted_results = sorted(results, key=_get_distance)

        threshold = initial_threshold
        for _ in range(_MAX_FILTER_ITERATIONS):
            if threshold > cap:
                break
            cutoff = _count_within_threshold(sorted_results, threshold)
            if cutoff >= top_k:
                return sorted_results[:cutoff]
            threshold += step
        # Final pass at cap
        cutoff = _count_within_threshold(sorted_results, cap)
        return sorted_results[:cutoff]

    def _fixed_filter(self, results: list[SearchChunk], threshold: float) -> list[SearchChunk]:
        """Simple fixed threshold filter - keep only results within distance threshold."""
        return [r for r in results if _get_distance(r) <= threshold]

    def get_chunks_by_source(self, source: str) -> list[SearchChunk]:
        """Return every chunk whose ``source`` equals *source*."""
        table = self.open_table(CHUNKS_TABLE)
        if table is None:
            return []
        escaped = escape_sql_string(source)
        try:
            rows = table.search().where(f"source = '{escaped}'").limit(None).to_list()
        except Exception:
            # FTS-enabled tables return a query builder that cannot
            # handle .where() on arbitrary columns; fall through to a
            # pyarrow.compute filter on the Arrow table so the source
            # match runs in C++ without materializing non-matching rows.
            log.debug("get_chunks_by_source search() failed, using Arrow fallback", exc_info=True)
            arrow_tbl = table.to_arrow()
            filtered = arrow_tbl.filter(pc.equal(arrow_tbl["source"], source))
            rows = filtered.to_pylist()
        return [SearchChunk(**r) for r in rows]

    def _delete_by_source_unlocked(self, source: str) -> None:
        """Delete all chunks from *source*. Caller must hold ``write_lock()``."""
        table = self.open_table(CHUNKS_TABLE)
        if table is not None:
            _safe_delete_unlocked(table, f"source = '{escape_sql_string(source)}'")

    def delete_by_source(self, source: str) -> None:
        """Delete all chunks from a given source file."""
        with write_lock():
            self._delete_by_source_unlocked(source)
        self._invalidate_source_cache()

    def get_sources(
        self,
        *,
        search: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[SourceRecord]:
        """Return source records, filtered by *search* and sliced by offset/limit."""
        table = self.open_table(SOURCES_TABLE)
        if table is None:
            return []
        query = table.search()
        where = _sources_search_filter(search)
        if where is not None:
            query = query.where(where)
        if offset:
            query = query.offset(offset)
        query = query.limit(limit)
        result: list[SourceRecord] = query.to_list()  # type: ignore[assignment]
        return result

    def count_sources(self, *, search: str | None = None) -> int:
        """Count tracked sources matching *search* without materializing rows."""
        table = self.open_table(SOURCES_TABLE)
        if table is None:
            return 0
        where = _sources_search_filter(search)
        count: int = table.count_rows() if where is None else table.count_rows(filter=where)
        return count

    def upsert_source(
        self,
        filename: str,
        file_hash: str,
        chunk_count: int,
        source_type: str = "document",
    ) -> None:
        """Add or update a source file tracking record."""
        with write_lock():
            db = self.get_db()
            table = ensure_table(db, SOURCES_TABLE, _sources_schema())
            _safe_delete_unlocked(table, f"filename = '{escape_sql_string(filename)}'")
            table.add(
                [
                    {
                        "filename": filename,
                        "file_hash": file_hash,
                        "ingested_at": datetime.now(UTC).isoformat(),
                        "chunk_count": chunk_count,
                        "source_type": source_type,
                    }
                ]
            )
        self._invalidate_source_cache()

    def _delete_source_unlocked(self, filename: str) -> None:
        """Remove the *filename* source record. Caller must hold ``write_lock()``."""
        table = self.open_table(SOURCES_TABLE)
        if table is not None:
            _safe_delete_unlocked(table, f"filename = '{escape_sql_string(filename)}'")

    def delete_source(self, filename: str) -> None:
        """Remove a source file tracking record."""
        with write_lock():
            self._delete_source_unlocked(filename)
        self._invalidate_source_cache()

    def _remove_one_unlocked(self, name: str) -> None:
        """Delete a document's chunks and its source record together.

        Both deletes run under the caller's single ``write_lock()`` so no
        reader can observe chunks whose source record is already gone.
        """
        self._delete_by_source_unlocked(name)
        self._delete_source_unlocked(name)

    def remove_documents(
        self,
        names: list[str],
        *,
        delete_files: bool = False,
        documents_dir: Path | None = None,
    ) -> RemoveResult:
        """Remove documents from the knowledge base by source name.
        Looks up known sources, deletes chunks and source records for each.
        If *delete_files* is True, resolves the path and verifies it is
        contained within *documents_dir* before unlinking (path traversal guard).

        Returns a RemoveResult with removed and not_found lists.
        """
        if documents_dir is None:
            documents_dir = self._config.documents_dir

        known = {s["filename"] for s in self.get_sources()}
        removed: list[str] = []
        not_found: list[str] = []

        for name in names:
            if name not in known:
                not_found.append(name)
                continue
            with write_lock():
                self._remove_one_unlocked(name)
            self._invalidate_source_cache()
            removed.append(name)
            if delete_files:
                try:
                    path = validate_path_within(documents_dir / name, documents_dir)
                except ValueError:
                    log.warning("Path traversal blocked: %s escapes %s", name, documents_dir)
                    continue
                if path.exists():
                    path.unlink()

        return RemoveResult(removed=removed, not_found=not_found)

    def clear_table(self, name: str, predicate: str) -> None:
        """Delete rows matching *predicate* from *name*. Acquires write lock."""
        with write_lock():
            table = self.open_table(name)
            if table is not None:
                _safe_delete_unlocked(table, predicate)

    def add_citations(self, records: list[CitationRecord]) -> int:
        """Add citation records to the store. Returns count added."""
        if not records:
            return 0
        with write_lock():
            db = self.get_db()
            table = ensure_table(db, CITATIONS_TABLE, _citations_schema())
            table.add(records)
        return len(records)

    def get_citations_for_wiki(self, wiki_source: str) -> list[CitationRecord]:
        """Get all citations for a wiki page."""
        table = self.open_table(CITATIONS_TABLE)
        if table is None:
            return []
        escaped = escape_sql_string(wiki_source)
        rows: list[CitationRecord] = table.search().where(f"wiki_source = '{escaped}'").to_list()
        return rows

    def get_citations_for_source(self, source_filename: str) -> list[CitationRecord]:
        """Get all citations that reference a source document (reverse lookup)."""
        table = self.open_table(CITATIONS_TABLE)
        if table is None:
            return []
        escaped = escape_sql_string(source_filename)
        rows: list[CitationRecord] = (
            table.search().where(f"source_filename = '{escaped}'").to_list()
        )
        return rows

    def delete_citations_for_wiki(self, wiki_source: str) -> None:
        """Delete all citations for a wiki page (used before regeneration)."""
        self.clear_table(
            CITATIONS_TABLE,
            f"wiki_source = '{escape_sql_string(wiki_source)}'",
        )

    def close(self) -> None:
        """Release the database connection and reset state."""
        self._db = None
        self._fts_ready = False

    def drop_all(self) -> None:
        """Drop all tables -- used by rebuild."""
        with write_lock():
            self._fts_ready = False
            db = self.get_db()
            for name in _table_names(db):
                db.drop_table(name)
        self._invalidate_source_cache()
