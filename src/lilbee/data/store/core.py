"""The ``Store`` class: high-level LanceDB read/write API used across lilbee."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from lilbee.core.config import (
    CHUNK_CONCEPTS_TABLE,
    CHUNKS_TABLE,
    CITATIONS_TABLE,
    MEMORIES_TABLE,
    META_TABLE,
    PAGE_TEXTS_TABLE,
    SOURCES_TABLE,
    Config,
)
from lilbee.core.security import validate_path_within
from lilbee.runtime.lock import LOCK_TIMEOUT, write_lock

from .lance_helpers import (
    _chunk_type_predicate,
    _has_fts_index,
    _has_vector_index,
    _safe_delete_unlocked,
    _sources_search_filter,
    _table_names,
    ensure_table,
    escape_sql_string,
    refs_compatible,
)
from .ranking import mmr_rerank
from .schema import _citations_schema, _meta_schema, _page_texts_schema, _sources_schema
from .types import (
    META_DELETE_ALL_PREDICATE,
    META_SCHEMA_VERSION,
    READ_CONSISTENCY_INTERVAL,
    SOURCE_STAT_UNKNOWN,
    ChunkType,
    ChunkWrite,
    CitationRecord,
    EmbeddingModelMismatchError,
    MemoryKind,
    MemoryRow,
    PageTextRecord,
    RemoveResult,
    SearchChunk,
    SourceRecord,
    SourceStat,
    SourceStatBackfill,
    SourceType,
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
    chunk_type: ChunkType | None = None,
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

# Vector ANN index. IVF_PQ compresses vectors so search scales to millions;
# refine_factor re-ranks the PQ candidates against full vectors to recover recall.
_VECTOR_METRIC = "cosine"
_ANN_INDEX_TYPE = "IVF_PQ"
_ANN_NPROBES_FLOOR = 20
_ANN_NPROBES_PARTITION_FRACTION = 0.05
_ANN_REFINE_FACTOR = 10

# Stat columns of ``_sources``; mirrors the field names in ``schema._sources_schema``
# and ``types.SourceRecord``. Legacy tables that predate these columns are migrated
# in place with the SOURCE_STAT_UNKNOWN sentinel.
_SOURCE_STAT_COLUMNS = ("size_bytes", "mtime_ns", "stat_captured_ns")

# (table, source column) pairs deleted when a source's rows are replaced. The
# concept nodes/edges tables carry no source column (corpus-level aggregates),
# so only the per-chunk concept mapping is source-scoped.
_PER_SOURCE_TABLES = (
    (CHUNKS_TABLE, "source"),
    (PAGE_TEXTS_TABLE, "source"),
    (CHUNK_CONCEPTS_TABLE, "chunk_source"),
)

# Stat backfills replace this many source rows per locked write: the first
# sync after a stat-column upgrade backfills every source, and an unchunked
# replace would join millions of filenames into one delete predicate.
_SOURCE_STAT_BATCH_ROWS = 2000


def _ann_nprobes(row_count: int) -> int:
    """Partitions to probe: a fixed fraction of the IVF partition count (~sqrt(N)), floored."""
    partitions = math.isqrt(max(row_count, 0))
    return max(_ANN_NPROBES_FLOOR, math.ceil(partitions * _ANN_NPROBES_PARTITION_FRACTION))


def _check_vector_dims(records: list[dict], embedding_dim: int) -> None:
    """Raise ``ValueError`` when any record's vector is not *embedding_dim* wide."""
    for rec in records:
        vec = rec.get("vector", [])
        if len(vec) != embedding_dim:
            raise ValueError(
                f"Vector dimension mismatch: expected {embedding_dim}, "
                f"got {len(vec)} (source={rec.get('source', '?')})"
            )


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

    def _write_lock(self, timeout: float = LOCK_TIMEOUT) -> AbstractContextManager[None]:
        """Acquire the write lock keyed on *this* store's data directory.

        A per-instance ``Lilbee`` writes to its own ``lancedb_dir``; locking the
        global ``cfg`` dir instead would leave those writes uncoordinated across
        processes.
        """
        return write_lock(self._config.lancedb_dir, timeout)

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
        rows = table.search().limit(None).to_list()
        if not rows:
            return None
        # _meta is meant to hold one row, but a swallowed delete on rewrite could
        # leave a stale one behind; take the newest so identity reads stay
        # deterministic rather than returning an arbitrary row.
        row = max(rows, key=lambda r: r["updated_at"])
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
        with self._write_lock():
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
            persisted_model=meta["embedding_model"],
            persisted_dim=meta["embedding_dim"],
            current_model=current_model,
            current_dim=current_dim,
        )

    def assert_embedding_compatible(self) -> None:
        """Run the full embedding-identity gate (legacy init, canonicalize, check).

        Mirrors the gate ``search`` applies. Callers that write under a fresh
        embedder (import) use this to fail before any destructive work when the
        store was built by a different model.
        """
        self.initialize_meta_if_legacy()
        self.canonicalize_meta_if_legacy()
        self._ensure_embedding_compat()

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
        with self._write_lock():
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
        with self._write_lock():
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

    def ensure_vector_index(self, *, force: bool = False) -> bool:
        """Build or refresh the ANN vector index when the corpus is large enough.

        Below ``cfg.ann_index_threshold`` (or when it is 0) the store keeps exact
        flat search, which is faster and exact for small vaults and is all a
        laptop needs. Once an index exists, ``optimize()`` folds new rows in.
        Pass ``force=True`` to build regardless of the threshold (publish flow).
        Returns True when an index was created or refreshed.
        """
        threshold = self._config.ann_index_threshold
        with self._write_lock():
            table = self.open_table(CHUNKS_TABLE)
            if table is None:
                return False
            if _has_vector_index(table):
                table.optimize()
                log.debug("Vector index optimized on '%s'", CHUNKS_TABLE)
                return True
            if not force and (threshold <= 0 or table.count_rows() < threshold):
                return False
            try:
                table.create_index(metric=_VECTOR_METRIC, index_type=_ANN_INDEX_TYPE)
                log.info("Vector ANN index created on '%s'", CHUNKS_TABLE)
                return True
            except Exception:
                log.warning(
                    "Vector ANN index build failed on '%s' at %d rows; search falls back "
                    "to exact flat scan, which is slow at this scale. Free up memory/disk "
                    "and re-run to rebuild the index.",
                    CHUNKS_TABLE,
                    table.count_rows(),
                    exc_info=True,
                )
                return False

    def add_chunks(self, records: list[dict]) -> int:
        """Add chunk records to the store. Returns count added.

        Raises ``EmbeddingModelMismatchError`` if the persisted ``_meta`` row was
        written under a different embedding model than the current ``cfg``. On the
        first write to a fresh store, ``_meta`` is initialized from the current cfg.

        The gate runs inside the write lock and uses a single cfg snapshot so a
        concurrent ``set_embedding_model`` cannot slip a write in past a stale
        compatibility check.
        """
        with self._write_lock():
            embedding_model = self._config.embedding_model
            embedding_dim = self._config.embedding_dim
            self._ensure_embedding_compat()
            self._fts_ready = False
            if not records:
                return 0
            _check_vector_dims(records, embedding_dim)
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
            return []
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
        chunk_type: ChunkType | None = None,
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
        query = table.search(query_vector).metric(_VECTOR_METRIC).limit(candidate_k)
        if _has_vector_index(table):
            # IVF_PQ is lossy; probe more partitions and refine against full
            # vectors so recall stays close to the exact flat scan.
            nprobes = _ann_nprobes(table.count_rows())
            query = query.nprobes(nprobes).refine_factor(_ANN_REFINE_FACTOR)
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

    def _delete_by_sources_unlocked(self, sources: list[str]) -> None:
        """Delete the sources' chunks, page texts, and chunk-concept rows.

        Caller must hold ``write_lock()``. One ``IN`` delete per table covers
        every source, so a batched flush pays a constant number of predicate
        deletes instead of one set per document. A delete failure propagates:
        swallowed, it would leave every flushed file silently stale; raised,
        the flush fails and the files replan on the next sync.
        """
        quoted = ", ".join(f"'{escape_sql_string(source)}'" for source in sources)
        for name, column in _PER_SOURCE_TABLES:
            table = self.open_table(name)
            if table is not None:
                table.delete(f"{column} IN ({quoted})")

    def _delete_by_source_unlocked(self, source: str) -> None:
        """Delete a single source's chunks, page texts, and chunk-concept rows."""
        self._delete_by_sources_unlocked([source])

    def delete_by_source(self, source: str) -> None:
        """Delete a source's chunks and page texts."""
        with self._write_lock():
            self._delete_by_source_unlocked(source)
        self._invalidate_source_cache()

    def add_page_texts(self, records: list[dict]) -> int:
        """Add per-page text rows (no vectors). Returns count added."""
        if not records:
            return 0
        with self._write_lock():
            db = self.get_db()
            table = ensure_table(db, PAGE_TEXTS_TABLE, _page_texts_schema())
            table.add(records)
        return len(records)

    def get_page_texts(self, source: str | None = None) -> list[PageTextRecord]:
        """Return per-page text rows, all or for a single *source*."""
        table = self.open_table(PAGE_TEXTS_TABLE)
        if table is None:
            return []
        query = table.search()
        if source is not None:
            query = query.where(f"source = '{escape_sql_string(source)}'")
        rows: list[PageTextRecord] = query.limit(None).to_list()
        return rows

    def page_text_sources(self) -> set[str]:
        """Return the distinct sources present in the page-text table."""
        table = self.open_table(PAGE_TEXTS_TABLE)
        if table is None:
            return set()
        return {row["source"] for row in table.search().select(["source"]).limit(None).to_list()}

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

    def _source_row(
        self,
        filename: str,
        file_hash: str,
        chunk_count: int,
        source_type: str,
        stat: SourceStat | None,
    ) -> dict:
        """Build one ``_sources`` row, defaulting absent stat to the unknown sentinel."""
        return {
            "filename": filename,
            "file_hash": file_hash,
            "ingested_at": datetime.now(UTC).isoformat(),
            "chunk_count": chunk_count,
            "source_type": source_type,
            "size_bytes": stat.size_bytes if stat else SOURCE_STAT_UNKNOWN,
            "mtime_ns": stat.mtime_ns if stat else SOURCE_STAT_UNKNOWN,
            "stat_captured_ns": stat.captured_ns if stat else SOURCE_STAT_UNKNOWN,
        }

    def _sources_table(self) -> lancedb.table.Table:
        """Open/create ``_sources``, adding the stat columns to pre-stat tables."""
        table = ensure_table(self.get_db(), SOURCES_TABLE, _sources_schema())
        missing = [name for name in _SOURCE_STAT_COLUMNS if name not in table.schema.names]
        if missing:
            table.add_columns({name: f"CAST({SOURCE_STAT_UNKNOWN} AS BIGINT)" for name in missing})
        return table

    def _replace_source_rows_unlocked(self, rows: list[dict]) -> None:
        """Replace source rows: one batched delete plus one batched add.

        Caller must hold ``write_lock()``. A per-file delete+add pair costs two
        LanceDB version commits, so bulk ingest folds every file in a flush into
        a single pair.
        """
        table = self._sources_table()
        filenames = ", ".join(f"'{escape_sql_string(r['filename'])}'" for r in rows)
        # Skip the add when the delete failed: adding over a stale row would leave
        # two _sources rows for one filename. The file replans on the next sync.
        if not _safe_delete_unlocked(table, f"filename IN ({filenames})"):
            return
        table.add(rows)

    def upsert_source(
        self,
        filename: str,
        file_hash: str,
        chunk_count: int,
        source_type: SourceType = SourceType.DOCUMENT,
        stat: SourceStat | None = None,
    ) -> None:
        """Add or update a source tracking record."""
        row = self._source_row(filename, file_hash, chunk_count, source_type, stat)
        with self._write_lock():
            self._replace_source_rows_unlocked([row])
        self._invalidate_source_cache()

    def update_source_stats(self, backfills: list[SourceStatBackfill]) -> None:
        """Record size/mtime for already-tracked sources in batched locked writes."""
        if not backfills:
            return
        for start in range(0, len(backfills), _SOURCE_STAT_BATCH_ROWS):
            rows = [
                {
                    **bf.record,
                    "size_bytes": bf.stat.size_bytes,
                    "mtime_ns": bf.stat.mtime_ns,
                    "stat_captured_ns": bf.stat.captured_ns,
                }
                for bf in backfills[start : start + _SOURCE_STAT_BATCH_ROWS]
            ]
            with self._write_lock():
                self._replace_source_rows_unlocked(rows)
        self._invalidate_source_cache()

    def optimize_sources(self) -> None:
        """Compact the sources table; per-flush upserts otherwise accrete tiny versions."""
        with self._write_lock():
            table = self.open_table(SOURCES_TABLE)
            if table is None:
                return
            try:
                table.optimize()
            except Exception:
                log.debug("Sources table optimize failed", exc_info=True)

    def write_chunks_batch(self, items: list[ChunkWrite]) -> int:
        """Write several documents' chunks in one locked transaction. Returns chunks added.

        One ``write_lock`` acquisition covers the batch's cleanup deletes, page
        texts, chunk add, and source upserts, so a reader never observes a
        half-applied batch. Page texts land after the cleanup and before the
        source rows, so a page-text failure leaves the rows stale and the files
        replan next sync; a document with no chunks still persists its page
        texts and source row. The embedding-identity gate and per-vector
        dimension check mirror ``add_chunks``; a dimension mismatch raises and
        the whole batch is rejected.
        """
        if not items:
            return 0
        with self._write_lock():
            embedding_model = self._config.embedding_model
            embedding_dim = self._config.embedding_dim
            self._ensure_embedding_compat()
            self._fts_ready = False
            all_records = [rec for it in items for rec in it.records]
            _check_vector_dims(all_records, embedding_dim)
            db = self.get_db()
            self._cleanup_batch_unlocked(items)
            self._add_page_texts_unlocked(db, items)
            self._add_chunk_records_unlocked(db, all_records, embedding_model, embedding_dim)
            self._replace_source_rows_unlocked(self._batch_source_rows(items))
        self._invalidate_source_cache()
        return len(all_records)

    def _cleanup_batch_unlocked(self, items: list[ChunkWrite]) -> None:
        """One ``IN`` delete per table for the flagged documents. Caller holds ``write_lock()``."""
        cleanup_sources = [it.source for it in items if it.needs_cleanup]
        if cleanup_sources:
            self._delete_by_sources_unlocked(cleanup_sources)

    def _add_page_texts_unlocked(self, db: lancedb.DBConnection, items: list[ChunkWrite]) -> None:
        """Add the batch's page-text rows. Caller holds ``write_lock()``."""
        page_rows = [row for it in items for row in (it.page_texts or [])]
        if page_rows:
            ensure_table(db, PAGE_TEXTS_TABLE, _page_texts_schema()).add(page_rows)

    def _add_chunk_records_unlocked(
        self,
        db: lancedb.DBConnection,
        all_records: list[dict],
        embedding_model: str,
        embedding_dim: int,
    ) -> None:
        """Add the batch's chunk rows, writing meta on first use. Caller holds ``write_lock()``."""
        if not all_records:
            return
        ensure_table(db, CHUNKS_TABLE, self._chunks_schema()).add(all_records)
        if self.get_meta() is None:
            self._write_meta_unlocked(embedding_model=embedding_model, embedding_dim=embedding_dim)

    def _batch_source_rows(self, items: list[ChunkWrite]) -> list[dict]:
        """One ``_sources`` row per batched document."""
        return [
            self._source_row(it.source, it.file_hash, len(it.records), SourceType.DOCUMENT, it.stat)
            for it in items
        ]

    def _delete_source_unlocked(self, filename: str) -> None:
        """Remove the *filename* source record. Caller must hold ``write_lock()``."""
        table = self.open_table(SOURCES_TABLE)
        if table is not None:
            _safe_delete_unlocked(table, f"filename = '{escape_sql_string(filename)}'")

    def delete_source(self, filename: str) -> None:
        """Remove a source file tracking record."""
        with self._write_lock():
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
            with self._write_lock():
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
        with self._write_lock():
            table = self.open_table(name)
            if table is not None:
                _safe_delete_unlocked(table, predicate)

    def add_citations(self, records: list[CitationRecord]) -> int:
        """Add citation records to the store. Returns count added."""
        if not records:
            return 0
        with self._write_lock():
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

    def _memories_schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field("id", pa.utf8()),
                pa.field("owner", pa.utf8()),
                pa.field("shared", pa.bool_()),
                pa.field("kind", pa.utf8()),
                pa.field("source", pa.utf8()),
                pa.field("text", pa.utf8()),
                pa.field("vector", pa.list_(pa.float32(), self._config.embedding_dim)),
                pa.field("created_at", pa.utf8()),
                pa.field("updated_at", pa.utf8()),
            ]
        )

    def _duplicate_memory_id_unlocked(
        self, table: lancedb.table.Table, record: MemoryRow
    ) -> str | None:
        """Return the id of a near-duplicate same-owner, same-kind memory, if any."""
        if table.count_rows() == 0:
            return None
        predicate = (
            f"owner = '{escape_sql_string(record.owner)}' "
            f"AND kind = '{escape_sql_string(record.kind)}'"
        )
        rows = table.search(record.vector).metric("cosine").where(predicate).limit(1).to_list()
        if rows and rows[0].get("_distance", 1.0) <= self._config.memory_dedup_distance:
            return str(rows[0]["id"])
        return None

    def _evict_overflow_unlocked(self, table: lancedb.table.Table, owner: str) -> None:
        """Delete oldest memories for *owner* so an incoming insert stays within the cap."""
        cap = self._config.memory_max_per_owner
        predicate = f"owner = '{escape_sql_string(owner)}'"
        rows = table.search().where(predicate).limit(None).to_list()
        if len(rows) < cap:
            return
        rows.sort(key=lambda r: r.get("created_at", ""))
        for row in rows[: len(rows) - (cap - 1)]:
            _safe_delete_unlocked(table, f"id = '{escape_sql_string(str(row['id']))}'")

    def add_memory(self, record: MemoryRow) -> str:
        """Insert *record*, or update the nearest same-owner duplicate in place.

        Returns the stored id. Raises ``EmbeddingModelMismatchError`` when the store
        was built under a different embedding model, and ``ValueError`` on a vector
        dimension mismatch.
        """
        if len(record.vector) != self._config.embedding_dim:
            raise ValueError(
                f"Memory vector dimension mismatch: expected "
                f"{self._config.embedding_dim}, got {len(record.vector)}"
            )
        with self._write_lock():
            embedding_model = self._config.embedding_model
            embedding_dim = self._config.embedding_dim
            self._ensure_embedding_compat()
            db = self.get_db()
            table = ensure_table(db, MEMORIES_TABLE, self._memories_schema())
            duplicate_id = self._duplicate_memory_id_unlocked(table, record)
            if duplicate_id is not None and _safe_delete_unlocked(
                table, f"id = '{escape_sql_string(duplicate_id)}'"
            ):
                # Only reuse the id once the old row is actually gone; a swallowed
                # delete failure would otherwise leave two rows with the same id.
                record.id = duplicate_id
            self._evict_overflow_unlocked(table, record.owner)
            table.add([record.model_dump(mode="json")])
            if self.get_meta() is None:
                self._write_meta_unlocked(
                    embedding_model=embedding_model, embedding_dim=embedding_dim
                )
            return record.id

    def get_memories(
        self,
        *,
        owner_predicate: str,
        kind: MemoryKind | None = None,
    ) -> list[MemoryRow]:
        """Return memories matching *owner_predicate* and optional *kind*, newest first."""
        table = self.open_table(MEMORIES_TABLE)
        if table is None:
            return []
        clauses = [f"({owner_predicate})"]
        if kind is not None:
            clauses.append(f"kind = '{escape_sql_string(kind)}'")
        rows = table.search().where(" AND ".join(clauses)).limit(None).to_list()
        memories = [MemoryRow(**r) for r in rows]
        memories.sort(key=lambda m: m.created_at, reverse=True)
        return memories

    def search_memories(
        self,
        query_vector: list[float],
        *,
        owner_predicate: str,
        top_k: int,
        max_distance: float,
    ) -> list[MemoryRow]:
        """Vector-recall FACT memories within *max_distance*, best first."""
        table = self.open_table(MEMORIES_TABLE)
        if table is None or top_k <= 0:
            return []
        self._ensure_embedding_compat()
        predicate = f"({owner_predicate}) AND kind = '{MemoryKind.FACT}'"
        rows = table.search(query_vector).metric("cosine").where(predicate).limit(top_k).to_list()
        return [MemoryRow(**r) for r in rows if r.get("_distance", 1.0) <= max_distance]

    def update_memory(self, memory_id: str, *, shared: bool, owner: str) -> bool:
        """Set the *shared* flag on *owner*'s memory. Returns True when found and owned.

        The ``owner`` predicate scopes the mutation to the caller's namespace so an
        agent cannot flip another owner's (or the human's) memory.
        """
        with self._write_lock():
            table = self.open_table(MEMORIES_TABLE)
            if table is None:
                return False
            predicate = self._owned_memory_predicate(memory_id, owner)
            rows = table.search().where(predicate).limit(1).to_list()
            if not rows:
                return False
            record = MemoryRow(**rows[0])
            record.shared = shared
            record.updated_at = datetime.now(UTC).isoformat()
            # If the delete fails, do not add the modified copy: that would leave
            # two rows for one id. Report not-updated instead.
            if not _safe_delete_unlocked(table, predicate):
                return False
            table.add([record.model_dump(mode="json")])
            return True

    def delete_memory(self, memory_id: str, *, owner: str) -> bool:
        """Delete *owner*'s memory by id. Returns True when a matching row was deleted.

        The ``owner`` predicate scopes the delete to the caller's namespace so an
        agent cannot destroy another owner's (or the human's) memory.
        """
        with self._write_lock():
            table = self.open_table(MEMORIES_TABLE)
            if table is None:
                return False
            predicate = self._owned_memory_predicate(memory_id, owner)
            if not table.search().where(predicate).limit(1).to_list():
                return False
            # Report the real outcome: a swallowed delete failure must not be
            # reported as a successful forget.
            return _safe_delete_unlocked(table, predicate)

    @staticmethod
    def _owned_memory_predicate(memory_id: str, owner: str) -> str:
        """SQL predicate matching a single memory id within *owner*'s namespace."""
        return f"id = '{escape_sql_string(memory_id)}' AND owner = '{escape_sql_string(owner)}'"

    def rebuild_memory_embeddings(self, embed: Callable[[list[str]], list[list[float]]]) -> int:
        """Re-embed every memory under the current model, recreating the table.

        The vector column dimension is immutable, so a different-dim model needs a
        fresh table; recreating unconditionally also covers the same-dim case. Memory
        text is human-authored and re-embeddable, so no data is lost. Returns the count.
        """
        table = self.open_table(MEMORIES_TABLE)
        if table is None:
            return 0
        rows = table.search().limit(None).to_list()
        if not rows:
            return 0
        memories = [MemoryRow(**r) for r in rows]
        vectors = embed([m.text for m in memories])
        for memory, vector in zip(memories, vectors, strict=True):
            memory.vector = vector
        with self._write_lock():
            db = self.get_db()
            db.drop_table(MEMORIES_TABLE)
            new_table = ensure_table(db, MEMORIES_TABLE, self._memories_schema())
            new_table.add([m.model_dump(mode="json") for m in memories])
        return len(memories)

    def close(self) -> None:
        """Release the database connection and reset state."""
        self._db = None
        self._fts_ready = False

    def drop_all(self) -> None:
        """Drop every table except ``_memories`` -- used by rebuild.

        Memory is user-authored data with no on-disk source, not derived from
        documents, so a rebuild preserves it. Only a factory reset (which deletes
        the data directory) clears it.
        """
        with self._write_lock():
            self._fts_ready = False
            db = self.get_db()
            for name in _table_names(db):
                if name == MEMORIES_TABLE:
                    continue
                db.drop_table(name)
        self._invalidate_source_cache()
