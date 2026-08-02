"""The ``Store`` class: high-level LanceDB read/write API used across lilbee."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable, Iterable, Sequence
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa

from lilbee.core.config import (
    CHUNK_CONCEPTS_TABLE,
    CHUNKS_TABLE,
    CITATIONS_TABLE,
    ENTITIES_TABLE,
    ENTITY_SCHEMA_TABLE,
    INGEST_SOURCE_COLUMNS,
    MEMORIES_TABLE,
    META_TABLE,
    PAGE_TEXTS_TABLE,
    SOURCES_TABLE,
    WIKI_MENTIONS_TABLE,
    Config,
)
from lilbee.core.vectors import Vector
from lilbee.retrieval.embedding_profiles import resolve_embedding_profile
from lilbee.runtime.lock import LOCK_TIMEOUT, LockTimeoutError, write_lock

from .fusion import adaptive_weight_scale, fuse_arms, normalized_bm25, vector_similarity
from .lance_helpers import (
    _CHUNK_COLUMN,
    _chunk_type_predicate,
    _has_fts_index,
    _has_scalar_index,
    _has_vector_index,
    _safe_delete_unlocked,
    _sources_search_filter,
    ensure_table,
    escape_sql_string,
    refs_compatible,
    table_names,
)
from .ranking import mmr_rerank
from .schema import (
    _citations_schema,
    _entity_schema_state_schema,
    _meta_schema,
    _page_texts_schema,
    _sources_schema,
    _wiki_mentions_schema,
)
from .types import (
    ENTITY_SCHEMA_DELETE_ALL_PREDICATE,
    META_DELETE_ALL_PREDICATE,
    META_SCHEMA_VERSION,
    READ_CONSISTENCY_INTERVAL,
    SOURCE_STAT_UNKNOWN,
    ChunkType,
    ChunkWrite,
    CitationRecord,
    EmbeddingModelMismatchError,
    EntitySchemaState,
    MemoryKind,
    MemoryRow,
    PageTextRecord,
    RemoveResult,
    SearchChunk,
    SourceMeta,
    SourceRecord,
    SourceStat,
    SourceStatBackfill,
    SourceType,
    StoreMeta,
)

if TYPE_CHECKING:
    import lance
    import lancedb
    import lancedb.table
    from lancedb.index import FTS

log = logging.getLogger(__name__)

# Batched ingest flushes contend with long store operations (a search-triggered
# FTS optimize can hold the lock past the interactive 30s), and failing the
# flush replans and re-embeds the whole batch. Give the batch path more
# patience than interactive writes before it gives up.
BATCH_LOCK_TIMEOUT = 120.0
# Lock budget for index builds reached from the read path: a query must not
# stall behind a long ingest, so it skips the build and retries next search.
_READ_LOCK_TIMEOUT = 2.0


def _drop_unsupported_far_rows(
    results: list[SearchChunk], max_distance: float
) -> list[SearchChunk]:
    """Apply ``max_distance`` to rows whose only signal is the vector arm.

    A row the BM25 arm also matched keeps lexical support regardless of its
    vector distance; dropping it on distance alone would re-bury exactly the
    identifier hits rank fusion exists to preserve.
    """
    if max_distance <= 0:
        return results
    return [
        r
        for r in results
        if r.bm25_score is not None or r.distance is None or r.distance <= max_distance
    ]


_MAX_THRESHOLD = 1.0
_MAX_FILTER_ITERATIONS = 20  # safety cap to prevent runaway loops


def _is_fts_position_overflow(exc: Exception) -> bool:
    """True when *exc* is LanceDB's positional-FTS list-encoding overflow.

    A positional index (built by an intermediate dev commit) raises e.g.
    "Max offset N exceeds length of values M" on optimize(); a positionless
    rebuild is the remediation. Matched on message because LanceDB raises it
    as a generic error type.
    """
    msg = str(exc).lower()
    return "offset" in msg and "exceeds" in msg


def _lexical_rows(
    table: lancedb.table.Table,
    query_text: str,
    limit: int,
    chunk_type: ChunkType | None,
    column: str = _CHUNK_COLUMN,
) -> list[SearchChunk]:
    """BM25 rows for *query_text* over a single FTS *column*.

    ``MatchQuery`` pins the column and matches plain terms, so an unpinned search
    cannot widen to the title index and a quoted span cannot reach LanceDB as a
    phrase (which the positionless index rejects). This is the one place FTS
    queries are built; every arm goes through it.
    """
    from lancedb.query import MatchQuery

    query = table.search(MatchQuery(query_text, column), query_type="fts").limit(limit)
    if chunk_type:
        query = query.where(_chunk_type_predicate(chunk_type))
    return [SearchChunk(**r) for r in query.to_list()]


# Vector ANN index. IVF_PQ compresses vectors so search scales to millions;
# refine_factor re-ranks the PQ candidates against full vectors to recover recall.
# The index type is carried by the lancedb IvfPq config at build time.
_VECTOR_METRIC = "cosine"
_ANN_NPROBES_FLOOR = 20
# Fraction of IVF partitions probed per query. 0.05 was the "fast" end of the
# recall/latency curve and measurably cost recall at scale: on the 8.8M-passage
# MS MARCO index it gave recall@100 of 67%, where probing more partitions
# reached 87% (docs/benchmarks/retrieval-msmarco.md). 0.15 is the "balanced /
# high-recall" range for IVF and recovers most of that gap; refine_factor below
# still re-ranks the survivors against full vectors.
_ANN_NPROBES_PARTITION_FRACTION = 0.15
_ANN_REFINE_FACTOR = 10

# Stat columns of ``_sources``; mirrors the field names in ``schema._sources_schema``
# and ``types.SourceRecord``. Legacy tables that predate these columns are migrated
# in place with the SOURCE_STAT_UNKNOWN sentinel.
_SOURCE_STAT_COLUMNS = ("size_bytes", "mtime_ns", "stat_captured_ns")

# Extraction-metadata columns of ``_sources``; nullable strings, so legacy
# tables migrate in place with NULL (meaning "extractor reported nothing").
_SOURCE_META_COLUMNS = ("title", "authors", "created_at")

# Document-title column of the chunks table; nullable so pre-title rows and
# writers that carry no title (wiki pages) read as NULL.
_TITLE_COLUMN = "title"

# (table, source column) pairs deleted when a source's rows are replaced. The
# concept nodes/edges tables carry no source column (corpus-level aggregates),
# so only the per-chunk concept mapping is source-scoped.
_PER_SOURCE_TABLES = tuple(
    (name, INGEST_SOURCE_COLUMNS[name])
    for name in (CHUNKS_TABLE, PAGE_TEXTS_TABLE, CHUNK_CONCEPTS_TABLE, ENTITIES_TABLE)
) + (
    # Per-(subject, source), so removing a source drops its mention evidence
    # here with its chunks; the wiki refresh only ever re-adds a source, never
    # has to remember to subtract a deleted one. Not in INGEST_SOURCE_COLUMNS
    # (a wiki table, not an ingest one), so its column is named directly.
    (WIKI_MENTIONS_TABLE, "source"),
)

# (table, source column) pairs re-keyed when a source is relocated (moved on
# disk, same content). Extends the per-source set with the wiki citation's raw
# source_filename so citations keep pointing at the source after a move.
_RELOCATABLE_TABLES = (
    *_PER_SOURCE_TABLES,
    (CITATIONS_TABLE, INGEST_SOURCE_COLUMNS[CITATIONS_TABLE]),
)

# Sentinel: relocation must leave the stored title untouched (extraction-derived).
_KEEP_TITLE = "\x00keep"

# Stat backfills replace this many source rows per locked write: the first
# sync after a stat-column upgrade backfills every source, and an unchunked
# replace would join millions of filenames into one delete predicate.
_SOURCE_STAT_BATCH_ROWS = 2000

# Rows per Arrow batch when the aggregate scan walks the whole chunks table;
# bounds the decoded-text working set while the scan stays columnar.
_TERM_SCAN_BATCH_ROWS = 20_000

# The title arm collapses each matched document to one row, so it over-fetches
# to gather enough distinct documents before deduping. Bounded so a title that
# hits a huge document can't scan the whole corpus.
_TITLE_FETCH_FACTOR = 20
_TITLE_MIN_FETCH = 200
_TITLE_FETCH_CEILING = 4096


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


def _citations_for_wiki_predicate(wiki_source: str) -> str:
    """SQL predicate selecting every citation row belonging to *wiki_source*."""
    return f"wiki_source = '{escape_sql_string(wiki_source)}'"


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
        self._title_fts_ready: bool = False
        self._doc_prefix_warned: bool = False
        # Scalar indexes (source/chunk_type) are built at ingest; a serve-only
        # store builds them lazily from the search path.
        self._scalar_ready: bool = False
        self._db: lancedb.DBConnection | None = None
        # Cache of {filename: ingested_at} rebuilt only when sources
        # mutate; callers (temporal filter) hit it per-query.
        self._source_ingested_cache: dict[str, str] | None = None

    def _fts_config(self) -> FTS:
        """Shared FTS index config: positionless (with_position=True overflows
        LanceDB's list encoding on optimize()) and stemmed for the configured
        corpus language."""
        from lancedb.index import FTS

        return FTS(with_position=False, language=self._config.fts_language)

    def _index_build_lock(self, blocking: bool) -> AbstractContextManager[None]:
        """Write lock for index builds: full budget from ingest, short from the read path."""
        return self._write_lock() if blocking else self._write_lock(_READ_LOCK_TIMEOUT)

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
                pa.field(_TITLE_COLUMN, pa.utf8()),
                pa.field("vector", pa.list_(pa.float32(), self._config.embedding_dim)),
            ]
        )

    def _chunks_table(self) -> lancedb.table.Table:
        """Open/create the chunks table, adding the title column to pre-title tables."""
        table = ensure_table(self.get_db(), CHUNKS_TABLE, self._chunks_schema())
        if _TITLE_COLUMN not in table.schema.names:
            table.add_columns({_TITLE_COLUMN: "CAST(NULL AS STRING)"})
            self._backfill_stem_titles_unlocked(table)
        return table

    def _backfill_stem_titles_unlocked(self, table: lancedb.table.Table) -> None:
        """Backfill filename-stem titles for pre-upgrade rows. Caller holds ``write_lock()``.

        Without this the title arm only matches documents ingested after the
        upgrade. Extracted (H1/EXIF) titles still need ``lilbee rebuild``;
        failure leaves NULLs, the pre-backfill behavior.
        """
        from lilbee.data.title import derive_title  # circular at module scope

        try:
            rows = table.search().select(["source"]).limit(None).to_list()
            sources = sorted({r["source"] for r in rows})
            filled = 0
            for source in sources:
                title = derive_title(source)
                if not title:
                    continue
                escaped = source.replace("'", "''")
                table.update(where=f"source = '{escaped}'", values={_TITLE_COLUMN: title})
                filled += 1
            log.info(
                "Backfilled filename titles for %d of %d existing sources; run "
                "`lilbee rebuild` to derive titles from document content",
                filled,
                len(sources),
            )
        except Exception:
            log.warning(
                "Title backfill failed; pre-upgrade rows keep NULL titles until `lilbee rebuild`",
                exc_info=True,
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

    def _warn_stale_doc_prefix(self) -> None:
        """Warn once when the embedding family's document prefix postdates this store.

        Queries would carry the family prefix while stored documents do not;
        the mismatch degrades quality silently until a rebuild re-embeds.
        """
        if self._doc_prefix_warned:
            return
        self._doc_prefix_warned = True
        meta = self.get_meta()
        if meta is None:
            return
        profile = resolve_embedding_profile(self._config.embedding_model)
        if profile.doc_prefix and meta["schema_version"] < profile.doc_prefix_since:
            log.warning(
                "This index predates '%s' document prefixes: queries are prefixed "
                "but stored documents are not. Run `lilbee rebuild` to re-embed.",
                self._config.embedding_model,
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
        if name not in table_names(db):
            return None
        return db.open_table(name)

    def ensure_fts_index(self, *, blocking: bool = True) -> None:
        """Create the chunks FTS index, or run ``optimize()`` once it exists.

        ``optimize()`` folds newly added rows into the FTS index and also
        runs LanceDB's default compaction + version pruning (default prune
        window: 7 days). Work scales with recent deltas rather than total
        chunk count, so large corpora no longer pay the full
        ``create_index(config=FTS(), replace=True)`` rebuild cost on every sync.

        ``blocking=False`` (the search path) marks an existing index ready
        without the lock and skips maintenance when another process holds it,
        so a long concurrent ingest cannot stall or fail a query.
        """
        probe = self.open_table(CHUNKS_TABLE)
        if probe is None:
            return
        if _has_fts_index(probe):
            self._fts_ready = True
        try:
            with self._index_build_lock(blocking):
                self._ensure_fts_index_unlocked()
        except LockTimeoutError:
            if blocking:
                raise
            log.debug("Skipped FTS index maintenance; another process holds the write lock")

    def _ensure_fts_index_unlocked(self) -> None:
        """Body of ``ensure_fts_index``. Caller holds ``write_lock()``."""
        table = self.open_table(CHUNKS_TABLE)
        if table is None:
            return
        try:
            if _has_fts_index(table):
                self._fts_ready = True
                try:
                    # One optimize folds new rows into every index on the table.
                    table.optimize()
                    log.debug("FTS index optimized on '%s'", CHUNKS_TABLE)
                except Exception as exc:
                    if _is_fts_position_overflow(exc):
                        # Positional indexes overflow on optimize(); rebuild
                        # positionless once.
                        self._rebuild_fts_positionless(table)
                    else:
                        log.warning(
                            "FTS optimize() failed; the existing index still serves hybrid search",
                            exc_info=True,
                        )
            else:
                # Positionless: with_position=True overflows LanceDB's list
                # encoding on optimize(), and nothing issues phrase queries.
                table.create_index(_CHUNK_COLUMN, config=self._fts_config(), replace=False)
                self._fts_ready = True
                log.debug("FTS index created on '%s'", CHUNKS_TABLE)
            # Only the opt-in title arm needs the title index.
            if self._config.title_search:
                self._ensure_title_fts_unlocked(table)
        except Exception:
            log.debug("FTS index ensure failed (empty table?)", exc_info=True)

    def _ensure_title_fts_unlocked(self, table: lancedb.table.Table) -> None:
        """Create the title FTS index when the column exists. Caller holds ``write_lock()``.

        Failure never blocks the chunk index: the title arm feature-detects the
        index per query, so a store without it simply searches without titles.
        """
        if _TITLE_COLUMN not in table.schema.names or _has_fts_index(table, _TITLE_COLUMN):
            self._title_fts_ready = _has_fts_index(table, _TITLE_COLUMN)
            return
        try:
            # Positionless for the same reason as the chunk index.
            table.create_index(_TITLE_COLUMN, config=self._fts_config(), replace=False)
            self._title_fts_ready = True
            log.debug("Title FTS index created on '%s'", CHUNKS_TABLE)
        except Exception:
            # Only reached with title_search enabled, so a silent failure means
            # the user's opted-in title arm quietly does nothing. Warn, don't hide.
            log.warning(
                "Title FTS index creation failed; the title-search arm will "
                "contribute nothing until it can be built",
                exc_info=True,
            )

    def ensure_title_fts_index(self, *, blocking: bool = True) -> None:
        """Build the title FTS index for a title_search toggle after startup.

        Without this, a process that latched ``_fts_ready`` before the toggle
        never builds the index and the title arm stays silently dead until the
        next ingest or restart.
        """
        table = self.open_table(CHUNKS_TABLE)
        if table is None:
            return
        if _has_fts_index(table, _TITLE_COLUMN):
            self._title_fts_ready = True
            return
        try:
            with self._index_build_lock(blocking):
                self._ensure_title_fts_unlocked(table)
        except LockTimeoutError:
            if blocking:
                raise
            log.debug("Skipped title FTS build; another process holds the write lock")

    def _rebuild_fts_positionless(self, table: lancedb.table.Table) -> None:
        """Replace positional FTS indexes with positionless ones. Caller holds the lock.

        The one-shot remediation for a store whose index was built
        ``with_position=True`` and now overflows on every ``optimize()``. The
        title index is rebuilt too when the title arm is enabled.
        """
        try:
            table.create_index(_CHUNK_COLUMN, config=self._fts_config(), replace=True)
            if self._config.title_search and _TITLE_COLUMN in table.schema.names:
                table.create_index(_TITLE_COLUMN, config=self._fts_config(), replace=True)
            log.warning("Rebuilt the FTS index positionless after a positional-index overflow")
        except Exception:
            log.warning(
                "Positionless FTS rebuild failed; the existing index still serves",
                exc_info=True,
            )

    # Tables and (column, index_type) pairs the query path filters by.
    # chunk_concepts serves the concept boost (ConceptGraph._chunk_concepts_from);
    # without its index every boosted query full-scans the table.
    _SCALAR_TARGETS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
        (CHUNKS_TABLE, (("source", "BTREE"), ("chunk_type", "BITMAP"))),
        (CHUNK_CONCEPTS_TABLE, (("chunk_source", "BTREE"),)),
    )

    def ensure_scalar_indexes(self, *, blocking: bool = True) -> None:
        """Build scalar indexes on the columns lilbee filters by.

        ``source`` and ``chunk_type`` predicates run as prefilters (LanceDB's
        default), but without an index each is a full-table scan. Readiness
        latches only when every target table exists and is covered, so a table
        created later (chunk_concepts under serve ordering) still gets its
        index on a following call. The lock is taken only when there is
        something to build; ``blocking=False`` (the search path) skips the
        build when another process holds it instead of stalling the query.
        """
        pending = []
        complete = True
        for name, columns in self._SCALAR_TARGETS:
            table = self.open_table(name)
            if table is None:
                complete = False
                continue
            names = table.schema.names
            if any(c in names and not _has_scalar_index(table, c) for c, _ in columns):
                pending.append((name, columns))
        if not pending:
            self._scalar_ready = complete
            return
        try:
            with self._index_build_lock(blocking):
                for name, columns in pending:
                    self._ensure_scalar_index_on(name, columns)
            self._scalar_ready = complete
        except LockTimeoutError:
            if blocking:
                raise
            log.debug("Skipped scalar index build; another process holds the write lock")

    def _ensure_scalar_index_on(
        self, table_name: str, columns: tuple[tuple[str, str], ...]
    ) -> None:
        """Build the given (column, index_type) scalar indexes on *table_name*.

        Caller holds ``write_lock()``. Each column gets its own try so one
        failure does not skip the rest; a failure on a populated table warns
        (the prefilter speedup is silently lost) while an empty table's is debug.
        """
        table = self.open_table(table_name)
        if table is None:
            return
        names = table.schema.names
        fail_level = logging.WARNING if table.count_rows() > 0 else logging.DEBUG
        for column, index_type in columns:
            if column not in names or _has_scalar_index(table, column):
                continue
            try:
                table.create_scalar_index(column, index_type=index_type, replace=False)
                log.debug("Scalar (%s) index created on '%s.%s'", index_type, table_name, column)
            except Exception:
                log.log(
                    fail_level,
                    "Scalar index create failed on '%s.%s'",
                    table_name,
                    column,
                    exc_info=True,
                )

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
            from lancedb.index import IvfPq

            try:
                table.create_index("vector", config=IvfPq(distance_type=_VECTOR_METRIC))
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

    def _add_chunks_unlocked(self, records: list[dict]) -> int:
        """Add chunk records and return the count. Caller must hold ``write_lock()``."""
        embedding_model = self._config.embedding_model
        embedding_dim = self._config.embedding_dim
        self._ensure_embedding_compat()
        self._fts_ready = False
        self._scalar_ready = False
        if not records:
            return 0
        _check_vector_dims(records, embedding_dim)
        table = self._chunks_table()
        table.add(records)
        if self.get_meta() is None:
            self._write_meta_unlocked(embedding_model=embedding_model, embedding_dim=embedding_dim)
        return len(records)

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
            return self._add_chunks_unlocked(records)

    def replace_chunks(self, records: list[dict], predicate: str) -> int:
        """Replace the chunk rows matching *predicate* with *records* under one write lock.

        Same compatibility and dimension gates as :meth:`add_chunks`, both run
        before the delete so a rejected write leaves the existing rows in place.
        The lock serializes writers; delete and add are separate commits, so a
        concurrent reader can briefly see the rows absent, and callers retry on
        a crash between the two. A delete failure propagates, following the same
        rule as :meth:`_delete_by_sources_unlocked`: swallowed, the caller would
        read a success-shaped result over rows still describing the old body.
        Returns the count added.
        """
        with self._write_lock():
            self._ensure_embedding_compat()
            _check_vector_dims(records, self._config.embedding_dim)
            table = self._chunks_table()
            table.delete(predicate)
            return self._add_chunks_unlocked(records)

    def _stamp_meta_unlocked(self, embedding_model: str, embedding_dim: int) -> None:
        """Write the embedder identity on the first write to a fresh store."""
        if self.get_meta() is None:
            self._write_meta_unlocked(embedding_model=embedding_model, embedding_dim=embedding_dim)

    def absorb_rows(self, name: str, rows: pa.Table) -> int:
        """Append an ingest shard's *rows* to table *name*, creating it if absent.

        The rows already carry their embeddings, so this is the merge path for a
        multi-GPU sync: the per-worker stores are folded in whole and the indexes
        are rebuilt corpus-wide afterwards.
        """
        with self._write_lock():
            self._ensure_embedding_compat()
            self._fts_ready = False
            self._scalar_ready = False
            ensure_table(self.get_db(), name, rows.schema).add(rows)
            self._stamp_meta_unlocked(self._config.embedding_model, self._config.embedding_dim)
            return int(rows.num_rows)

    def _assert_schemas_match(
        self,
        name: str,
        target: Path,
        shard_tables: list[Path],
        sources: Sequence[lance.LanceDataset],
    ) -> None:
        """Refuse shards whose schema differs from the table's.

        A mismatched vector width is the realistic case: two workers built with
        different embedding models. Adoption commits fragment metadata and never
        passes the rows through a writer, so nothing else would notice.
        """
        import lance

        expected = lance.dataset(str(target)).schema
        for shard_table, source in zip(shard_tables, sources, strict=True):
            if not source.schema.equals(expected):
                raise ValueError(
                    f"Cannot adopt {shard_table} into {name}: its schema does not match "
                    f"the index. A shard built with a different embedding model cannot be "
                    f"folded in; re-ingest it with the model this index was built on."
                )

    def adopt_fragments(self, name: str, shard_tables: list[Path]) -> int:
        """Take over *shard_tables*' data files for table *name* without copying rows.

        Every fragment's data file is hard-linked into this table's directory and
        the whole set committed as one metadata-only append, so the rows are never
        read or rewritten and the bytes exist once on disk with two names. That is
        the difference between a merge that costs the corpus and one that costs its
        fragment count: the copy path rewrites every vector, and because the shard
        stores are kept as resume state the corpus would then be on disk twice.

        Whole-fragment only, so this serves the first full merge. A re-sync merges
        named sources, where a fragment holds both touched and untouched rows and
        the scoped row copy is both correct and already cheap.

        Returns the rows adopted. Raises ValueError when a shard's schema differs
        from the table's, and OSError when a shard is on another filesystem (hard
        links cannot cross one) or a data file name collides. The parent is left
        as it was in every failure: schemas are checked before anything is linked,
        links made before a failure are removed, and the commit happens once at
        the end.
        """
        import lance

        with self._write_lock():
            self._ensure_embedding_compat()
            target = self._config.lancedb_dir / f"{name}.lance"
            sources = [lance.dataset(str(shard_table)) for shard_table in shard_tables]
            if not sources:
                return 0
            if name not in table_names(self.get_db()):
                ensure_table(self.get_db(), name, sources[0].schema)
            # Before anything is linked. Committing a fragment whose schema does
            # not match writes an index that reads back as a panic inside Arrow
            # rather than an error: the row copy is rejected by the writer, and
            # adoption has no writer to reject it.
            self._assert_schemas_match(name, target, shard_tables, sources)
            # A table created empty has no data directory yet: nothing has been
            # written into it, and the links below need somewhere to go.
            (target / "data").mkdir(parents=True, exist_ok=True)
            adopted: list[lance.FragmentMetadata] = []
            linked: list[Path] = []
            rows = 0
            try:
                for shard_table, source in zip(shard_tables, sources, strict=True):
                    for fragment in source.get_fragments():
                        meta = fragment.metadata
                        for data_file in meta.files:
                            filename = Path(data_file.path).name
                            link = target / "data" / filename
                            os.link(shard_table / "data" / filename, link)
                            linked.append(link)
                        adopted.append(meta)
                    rows += source.count_rows()
            except OSError:
                # A link made before the failure is a file the manifest never
                # names, so nothing would ever remove it. The caller falls back
                # to copying rows.
                for link in linked:
                    link.unlink(missing_ok=True)
                raise
            if not adopted:
                return 0
            self._fts_ready = False
            self._scalar_ready = False
            existing = lance.dataset(str(target))
            lance.LanceDataset.commit(
                str(target),
                lance.LanceOperation.Append(adopted),
                read_version=existing.version,
            )
            self._stamp_meta_unlocked(self._config.embedding_model, self._config.embedding_dim)
            return rows

    def bm25_probe(
        self, query_text: str, top_k: int = 5, chunk_type: ChunkType | None = None
    ) -> list[SearchChunk]:
        """Quick BM25-only search for confidence checking. Returns up to top_k results.

        When *chunk_type* is set, only chunks of that type ("raw" or "wiki") are returned.
        """
        table = self.open_table(CHUNKS_TABLE)
        if table is None:
            return []
        if not self._fts_ready:
            self.ensure_fts_index(blocking=False)
        if not self._fts_ready:
            return []
        try:
            results = _lexical_rows(table, query_text, top_k, chunk_type)
            norms = normalized_bm25([r.bm25_score or 0.0 for r in results])
            return [
                r.model_copy(update={"score": norm}) for r, norm in zip(results, norms, strict=True)
            ]
        except Exception:
            log.debug("BM25 probe failed", exc_info=True)
            return []

    def search(
        self,
        query_vector: Vector,
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
        self._warn_stale_doc_prefix()

        if not self._scalar_ready:
            # A serve-only store never ran ingest, where scalar indexes are
            # built; without them the source/chunk_type prefilters full-scan.
            self.ensure_scalar_indexes(blocking=False)

        if query_text and not self._fts_ready:
            self.ensure_fts_index(blocking=False)
        if query_text and self._config.title_search and not self._title_fts_ready:
            self.ensure_title_fts_index(blocking=False)

        if query_text and self._fts_ready:
            try:
                return self._hybrid_search(
                    table, query_text, query_vector, top_k, max_distance, chunk_type
                )
            except Exception:
                # Falling back changes recall characteristics for the query;
                # a corpus-wide FTS breakage must not present as silence.
                log.warning("Hybrid search failed, falling back to vector-only", exc_info=True)

        rows = self._vector_arm(
            table, query_vector, top_k * self._config.candidate_multiplier, chunk_type
        )
        log.debug(
            "Vector search: query=%r, candidates=%d, max_distance=%.2f",
            query_text or "vector-only",
            len(rows),
            max_distance,
        )
        if rows:
            log.debug("Top 5 distances: %s", [r.distance for r in rows[:5]])
        results = self._filter_and_rerank(rows, query_vector, top_k, max_distance)
        return [
            r.model_copy(
                update={"score": vector_similarity(r.distance) if r.distance is not None else 0.0}
            )
            for r in results
        ]

    def _vector_arm(
        self,
        table: lancedb.table.Table,
        query_vector: Vector,
        limit: int,
        chunk_type: ChunkType | None,
    ) -> list[SearchChunk]:
        """Vector-arm candidates with the ANN recall recovery applied.

        When ``chunk_type`` is set, the predicate is pushed into the query so
        the limit applies *after* the type filter; post-filtering would
        silently starve wiki-only queries whose matches live past the window.
        """
        query = table.search(query_vector).metric(_VECTOR_METRIC).limit(limit)
        if _has_vector_index(table):
            # IVF_PQ is lossy; probe more partitions and refine against full
            # vectors so recall stays close to the exact flat scan.
            query = query.nprobes(_ann_nprobes(table.count_rows()))
            query = query.refine_factor(_ANN_REFINE_FACTOR)
        if chunk_type:
            query = query.where(_chunk_type_predicate(chunk_type))
        return [SearchChunk(**r) for r in query.to_list()]

    def _fts_arm(
        self,
        table: lancedb.table.Table,
        query_text: str,
        limit: int,
        chunk_type: ChunkType | None,
    ) -> list[SearchChunk]:
        """BM25-arm candidates over the chunk text."""
        return _lexical_rows(table, query_text, limit, chunk_type)

    def _title_arm(
        self,
        table: lancedb.table.Table,
        query_text: str,
        limit: int,
        chunk_type: ChunkType | None,
    ) -> list[SearchChunk]:
        """One BM25 row per document whose title matches, in title-relevance order.

        Every chunk of a document carries the same title, so all of its chunks
        tie on BM25 and a plain ``limit`` would return an arbitrary tie-ordered
        subset of a single document. Instead this over-fetches, collapses each
        source to one deterministic representative (its first chunk), and returns
        the top *limit* documents ordered by title score -- so "a query naming a
        document by title surfaces its chunks" holds as one stable row per doc.

        Empty when the store predates the title column or its FTS index (old
        indexes keep working) and empty on any query-time failure: the optional
        title arm must never take down the healthy chunk arm, so its failure
        degrades to no-titles, mirroring ``bm25_probe``.
        """
        if not _has_fts_index(table, _TITLE_COLUMN):
            return []
        # Every chunk of one document ties on title BM25, so a fixed window can
        # fill up with a single long document's chunks and starve every other
        # title-matching document. Widen the fetch until enough distinct
        # documents surface, the matches run out, or the ceiling is hit.
        fetch = max(limit * _TITLE_FETCH_FACTOR, _TITLE_MIN_FETCH)
        while True:
            try:
                rows = _lexical_rows(table, query_text, fetch, chunk_type, column=_TITLE_COLUMN)
            except Exception:
                log.debug("Title arm search failed; contributing no title rows", exc_info=True)
                return []
            best: dict[str, SearchChunk] = {}
            for row in rows:
                seen = best.get(row.source)
                if seen is None or row.chunk_index < seen.chunk_index:
                    best[row.source] = row
            if len(best) >= limit or len(rows) < fetch or fetch >= _TITLE_FETCH_CEILING:
                break
            fetch = min(fetch * 4, _TITLE_FETCH_CEILING)
        ordered = sorted(best.values(), key=lambda r: (-(r.bm25_score or 0.0), r.source))
        return ordered[:limit]

    def _hybrid_search(
        self,
        table: lancedb.table.Table,
        query_text: str,
        query_vector: Vector,
        top_k: int,
        max_distance: float,
        chunk_type: ChunkType | None = None,
    ) -> list[SearchChunk]:
        """Multi-arm retrieval fused by weighted reciprocal rank; the fused ordering is final.

        A vector arm and a chunk-BM25 arm always run; a title-BM25 arm joins
        when ``cfg.title_search`` is on. Each row's fused score is the
        weight-normalized sum of its arm contributions: the vector arm has
        weight 1.0, the lexical arm ``cfg.lexical_fusion_weight`` (scaled per
        query when ``cfg.adaptive_fusion`` is on), the title arm
        ``cfg.title_search_weight``. So a row a single peaked arm is certain
        about scores that arm's share of the total weight, not a fixed 0.5.

        Each arm fetches exactly ``top_k`` rows. Deeper pools measurably hurt
        rank fusion by flooding the fused top-k with both-arm mediocrity and
        burying single-arm certainty (lexical identifier hits above all). No
        MMR runs here: lexical passages are often mutually similar, which MMR
        penalizes, trading relevant hits for off-topic neighbors.

        Title rows carry ``bm25_score``, so a title match counts as lexical
        support for the distance exemption like any other lexical hit.
        """
        title_rows: list[SearchChunk] = []
        if self._config.title_search:
            title_rows = self._title_arm(table, query_text, top_k, chunk_type)
        vector_rows = self._vector_arm(table, query_vector, top_k, chunk_type)
        base_lexical_weight = self._config.lexical_fusion_weight
        base_title_weight = self._config.title_search_weight
        lexical_weight = base_lexical_weight
        title_weight = base_title_weight
        if self._config.adaptive_fusion:
            # Quiet the lexical arms per query by vector confidence. The title
            # arm is lexical too, so the same factor scales it.
            scale = adaptive_weight_scale(vector_rows, self._config.adaptive_fusion_margin)
            lexical_weight = base_lexical_weight * scale
            title_weight = base_title_weight * scale
        fused = fuse_arms(
            vector_rows,
            self._fts_arm(table, query_text, top_k, chunk_type),
            title_rows,
            lexical_weight=lexical_weight,
            title_weight=title_weight,
        )
        fused = _drop_unsupported_far_rows(fused, max_distance)
        return fused[:top_k]

    def _filter_and_rerank(
        self,
        results: list[SearchChunk],
        query_vector: Vector,
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

    def add_entities(self, records: list[dict]) -> int:
        """Append typed entity rows; creates the table on first write.

        Additive to the store: existing tables and schemas are untouched, and
        stores without this table behave as if nothing was ever extracted.
        """
        if not records:
            return 0
        from lilbee.retrieval.entities.schema import _entities_schema

        with self._write_lock():
            db = self.get_db()
            table = ensure_table(db, ENTITIES_TABLE, _entities_schema())
            table.add(records)
            return len(records)

    def entity_schema_state(self) -> EntitySchemaState | None:
        """The persisted entity schema row, or ``None`` when never induced.

        The schema is machine state induced from the corpus and lives inside
        the index, so it travels with the data.
        """
        table = self.open_table(ENTITY_SCHEMA_TABLE)
        if table is None:
            return None
        rows = table.search().limit(None).to_list()
        if not rows:
            return None
        # One row by contract; take the newest if a rewrite ever left a stale one.
        row = max(rows, key=lambda r: r["updated_at"])
        return EntitySchemaState(
            schema_json=str(row["schema_json"]),
            applied=bool(row["applied"]),
            source_count=int(row["source_count"]),
            updated_at=str(row["updated_at"]),
        )

    def save_entity_schema(self, schema_json: str, *, applied: bool, source_count: int) -> None:
        """Overwrite the single persisted entity schema row."""
        with self._write_lock():
            db = self.get_db()
            table = ensure_table(db, ENTITY_SCHEMA_TABLE, _entity_schema_state_schema())
            _safe_delete_unlocked(table, ENTITY_SCHEMA_DELETE_ALL_PREDICATE)
            table.add(
                [
                    {
                        "schema_json": schema_json,
                        "applied": applied,
                        "source_count": source_count,
                        "updated_at": datetime.now(UTC).isoformat(),
                    }
                ]
            )

    def mark_entity_schema_applied(self) -> None:
        """Record that a full extraction pass completed under the stored schema."""
        state = self.entity_schema_state()
        if state is None:
            return
        self.save_entity_schema(
            state["schema_json"], applied=True, source_count=state["source_count"]
        )

    def entity_value_counts(self, entity_type: str) -> tuple[int, int]:
        """(mentions, distinct normalized values) for one entity type.

        Full scan by design: a count is a corpus property. Streaming batches
        keep memory flat at any corpus size.
        """
        table = self.open_table(ENTITIES_TABLE)
        if table is None:
            return 0, 0
        mentions = 0
        values: set[str] = set()
        arrow = table.to_arrow().select(["type", "normalized_value"])
        for batch in arrow.to_batches(max_chunksize=_TERM_SCAN_BATCH_ROWS):
            types = batch.column("type").to_pylist()
            vals = batch.column("normalized_value").to_pylist()
            for t_, v in zip(types, vals, strict=True):
                if t_ == entity_type:
                    mentions += 1
                    values.add(v)
        return mentions, len(values)

    def entity_association_counts(self, counted: str, grouped_by: str) -> dict[str, int]:
        """Distinct *counted*-type values co-occurring with each *grouped_by* value.

        Co-occurrence is per chunk: two entities extracted from the same
        ``(source, chunk_index)`` are associated. This is the GROUP BY that
        answers "how many X is each Y associated with".
        """
        table = self.open_table(ENTITIES_TABLE)
        if table is None:
            return {}
        per_chunk: dict[tuple[str, int], tuple[set[str], set[str]]] = {}
        arrow = table.to_arrow().select(["type", "normalized_value", "source", "chunk_index"])
        for batch in arrow.to_batches(max_chunksize=_TERM_SCAN_BATCH_ROWS):
            rows = zip(
                batch.column("type").to_pylist(),
                batch.column("normalized_value").to_pylist(),
                batch.column("source").to_pylist(),
                batch.column("chunk_index").to_pylist(),
                strict=True,
            )
            for t_, v, src, idx in rows:
                if t_ not in (counted, grouped_by):
                    continue
                counted_vals, group_vals = per_chunk.setdefault((src, idx), (set(), set()))
                (counted_vals if t_ == counted else group_vals).add(v)
        associations: dict[str, set[str]] = {}
        for counted_vals, group_vals in per_chunk.values():
            for group_value in group_vals:
                associations.setdefault(group_value, set()).update(counted_vals)
        return {k: len(v) for k, v in sorted(associations.items())}

    def count_term_mentions(self, term: str) -> tuple[int, int]:
        """(matching chunks, distinct matching sources) for a case-insensitive
        substring scan of the WHOLE chunks table.

        This is deliberately a full scan, not a top-k search: a count is a
        corpus property, and any retrieval shortcut undercounts it. Streaming
        Arrow batches keeps the working set to one batch of text at a time,
        so cost is linear in corpus size and memory stays flat.
        """
        table = self.open_table(CHUNKS_TABLE)
        if table is None:
            return 0, 0
        needle = term.lower()
        chunk_hits = 0
        sources: set[str] = set()
        arrow = table.to_arrow().select(["source", "chunk"])
        for batch in arrow.to_batches(max_chunksize=_TERM_SCAN_BATCH_ROWS):
            texts = batch.column("chunk").to_pylist()
            names = batch.column("source").to_pylist()
            for name, text in zip(names, texts, strict=True):
                if text and needle in text.lower():
                    chunk_hits += 1
                    sources.add(name)
        return chunk_hits, len(sources)

    def count_chunks(self) -> int:
        """Total chunks in the store."""
        table = self.open_table(CHUNKS_TABLE)
        return table.count_rows() if table is not None else 0

    def get_chunks_by_source(self, source: str) -> list[SearchChunk]:
        """Return every chunk whose ``source`` equals *source*.

        The database does the filtering, so only the matching rows are read.
        A query failure raises rather than falling back to a whole-table scan:
        a document's chunks are a bounded read, and the scan that would rescue
        it costs the entire index, vectors included, in memory.
        """
        table = self.open_table(CHUNKS_TABLE)
        if table is None:
            return []
        escaped = escape_sql_string(source)
        rows = table.search().where(f"source = '{escaped}'").limit(None).to_list()
        return [SearchChunk(**r) for r in rows]

    def get_chunks_by_indices(self, source: str, indices: Sequence[int]) -> list[SearchChunk]:
        """Return *source*'s chunks whose ``chunk_index`` is in *indices*.

        Rows come back in ``chunk_index`` order; indices past either end of
        the document are simply absent from the result. Filtering happens in
        the database for the same reason as :meth:`get_chunks_by_source`:
        neighbor expansion runs once per hit source per query, so a
        whole-table rescue would spike memory on the hottest path there is.
        """
        if not indices:
            return []
        table = self.open_table(CHUNKS_TABLE)
        if table is None:
            return []
        escaped = escape_sql_string(source)
        wanted = ", ".join(str(int(i)) for i in indices)
        predicate = f"source = '{escaped}' AND chunk_index IN ({wanted})"
        rows = table.search().where(predicate).limit(None).to_list()
        return sorted((SearchChunk(**r) for r in rows), key=lambda c: c.chunk_index)

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

    def page_texts_arrow(self, source: str | None = None) -> pa.Table:
        """Return per-page text rows as an Arrow table in a single scan.

        The columnar sibling of :meth:`get_page_texts`: the export path keeps the
        whole set in Arrow (no per-row Python objects) from read through file
        write. Empty with the canonical schema when the table or *source* is empty.
        """
        table = self.open_table(PAGE_TEXTS_TABLE)
        if table is None:
            return _page_texts_schema().empty_table()
        query = table.search().select(["source", "page", "text", "content_type"])
        if source is not None:
            query = query.where(f"source = '{escape_sql_string(source)}'")
        return query.limit(None).to_arrow()

    def sources_arrow(self) -> pa.Table:
        """Return each tracked source's extraction metadata as an Arrow table.

        The columnar sibling of :meth:`get_sources`, keyed by ``source`` so it
        joins straight onto the page-text table. ``get_sources`` builds a dict per
        source, which on a corpus of millions of single-page documents costs more
        than the text being exported. An index written before the metadata columns
        existed gets them as nulls rather than missing, so the join has one shape.
        """
        import pyarrow as pa
        import pyarrow.compute as pc

        table = self.open_table(SOURCES_TABLE)
        columns = ["source", *SourceMeta._fields]
        if table is None:
            return pa.schema([pa.field(name, pa.utf8()) for name in columns]).empty_table()
        present = [name for name in SourceMeta._fields if name in table.schema.names]
        arrow = table.search().select(["filename", *present]).limit(None).to_arrow()
        arrow = arrow.rename_columns(["source", *present])
        for name in SourceMeta._fields:
            if name not in present:
                arrow = arrow.append_column(name, pa.nulls(arrow.num_rows, pa.utf8()))
        arrow = arrow.select(columns)
        if pc.count_distinct(arrow.column("source")).as_py() == arrow.num_rows:
            return arrow
        # One row per source, so a caller joining on it cannot fan out. A doubled
        # row (a source re-merged from a shard) would otherwise multiply every page
        # it owns. Last wins, matching the dict this replaced; single-threaded
        # because that is the only execution mode with an ordered aggregate.
        grouped = arrow.group_by("source", use_threads=False).aggregate(
            [(name, "last") for name in SourceMeta._fields]
        )
        renamed = grouped.rename_columns(
            [name.removesuffix("_last") for name in grouped.schema.names]
        )
        return renamed.select(columns)

    def wiki_chunk_sources(self) -> set[str]:
        """Return the distinct sources of the chunk rows written by the wiki layer."""
        table = self.open_table(CHUNKS_TABLE)
        if table is None:
            return set()
        rows = (
            table.search()
            .where(f"chunk_type = '{ChunkType.WIKI}'")
            .select(["source"])
            .limit(None)
            .to_list()
        )
        return {row["source"] for row in rows}

    def wiki_citation_sources(self) -> set[str]:
        """Return the distinct wiki_source values present in the citations table."""
        table = self.open_table(CITATIONS_TABLE)
        if table is None:
            return set()
        rows = table.search().select(["wiki_source"]).limit(None).to_list()
        return {row["wiki_source"] for row in rows}

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
        where = _sources_search_filter(search, include_title="title" in table.schema.names)
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
        where = _sources_search_filter(search, include_title="title" in table.schema.names)
        count: int = table.count_rows() if where is None else table.count_rows(filter=where)
        return count

    def _source_row(
        self,
        filename: str,
        file_hash: str,
        chunk_count: int,
        source_type: str,
        stat: SourceStat | None,
        meta: SourceMeta | None = None,
    ) -> dict:
        """Build one ``_sources`` row, defaulting absent stat to the unknown sentinel.

        Absent extraction metadata persists as NULL, matching rows written
        before the metadata columns existed.
        """
        meta = meta or SourceMeta()
        return {
            "filename": filename,
            "file_hash": file_hash,
            "ingested_at": datetime.now(UTC).isoformat(),
            "chunk_count": chunk_count,
            "source_type": source_type,
            "size_bytes": stat.size_bytes if stat else SOURCE_STAT_UNKNOWN,
            "mtime_ns": stat.mtime_ns if stat else SOURCE_STAT_UNKNOWN,
            "stat_captured_ns": stat.captured_ns if stat else SOURCE_STAT_UNKNOWN,
            "title": meta.title or None,
            "authors": meta.authors or None,
            "created_at": meta.created_at or None,
        }

    def _sources_table(self) -> lancedb.table.Table:
        """Open/create ``_sources``, adding the stat and metadata columns to older tables."""
        table = ensure_table(self.get_db(), SOURCES_TABLE, _sources_schema())
        defaults = {name: f"CAST({SOURCE_STAT_UNKNOWN} AS BIGINT)" for name in _SOURCE_STAT_COLUMNS}
        defaults |= {name: "CAST(NULL AS STRING)" for name in _SOURCE_META_COLUMNS}
        missing = {name: sql for name, sql in defaults.items() if name not in table.schema.names}
        if missing:
            table.add_columns(missing)
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
        meta: SourceMeta | None = None,
    ) -> None:
        """Add or update a source tracking record."""
        row = self._source_row(filename, file_hash, chunk_count, source_type, stat, meta)
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
        with self._write_lock(timeout=BATCH_LOCK_TIMEOUT):
            embedding_model = self._config.embedding_model
            embedding_dim = self._config.embedding_dim
            self._ensure_embedding_compat()
            self._fts_ready = False
            self._scalar_ready = False
            all_records = [rec for it in items for rec in it.records]
            _check_vector_dims(all_records, embedding_dim)
            db = self.get_db()
            self._cleanup_batch_unlocked(items)
            self._add_page_texts_unlocked(db, items)
            self._add_chunk_records_unlocked(all_records, embedding_model, embedding_dim)
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
        all_records: list[dict],
        embedding_model: str,
        embedding_dim: int,
    ) -> None:
        """Add the batch's chunk rows, writing meta on first use. Caller holds ``write_lock()``."""
        if not all_records:
            return
        self._chunks_table().add(all_records)
        if self.get_meta() is None:
            self._write_meta_unlocked(embedding_model=embedding_model, embedding_dim=embedding_dim)

    def _batch_source_rows(self, items: list[ChunkWrite]) -> list[dict]:
        """One ``_sources`` row per batched document."""
        return [
            self._source_row(
                it.source, it.file_hash, len(it.records), it.source_type, it.stat, it.meta
            )
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

    def _remove_many_unlocked(self, names: list[str]) -> None:
        """Delete the documents' chunks and source records together.

        All deletes run under the caller's single ``write_lock()`` so no
        reader can observe chunks whose source record is already gone; one
        ``IN`` delete per table covers the whole set.
        """
        self._delete_by_sources_unlocked(names)
        quoted = ", ".join(f"'{escape_sql_string(name)}'" for name in names)
        table = self.open_table(SOURCES_TABLE)
        if table is not None:
            _safe_delete_unlocked(table, f"filename IN ({quoted})")

    def relocate_sources(self, moves: list[tuple[str, str, SourceStat | None]]) -> None:
        """Re-key moved sources from old filename to new, preserving their chunks.

        A source whose file moved (same content hash, new path) keeps its chunks
        and embeddings; only its filename key and disk stat change. Each per-source
        table's source column, the citation source_filename, and the sources row are
        updated in place under one write lock, so a move costs no re-extraction or
        re-embedding. ``moves`` is ``(old_name, new_name, new_stat)`` tuples.

        Each table is opened once; the re-key is then a targeted per-move update.
        A single-statement batch would need a ``CASE`` expression, which LanceDB's
        update SQL does not support, and a delete+re-add across the vector tables is
        not worth its risk for what is a rare mass relabel.
        """
        if not moves:
            return
        from lilbee.data.title import derive_title  # circular at module scope

        with self._write_lock():
            tables = [(self.open_table(name), column) for name, column in _RELOCATABLE_TABLES]
            sources = self.open_table(SOURCES_TABLE)
            for old, new, stat in moves:
                where_old = f"= '{escape_sql_string(old)}'"
                new_title = self._relocated_title(sources, old, new, derive_title)
                for table, column in tables:
                    if table is None:
                        continue
                    values: dict[str, object] = {column: new}
                    # Stem titles track the filename; re-derive them on the same
                    # handle and statement as the re-key.
                    if new_title is not _KEEP_TITLE and _TITLE_COLUMN in table.schema.names:
                        values[_TITLE_COLUMN] = new_title
                    table.update(where=f"{column} {where_old}", values=values)
                if sources is not None:
                    row_values: dict[str, object] = {"filename": new}
                    if new_title is not _KEEP_TITLE:
                        row_values["title"] = new_title
                    if stat is not None:
                        row_values["size_bytes"] = stat.size_bytes
                        row_values["mtime_ns"] = stat.mtime_ns
                        row_values["stat_captured_ns"] = stat.captured_ns
                    sources.update(where=f"filename {where_old}", values=row_values)
        self._invalidate_source_cache()

    def _relocated_title(
        self,
        sources: lancedb.table.Table | None,
        old: str,
        new: str,
        derive: Callable[[str], str],
    ) -> str | None:
        """New title for a moved source, or ``_KEEP_TITLE`` when it must not change.

        Extraction-derived titles survive a move (the content is unchanged);
        a stem-derived title tracks the filename it was derived from, so it is
        re-derived from the new name instead of matching the old one forever.
        """
        if sources is None:
            return _KEEP_TITLE
        try:
            rows = (
                sources.search()
                .where(f"filename = '{escape_sql_string(old)}'")
                .select(["title"])
                .limit(1)
                .to_list()
            )
        except Exception:
            return _KEEP_TITLE
        if not rows or "title" not in rows[0]:
            return _KEEP_TITLE
        stored = rows[0]["title"] or ""
        if stored != (derive(old) or ""):
            return _KEEP_TITLE
        return derive(new) or None

    def remove_documents(self, names: list[str]) -> RemoveResult:
        """Remove documents from the knowledge base by source name.

        Looks up known sources and deletes their chunks and source records. Never
        touches files on disk: source bytes are the user's, and a linked-in corpus
        must never be deleted. Durable, file-aware removal (skip-markers, unlinking
        a top-level link) lives in :func:`lilbee.app.ingest.remove_documents_durably`.

        Returns a RemoveResult with removed and not_found lists.
        """
        known = {s["filename"] for s in self.get_sources()}
        removed = [name for name in names if name in known]
        not_found = [name for name in names if name not in known]

        if removed:
            # One lock acquisition and one IN-delete per table for the whole
            # set, mirroring the batched flush path, instead of a LanceDB
            # version commit per document.
            with self._write_lock():
                self._remove_many_unlocked(removed)
            self._invalidate_source_cache()

        return RemoveResult(removed=removed, not_found=not_found)

    def clear_table(self, name: str, predicate: str) -> bool:
        """Delete rows matching *predicate* from *name*. Acquires write lock.

        Returns whether the delete succeeded, so a caller recording the
        outcome (prune, the legacy migration) does not report success over a
        swallowed failure.
        """
        with self._write_lock():
            table = self.open_table(name)
            if table is None:
                return True
            return _safe_delete_unlocked(table, predicate)

    def clear_and_add(self, name: str, schema: pa.Schema, rows: list[dict], predicate: str) -> None:
        """Replace the rows matching *predicate* with *rows* in one locked write.

        Delete and add run under a single write lock, so a reader never observes
        the table emptied mid-rebuild. A delete failure propagates, following the
        same rule as :meth:`_delete_by_sources_unlocked`: adding over rows whose
        predecessors are still there duplicates them, and reporting success would
        leave the caller acting on state it thinks it replaced.
        """
        with self._write_lock():
            db = self.get_db()
            table = ensure_table(db, name, schema)
            table.delete(predicate)
            if rows:
                table.add(rows)

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

    def delete_citations_for_wiki(self, wiki_source: str) -> bool:
        """Delete all citations for a wiki page. Returns whether the delete succeeded."""
        return self.clear_table(CITATIONS_TABLE, _citations_for_wiki_predicate(wiki_source))

    def delete_all_wiki_rows(self) -> bool:
        """Delete every wiki chunk row, every citation, and every mention.
        Returns whether all deletes succeeded, so a caller cannot report a wipe
        over a swallowed failure. Only the wiki layer writes citations and
        mentions, so wiping it empties those tables outright. All deletes run
        before the results combine.
        """
        chunks_cleared = self.clear_table(CHUNKS_TABLE, f"chunk_type = '{ChunkType.WIKI}'")
        citations_cleared = self.clear_table(CITATIONS_TABLE, "1 = 1")
        mentions_cleared = self.clear_wiki_mentions()
        return chunks_cleared and citations_cleared and mentions_cleared

    def replace_citations_for_wiki(self, wiki_source: str, records: list[CitationRecord]) -> None:
        """Swap a wiki page's citations for *records* under one write lock.

        The lock keeps another writer from landing between the delete and the
        add; the two remain separate commits, so a crash in between leaves the
        rows absent until the page is regenerated or accepted again.
        """
        self.clear_and_add(
            CITATIONS_TABLE,
            _citations_schema(),
            [dict(rec) for rec in records],
            _citations_for_wiki_predicate(wiki_source),
        )

    def replace_wiki_mentions_for_source(self, source: str, rows: list[dict]) -> None:
        """Swap one source's wiki mention rows for *rows* under one write lock.

        A source contributes its whole mention set at once, so an incremental
        wiki refresh replaces exactly that source's rows and leaves every other
        source's evidence in place for the corpus-wide aggregate.
        """
        escaped = escape_sql_string(source)
        self.clear_and_add(
            WIKI_MENTIONS_TABLE,
            _wiki_mentions_schema(),
            rows,
            f"source = '{escaped}'",
        )

    def wiki_mention_rows(self, slugs: Iterable[str] | None = None) -> list[dict]:
        """Every wiki mention row, or only those for *slugs*.

        The wiki aggregates these across sources to rebuild the stub index. A
        full refresh reads them all; an incremental one reads only the slugs it
        touched, to recompute their corpus-wide totals without a full scan.
        """
        table = self.open_table(WIKI_MENTIONS_TABLE)
        if table is None:
            return []
        query = table.search()
        if slugs is not None:
            wanted = list(slugs)
            if not wanted:
                return []
            joined = ", ".join(f"'{escape_sql_string(s)}'" for s in wanted)
            query = query.where(f"slug IN ({joined})")
        rows: list[dict] = query.limit(None).to_list()
        return rows

    def clear_wiki_mentions(self) -> bool:
        """Drop every wiki mention row (a full rebuild starts from nothing)."""
        return self.clear_table(WIKI_MENTIONS_TABLE, "1 = 1")

    def has_wiki_mentions(self) -> bool:
        """Whether any wiki mention row exists.

        A cold store or one migrated from the file-only index has none, which
        forces a refresh to rebuild in full and seed the table before an
        incremental pass can aggregate over it.
        """
        table = self.open_table(WIKI_MENTIONS_TABLE)
        return table is not None and table.count_rows() > 0

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
        query_vector: Vector,
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

    def rebuild_memory_embeddings(self, embed: Callable[[list[str]], list[Vector]]) -> int:
        """Re-embed every memory under the current model, recreating the table.

        The vector column dimension is immutable, so a different-dim model needs a
        fresh table; recreating unconditionally also covers the same-dim case. Memory
        text is human-authored and re-embeddable, so no data is lost. Returns the count.

        The snapshot, embed, and table rebuild all run under the write lock so a
        concurrent ``add_memory`` cannot commit into the read-then-drop window and
        be erased; it either lands before the snapshot or blocks until the rebuild
        finishes.
        """
        with self._write_lock():
            table = self.open_table(MEMORIES_TABLE)
            if table is None:
                return 0
            rows = table.search().limit(None).to_list()
            if not rows:
                return 0
            memories = [MemoryRow(**r) for r in rows]
            vectors = embed([m.text for m in memories])
            for memory, vector in zip(memories, vectors, strict=True):
                # MemoryRow serializes to JSON on write, which has no ndarray encoding.
                memory.vector = vector.tolist()
            db = self.get_db()
            db.drop_table(MEMORIES_TABLE)
            new_table = ensure_table(db, MEMORIES_TABLE, self._memories_schema())
            new_table.add([m.model_dump(mode="json") for m in memories])
        return len(memories)

    def close(self) -> None:
        """Release the database connection and reset state."""
        self._db = None
        self._fts_ready = False
        self._title_fts_ready = False
        self._scalar_ready = False

    def drop_all(self) -> None:
        """Drop every table except ``_memories`` -- used by rebuild.

        Memory is user-authored data with no on-disk source, not derived from
        documents, so a rebuild preserves it. Only a factory reset (which deletes
        the data directory) clears it.
        """
        with self._write_lock():
            self._fts_ready = False
            self._title_fts_ready = False
            self._scalar_ready = False
            db = self.get_db()
            for name in table_names(db):
                if name == MEMORIES_TABLE:
                    continue
                db.drop_table(name)
        self._invalidate_source_cache()
