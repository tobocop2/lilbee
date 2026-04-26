"""LanceDB vector store operations."""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    import lancedb
    import lancedb.table

from lilbee.config import (
    CHUNKS_TABLE,
    CITATIONS_TABLE,
    META_TABLE,
    SOURCES_TABLE,
    Config,
    cfg,
)
from lilbee.lock import write_lock
from lilbee.security import validate_path_within

log = logging.getLogger(__name__)


def install_lancedb_thread_error_suppressor() -> None:
    """Install a ``threading.excepthook`` that swallows lancedb shutdown noise.
    lancedb has no ``close()`` API and its internal event loop thread crashes
    during Python interpreter teardown. The exception is harmless (the process
    is exiting anyway) but pollutes CLI/TUI output. This is opt-in so importing
    ``lilbee.store`` has no hidden side effects; call it once from the CLI/TUI
    bootstrap.
    """
    original = threading.excepthook

    def _hook(args: threading.ExceptHookArgs) -> None:
        if args.thread and "LanceDB" in args.thread.name:
            return
        original(args)

    threading.excepthook = _hook


# How often readers re-check the manifest for new versions from other processes.
# Zero means strong consistency (every read checks); higher values reduce disk I/O
# on slow media (HDD) at the cost of serving slightly stale data.
READ_CONSISTENCY_INTERVAL = timedelta(seconds=5)

# Values for the ``chunk_type`` column. Everything goes in as raw except wiki
# pages written by the wiki producer; callers filter with ``Store.search(chunk_type=...)``.
CHUNK_TYPE_RAW = "raw"
CHUNK_TYPE_WIKI = "wiki"


class SearchScope(StrEnum):
    """What the user wants to search over.

    Values are used as-is on CLI flags, MCP params, and HTTP query strings.
    ``BOTH`` resolves to a ``None`` ``chunk_type`` (no filter); the two
    others map 1:1 to the chunks-table values.
    """

    RAW = CHUNK_TYPE_RAW
    WIKI = CHUNK_TYPE_WIKI
    BOTH = "both"


def scope_to_chunk_type(scope: SearchScope | str | None) -> str | None:
    """Translate a user-facing scope into a ``Store.search`` ``chunk_type`` arg.

    ``None``/``"both"`` → no filter. ``"raw"`` / ``"wiki"`` → the matching
    chunks-table value. Raises ``ValueError`` on any other string.
    """
    if scope is None:
        return None
    normalized = SearchScope(scope)
    if normalized is SearchScope.BOTH:
        return None
    return normalized.value


class SearchChunk(BaseModel):
    """A search result from LanceDB.
    Hybrid results have ``relevance_score`` set (higher = better).
    Vector-only results have ``distance`` set (lower = better).
    """

    model_config = ConfigDict(populate_by_name=True)

    source: str
    content_type: str
    chunk_type: str = CHUNK_TYPE_RAW

    @field_validator("chunk_type", mode="before")
    @classmethod
    def _coerce_none_chunk_type(cls, v: str | None) -> str:
        """LanceDB rows from before the chunk_type column was added return None."""
        return v if v is not None else CHUNK_TYPE_RAW

    page_start: int
    page_end: int
    line_start: int
    line_end: int
    chunk: str
    chunk_index: int
    vector: list[float] = Field(repr=False)
    distance: float | None = Field(None, alias="_distance")
    relevance_score: float | None = Field(None, alias="_relevance_score")


def cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def mmr_rerank(
    query_vector: list[float],
    results: list[SearchChunk],
    top_k: int,
    mmr_lambda: float | None = None,
) -> list[SearchChunk]:
    """Maximal Marginal Relevance — select diverse results.
    Algorithm: Carbonell & Goldstein 1998,
    "The Use of MMR, Diversity-Based Reranking for Reordering Documents
    and Producing Summaries."

    ``mmr_lambda`` controls the relevance/diversity tradeoff:
    0.0 = maximum diversity, 1.0 = pure relevance.
    Defaults to ``cfg.mmr_lambda`` (0.5).

    Complexity: O(top_k · N · D) time, O(N · D) space for N candidates
    of dimension D. Each outer iteration updates a running max-redundancy
    vector via one matmul rather than recomputing pairs pairwise.
    Candidate vectors run through numpy in ``float32``, which can pick a
    different candidate than the pure-Python ``float64`` loop on
    ties within ~1e-7; distinct in principle, unobservable in practice
    since sub-float32 differences are below retrieval signal.
    """
    if mmr_lambda is None:
        mmr_lambda = cfg.mmr_lambda
    if len(results) <= top_k:
        return results

    candidate_vecs = np.asarray([r.vector for r in results], dtype=np.float32)
    query = np.asarray(query_vector, dtype=np.float32)
    # L2-normalize once so cosine becomes a plain dot product.
    cand_norms = np.linalg.norm(candidate_vecs, axis=1, keepdims=True)
    cand_norms[cand_norms == 0] = 1.0
    cand_unit = candidate_vecs / cand_norms
    query_norm = float(np.linalg.norm(query)) or 1.0
    query_unit = query / query_norm

    relevance = cand_unit @ query_unit  # shape (N,)

    n = len(results)
    max_redundancy = np.zeros(n, dtype=np.float32)
    available = np.ones(n, dtype=bool)
    selected: list[SearchChunk] = []

    for picks in range(top_k):
        redundancy_term = max_redundancy if picks > 0 else np.zeros(n, dtype=np.float32)
        score = mmr_lambda * relevance - (1.0 - mmr_lambda) * redundancy_term
        # Mask already-picked candidates so argmax skips them.
        score = np.where(available, score, -np.inf)
        best = int(np.argmax(score))
        selected.append(results[best])
        available[best] = False
        # Update running max redundancy against the newly-selected vector.
        similarity = cand_unit @ cand_unit[best]
        max_redundancy = np.maximum(max_redundancy, similarity)

    return selected


class SourceRecord(TypedDict):
    """A tracked source document record."""

    filename: str
    file_hash: str
    ingested_at: str
    chunk_count: int
    source_type: str


class CitationRecord(TypedDict):
    """A citation linking a wiki chunk to a specific source location."""

    wiki_source: str
    wiki_chunk_index: int
    citation_key: str
    claim_type: str
    source_filename: str
    source_hash: str
    page_start: int
    page_end: int
    line_start: int
    line_end: int
    excerpt: str
    created_at: str


class StoreMeta(TypedDict):
    """Single-row store metadata recording the embedding model used to build the store.

    Compatibility is checked before every read and write. When ``cfg.embedding_model``
    or ``cfg.embedding_dim`` drifts from the persisted row, the store refuses to serve
    until ``lilbee rebuild`` (CLI) or ``POST /api/sync {"force_rebuild": true}`` (HTTP)
    rewrites the chunks under the new model.
    """

    embedding_model: str
    embedding_dim: int
    schema_version: int
    updated_at: str


# ``schema_version`` is an integer for forward-compat. Bump only if we ever need to
# add or rename a meta column without forcing every store to drop_all.
META_SCHEMA_VERSION = 1


class EmbeddingModelMismatchError(RuntimeError):
    """Raised when stored vectors were built with a different embedding model than ``cfg``.

    Carries a user-facing message naming both the persisted and the configured model and
    pointing at the two recovery paths (``lilbee rebuild`` and ``POST /api/sync`` with
    ``force_rebuild=true``).
    """


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


def _meta_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("embedding_model", pa.utf8()),
            pa.field("embedding_dim", pa.int32()),
            pa.field("schema_version", pa.int32()),
            pa.field("updated_at", pa.utf8()),
        ]
    )


def _sources_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("filename", pa.utf8()),
            pa.field("file_hash", pa.utf8()),
            pa.field("ingested_at", pa.utf8()),
            pa.field("chunk_count", pa.int32()),
            pa.field("source_type", pa.utf8()),
        ]
    )


def _citations_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("wiki_source", pa.utf8()),
            pa.field("wiki_chunk_index", pa.int32()),
            pa.field("citation_key", pa.utf8()),
            pa.field("claim_type", pa.utf8()),
            pa.field("source_filename", pa.utf8()),
            pa.field("source_hash", pa.utf8()),
            pa.field("page_start", pa.int32()),
            pa.field("page_end", pa.int32()),
            pa.field("line_start", pa.int32()),
            pa.field("line_end", pa.int32()),
            pa.field("excerpt", pa.utf8()),
            pa.field("created_at", pa.utf8()),
        ]
    )


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


@dataclass
class RemoveResult:
    """Result of a remove_documents operation."""

    removed: list[str]
    not_found: list[str]


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


class Store:
    """LanceDB vector store — wraps all DB operations with config-driven defaults."""

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

    def _write_meta_unlocked(self) -> None:
        """Overwrite the single ``_meta`` row with the current cfg snapshot.

        Caller must hold ``write_lock()``. Maintains the single-row invariant via
        delete-then-insert (mirrors ``upsert_source``).
        """
        db = self.get_db()
        table = ensure_table(db, META_TABLE, _meta_schema())
        _safe_delete_unlocked(table, "schema_version >= 0")
        table.add(
            [
                {
                    "embedding_model": self._config.embedding_model,
                    "embedding_dim": self._config.embedding_dim,
                    "schema_version": META_SCHEMA_VERSION,
                    "updated_at": datetime.now(UTC).isoformat(),
                }
            ]
        )

    def _ensure_embedding_compat(self) -> None:
        """Refuse to operate when the persisted embedding identity drifts from cfg.

        Behavior matrix:
        - ``_meta`` row matches cfg: no-op (the common case).
        - ``_meta`` empty AND chunks table empty: no-op. Will be initialized on first
          ``add_chunks``.
        - ``_meta`` empty AND chunks present (legacy store from before this gate
          existed): write ``_meta`` from current cfg with a one-time warning. This
          trusts that current cfg matches the actual stored vectors. Pre-upgrade
          silent corruption is not retroactively detectable, but the gate is active
          from this point forward.
        - ``_meta`` present and mismatches cfg: raise ``EmbeddingModelMismatchError``.
        """
        meta = self.get_meta()
        if meta is not None:
            if (
                meta["embedding_model"] == self._config.embedding_model
                and meta["embedding_dim"] == self._config.embedding_dim
            ):
                return
            raise EmbeddingModelMismatchError(
                _embedding_mismatch_message(
                    persisted_model=meta["embedding_model"],
                    persisted_dim=meta["embedding_dim"],
                    current_model=self._config.embedding_model,
                    current_dim=self._config.embedding_dim,
                )
            )
        chunks = self.open_table(CHUNKS_TABLE)
        if chunks is None or chunks.count_rows() == 0:
            return
        log.warning(
            "Legacy store has chunks but no _meta row. Initializing _meta from "
            "the current configuration (embedding_model=%s, embedding_dim=%d). "
            "If you changed embedding_model before upgrading, run `lilbee rebuild` "
            "to ensure the store is consistent.",
            self._config.embedding_model,
            self._config.embedding_dim,
        )
        with write_lock():
            self._write_meta_unlocked()

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
        """
        self._ensure_embedding_compat()
        with write_lock():
            self._fts_ready = False
            if not records:
                return 0
            for rec in records:
                vec = rec.get("vector", [])
                if len(vec) != self._config.embedding_dim:
                    raise ValueError(
                        f"Vector dimension mismatch: expected {self._config.embedding_dim}, "
                        f"got {len(vec)} (source={rec.get('source', '?')})"
                    )
            db = self.get_db()
            table = ensure_table(db, CHUNKS_TABLE, self._chunks_schema())
            table.add(records)
            if self.get_meta() is None:
                self._write_meta_unlocked()
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
        if max_distance > 0:
            before = len(results)
            if self._config.adaptive_threshold:
                results = self._adaptive_filter(results, top_k, max_distance)
                log.debug(
                    "After adaptive filter: %d/%d results, threshold=%.2f",
                    len(results),
                    before,
                    max_distance,
                )
            else:
                results = self._fixed_filter(results, max_distance)
                log.debug(
                    "After fixed filter: %d/%d results, threshold=%.2f",
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

    def delete_by_source(self, source: str) -> None:
        """Delete all chunks from a given source file."""
        with write_lock():
            table = self.open_table(CHUNKS_TABLE)
            if table is not None:
                _safe_delete_unlocked(table, f"source = '{escape_sql_string(source)}'")
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

    def delete_source(self, filename: str) -> None:
        """Remove a source file tracking record."""
        with write_lock():
            table = self.open_table(SOURCES_TABLE)
            if table is not None:
                _safe_delete_unlocked(table, f"filename = '{escape_sql_string(filename)}'")
        self._invalidate_source_cache()

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
            self.delete_by_source(name)
            self.delete_source(name)
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
