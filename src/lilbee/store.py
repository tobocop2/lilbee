"""LanceDB vector store operations."""

import logging
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypedDict

import lancedb
import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field

from lilbee.config import CHUNKS_TABLE, SOURCES_TABLE, Config, cfg
from lilbee.lock import write_lock

log = logging.getLogger(__name__)

# How often readers re-check the manifest for new versions from other processes.
# Zero means strong consistency (every read checks); higher values reduce disk I/O
# on slow media (HDD) at the cost of serving slightly stale data.
READ_CONSISTENCY_INTERVAL = timedelta(seconds=5)


class _FtsState:
    """Ephemeral FTS index state — resets on process start."""

    ready: bool = False


class SearchChunk(BaseModel):
    """A search result from LanceDB.

    Hybrid results have ``relevance_score`` set (higher = better).
    Vector-only results have ``distance`` set (lower = better).
    """

    model_config = ConfigDict(populate_by_name=True)

    source: str
    content_type: str
    page_start: int
    page_end: int
    line_start: int
    line_end: int
    chunk: str
    chunk_index: int
    vector: list[float] = Field(repr=False)
    distance: float | None = Field(None, alias="_distance")
    relevance_score: float | None = Field(None, alias="_relevance_score")


def _cosine_sim(a: list[float], b: list[float]) -> float:
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
    """
    if mmr_lambda is None:
        mmr_lambda = cfg.mmr_lambda
    if len(results) <= top_k:
        return results

    selected: list[SearchChunk] = []
    remaining = list(results)

    for _ in range(top_k):
        best_score = -float("inf")
        best_idx = 0
        for i, candidate in enumerate(remaining):
            relevance = _cosine_sim(query_vector, candidate.vector)
            redundancy = 0.0
            if selected:
                redundancy = max(_cosine_sim(candidate.vector, s.vector) for s in selected)
            score = mmr_lambda * relevance - (1 - mmr_lambda) * redundancy
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(remaining.pop(best_idx))

    return selected


class SourceRecord(TypedDict):
    """A tracked source document record."""

    filename: str
    file_hash: str
    ingested_at: str
    chunk_count: int


def _sources_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("filename", pa.utf8()),
            pa.field("file_hash", pa.utf8()),
            pa.field("ingested_at", pa.utf8()),
            pa.field("chunk_count", pa.int32()),
        ]
    )


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
    return value.replace("'", "''")


def _hybrid_search(
    table: lancedb.table.Table,
    query_text: str,
    query_vector: list[float],
    top_k: int,
) -> list[SearchChunk]:
    """Run hybrid (vector + FTS) search with RRF reranking."""
    from lancedb.rerankers import RRFReranker

    rows = (
        table.search(query_type="hybrid")
        .vector(query_vector)
        .text(query_text)
        .rerank(RRFReranker())
        .limit(top_k)
        .to_list()
    )
    return [SearchChunk(**r) for r in rows]


class RemoveResult:
    """Result of a remove_documents operation."""

    def __init__(self, removed: list[str], not_found: list[str]) -> None:
        self.removed = removed
        self.not_found = not_found


_MAX_THRESHOLD = 1.0
_MAX_FILTER_ITERATIONS = 20  # safety cap to prevent runaway loops


class Store:
    """LanceDB vector store — wraps all DB operations with config-driven defaults."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._fts = _FtsState()
        self._db: lancedb.DBConnection | None = None

    def _chunks_schema(self) -> pa.Schema:
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
                pa.field("vector", pa.list_(pa.float32(), self._config.embedding_dim)),
            ]
        )

    def get_db(self) -> lancedb.DBConnection:
        if self._db is None:
            self._config.lancedb_dir.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(
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
        """Create or replace the FTS index on the chunks table.

        No-op when the table doesn't exist or is empty.  Sets _fts.ready
        on success so hybrid_search can be used.
        """
        with write_lock():
            table = self.open_table(CHUNKS_TABLE)
            if table is None:
                return
            try:
                table.create_fts_index("chunk", replace=True)
                self._fts.ready = True
                log.debug("FTS index created/replaced on '%s'", CHUNKS_TABLE)
            except Exception:
                log.debug("FTS index creation failed (empty table?)", exc_info=True)

    def add_chunks(self, records: list[dict]) -> int:
        """Add chunk records to the store. Returns count added."""
        with write_lock():
            self._fts.ready = False
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
            return len(records)

    def bm25_probe(self, query_text: str, top_k: int = 5) -> list[SearchChunk]:
        """Quick BM25-only search for confidence checking. Returns up to top_k results."""
        table = self.open_table(CHUNKS_TABLE)
        if table is None:
            return []
        if not self._fts.ready:
            self.ensure_fts_index()
        if not self._fts.ready:
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
    ) -> list[SearchChunk]:
        """Search for similar chunks — hybrid when FTS available, else vector-only.

        Results with distance > max_distance are filtered out (vector-only path).
        Pass max_distance=0 to disable filtering.
        """
        if top_k is None:
            top_k = self._config.top_k
        if max_distance is None:
            max_distance = self._config.max_distance
        table = self.open_table(CHUNKS_TABLE)
        if table is None:
            return []

        if query_text and not self._fts.ready:
            self.ensure_fts_index()

        if query_text and self._fts.ready:
            try:
                return _hybrid_search(table, query_text, query_vector, top_k)
            except Exception:
                log.debug("Hybrid search failed, falling back to vector-only", exc_info=True)

        candidate_k = top_k * self._config.candidate_multiplier
        rows = table.search(query_vector).metric("cosine").limit(candidate_k).to_list()
        results = [SearchChunk(**r) for r in rows]
        if max_distance > 0:
            results = self._adaptive_filter(results, top_k, max_distance)
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

        Step size is ``self._config.adaptive_threshold_step`` (default 0.2).
        Stops after ``_MAX_FILTER_ITERATIONS`` to prevent runaway loops.
        """
        cap = max(initial_threshold, _MAX_THRESHOLD)
        step = self._config.adaptive_threshold_step
        threshold = initial_threshold
        for _ in range(_MAX_FILTER_ITERATIONS):
            if threshold > cap:
                break
            filtered = [
                r
                for r in results
                if (r.distance if r.distance is not None else float("inf")) <= threshold
            ]
            if len(filtered) >= top_k:
                return filtered
            threshold += step
        return [
            r for r in results if (r.distance if r.distance is not None else float("inf")) <= cap
        ]

    def get_chunks_by_source(self, source: str) -> list[SearchChunk]:
        """Return all chunks for a given source file."""
        table = self.open_table(CHUNKS_TABLE)
        if table is None:
            return []
        escaped = escape_sql_string(source)
        rows = table.search().where(f"source = '{escaped}'").to_list()
        return [SearchChunk(**r) for r in rows]

    def delete_by_source(self, source: str) -> None:
        """Delete all chunks from a given source file."""
        with write_lock():
            table = self.open_table(CHUNKS_TABLE)
            if table is not None:
                _safe_delete_unlocked(table, f"source = '{escape_sql_string(source)}'")

    def get_sources(self) -> list[SourceRecord]:
        """Get all tracked source file records."""
        table = self.open_table(SOURCES_TABLE)
        if table is None:
            return []
        result: list[SourceRecord] = table.to_arrow().to_pylist()  # type: ignore[assignment]
        return result

    def upsert_source(self, filename: str, file_hash: str, chunk_count: int) -> None:
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
                    }
                ]
            )

    def delete_source(self, filename: str) -> None:
        """Remove a source file tracking record."""
        with write_lock():
            table = self.open_table(SOURCES_TABLE)
            if table is not None:
                _safe_delete_unlocked(table, f"filename = '{escape_sql_string(filename)}'")

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
                path = (documents_dir / name).resolve()
                if not path.is_relative_to(documents_dir.resolve()):
                    log.warning("Path traversal blocked: %s escapes %s", name, documents_dir)
                    continue
                if path.exists():
                    path.unlink()

        return RemoveResult(removed=removed, not_found=not_found)

    def drop_all(self) -> None:
        """Drop all tables — used by rebuild."""
        with write_lock():
            self._fts.ready = False
            db = self.get_db()
            for name in _table_names(db):
                db.drop_table(name)


# ---------------------------------------------------------------------------
# Module-level convenience API -- delegates through runtime singleton
# ---------------------------------------------------------------------------


def get_db() -> lancedb.DBConnection:
    from lilbee.runtime import get_store

    return get_store().get_db()


def open_table(name: str) -> lancedb.table.Table | None:
    from lilbee.runtime import get_store

    return get_store().open_table(name)


def ensure_fts_index() -> None:
    from lilbee.runtime import get_store

    get_store().ensure_fts_index()


def add_chunks(records: list[dict]) -> int:
    from lilbee.runtime import get_store

    return get_store().add_chunks(records)


def bm25_probe(query_text: str, top_k: int = 5) -> list[SearchChunk]:
    from lilbee.runtime import get_store

    return get_store().bm25_probe(query_text, top_k)


def search(
    query_vector: list[float],
    top_k: int | None = None,
    max_distance: float | None = None,
    query_text: str | None = None,
) -> list[SearchChunk]:
    from lilbee.runtime import get_store

    return get_store().search(query_vector, top_k, max_distance, query_text)


def get_chunks_by_source(source: str) -> list[SearchChunk]:
    from lilbee.runtime import get_store

    return get_store().get_chunks_by_source(source)


def delete_by_source(source: str) -> None:
    from lilbee.runtime import get_store

    get_store().delete_by_source(source)


def delete_source(filename: str) -> None:
    from lilbee.runtime import get_store

    get_store().delete_source(filename)


def get_sources() -> list[SourceRecord]:
    from lilbee.runtime import get_store

    return get_store().get_sources()


def upsert_source(filename: str, file_hash: str, chunk_count: int) -> None:
    from lilbee.runtime import get_store

    get_store().upsert_source(filename, file_hash, chunk_count)


def remove_documents(
    names: list[str],
    *,
    delete_files: bool = False,
    documents_dir: Path | None = None,
) -> RemoveResult:
    from lilbee.runtime import get_store

    return get_store().remove_documents(
        names, delete_files=delete_files, documents_dir=documents_dir
    )


def drop_all() -> None:
    from lilbee.runtime import get_store

    get_store().drop_all()
