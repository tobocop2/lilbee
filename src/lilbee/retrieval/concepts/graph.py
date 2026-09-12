"""ConceptGraph: extracts, stores, and queries concept relationships."""

from __future__ import annotations

import logging
import threading
from collections import Counter
from collections.abc import Collection, Iterator
from typing import TYPE_CHECKING, Any, NamedTuple

import pyarrow as pa
import pyarrow.compute as pc

if TYPE_CHECKING:
    import lancedb.table

from lilbee.core.config import (
    CHUNK_CONCEPTS_TABLE,
    CONCEPT_EDGES_TABLE,
    CONCEPT_NODES_TABLE,
    Config,
)
from lilbee.data import store as data_store
from lilbee.data.store import ConceptRecords, Store, escape_sql_string
from lilbee.retrieval.concepts.community import Community, _compute_pmi, _leiden_partition
from lilbee.retrieval.concepts.nlp import _ensure_spacy_model, _filter_noun_chunks
from lilbee.retrieval.concepts.schema import (
    _chunk_concepts_schema,
    _concept_edges_schema,
    _concept_nodes_schema,
)
from lilbee.runtime import lock

log = logging.getLogger(__name__)

# Rows per record batch when scanning a concept table; bounds the Python-dict
# working set while the columnar Arrow data stays compact.
_TABLE_SCAN_BATCH_ROWS = 50_000

_CONCEPT_TABLES = (CONCEPT_NODES_TABLE, CONCEPT_EDGES_TABLE, CHUNK_CONCEPTS_TABLE)


def _iter_row_batches(table: lancedb.table.Table) -> Iterator[list[dict[str, Any]]]:
    """Yield a table's rows as bounded-size lists of dicts."""
    for batch in table.to_arrow().to_batches(max_chunksize=_TABLE_SCAN_BATCH_ROWS):
        yield batch.to_pylist()


def _quoted(values: Collection[str]) -> str:
    """SQL string list for an IN predicate, in stable order."""
    return ", ".join(f"'{escape_sql_string(value)}'" for value in sorted(values))


def _distinct_chunk_count(cc_table: Any) -> int:
    """Chunks carrying at least one concept."""
    keys: set[tuple[str, int]] = set()
    for rows in _iter_row_batches(cc_table):
        for row in rows:
            keys.add((row["chunk_source"], row["chunk_index"]))
    return len(keys)


def _affected_predicates(
    touched: set[str], affected_ids: set[int], affected: set[str]
) -> tuple[str, str]:
    """Scoped delete predicates covering the touched communities."""
    parts = [f"concept IN ({_quoted(touched)})"]
    if affected_ids:
        parts.append(f"cluster_id IN ({', '.join(str(cid) for cid in sorted(affected_ids))})")
    edge_list = _quoted(affected)
    return " OR ".join(parts), f"source IN ({edge_list}) OR target IN ({edge_list})"


class _PmiInputs(NamedTuple):
    """Corpus-wide counts PMI is computed from (see _corpus_pmi_inputs)."""

    cooccurrences: Counter[tuple[str, str]]
    concept_counts: Counter[str]
    total_chunks: int
    # A missing map means concepts were never built; an existing but empty one
    # means the corpus was emptied. Both give total_chunks == 0, and only the
    # second should clear the graph.
    map_exists: bool


class ConceptGraph:
    """Concept graph -- extracts, stores, and queries concept relationships."""

    def __init__(self, config: Config, store: Store) -> None:
        self._config = config
        self._store = store
        self._nlp: Any = None
        self._nlp_unavailable: bool = False
        # A spaCy Language is not safe for concurrent processing (shared Vocab /
        # StringStore). ConceptGraph is a Services singleton, so serialize every
        # nlp() / nlp.pipe() call on the shared daemon behind this lock.
        self._nlp_lock = threading.Lock()
        # Single-entry memo: one search() extracts concepts for the same query
        # twice (expansion + boost), so cache the last (text, max) -> result to
        # spare the second spaCy pass without unbounded growth.
        self._last_extract: tuple[tuple[str, int], list[str]] | None = None

    def _ensure_nlp(self) -> Any | None:
        """Lazy-load and cache the spaCy model. Returns None if unavailable."""
        if self._nlp is None and not self._nlp_unavailable:
            # Double-checked under _nlp_lock so two concurrent first-callers don't
            # each load en_core_web_sm (the loser would just be discarded).
            with self._nlp_lock:
                if self._nlp is None and not self._nlp_unavailable:
                    try:
                        self._nlp = _ensure_spacy_model()
                    except ImportError:
                        log.warning("Concept graph disabled: spaCy model unavailable")
                        self._nlp_unavailable = True
        return self._nlp

    def extract_concepts(self, text: str, max_concepts: int | None = None) -> list[str]:
        """Extract noun-phrase concepts from text via spaCy."""
        if max_concepts is None:
            max_concepts = self._config.concept_max_per_chunk
        if not text.strip():
            return []
        nlp = self._ensure_nlp()
        if nlp is None:
            return []
        cache_key = (text, max_concepts)
        with self._nlp_lock:
            if self._last_extract is not None and self._last_extract[0] == cache_key:
                return self._last_extract[1]
            doc = nlp(text)
            result = _filter_noun_chunks(doc, max_concepts)
            self._last_extract = (cache_key, result)
            return result

    def extract_concepts_batch(self, texts: list[str]) -> list[list[str]]:
        """Batch-extract concepts from multiple texts."""
        if not texts:
            return []
        nlp = self._ensure_nlp()
        if nlp is None:
            return [[] for _ in texts]
        max_concepts = self._config.concept_max_per_chunk
        # Hold the lock across the full pipe iteration: nlp.pipe is lazy, so the
        # actual parsing happens as the comprehension consumes it.
        with self._nlp_lock:
            return [_filter_noun_chunks(doc, max_concepts) for doc in nlp.pipe(texts)]

    def build_concept_records(
        self, chunk_ids: list[tuple[str, int]], concept_lists: list[list[str]]
    ) -> ConceptRecords:
        """Build co-occurrence graph rows from chunk concepts; no store access.

        Edge weights are raw co-occurrence counts (the edges carry graph
        connectivity), not PMI. Corpus PMI for clustering is recomputed from the
        chunk_concepts map in :meth:`rebuild_clusters`: computing PMI per file (with
        one file's chunk count as the denominator) and summing the per-file weights
        inflates pairs that recur across many small files, which is not corpus PMI.
        """
        cooccurrences: Counter[tuple[str, str]] = Counter()
        concept_counts: Counter[str] = Counter()
        chunk_concept_records: list[dict[str, Any]] = []

        for (source, idx), concepts in zip(chunk_ids, concept_lists, strict=True):
            for c in concepts:
                concept_counts[c] += 1
                chunk_concept_records.append(
                    {"chunk_source": source, "chunk_index": idx, "concept": c}
                )
            for i, a in enumerate(concepts):
                for b in concepts[i + 1 :]:
                    pair = (min(a, b), max(a, b))
                    cooccurrences[pair] += 1

        return ConceptRecords(
            nodes=[
                {"concept": c, "cluster_id": 0, "degree": count}
                for c, count in concept_counts.items()
            ],
            edges=[
                {"source": a, "target": b, "weight": float(count)}
                for (a, b), count in cooccurrences.items()
            ],
            chunk_concepts=chunk_concept_records,
        )

    def write_concept_records(self, records: ConceptRecords) -> None:
        """Write batched concept rows: one lock acquisition, at most one add per table."""
        with lock.write_lock(self._config.lancedb_dir):
            db = self._store.get_db()
            # Always create tables so get_graph() returns True even when
            # concept extraction yields no results for the current corpus.
            nodes_tbl = data_store.ensure_table(db, CONCEPT_NODES_TABLE, _concept_nodes_schema())
            edges_tbl = data_store.ensure_table(db, CONCEPT_EDGES_TABLE, _concept_edges_schema())
            cc_tbl = data_store.ensure_table(db, CHUNK_CONCEPTS_TABLE, _chunk_concepts_schema())
            if records.nodes:
                nodes_tbl.add(records.nodes)
            if records.edges:
                edges_tbl.add(records.edges)
            if records.chunk_concepts:
                cc_tbl.add(records.chunk_concepts)

    def boost_results(self, results: list[Any], query_concepts: list[str]) -> list[Any]:
        """Boost search results whose chunks overlap with query concepts.

        One batched chunk_concepts query serves the whole result set, grouped
        back per chunk in Python, so the boost costs one query rather than one
        per result.
        """
        if not query_concepts or not results:
            return results
        table = self._store.open_table(CHUNK_CONCEPTS_TABLE)
        if table is None:
            return results
        query_set = set(query_concepts)
        concepts_by_chunk = self._chunk_concepts_batch(
            table, {(r.source, r.chunk_index) for r in results}
        )
        boosted: list[Any] = []
        for r in results:
            chunk_concepts = concepts_by_chunk.get((r.source, r.chunk_index), set())
            overlap = len(query_set & chunk_concepts)
            if overlap > 0:
                boost = (overlap / len(query_set)) * self._config.concept_boost_weight
                r = r.model_copy()
                if r.score is not None:
                    # Canonical [0, 1] space: the boost weight is directly
                    # comparable to an arm's fusion weight. (Added to a raw
                    # RRF score, whose whole range is ~0.017, the same 0.3
                    # default swamped hybrid ranking outright.)
                    r.score = min(1.0, r.score + boost)
            boosted.append(r)
        return boosted

    def get_chunk_concepts(self, source: str, chunk_index: int) -> list[str]:
        """Get concepts associated with a specific chunk."""
        table = self._store.open_table(CHUNK_CONCEPTS_TABLE)
        if table is None:
            return []
        escaped = escape_sql_string(source)
        try:
            rows = (
                table.search()
                .where(f"chunk_source = '{escaped}' AND chunk_index = {int(chunk_index)}")
                .to_list()
            )
        except Exception:
            log.debug("get_chunk_concepts query failed for %r", source, exc_info=True)
            return []
        return [r["concept"] for r in rows]

    @staticmethod
    def _chunk_concepts_batch(
        table: Any, chunks: set[tuple[str, int]]
    ) -> dict[tuple[str, int], set[str]]:
        """Fetch many chunks' concepts in one query, keyed by (source, index).

        The predicate is the cross product of the distinct sources and
        indexes -- a cheap superset -- and rows are filtered back to the
        exact requested pairs in Python.
        """
        sources = ", ".join(f"'{escape_sql_string(s)}'" for s in sorted({s for s, _ in chunks}))
        indexes = ", ".join(str(int(i)) for i in sorted({i for _, i in chunks}))
        try:
            rows = (
                table.search()
                .where(f"chunk_source IN ({sources}) AND chunk_index IN ({indexes})")
                .to_list()
            )
        except Exception:
            log.debug("chunk concepts batch query failed", exc_info=True)
            return {}
        concepts_by_chunk: dict[tuple[str, int], set[str]] = {}
        for row in rows:
            key = (row["chunk_source"], row["chunk_index"])
            if key in chunks:
                concepts_by_chunk.setdefault(key, set()).add(row["concept"])
        return concepts_by_chunk

    def expand_query(self, query: str) -> list[str]:
        """Expand a query with related concepts from the graph."""
        concepts = self.extract_concepts(query)
        if not concepts:
            return []
        related: list[str] = []
        seen = set(concepts)
        for concept in concepts:
            for neighbor in self.get_related_concepts(concept):
                if neighbor not in seen:
                    related.append(neighbor)
                    seen.add(neighbor)
        return related

    def get_related_concepts(self, concept: str, depth: int = 1) -> list[str]:
        """Find concepts related to *concept* via graph edges, up to *depth* hops.

        One batched query per depth level: O(depth) DB round-trips,
        independent of frontier size.
        """
        table = self._store.open_table(CONCEPT_EDGES_TABLE)
        if table is None:
            return []
        visited: set[str] = {concept}
        frontier: list[str] = [concept]
        for _ in range(depth):
            if not frontier:
                break
            escaped_list = ", ".join(f"'{escape_sql_string(n)}'" for n in frontier)
            try:
                rows = (
                    table.search()
                    .where(f"source IN ({escaped_list}) OR target IN ({escaped_list})")
                    .to_list()
                )
            except Exception:
                log.debug(
                    "concept expand batch failed at frontier size %d",
                    len(frontier),
                    exc_info=True,
                )
                break
            next_frontier: list[str] = []
            for row in rows:
                for endpoint in (row["source"], row["target"]):
                    if endpoint not in visited:
                        visited.add(endpoint)
                        next_frontier.append(endpoint)
            frontier = next_frontier
        return [c for c in visited if c != concept]

    def top_communities(self, k: int = 10) -> list[Community]:
        """Return the *k* largest concept communities.

        Uses ``pyarrow.compute.value_counts`` to pick the top-k
        cluster_ids in columnar memory, then materializes only those
        clusters' members. Peak Python memory scales with members of
        the top *k* clusters, not the total node count.
        """
        table = self._store.open_table(CONCEPT_NODES_TABLE)
        if table is None:
            return []
        arrow_tbl = table.to_arrow()
        if arrow_tbl.num_rows == 0:
            return []
        counts = pc.value_counts(arrow_tbl["cluster_id"]).to_pylist()
        top = sorted(counts, key=lambda entry: entry["counts"], reverse=True)[:k]
        top_ids = [entry["values"] for entry in top if entry["values"] is not None]
        if not top_ids:
            return []
        member_rows = arrow_tbl.filter(
            pc.is_in(arrow_tbl["cluster_id"], value_set=pa.array(top_ids))
        ).to_pylist()
        by_cluster: dict[int, list[str]] = {}
        for row in member_rows:
            by_cluster.setdefault(row["cluster_id"], []).append(row["concept"])
        return [
            Community(
                cluster_id=cid,
                size=len(by_cluster.get(cid, [])),
                concepts=by_cluster.get(cid, []),
            )
            for cid in top_ids
            if by_cluster.get(cid)
        ]

    def _corpus_pmi_inputs(
        self,
    ) -> _PmiInputs:
        """Co-occurrence counts, concept document-frequencies, and chunk count,
        all derived from the chunk_concepts table.

        chunk_concepts is the ground-truth concept<->chunk map: it is source-scoped
        (re-ingesting a source replaces its rows) and its schema is stable, so PMI
        computed from it stays correct across re-ingests and version upgrades. The
        edge table accrues per-file appends between rebuilds and those weights are
        per-file co-occurrence counts, not corpus PMI, so it is not a safe source
        for these corpus counts.

        Concepts are de-duplicated per chunk, so a concept (or pair) counts once per
        distinct chunk it appears in -- the document frequency PMI is defined on.
        """
        cooccurrences: Counter[tuple[str, str]] = Counter()
        concept_counts: Counter[str] = Counter()
        table = self._store.open_table(CHUNK_CONCEPTS_TABLE)
        if table is None:
            return _PmiInputs(cooccurrences, concept_counts, 0, map_exists=False)
        per_chunk: dict[tuple[str, int], set[str]] = {}
        for rows in _iter_row_batches(table):
            for row in rows:
                key = (row["chunk_source"], row["chunk_index"])
                per_chunk.setdefault(key, set()).add(row["concept"])
        for concepts in per_chunk.values():
            ordered = sorted(concepts)
            for c in ordered:
                concept_counts[c] += 1
            for i, a in enumerate(ordered):
                for b in ordered[i + 1 :]:
                    cooccurrences[(a, b)] += 1
        return _PmiInputs(cooccurrences, concept_counts, len(per_chunk), map_exists=True)

    def rebuild_clusters(self, added: Collection[str] = (), updated: Collection[str] = ()) -> None:
        """Re-run Leiden clustering; scoped to changed sources when any are named.

        Empty *added*/*updated* re-partitions the whole corpus. Otherwise only
        the communities touching the changed sources' concepts are re-run and
        merged back; untouched clusters keep their rows. Removals pass nothing:
        their rows are already gone, so only a full pass clears their nodes.
        """
        changed = set(added) | set(updated)
        if changed and self._recluster_changed(changed, sweep_orphans=bool(updated)):
            return
        self._rebuild_all()

    def _recluster_changed(self, changed: set[str], *, sweep_orphans: bool) -> bool:
        """Re-partition the communities touching *changed*. False needs a full pass."""
        nodes_table = self._store.open_table(CONCEPT_NODES_TABLE)
        if nodes_table is None:
            return False
        node_rows = nodes_table.to_arrow().to_pylist()
        if not node_rows:
            return False
        cc_table = self._store.open_table(CHUNK_CONCEPTS_TABLE)
        if cc_table is None:
            return True
        touched = self._touched_concepts(cc_table, changed)
        if touched:
            self._recluster_affected(cc_table, node_rows, touched)
        if sweep_orphans:
            self._sweep_orphan_concepts(cc_table, node_rows)
        return True

    @staticmethod
    def _touched_concepts(cc_table: Any, changed: set[str]) -> set[str]:
        """Distinct concepts the changed sources' chunks currently carry."""
        rows = cc_table.search().where(f"chunk_source IN ({_quoted(changed)})").to_list()
        return {row["concept"] for row in rows}

    def _recluster_affected(
        self, cc_table: Any, node_rows: list[dict[str, Any]], touched: set[str]
    ) -> None:
        """Re-partition the touched communities and merge the result back."""
        concept_to_cluster = {row["concept"]: row["cluster_id"] for row in node_rows}
        affected_ids = {concept_to_cluster[c] for c in touched if c in concept_to_cluster}
        affected = set(touched)
        affected.update(c for c, cid in concept_to_cluster.items() if cid in affected_ids)
        counts, cooccurrences = self._affected_pmi_inputs(cc_table, affected)
        pmi_weights = _compute_pmi(cooccurrences, counts, _distinct_chunk_count(cc_table))
        node_records: list[dict[str, Any]] = []
        edge_rows = [{"source": a, "target": b, "weight": w} for (a, b), w in pmi_weights.items()]
        if pmi_weights:
            partition, degree_map = _leiden_partition(edge_rows)
            next_id = max(row["cluster_id"] for row in node_rows) + 1
            node_records = [
                {"concept": c, "cluster_id": next_id + cid, "degree": degree_map.get(c, 0)}
                for c, cid in partition.items()
                if c in affected
            ]
        nodes_predicate, edges_predicate = _affected_predicates(touched, affected_ids, affected)
        self._store.clear_and_add(
            CONCEPT_NODES_TABLE, _concept_nodes_schema(), node_records, nodes_predicate
        )
        self._store.clear_and_add(
            CONCEPT_EDGES_TABLE, _concept_edges_schema(), edge_rows, edges_predicate
        )

    def _affected_pmi_inputs(
        self, cc_table: Any, affected: set[str]
    ) -> tuple[Counter[str], Counter[tuple[str, str]]]:
        """Corpus counts and co-occurrences for pairs touching *affected*.

        A pair's chunks all hold its affected endpoint, so the neighborhood
        carries every affected pair's co-occurrence count exactly; only the
        outside neighbors' document frequencies need a second count.
        """
        chunk_rows = cc_table.search().where(f"concept IN ({_quoted(affected)})").to_list()
        chunk_keys = {(row["chunk_source"], row["chunk_index"]) for row in chunk_rows}
        concepts_by_chunk = self._chunk_concepts_batch(cc_table, chunk_keys)
        counts: Counter[str] = Counter()
        cooccurrences: Counter[tuple[str, str]] = Counter()
        for concepts in concepts_by_chunk.values():
            ordered = sorted(concepts)
            for concept in ordered:
                if concept in affected:
                    counts[concept] += 1
            for i, first in enumerate(ordered):
                for second in ordered[i + 1 :]:
                    if first in affected or second in affected:
                        cooccurrences[(first, second)] += 1
        neighbors = {c for concepts in concepts_by_chunk.values() for c in concepts} - affected
        if neighbors:
            count_rows = cc_table.search().where(f"concept IN ({_quoted(neighbors)})").to_list()
            seen: dict[str, set[tuple[str, int]]] = {}
            for row in count_rows:
                seen.setdefault(row["concept"], set()).add(
                    (row["chunk_source"], row["chunk_index"])
                )
            for concept, keys in seen.items():
                counts[concept] = len(keys)
        return counts, cooccurrences

    def _sweep_orphan_concepts(self, cc_table: Any, node_rows: list[dict[str, Any]]) -> None:
        """Delete nodes and edges for concepts no chunk carries anymore."""
        live = {row["concept"] for rows in _iter_row_batches(cc_table) for row in rows}
        orphans = {row["concept"] for row in node_rows} - live
        if orphans:
            quoted = _quoted(orphans)
            self._store.clear_and_add(
                CONCEPT_NODES_TABLE, _concept_nodes_schema(), [], f"concept IN ({quoted})"
            )
            self._store.clear_and_add(
                CONCEPT_EDGES_TABLE,
                _concept_edges_schema(),
                [],
                f"source IN ({quoted}) OR target IN ({quoted})",
            )

    def _rebuild_all(self) -> None:
        """Recompute corpus PMI from the chunk_concepts map, re-run Leiden, compact.

        PMI is a corpus-level statistic, so it is computed once over corpus-wide
        co-occurrence and concept counts (see :meth:`_corpus_pmi_inputs`) rather
        than per file; summing per-file PMI would inflate pairs that recur across
        many small files.

        Both the nodes and the edges tables are replaced with the freshly
        computed corpus graph. Per-file writes only ever append edges, so
        without this rewrite the edges table grows monotonically across syncs
        and expand_query keeps serving edges for concepts that left the corpus.
        """
        cooccurrences, concept_counts, total_chunks, map_exists = self._corpus_pmi_inputs()
        if total_chunks == 0:
            # The corpus was emptied: leaving the last graph in place would keep
            # expansion serving concepts no document carries any more.
            if map_exists:
                self._clear_graph()
            return
        if not cooccurrences:
            # Chunks remain but no concept pair co-occurs: the previous graph
            # is stale, not still valid.
            self._clear_graph()
            return
        pmi_weights = _compute_pmi(cooccurrences, concept_counts, total_chunks)
        if not pmi_weights:
            # Every pair co-occurred at or below chance: no edge set to
            # cluster, and the previous graph no longer describes the corpus.
            self._clear_graph()
            return
        edge_rows = [{"source": a, "target": b, "weight": w} for (a, b), w in pmi_weights.items()]

        partition, degree_map = _leiden_partition(edge_rows)

        node_records = [
            {
                "concept": node,
                "cluster_id": cluster_id,
                "degree": degree_map.get(node, 0),
            }
            for node, cluster_id in partition.items()
        ]

        # Delete the old rows and add the new ones under one lock per table so
        # a reader never sees a table emptied while get_graph() still reports
        # it present (which would blank top_communities / cluster labels).
        self._store.clear_and_add(
            CONCEPT_NODES_TABLE, _concept_nodes_schema(), node_records, "concept IS NOT NULL"
        )
        self._store.clear_and_add(
            CONCEPT_EDGES_TABLE, _concept_edges_schema(), edge_rows, "source IS NOT NULL"
        )
        self.compact_tables()

    def _clear_graph(self) -> None:
        """Drop every node and edge, keeping both tables present for readers."""
        self._store.clear_and_add(
            CONCEPT_NODES_TABLE, _concept_nodes_schema(), [], "concept IS NOT NULL"
        )
        self._store.clear_and_add(
            CONCEPT_EDGES_TABLE, _concept_edges_schema(), [], "source IS NOT NULL"
        )

    def compact_tables(self) -> None:
        """Compact the concept tables; per-file adds otherwise accrete tiny versions."""
        with lock.write_lock(self._config.lancedb_dir):
            for name in _CONCEPT_TABLES:
                table = self._store.open_table(name)
                if table is None:
                    continue
                try:
                    table.optimize()
                except Exception:
                    log.debug("Concept table optimize failed on '%s'", name, exc_info=True)

    def get_cluster_sources(self, min_sources: int = 3) -> dict[int, set[str]]:
        """Return clusters that span at least *min_sources* distinct sources.
        Joins concept_nodes (concept -> cluster_id) with chunk_concepts
        (concept -> chunk_source) to find which document sources each
        cluster touches.
        """
        nodes_table = self._store.open_table(CONCEPT_NODES_TABLE)
        cc_table = self._store.open_table(CHUNK_CONCEPTS_TABLE)
        if nodes_table is None or cc_table is None:
            return {}

        concept_to_cluster: dict[str, int] = {}
        for node_rows in _iter_row_batches(nodes_table):
            for row in node_rows:
                concept_to_cluster[row["concept"]] = row["cluster_id"]

        cluster_sources: dict[int, set[str]] = {}
        for cc_rows in _iter_row_batches(cc_table):
            for row in cc_rows:
                cid = concept_to_cluster.get(row["concept"])
                if cid is None:
                    continue
                cluster_sources.setdefault(cid, set()).add(row["chunk_source"])

        return {
            cid: sources for cid, sources in cluster_sources.items() if len(sources) >= min_sources
        }

    def get_cluster_label(self, cluster_id: int) -> str:
        """Return a human-readable label for *cluster_id* (highest-degree concept)."""
        table = self._store.open_table(CONCEPT_NODES_TABLE)
        if table is None:
            return f"cluster-{cluster_id}"
        try:
            rows = table.search().where(f"cluster_id = {int(cluster_id)}").to_list()
        except Exception:
            log.debug("get_cluster_label query failed", exc_info=True)
            return f"cluster-{cluster_id}"
        if not rows:
            return f"cluster-{cluster_id}"
        best = max(rows, key=lambda r: r["degree"])
        return str(best["concept"])

    def get_graph(self) -> bool:
        """Check whether a concept graph exists in the store."""
        if not self._config.concept_graph:
            return False
        return self._store.open_table(CONCEPT_NODES_TABLE) is not None

    def reset_nlp_cache(self) -> None:
        """Clear the spaCy model cache. For testing only."""
        self._nlp = None
        self._nlp_unavailable = False
