"""ConceptGraph: extracts, stores, and queries concept relationships."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from lilbee.concepts.community import Community, _compute_pmi, _leiden_partition
from lilbee.concepts.nlp import _ensure_spacy_model, _filter_noun_chunks
from lilbee.concepts.schema import (
    _chunk_concepts_schema,
    _concept_edges_schema,
    _concept_nodes_schema,
)
from lilbee.core.config import (
    CHUNK_CONCEPTS_TABLE,
    CONCEPT_EDGES_TABLE,
    CONCEPT_NODES_TABLE,
    Config,
)
from lilbee.data.store import Store, escape_sql_string

log = logging.getLogger(__name__)


class ConceptGraph:
    """Concept graph -- extracts, stores, and queries concept relationships."""

    def __init__(self, config: Config, store: Store) -> None:
        self._config = config
        self._store = store
        self._nlp: Any = None
        self._nlp_unavailable: bool = False

    def _ensure_nlp(self) -> Any | None:
        """Lazy-load and cache the spaCy model. Returns None if unavailable."""
        if self._nlp_unavailable:
            return None
        if self._nlp is None:
            try:
                self._nlp = _ensure_spacy_model()
            except ImportError:
                log.warning("Concept graph disabled: spaCy model unavailable")
                self._nlp_unavailable = True
                return None
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
        doc = nlp(text)
        return _filter_noun_chunks(doc, max_concepts)

    def extract_concepts_batch(self, texts: list[str]) -> list[list[str]]:
        """Batch-extract concepts from multiple texts."""
        if not texts:
            return []
        nlp = self._ensure_nlp()
        if nlp is None:
            return [[] for _ in texts]
        max_concepts = self._config.concept_max_per_chunk
        return [_filter_noun_chunks(doc, max_concepts) for doc in nlp.pipe(texts)]

    def build_from_chunks(
        self, chunk_ids: list[tuple[str, int]], concept_lists: list[list[str]]
    ) -> None:
        """Build co-occurrence graph from chunk concepts, compute PMI, store tables."""
        from lilbee.data.store import ensure_table
        from lilbee.lock import write_lock

        if not chunk_ids:
            return

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

        pmi_weights = _compute_pmi(cooccurrences, concept_counts, len(chunk_ids))

        edge_records = [
            {"source": a, "target": b, "weight": w} for (a, b), w in pmi_weights.items()
        ]

        node_records = [
            {"concept": c, "cluster_id": 0, "degree": count} for c, count in concept_counts.items()
        ]

        with write_lock():
            db = self._store.get_db()
            # Always create tables so get_graph() returns True even when
            # concept extraction yields no results for the current corpus.
            nodes_tbl = ensure_table(db, CONCEPT_NODES_TABLE, _concept_nodes_schema())
            edges_tbl = ensure_table(db, CONCEPT_EDGES_TABLE, _concept_edges_schema())
            cc_tbl = ensure_table(db, CHUNK_CONCEPTS_TABLE, _chunk_concepts_schema())
            if node_records:
                nodes_tbl.add(node_records)
            if edge_records:
                edges_tbl.add(edge_records)
            if chunk_concept_records:
                cc_tbl.add(chunk_concept_records)

    def boost_results(self, results: list[Any], query_concepts: list[str]) -> list[Any]:
        """Boost search results whose chunks overlap with query concepts."""
        if not query_concepts or not results:
            return results
        query_set = set(query_concepts)
        boosted: list[Any] = []
        for r in results:
            chunk_concepts = set(self.get_chunk_concepts(r.source, r.chunk_index))
            overlap = len(query_set & chunk_concepts)
            if overlap > 0:
                boost = (overlap / len(query_set)) * self._config.concept_boost_weight
                r = r.model_copy()
                if r.relevance_score is not None:
                    r.relevance_score = r.relevance_score + boost
                elif r.distance is not None:
                    r.distance = max(self._config.concept_boost_floor, r.distance - boost)
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
                .where(f"chunk_source = '{escaped}' AND chunk_index = {chunk_index}")
                .to_list()
            )
            return [r["concept"] for r in rows]
        except Exception:
            return []

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

        One batched query per depth level — O(depth) DB round-trips,
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

    def rebuild_clusters(self) -> None:
        """Re-run Leiden clustering on the existing edge table."""
        from lilbee.data.store import ensure_table
        from lilbee.lock import write_lock

        edges_table = self._store.open_table(CONCEPT_EDGES_TABLE)
        if edges_table is None:
            return
        edge_rows = edges_table.to_arrow().to_pylist()
        if not edge_rows:
            return

        partition, degree_map = _leiden_partition(edge_rows)

        node_records = [
            {
                "concept": node,
                "cluster_id": cluster_id,
                "degree": degree_map.get(node, 0),
            }
            for node, cluster_id in partition.items()
        ]

        self._store.clear_table(CONCEPT_NODES_TABLE, "concept IS NOT NULL")
        if node_records:
            with write_lock():
                db = self._store.get_db()
                nodes_table = ensure_table(db, CONCEPT_NODES_TABLE, _concept_nodes_schema())
                nodes_table.add(node_records)

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

        node_rows = nodes_table.to_arrow().to_pylist()
        concept_to_cluster: dict[str, int] = {r["concept"]: r["cluster_id"] for r in node_rows}

        cc_rows = cc_table.to_arrow().to_pylist()
        cluster_sources: dict[int, set[str]] = {}
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
