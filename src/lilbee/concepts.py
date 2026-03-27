"""Concept graph for LazyGraphRAG-style index-time knowledge extraction.

Extracts noun-phrase concepts from chunks via spaCy, builds a PPMI-weighted
co-occurrence graph, and clusters with Leiden (graspologic-native). Used to
boost search results by concept overlap and expand queries via graph traversal.

Requires optional ``graph`` extra: ``pip install lilbee[graph]``.
When dependencies are missing, all public functions degrade gracefully.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from lilbee.config import (
    CHUNK_CONCEPTS_TABLE,
    CONCEPT_EDGES_TABLE,
    CONCEPT_NODES_TABLE,
    Config,
    cfg,
)

log = logging.getLogger(__name__)

# Minimum concept length (chars) to filter noise like single letters
_MIN_CONCEPT_LEN = 2

# Minimum edge weight for Leiden clustering — avoids zero/negative weights
# which graspologic-native rejects
_MIN_LEIDEN_WEIGHT = 0.01


def concepts_available() -> bool:
    """Check if concept graph dependencies (spacy, graspologic) are installed."""
    try:
        import graspologic_native  # noqa: F401
        import spacy  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass
class Community:
    """A cluster of related concepts from Leiden partitioning."""

    cluster_id: int
    size: int
    concepts: list[str]


def _ensure_spacy_model() -> Any:
    """Load the spaCy model, auto-downloading on first use."""
    import spacy

    model_name = "en_core_web_sm"
    try:
        return spacy.load(model_name)
    except OSError:
        log.info("Downloading spaCy model '%s'...", model_name)
        from spacy.cli import download

        download(model_name)
        return spacy.load(model_name)


_nlp: Any = None


def _get_nlp() -> Any:
    """Lazy-load and cache the spaCy model."""
    global _nlp
    if _nlp is None:
        _nlp = _ensure_spacy_model()
    return _nlp


def _filter_noun_chunks(doc: Any, max_concepts: int) -> list[str]:
    """Extract deduplicated, filtered noun chunks from a spaCy doc."""
    seen: set[str] = set()
    concepts: list[str] = []
    for chunk in doc.noun_chunks:
        concept = chunk.text.lower().strip()
        if len(concept) < _MIN_CONCEPT_LEN:
            continue
        if concept in seen:
            continue
        seen.add(concept)
        concepts.append(concept)
        if len(concepts) >= max_concepts:
            break
    return concepts


def _concept_nodes_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("concept", pa.utf8()),
            pa.field("cluster_id", pa.int32()),
            pa.field("degree", pa.int32()),
        ]
    )


def _concept_edges_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("source", pa.utf8()),
            pa.field("target", pa.utf8()),
            pa.field("weight", pa.float32()),
        ]
    )


def _chunk_concepts_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("chunk_source", pa.utf8()),
            pa.field("chunk_index", pa.int32()),
            pa.field("concept", pa.utf8()),
        ]
    )


def _compute_pmi(
    cooccurrences: Counter[tuple[str, str]],
    concept_counts: Counter[str],
    total_chunks: int,
) -> dict[tuple[str, str], float]:
    """Compute PPMI (Positive PMI) weights for concept co-occurrence pairs.

    PPMI = max(0, log2(P(a,b) / (P(a) * P(b)))).
    Negative values are clamped to zero to discard anti-correlated pairs.
    """
    pmi: dict[tuple[str, str], float] = {}
    for (a, b), count in cooccurrences.items():
        p_ab = count / total_chunks
        p_a = concept_counts[a] / total_chunks
        p_b = concept_counts[b] / total_chunks
        pmi[(a, b)] = max(0.0, math.log2(p_ab / (p_a * p_b)))
    return pmi


def _leiden_partition(
    edge_rows: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    """Run Leiden clustering on edge rows. Returns (partition, degree_map).

    Uses graspologic-native's Rust implementation for performance.
    """
    from graspologic_native import leiden

    edges: list[tuple[str, str, float]] = [
        (row["source"], row["target"], max(_MIN_LEIDEN_WEIGHT, row["weight"])) for row in edge_rows
    ]
    _modularity, partition = leiden(edges=edges)  # type: ignore[call-arg]

    degree_map: dict[str, int] = Counter()
    for row in edge_rows:
        degree_map[row["source"]] += 1
        degree_map[row["target"]] += 1
    return partition, dict(degree_map)


class ConceptGraph:
    """Concept graph — extracts, stores, and queries concept co-occurrence data."""

    def __init__(self, config: Config) -> None:
        self._config = config

    def extract_concepts(self, text: str, max_concepts: int | None = None) -> list[str]:
        """Extract deduplicated noun-phrase concepts from text."""
        if max_concepts is None:
            max_concepts = self._config.concept_max_per_chunk
        if not text.strip():
            return []
        nlp = _get_nlp()
        doc = nlp(text)
        return _filter_noun_chunks(doc, max_concepts)

    def extract_concepts_batch(self, texts: list[str]) -> list[list[str]]:
        """Batch-extract concepts using spaCy's ``nlp.pipe()``."""
        if not texts:
            return []
        max_concepts = self._config.concept_max_per_chunk
        nlp = _get_nlp()
        return [_filter_noun_chunks(doc, max_concepts) for doc in nlp.pipe(texts)]

    def boost_results(
        self,
        results: list[Any],
        query_concepts: list[str],
    ) -> list[Any]:
        """Adjust search result scores by concept overlap with query."""
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
                    r.distance = max(0.0, r.distance - boost)
            boosted.append(r)
        return boosted

    def get_chunk_concepts(self, source: str, chunk_index: int) -> list[str]:
        """Look up concepts for a specific chunk."""
        from lilbee.store import escape_sql_string, open_table

        table = open_table(CHUNK_CONCEPTS_TABLE)
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
        """Extract concepts from query and find related concepts via graph."""
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
        """BFS traversal of edges to find related concepts."""
        from lilbee.store import escape_sql_string, open_table

        table = open_table(CONCEPT_EDGES_TABLE)
        if table is None:
            return []
        visited: set[str] = {concept}
        frontier: list[str] = [concept]
        for _ in range(depth):
            next_frontier: list[str] = []
            for node in frontier:
                escaped = escape_sql_string(node)
                try:
                    rows = (
                        table.search()
                        .where(f"source = '{escaped}' OR target = '{escaped}'")
                        .to_list()
                    )
                except Exception:
                    continue
                for row in rows:
                    neighbor = row["target"] if row["source"] == node else row["source"]
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier
        return [c for c in visited if c != concept]

    def get_graph(self) -> bool:
        """Return whether concept graph tables exist."""
        if not self._config.concept_graph:
            return False
        from lilbee.store import open_table

        return open_table(CONCEPT_NODES_TABLE) is not None


# ---------------------------------------------------------------------------
# Backwards-compatible module-level API
# ---------------------------------------------------------------------------


def extract_concepts(text: str, max_concepts: int | None = None) -> list[str]:
    if max_concepts is None:
        max_concepts = cfg.concept_max_per_chunk
    if not text.strip():
        return []
    nlp = _get_nlp()
    doc = nlp(text)
    return _filter_noun_chunks(doc, max_concepts)


def extract_concepts_batch(texts: list[str]) -> list[list[str]]:
    if not texts:
        return []
    max_concepts = cfg.concept_max_per_chunk
    nlp = _get_nlp()
    return [_filter_noun_chunks(doc, max_concepts) for doc in nlp.pipe(texts)]


def build_from_chunks(
    chunk_ids: list[tuple[str, int]],
    concept_lists: list[list[str]],
) -> None:
    """Build co-occurrence graph from chunk concepts, compute PMI, store tables."""
    from lilbee.lock import write_lock
    from lilbee.store import ensure_table, get_db

    if not chunk_ids:
        return

    cooccurrences: Counter[tuple[str, str]] = Counter()
    concept_counts: Counter[str] = Counter()
    chunk_concept_records: list[dict[str, Any]] = []

    for (source, idx), concepts in zip(chunk_ids, concept_lists, strict=True):
        for c in concepts:
            concept_counts[c] += 1
            chunk_concept_records.append({"chunk_source": source, "chunk_index": idx, "concept": c})
        for i, a in enumerate(concepts):
            for b in concepts[i + 1 :]:
                pair = (min(a, b), max(a, b))
                cooccurrences[pair] += 1

    pmi_weights = _compute_pmi(cooccurrences, concept_counts, len(chunk_ids))

    edge_records = [{"source": a, "target": b, "weight": w} for (a, b), w in pmi_weights.items()]

    node_records = [
        {"concept": c, "cluster_id": 0, "degree": count} for c, count in concept_counts.items()
    ]

    with write_lock():
        db = get_db()
        if node_records:
            table = ensure_table(db, CONCEPT_NODES_TABLE, _concept_nodes_schema())
            table.add(node_records)
        if edge_records:
            table = ensure_table(db, CONCEPT_EDGES_TABLE, _concept_edges_schema())
            table.add(edge_records)
        if chunk_concept_records:
            table = ensure_table(db, CHUNK_CONCEPTS_TABLE, _chunk_concepts_schema())
            table.add(chunk_concept_records)


def boost_results(
    results: list[Any],
    query_concepts: list[str],
) -> list[Any]:
    if not query_concepts or not results:
        return results
    query_set = set(query_concepts)
    boosted: list[Any] = []
    for r in results:
        chunk_concepts = set(get_chunk_concepts(r.source, r.chunk_index))
        overlap = len(query_set & chunk_concepts)
        if overlap > 0:
            boost = (overlap / len(query_set)) * cfg.concept_boost_weight
            r = r.model_copy()
            if r.relevance_score is not None:
                r.relevance_score = r.relevance_score + boost
            elif r.distance is not None:
                r.distance = max(0.0, r.distance - boost)
        boosted.append(r)
    return boosted


def get_chunk_concepts(source: str, chunk_index: int) -> list[str]:
    from lilbee.store import escape_sql_string, open_table

    table = open_table(CHUNK_CONCEPTS_TABLE)
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


def expand_query(query: str) -> list[str]:
    concepts = extract_concepts(query)
    if not concepts:
        return []
    related: list[str] = []
    seen = set(concepts)
    for concept in concepts:
        for neighbor in get_related_concepts(concept):
            if neighbor not in seen:
                related.append(neighbor)
                seen.add(neighbor)
    return related


def get_related_concepts(concept: str, depth: int = 1) -> list[str]:
    from lilbee.store import escape_sql_string, open_table

    table = open_table(CONCEPT_EDGES_TABLE)
    if table is None:
        return []
    visited: set[str] = {concept}
    frontier: list[str] = [concept]
    for _ in range(depth):
        next_frontier: list[str] = []
        for node in frontier:
            escaped = escape_sql_string(node)
            try:
                rows = (
                    table.search()
                    .where(f"source = '{escaped}' OR target = '{escaped}'")
                    .to_list()
                )
            except Exception:
                continue
            for row in rows:
                neighbor = row["target"] if row["source"] == node else row["source"]
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.append(neighbor)
        frontier = next_frontier
    return [c for c in visited if c != concept]


def top_communities(k: int = 10) -> list[Community]:
    from lilbee.store import open_table

    table = open_table(CONCEPT_NODES_TABLE)
    if table is None:
        return []
    rows = table.to_arrow().to_pylist()
    clusters: dict[int, list[str]] = {}
    for row in rows:
        cid = row["cluster_id"]
        clusters.setdefault(cid, []).append(row["concept"])
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    return [
        Community(cluster_id=cid, size=len(concepts), concepts=concepts)
        for cid, concepts in sorted_clusters[:k]
    ]


def rebuild_clusters() -> None:
    from lilbee.lock import write_lock
    from lilbee.store import ensure_table, get_db, open_table, safe_delete

    edges_table = open_table(CONCEPT_EDGES_TABLE)
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

    db = get_db()
    nodes_table = ensure_table(db, CONCEPT_NODES_TABLE, _concept_nodes_schema())
    safe_delete(nodes_table, "concept IS NOT NULL")
    with write_lock():
        if node_records:
            nodes_table.add(node_records)


def get_graph() -> bool:
    if not cfg.concept_graph:
        return False
    from lilbee.store import open_table

    return open_table(CONCEPT_NODES_TABLE) is not None


def reset_graph() -> None:
    """Clear the spaCy model cache. For testing only."""
    global _nlp
    _nlp = None
