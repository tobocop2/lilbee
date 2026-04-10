"""Embedding-space source clusterer — default wiki synthesis backend.

Clusters source documents by computing one mean embedding per source,
then grouping sources whose mean embeddings are within a cosine similarity
threshold. Uses single-linkage agglomerative merging (connected components
under the threshold). Pure Python, no optional dependencies.

This is the default ``SourceClusterer`` implementation so the wiki
synthesis feature works without the optional ``[graph]`` extra. The
component search is O(n^2) in the number of sources, which is fine for
local knowledge bases up to a few thousand documents.
"""

from __future__ import annotations

import logging
import math
from collections import Counter

from lilbee.clustering import SourceCluster
from lilbee.config import CHUNKS_TABLE, Config
from lilbee.store import Store

log = logging.getLogger(__name__)

# Words shorter than this are excluded from heuristic labels — they tend to
# be articles, prepositions, and other low-signal tokens. 5 keeps useful
# technical terms (python, rust, kafka) while dropping the/and/with/from.
_MIN_LABEL_WORD_LEN = 5

# Number of words combined into the heuristic label.
_MAX_LABEL_CANDIDATES = 3


def _normalized_mean(vectors: list[list[float]]) -> list[float] | None:
    """Return the L2-normalized mean of a list of vectors, or None if degenerate."""
    if not vectors:
        return None
    dim = len(vectors[0])
    sums = [0.0] * dim
    for vec in vectors:
        for i, value in enumerate(vec):
            sums[i] += value
    count = float(len(vectors))
    mean = [v / count for v in sums]
    norm = math.sqrt(sum(v * v for v in mean))
    if norm == 0.0:
        return None
    return [v / norm for v in mean]


def _aggregate_by_source(
    rows: list[dict[str, object]],
) -> tuple[dict[str, list[list[float]]], dict[str, list[str]]]:
    """Single-pass collection of vectors and chunk text grouped by source."""
    vectors: dict[str, list[list[float]]] = {}
    texts: dict[str, list[str]] = {}
    for row in rows:
        source = row.get("source")
        if not isinstance(source, str):
            continue
        vector = row.get("vector")
        if isinstance(vector, (list, tuple)):
            vectors.setdefault(source, []).append([float(v) for v in vector])
        chunk_text = row.get("chunk")
        if isinstance(chunk_text, str):
            texts.setdefault(source, []).append(chunk_text)
    return vectors, texts


def _mean_embeddings(
    vectors_by_source: dict[str, list[list[float]]],
) -> tuple[list[str], list[list[float]]]:
    """Compute one L2-normalized mean embedding per source."""
    sources: list[str] = []
    means: list[list[float]] = []
    for source, vectors in vectors_by_source.items():
        normalized = _normalized_mean(vectors)
        if normalized is None:
            continue
        sources.append(source)
        means.append(normalized)
    return sources, means


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two already-normalized vectors (= dot product)."""
    return sum(x * y for x, y in zip(a, b, strict=True))


def _connected_components(
    embeddings: list[list[float]],
    threshold: float,
) -> list[list[int]]:
    """Return connected components of a cosine-similarity graph above ``threshold``.

    Iterative depth-first traversal. Two nodes share an edge when their
    embeddings are within ``threshold`` cosine similarity of each other.
    """
    n = len(embeddings)
    visited = [False] * n
    components: list[list[int]] = []
    for start in range(n):
        if visited[start]:
            continue
        component: list[int] = []
        stack = [start]
        visited[start] = True
        while stack:
            node = stack.pop()
            component.append(node)
            for other in range(n):
                if visited[other]:
                    continue
                if _cosine_similarity(embeddings[node], embeddings[other]) >= threshold:
                    visited[other] = True
                    stack.append(other)
        components.append(component)
    return components


def _heuristic_label(
    sources: list[str],
    texts_by_source: dict[str, list[str]],
    fallback: str,
) -> str:
    """Pick a short topic label from the most common content words in the cluster."""
    counts: Counter[str] = Counter()
    for source in sources:
        for chunk_text in texts_by_source.get(source, []):
            for raw in chunk_text.lower().split():
                word = "".join(ch for ch in raw if ch.isalnum())
                if len(word) < _MIN_LABEL_WORD_LEN:
                    continue
                counts[word] += 1
    if not counts:
        return fallback
    return " ".join(word for word, _ in counts.most_common(_MAX_LABEL_CANDIDATES))


class EmbeddingClusterer:
    """Cluster sources by mean-embedding cosine similarity.

    Implements :class:`~lilbee.clustering.SourceClusterer` without any
    optional dependencies. Clustering is single-linkage agglomerative:
    two sources join the same cluster when their mean embeddings are
    within ``config.wiki_clusterer_threshold`` cosine similarity.
    """

    def __init__(self, config: Config, store: Store) -> None:
        self._config = config
        self._store = store

    def available(self) -> bool:
        """Available whenever the chunks table exists and is non-empty."""
        table = self._store.open_table(CHUNKS_TABLE)
        if table is None:
            return False
        try:
            return bool(table.count_rows())
        except Exception:
            log.debug("count_rows() failed on chunks table", exc_info=True)
            return True

    def get_clusters(self, min_sources: int = 3) -> list[SourceCluster]:
        """Compute clusters of sources above the similarity threshold."""
        table = self._store.open_table(CHUNKS_TABLE)
        if table is None:
            return []

        rows = table.to_arrow().to_pylist()
        vectors_by_source, texts_by_source = _aggregate_by_source(rows)
        sources, embeddings = _mean_embeddings(vectors_by_source)
        if len(sources) < min_sources:
            return []

        components = _connected_components(embeddings, self._config.wiki_clusterer_threshold)

        clusters: list[SourceCluster] = []
        for idx, component in enumerate(components):
            member_sources = sorted(sources[i] for i in component)
            if len(member_sources) < min_sources:
                continue
            cluster_id = f"embedding-{idx}"
            label = _heuristic_label(member_sources, texts_by_source, fallback=cluster_id)
            clusters.append(
                SourceCluster(
                    cluster_id=cluster_id,
                    label=label,
                    sources=frozenset(member_sources),
                )
            )
        return clusters
