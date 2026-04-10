"""Chunk-level mutual-kNN source clusterer — default wiki synthesis backend.

Pipeline:

1. Load every chunk's (source, chunk_index, text, vector) from LanceDB.
2. Build a float32 matrix ``V`` and L2-normalize rows.
3. Compute a mutual k-nearest-neighbors graph over chunks using blocked
   similarity. ``k`` auto-scales from corpus size unless
   ``config.wiki_clusterer_k`` is set.
4. Run asynchronous Label Propagation (Raghavan et al. 2007) over the
   mutual-kNN graph to obtain chunk-level communities.
5. Aggregate chunk communities into source communities, requiring a
   source to contribute at least ``min(3, ceil(0.2 * total_chunks))``
   chunks before it joins a cluster. This keeps a single stray chunk
   from dragging a whole document into an unrelated cluster.
6. Filter communities that span fewer than ``min_sources`` distinct
   sources, then label each cluster with TF-IDF scoring over member
   chunk text against corpus-wide document frequency.

Mutual-kNN is itself hub-robust: a pathological hub ends up in many
one-way neighborhoods but can reciprocate at most ``k`` of them, so
hub-driven bridging across topics is broken at the graph-construction
step without any post-hoc similarity rescaling.

Uses numpy for the O(N²) similarity kernel. numpy is listed in
``pyproject.toml`` as a direct dependency; it was already required
transitively by lancedb but the clusterer imports it at module top
level, so the dependency is made explicit.

The implementation is blocked in row chunks of ``_BLOCK_SIZE`` to keep
peak memory bounded regardless of corpus size, so it scales comfortably
to tens of thousands of chunks on a laptop.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from lilbee.clustering import SourceCluster
from lilbee.config import CHUNKS_TABLE, Config
from lilbee.store import Store
from lilbee.text import tokenize

log = logging.getLogger(__name__)

# Block size for the similarity kernel. With N=10000 and D=768 this caps
# peak float32 memory at block * N * 4 bytes ~= 40 MB.
_BLOCK_SIZE = 1024

# Label Propagation hard iteration cap. Convergence is typically reached
# in well under 10 passes on real corpora.
_MAX_LPA_ITERATIONS = 30

# Minimum non-zero L2 norm for a row vector to be kept.
_MIN_VECTOR_NORM = 1e-12

# Source-membership thresholds. A source joins a chunk community when it
# contributes at least `min(_MIN_SOURCE_CHUNKS, ceil(total * _MIN_SOURCE_FRACTION))`
# of its chunks. The stricter (smaller) side wins, so a single stray chunk
# from a long document never pulls the whole source into an unrelated cluster.
_MIN_SOURCE_CHUNKS = 3
_MIN_SOURCE_FRACTION = 0.2

# TF-IDF labeling knobs.
_LABEL_TOP_TERMS = 3
_LABEL_SHORT_CHUNK_CAP = 20  # chunks shorter than this are down-weighted

# kNN auto-scaling bounds. Formula: clamp(round(log2(N)+2), _MIN_K, _MAX_K).
_MIN_K = 5
_MAX_K = 20


@dataclass
class _ChunkRecord:
    """Lightweight view of one chunk row used by the clusterer."""

    source: str
    chunk_index: int
    text: str
    tokens: list[str] = field(default_factory=list)


def _auto_k(n: int) -> int:
    """Pick a neighborhood size from corpus size via ``clamp(log2(N)+2)``."""
    if n <= 1:
        return _MIN_K
    raw = round(math.log2(max(n, 4)) + 2)
    return max(_MIN_K, min(_MAX_K, raw))


def _load_chunk_records(
    store: Store,
) -> tuple[list[_ChunkRecord], np.ndarray]:
    """Scan the chunks table once and return records plus a float32 matrix.

    Rows with an unparseable vector are skipped. Rows are sorted by
    ``(source, chunk_index)`` so downstream cluster IDs are stable
    regardless of LanceDB's row return order. Records are tokenized
    once here so TF-IDF labeling does not re-tokenize.
    """
    table = store.open_table(CHUNKS_TABLE)
    if table is None:
        return [], np.zeros((0, 0), dtype=np.float32)

    arrow = table.to_arrow().to_pylist()
    pairs: list[tuple[_ChunkRecord, list[float]]] = []
    for row in arrow:
        vector = row.get("vector")
        if not isinstance(vector, (list, tuple)):
            continue
        source = row.get("source")
        if not isinstance(source, str):
            continue
        chunk_text = row.get("chunk") if isinstance(row.get("chunk"), str) else ""
        chunk_index = int(row.get("chunk_index") or 0)
        record = _ChunkRecord(
            source=source,
            chunk_index=chunk_index,
            text=chunk_text,
            tokens=tokenize(chunk_text),
        )
        pairs.append((record, [float(v) for v in vector]))

    if not pairs:
        return [], np.zeros((0, 0), dtype=np.float32)

    pairs.sort(key=lambda pair: (pair[0].source, pair[0].chunk_index))
    records = [record for record, _ in pairs]
    vectors = [vec for _, vec in pairs]
    return records, np.asarray(vectors, dtype=np.float32)


def _normalize_rows(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (normalized_matrix, keep_mask). Zero-norm rows are dropped."""
    if matrix.size == 0:
        return matrix, np.zeros(0, dtype=bool)
    norms = np.linalg.norm(matrix, axis=1)
    keep = norms > _MIN_VECTOR_NORM
    if not keep.all():
        matrix = matrix[keep]
        norms = norms[keep]
    return matrix / norms[:, None], keep


def _mutual_knn(matrix: np.ndarray, k: int) -> dict[int, set[int]]:
    """Build a mutual k-nearest-neighbors graph over L2-normalized rows.

    Computes similarity in row blocks so peak memory stays bounded.
    Self-similarity is masked so each row's neighbors exclude itself.
    Mutuality is enforced by keeping only edges ``(i, j)`` where
    ``j`` is in row ``i``'s top-k AND ``i`` is in row ``j``'s top-k;
    this single rule breaks hub-driven bridging without any extra
    similarity rescaling.
    """
    n = matrix.shape[0]
    if n == 0 or k <= 0:
        return {}

    effective_k = min(k, n - 1)
    if effective_k <= 0:
        return {i: set() for i in range(n)}

    top_neighbors: list[set[int]] = [set() for _ in range(n)]

    for start in range(0, n, _BLOCK_SIZE):
        stop = min(start + _BLOCK_SIZE, n)
        sim_block = matrix[start:stop] @ matrix.T  # (block, n)
        # Mask self-similarity so each row's own index cannot be returned.
        block_rows = np.arange(stop - start)
        sim_block[block_rows, np.arange(start, stop)] = -math.inf
        neighbor_idx = np.argpartition(-sim_block, effective_k - 1, axis=1)[:, :effective_k]
        for local_row, global_row in enumerate(range(start, stop)):
            top_neighbors[global_row] = set(neighbor_idx[local_row].tolist())

    mutual: dict[int, set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        for j in top_neighbors[i]:
            if i in top_neighbors[j]:
                mutual[i].add(j)
    return mutual


def _label_propagation(
    adjacency: dict[int, set[int]],
    order: list[int],
) -> list[int]:
    """Async Label Propagation with deterministic min-label tie-breaking.

    Each node adopts the most common label among its neighbors. Ties are
    broken by smallest label id so the outcome is reproducible across
    runs on the same corpus. The caller is responsible for passing a
    complete ``adjacency`` (one entry per node 0..n-1) and an ``order``
    that covers every node — ``get_clusters`` guarantees both.
    """
    n = len(adjacency)
    labels = list(range(n))
    for _ in range(_MAX_LPA_ITERATIONS):
        changed = False
        for node in order:
            neighbors = adjacency.get(node)
            if not neighbors:
                continue
            counts: Counter[int] = Counter(labels[j] for j in neighbors)
            top_count = max(counts.values())
            best = min(label for label, count in counts.items() if count == top_count)
            if labels[node] != best:
                labels[node] = best
                changed = True
        if not changed:
            break
    return labels


def _communities_by_label(labels: list[int]) -> dict[int, list[int]]:
    """Group node indices by their final community label."""
    communities: dict[int, list[int]] = {}
    for node, label in enumerate(labels):
        communities.setdefault(label, []).append(node)
    return communities


def _source_totals(records: list[_ChunkRecord]) -> dict[str, int]:
    """Return the total chunk count per source across the whole corpus."""
    totals: dict[str, int] = {}
    for record in records:
        totals[record.source] = totals.get(record.source, 0) + 1
    return totals


def _filter_sources(
    member_indices: list[int],
    records: list[_ChunkRecord],
    source_totals: dict[str, int],
) -> frozenset[str]:
    """Apply the source-membership threshold to a community's members."""
    per_source: dict[str, int] = {}
    for idx in member_indices:
        source = records[idx].source
        per_source[source] = per_source.get(source, 0) + 1
    kept: set[str] = set()
    for source, count in per_source.items():
        total = source_totals.get(source, count)
        fractional_cutoff = math.ceil(total * _MIN_SOURCE_FRACTION)
        cutoff = min(_MIN_SOURCE_CHUNKS, fractional_cutoff)
        if count >= cutoff:
            kept.add(source)
    return frozenset(kept)


def _corpus_document_frequency(records: list[_ChunkRecord]) -> dict[str, int]:
    """Compute document frequency (chunk count containing term) for every term."""
    df: dict[str, int] = {}
    for record in records:
        for term in set(record.tokens):
            df[term] = df.get(term, 0) + 1
    return df


def _label_community(
    member_indices: list[int],
    records: list[_ChunkRecord],
    df: dict[str, int],
    total_chunks: int,
    fallback: str,
) -> str:
    """Pick a topic label for a community using sublinear TF-IDF scoring."""
    tf: dict[str, float] = {}
    for idx in member_indices:
        tokens = records[idx].tokens
        if not tokens:
            continue
        weight = min(1.0, len(tokens) / _LABEL_SHORT_CHUNK_CAP)
        counts: Counter[str] = Counter(tokens)
        for term, count in counts.items():
            tf[term] = tf.get(term, 0.0) + weight * (1.0 + math.log(count))

    scored: list[tuple[float, str]] = []
    for term, term_tf in tf.items():
        idf = math.log(total_chunks / (1 + df.get(term, 0)))
        if idf <= 0:
            continue
        scored.append((term_tf * idf, term))
    if not scored:
        return fallback

    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return " ".join(term for _, term in scored[:_LABEL_TOP_TERMS])


def _build_clusters(
    communities: dict[int, list[int]],
    records: list[_ChunkRecord],
    source_totals: dict[str, int],
    df: dict[str, int],
    min_sources: int,
) -> tuple[list[SourceCluster], int]:
    """Turn raw chunk communities into published source clusters.

    Returns ``(clusters, noise_chunk_count)`` where ``noise_chunk_count``
    is the number of chunks whose community failed the source filter.
    """
    ordered = sorted(communities.items(), key=lambda pair: (-len(pair[1]), pair[0]))
    total_chunks = len(records)
    clusters: list[SourceCluster] = []
    noise = 0
    for idx, (_, members) in enumerate(ordered):
        kept_sources = _filter_sources(members, records, source_totals)
        if len(kept_sources) < min_sources:
            noise += len(members)
            continue
        cluster_id = f"embedding-{idx}"
        label = _label_community(members, records, df, total_chunks, fallback=cluster_id)
        clusters.append(
            SourceCluster(
                cluster_id=cluster_id,
                label=label,
                sources=kept_sources,
            )
        )
    return clusters, noise


def _warn_if_undersegmented(
    clusters: list[SourceCluster],
    source_totals: dict[str, int],
) -> None:
    """Warn when a single cluster covers more than half the corpus sources."""
    if not clusters or not source_totals:
        return
    total_sources = len(source_totals)
    for cluster in clusters:
        if len(cluster.sources) * 2 > total_sources:
            log.warning(
                "wiki clustering: cluster %r covers %d/%d sources; "
                "consider lowering wiki_clusterer_k or check embedding quality",
                cluster.label,
                len(cluster.sources),
                total_sources,
            )
            break


class EmbeddingClusterer:
    """Chunk-level mutual-kNN clusterer with TF-IDF labels."""

    def __init__(self, config: Config, store: Store) -> None:
        self._config = config
        self._store = store

    def available(self) -> bool:
        """Clusterer is available when the chunks table has any rows."""
        table = self._store.open_table(CHUNKS_TABLE)
        if table is None:
            return False
        try:
            return bool(table.count_rows())
        except Exception:
            log.debug("count_rows() failed on chunks table", exc_info=True)
            return True

    def get_clusters(self, min_sources: int = 3) -> list[SourceCluster]:
        """Return chunk-level communities projected to source clusters."""
        records, matrix = _load_chunk_records(self._store)
        if not records:
            return []

        matrix, keep_mask = _normalize_rows(matrix)
        records = [record for record, keep in zip(records, keep_mask, strict=True) if keep]
        if not records:
            return []

        configured_k = self._config.wiki_clusterer_k
        k = configured_k if configured_k > 0 else _auto_k(len(records))
        adjacency = _mutual_knn(matrix, k)
        labels = _label_propagation(adjacency, order=list(range(len(records))))
        communities = _communities_by_label(labels)

        totals = _source_totals(records)
        df = _corpus_document_frequency(records)
        clusters, noise = _build_clusters(communities, records, totals, df, min_sources)

        log.info(
            "wiki clustering: N=%d k=%d communities=%d kept=%d noise=%d",
            len(records),
            k,
            len(communities),
            len(clusters),
            noise,
        )
        _warn_if_undersegmented(clusters, totals)
        return clusters
