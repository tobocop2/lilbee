"""Chunk-level mutual-kNN clusterer with TF-IDF labels.

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
"""

from __future__ import annotations

import logging

from lilbee.core.config import CHUNKS_TABLE, Config
from lilbee.data.store import Store
from lilbee.retrieval.clustering import SourceCluster
from lilbee.retrieval.clustering_embedding.helpers import (
    _build_clusters,
    _corpus_document_frequency,
    _load_chunk_records,
    _source_totals,
    auto_k,
    communities_by_label,
    label_propagation,
    mutual_knn,
    normalize_rows,
)

log = logging.getLogger(__name__)


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
        """Clusterer is available when the chunks table has any rows.

        ``count_rows()`` is a LanceDB call that can raise on transient
        backend issues (concurrent compaction, schema rewrites). When
        it does, we optimistically report available=True and let
        ``get_clusters`` surface the real error on the next scan: the
        alternative would silently disable wiki synthesis without the
        user seeing why. A WARNING is emitted so the failure is still
        visible at the default log level.
        """
        table = self._store.open_table(CHUNKS_TABLE)
        if table is None:
            return False
        try:
            return bool(table.count_rows())
        except Exception:
            log.warning(
                "count_rows() failed on chunks table; reporting available=True "
                "optimistically and deferring the error to get_clusters",
                exc_info=True,
            )
            return True

    def get_clusters(self, min_sources: int = 3) -> list[SourceCluster]:
        """Return chunk-level communities projected to source clusters."""
        records, matrix = _load_chunk_records(self._store)
        if not records:
            return []

        matrix, keep_mask = normalize_rows(matrix)
        records = [record for record, keep in zip(records, keep_mask, strict=True) if keep]
        if not records:
            return []

        configured_k = self._config.wiki_clusterer_k
        k = configured_k if configured_k > 0 else auto_k(len(records))
        adjacency = mutual_knn(matrix, k)
        if not any(adjacency.values()):
            # WARNING (not INFO) so users see why synthesis produced zero
            # pages at the default log level: matches the other degenerate
            # clustering outcome, ``_warn_if_undersegmented``.
            log.warning(
                "wiki clustering: N=%d k=%d no mutual edges: skipping synthesis",
                len(records),
                k,
            )
            return []
        labels = label_propagation(adjacency, order=list(range(len(records))))
        communities = communities_by_label(labels)

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
