"""SourceClusterer adapter backed by the concept graph."""

from __future__ import annotations

from lilbee.clustering import SourceCluster
from lilbee.concepts.graph import ConceptGraph
from lilbee.concepts.nlp import concepts_available
from lilbee.config import Config
from lilbee.store import Store


class ConceptGraphClusterer:
    """SourceClusterer backed by the concept graph (requires ``[graph]`` extra).

    Wraps :class:`ConceptGraph` so the wiki synthesis layer can consume
    concept-based clusters through the generic ``SourceClusterer`` protocol
    without importing ``ConceptGraph`` directly. Leaves ``ConceptGraph``
    unchanged.
    """

    def __init__(self, config: Config, store: Store) -> None:
        self._graph = ConceptGraph(config, store)

    def available(self) -> bool:
        """Concept-graph clustering needs both dependencies and a built graph."""
        return bool(concepts_available() and self._graph.get_graph())

    def get_clusters(self, min_sources: int = 3) -> list[SourceCluster]:
        """Expose concept clusters as generic :class:`SourceCluster` values."""
        cluster_sources = self._graph.get_cluster_sources(min_sources=min_sources)
        return [
            SourceCluster(
                cluster_id=f"concept-{cid}",
                label=self._graph.get_cluster_label(cid),
                sources=frozenset(sources),
            )
            for cid, sources in cluster_sources.items()
            if len(sources) >= min_sources
        ]
