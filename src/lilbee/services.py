"""Typed service container — single point of access for all singletons.

All runtime dependencies (provider, store, embedder, reranker, concepts,
clusterer, searcher) are created lazily on first call to ``get_services()``
and cached for the process lifetime. Tests call ``reset_services()``
between runs.
"""

from __future__ import annotations

import atexit
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lilbee.clustering import SourceClusterer
    from lilbee.concepts import ConceptGraph
    from lilbee.config import Config
    from lilbee.embedder import Embedder
    from lilbee.providers.base import LLMProvider
    from lilbee.query import Searcher
    from lilbee.registry import ModelRegistry
    from lilbee.reranker import Reranker
    from lilbee.store import Store

log = logging.getLogger(__name__)

_CLUSTERER_EMBEDDING = "embedding"
_CLUSTERER_CONCEPTS = "concepts"


@dataclass(frozen=True)
class Services:
    """Holds all runtime service instances."""

    provider: LLMProvider
    store: Store
    embedder: Embedder
    reranker: Reranker
    concepts: ConceptGraph
    clusterer: SourceClusterer
    searcher: Searcher
    registry: ModelRegistry


_svc: Services | None = None


def _build_clusterer(cfg: Config, store: Store) -> SourceClusterer:
    """Select a :class:`SourceClusterer` backend with safe fallback.

    Respects ``cfg.wiki_clusterer``. If the user asked for the concept
    graph but the optional ``[graph]`` extras are missing or the graph
    has not been built, logs a warning and falls back to the embedding
    clusterer so wiki synthesis keeps working.
    """
    from lilbee.clustering_embedding import EmbeddingClusterer

    choice = cfg.wiki_clusterer
    if choice == _CLUSTERER_CONCEPTS:
        from lilbee.concepts import ConceptGraphClusterer

        graph_clusterer = ConceptGraphClusterer(cfg, store)
        if graph_clusterer.available():
            return graph_clusterer
        log.warning(
            "wiki_clusterer=concepts but the [graph] extra is not installed or "
            "the concept graph has not been built. Falling back to the "
            "embedding clusterer."
        )
    return EmbeddingClusterer(cfg, store)


def get_services() -> Services:
    """Return the cached Services singleton, creating on first call."""
    global _svc
    if _svc is not None:
        return _svc

    from lilbee.concepts import ConceptGraph
    from lilbee.config import cfg
    from lilbee.embedder import Embedder
    from lilbee.providers.factory import create_provider
    from lilbee.query import Searcher
    from lilbee.registry import ModelRegistry
    from lilbee.reranker import Reranker
    from lilbee.store import Store

    provider = create_provider(cfg)
    store = Store(cfg)
    embedder = Embedder(cfg, provider)
    reranker = Reranker(cfg)
    concepts = ConceptGraph(cfg, store)
    clusterer = _build_clusterer(cfg, store)
    registry = ModelRegistry(cfg.models_dir)
    searcher = Searcher(cfg, provider, store, embedder, reranker, concepts)
    _svc = Services(
        provider=provider,
        store=store,
        embedder=embedder,
        reranker=reranker,
        concepts=concepts,
        clusterer=clusterer,
        searcher=searcher,
        registry=registry,
    )
    return _svc


def set_services(services: Services | None) -> None:
    """Replace the cached Services singleton (for testing)."""
    global _svc
    _svc = services


def reset_services() -> None:
    """Shut down and discard all cached instances."""
    global _svc
    if _svc is not None:
        _svc.provider.shutdown()
        _svc.store.close()
    _svc = None


atexit.register(reset_services)
