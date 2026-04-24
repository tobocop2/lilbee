"""Typed service container — single point of access for all singletons.

All runtime dependencies (provider, store, embedder, reranker, concepts,
clusterer, searcher) are created lazily on first call to ``get_services()``
and cached for the process lifetime. Tests call ``reset_services()``
between runs.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lilbee.clustering import Clusterer
    from lilbee.concepts import ConceptGraph
    from lilbee.embedder import Embedder
    from lilbee.providers.base import LLMProvider
    from lilbee.query import Searcher
    from lilbee.registry import ModelRegistry
    from lilbee.reranker import Reranker
    from lilbee.store import Store


@dataclass(frozen=True)
class Services:
    """Holds all runtime service instances."""

    provider: LLMProvider
    store: Store
    embedder: Embedder
    reranker: Reranker
    concepts: ConceptGraph
    clusterer: Clusterer
    searcher: Searcher
    registry: ModelRegistry


_svc: Services | None = None


def get_services() -> Services:
    """Return the cached Services singleton, creating on first call.

    Service modules are imported inside the function to keep CLI
    startup fast: ``services`` is on every CLI import path, and the
    concrete service modules transitively pull in heavy libraries
    (llama-cpp, lancedb, sentence-transformers). Deferring the loads
    until first ``get_services()`` call makes ``lilbee --help`` and
    TUI splash render in milliseconds instead of seconds.
    """
    global _svc
    if _svc is not None:
        return _svc

    from lilbee.clustering import Clusterer
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
    clusterer = Clusterer(cfg, store)
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
