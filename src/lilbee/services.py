"""Typed service container — single point of access for all singletons.

All runtime dependencies (provider, store, embedder, reranker, concepts,
searcher) are created lazily on first call to ``get_services()`` and cached for the
process lifetime.  Tests call ``reset_services()`` between runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lilbee.concepts import ConceptGraph
    from lilbee.embedder import Embedder
    from lilbee.providers.base import LLMProvider
    from lilbee.query import Searcher
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
    searcher: Searcher


_svc: Services | None = None


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
    from lilbee.reranker import Reranker
    from lilbee.store import Store

    provider = create_provider(cfg)
    store = Store(cfg)
    embedder = Embedder(cfg, provider)
    reranker = Reranker(cfg)
    concepts = ConceptGraph(cfg, store)
    searcher = Searcher(cfg, provider, store, embedder, reranker, concepts)
    _svc = Services(
        provider=provider,
        store=store,
        embedder=embedder,
        reranker=reranker,
        concepts=concepts,
        searcher=searcher,
    )
    return _svc


def reset_services() -> None:
    """Shut down and discard all cached instances."""
    global _svc
    if _svc is not None:
        _svc.provider.shutdown()
    _svc = None
