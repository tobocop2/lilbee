"""Composition root -- lazy singletons for Store, Embedder, Provider, and Searcher.

All CLI entry points, MCP tools, and ingest functions resolve their
dependencies through this module. The ``get_*`` functions create instances
on first call using the global ``cfg`` and cache them for the process lifetime.

Tests call ``reset()`` to discard cached instances between runs.
"""

from __future__ import annotations

from lilbee.config import cfg
from lilbee.embedder import Embedder
from lilbee.providers.base import LLMProvider
from lilbee.providers.factory import create_provider
from lilbee.query import Searcher
from lilbee.store import Store

_provider: LLMProvider | None = None
_store: Store | None = None
_embedder: Embedder | None = None
_searcher: Searcher | None = None


def get_provider() -> LLMProvider:
    """Return the configured LLM provider singleton."""
    global _provider
    if _provider is None:
        _provider = create_provider(cfg)
    return _provider


def get_store() -> Store:
    """Return the Store singleton (backed by cfg)."""
    global _store
    if _store is None:
        _store = Store(cfg)
    return _store


def get_embedder() -> Embedder:
    """Return the Embedder singleton."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder(cfg, get_provider())
    return _embedder


def get_searcher() -> Searcher:
    """Return the Searcher singleton."""
    global _searcher
    if _searcher is None:
        _searcher = Searcher(cfg, get_provider(), get_store(), get_embedder())
    return _searcher


def reset() -> None:
    """Shut down and discard all cached singletons.

    Calls shutdown() on the provider (if any) to release resources.
    """
    global _provider, _store, _embedder, _searcher
    if _provider is not None:
        _provider.shutdown()
    _provider = None
    _store = None
    _embedder = None
    _searcher = None
