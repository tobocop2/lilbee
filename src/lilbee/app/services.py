"""Typed service container: single point of access for all singletons.

All runtime dependencies (provider, store, embedder, reranker, concepts,
clusterer, searcher, worker pool) are created lazily on first call to
``get_services()`` and cached for the process lifetime. Tests call
``reset_services()`` between runs.
"""

from __future__ import annotations

import asyncio
import atexit
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lilbee.catalog.hf_client import HfClient
    from lilbee.data.store import Store
    from lilbee.modelhub.model_manager import ModelManager
    from lilbee.modelhub.model_manager.discovery import KnownModelCache
    from lilbee.modelhub.registry import ModelRegistry
    from lilbee.providers.base import LLMProvider
    from lilbee.providers.roles import WorkerRole
    from lilbee.retrieval.clustering import Clusterer
    from lilbee.retrieval.concepts import ConceptGraph
    from lilbee.retrieval.embedder import Embedder
    from lilbee.retrieval.query import Searcher
    from lilbee.retrieval.reranker import Reranker
    from lilbee.runtime.ingest_lock import IngestLockRegistry


@dataclass
class CrawlerSyncState:
    """Process-wide sync coordination state (lock + last-run timestamp)."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    last_run: float = 0.0


@dataclass(frozen=True)
class Services:
    """Holds all runtime service instances.

    Inference lifecycle (cancel, per-role reload, spawn notifications) is owned
    by the provider, which manages the llama-server fleet. Services exposes thin
    pass-throughs so callers (Ctrl+C, the chat-stream cancel action, the settings
    and model-bar pickers, the TUI task bar) need not reach into the provider's
    API. ``cancel_inference()`` is the canonical cancel entry point.
    """

    provider: LLMProvider
    store: Store
    embedder: Embedder
    reranker: Reranker
    concepts: ConceptGraph
    clusterer: Clusterer
    searcher: Searcher
    registry: ModelRegistry
    hf_client: HfClient
    ingest_lock_registry: IngestLockRegistry
    model_manager: ModelManager
    crawler_semaphore: asyncio.Semaphore | None
    crawler_sync_state: CrawlerSyncState
    known_models: KnownModelCache

    def cancel_inference(self) -> None:
        """Interrupt any in-flight generation. Idempotent.

        The fleet engine stops a llama-server by client disconnect (the chat
        worker closes the active stream), so this is a no-op there; it stays the
        canonical entry point in case a backend needs an explicit interrupt.
        """
        self.provider.cancel_inference()

    def reload_role(self, role_name: WorkerRole) -> None:
        """Respawn only *role_name*'s model server so it picks up changed cfg.

        Other roles' servers and any in-flight stream they own are untouched. Use
        when one role-bound model setting changed (e.g. embedding_model). The
        respawn runs off the caller's thread, so this returns immediately.
        """
        self.provider.reload_role(role_name)

    def add_pool_listener(
        self,
        *,
        on_spawning: Callable[[WorkerRole], None] | None = None,
        on_spawned: Callable[[WorkerRole], None] | None = None,
    ) -> None:
        """Subscribe to server spawn lifecycle events.

        Forwards to :meth:`LLMProvider.add_spawn_listener`. The TUI uses this to
        surface "Starting <role>..." / "<role> ready" notifications when a role's
        server (re)spawns (cold start after a non-eager boot, or a reload).
        """
        self.provider.add_spawn_listener(on_spawning=on_spawning, on_spawned=on_spawned)


_svc: Services | None = None
"""Cached singleton, set on first ``get_services()`` call.

Concurrency contract: lilbee runs the asyncio loop on a single worker
thread + Textual's main thread. ``get_services()`` is idempotent (the
``if _svc is not None: return`` early-out covers re-entry from a
background thread). Tests that need a custom container call
``set_services(make_mock_services(...))`` explicitly; ``peek_services()``
is the read-only inspector for cleanup fixtures. The Services dataclass
itself is logically immutable post-construction (its fields are
references to long-lived service objects), so concurrent reads are safe
without a lock.
"""


def get_services() -> Services:
    """Return the cached Services singleton, creating on first call.

    Service modules are imported inside the function to keep CLI
    startup fast: ``services`` is on every CLI import path, and the
    concrete service modules transitively pull in heavy libraries
    (llama-cpp, lancedb, kreuzberg). Deferring the loads until first
    ``get_services()`` call makes ``lilbee --help`` and TUI splash
    render in milliseconds instead of seconds.
    """
    global _svc
    if _svc is not None:
        return _svc

    from lilbee.app.settings import reconcile_embedding_dim
    from lilbee.catalog.hf_client import HfClient
    from lilbee.core.config import cfg
    from lilbee.data.store import Store
    from lilbee.modelhub.model_manager import ModelManager
    from lilbee.modelhub.model_manager.discovery import KnownModelCache
    from lilbee.modelhub.registry import ModelRegistry
    from lilbee.providers.factory import create_provider
    from lilbee.retrieval.clustering import Clusterer
    from lilbee.retrieval.concepts import ConceptGraph
    from lilbee.retrieval.embedder import Embedder
    from lilbee.retrieval.query import Searcher
    from lilbee.retrieval.reranker import Reranker
    from lilbee.runtime.ingest_lock import IngestLockRegistry

    provider = create_provider(cfg)
    registry = ModelRegistry(cfg.models_dir)
    # A config-file embedding_model with no embedding_dim would otherwise build the
    # store at the stale 768 default; pin the width to the embedder before Store().
    # Pass the registry so resolution doesn't re-enter this half-built get_services.
    reconcile_embedding_dim(registry)
    store = Store(cfg)
    embedder = Embedder(cfg, provider)
    reranker = Reranker(cfg)
    concepts = ConceptGraph(cfg, store)
    clusterer = Clusterer(cfg, store)
    searcher = Searcher(cfg, provider, store, embedder, reranker, concepts)
    hf_client = HfClient()
    ingest_lock_registry = IngestLockRegistry()
    model_manager = ModelManager(cfg.models_dir)
    crawler_semaphore = (
        asyncio.Semaphore(cfg.crawl_max_concurrent) if cfg.crawl_max_concurrent > 0 else None
    )
    crawler_sync_state = CrawlerSyncState()
    known_models = KnownModelCache()
    _svc = Services(
        provider=provider,
        store=store,
        embedder=embedder,
        reranker=reranker,
        concepts=concepts,
        clusterer=clusterer,
        searcher=searcher,
        registry=registry,
        hf_client=hf_client,
        ingest_lock_registry=ingest_lock_registry,
        model_manager=model_manager,
        crawler_semaphore=crawler_semaphore,
        crawler_sync_state=crawler_sync_state,
        known_models=known_models,
    )
    # Eager start is the default: pay the spawn cost per role server at TUI mount
    # so the first user action lands on a warm fleet. Roles whose model is unset
    # are skipped, so a setup with only chat + embed never spawns rerank or
    # vision. Set ``cfg.worker_pool_eager_start = false`` for headless scripts
    # where mount time matters more than first-call latency.
    if cfg.worker_pool_eager_start:
        from contextlib import suppress

        with suppress(Exception):
            provider.warm_up_pool()
    return _svc


def set_services(services: Services | None) -> None:
    """Replace the cached Services singleton (for testing)."""
    global _svc
    _svc = services


def peek_services() -> Services | None:
    """Return the cached Services container, or None if not yet initialized.

    Public read-only accessor for test cleanup helpers that need to
    inspect the singleton without forcing initialization.
    """
    return _svc


def reset_services() -> None:
    """Shut down and discard all cached instances."""
    global _svc
    if _svc is not None:
        _svc.provider.shutdown()
        _svc.store.close()
    _svc = None


def reset_store() -> None:
    """Close and rebuild only the Store and its dependents; keep providers loaded.

    Used after a data-dir wipe (``/reset``) where the LanceDB handle is invalid
    but the running provider/embedder/reranker are still good. Avoids the
    multi-second reload cost of ``reset_services()``.
    """
    global _svc
    if _svc is None:
        return
    from dataclasses import replace

    from lilbee.core.config import cfg
    from lilbee.data.store import Store
    from lilbee.retrieval.clustering import Clusterer
    from lilbee.retrieval.concepts import ConceptGraph
    from lilbee.retrieval.query import Searcher

    _svc.store.close()
    store = Store(cfg)
    concepts = ConceptGraph(cfg, store)
    clusterer = Clusterer(cfg, store)
    searcher = Searcher(cfg, _svc.provider, store, _svc.embedder, _svc.reranker, concepts)
    _svc = replace(
        _svc,
        store=store,
        concepts=concepts,
        clusterer=clusterer,
        searcher=searcher,
    )


atexit.register(reset_services)
