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

from lilbee.providers.worker.pool import shutdown_pool_runtime

_RELOAD_CLOSE_TIMEOUT_S = 5.0
"""Wall-clock budget for closing a detached worker channel during reload_role.

Matches ``_DEFAULT_SHUTDOWN_TIMEOUT_S`` in ``providers.worker.pool``: a worker
that does not ack SHUTDOWN within this window is terminated so the new model
load is not blocked.
"""

if TYPE_CHECKING:
    from lilbee.catalog.hf_client import HfClient
    from lilbee.data.store import Store
    from lilbee.modelhub.model_manager import ModelManager
    from lilbee.modelhub.model_manager.discovery import KnownModelCache
    from lilbee.modelhub.registry import ModelRegistry
    from lilbee.providers.base import LLMProvider
    from lilbee.providers.worker.health_ticker import HealthTickerHandle
    from lilbee.providers.worker.pool import PoolRuntime, WorkerPool
    from lilbee.providers.worker.transport import WorkerRole
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

    The worker pool sits on Services (not on the provider) so any
    subsystem can reach it for cancellation, health checks, or
    diagnostics without crossing into ``LlamaCppProvider``'s private
    API. ``cancel_inference()`` is the canonical entry point used by
    Ctrl+C and the chat-stream cancel action.
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
    worker_pool: WorkerPool
    pool_runtime: PoolRuntime
    pool_health_ticker: HealthTickerHandle
    known_models: KnownModelCache

    def cancel_inference(self) -> None:
        """Flip the abort flag on every registered worker pool role. Idempotent."""
        for role_name in self.worker_pool.registered_roles:
            self.worker_pool.accessor(role_name).cancel()

    def reload_role(self, role_name: WorkerRole) -> None:
        """Drop *role_name*'s current worker so the next call lazy-respawns with cfg.

        Detaches the channel synchronously (subsequent calls see no live worker),
        then closes the old channel in the background on the pool runtime so the
        caller's event loop is not stalled. Other roles' workers and any
        in-flight stream they own are untouched. Use when only one role-bound
        model setting has changed (e.g. embedding_model).
        """
        channel = self.worker_pool.detach_channel(role_name)
        if channel is None:
            return

        async def _close() -> None:
            await channel.close(timeout=_RELOAD_CLOSE_TIMEOUT_S)

        self.pool_runtime.submit(_close())

    def add_pool_listener(
        self,
        *,
        on_spawning: Callable[[WorkerRole], None] | None = None,
        on_spawned: Callable[[WorkerRole], None] | None = None,
    ) -> None:
        """Subscribe to worker spawn lifecycle events.

        Forwards directly to :meth:`WorkerPool.add_listener`. The TUI uses this
        to surface "Starting <role> worker..." / "<role> worker ready"
        notifications during the cold-start window.
        """
        self.worker_pool.add_listener(on_spawning=on_spawning, on_spawned=on_spawned)


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

    from lilbee.catalog.hf_client import HfClient
    from lilbee.core.config import cfg
    from lilbee.data.store import Store
    from lilbee.modelhub.model_manager import ModelManager
    from lilbee.modelhub.model_manager.discovery import KnownModelCache
    from lilbee.modelhub.registry import ModelRegistry
    from lilbee.providers.factory import create_provider
    from lilbee.providers.worker.health_ticker import start_health_ticker
    from lilbee.providers.worker.pool import PoolRuntime, WorkerPool
    from lilbee.providers.worker.transport import default_spawner
    from lilbee.retrieval.clustering import Clusterer
    from lilbee.retrieval.concepts import ConceptGraph
    from lilbee.retrieval.embedder import Embedder
    from lilbee.retrieval.query import Searcher
    from lilbee.retrieval.reranker import Reranker
    from lilbee.runtime.asyncio_loop import get_loop
    from lilbee.runtime.ingest_lock import IngestLockRegistry

    worker_pool = WorkerPool(
        spawner=default_spawner(),
        max_idle_s=cfg.worker_pool_max_idle_s,
    )
    pool_runtime = PoolRuntime()
    provider = create_provider(cfg)
    store = Store(cfg)
    embedder = Embedder(cfg, provider)
    reranker = Reranker(cfg)
    concepts = ConceptGraph(cfg, store)
    clusterer = Clusterer(cfg, store)
    registry = ModelRegistry(cfg.models_dir)
    searcher = Searcher(cfg, provider, store, embedder, reranker, concepts)
    hf_client = HfClient()
    ingest_lock_registry = IngestLockRegistry()
    model_manager = ModelManager(cfg.models_dir, cfg.remote_base_url)
    crawler_semaphore = (
        asyncio.Semaphore(cfg.crawl_max_concurrent) if cfg.crawl_max_concurrent > 0 else None
    )
    crawler_sync_state = CrawlerSyncState()
    pool_health_ticker: HealthTickerHandle = start_health_ticker(
        worker_pool, pool_runtime, get_loop()
    )
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
        worker_pool=worker_pool,
        pool_runtime=pool_runtime,
        pool_health_ticker=pool_health_ticker,
        known_models=known_models,
    )
    # Eager start is the default: pay 1-3 s per worker at TUI mount so the
    # first user action lands on a warm pool. Roles whose model is unset are
    # skipped, so a setup with only chat + embed never spawns rerank or
    # vision. Set ``cfg.worker_pool_eager_start = false`` for headless
    # scripts where mount time matters more than first-call latency.
    if cfg.worker_pool_eager_start:
        from contextlib import suppress

        with suppress(Exception):
            provider.warm_up_pool()
            pool_runtime.start()
            pool_runtime.run_sync(worker_pool.start_eager(), timeout=30.0)
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
        shutdown_pool_runtime(_svc.worker_pool, _svc.pool_runtime, _svc.pool_health_ticker)
        _svc.provider.shutdown()
        _svc.store.close()
    _svc = None


def reset_store() -> None:
    """Close and rebuild only the Store and its dependents; keep providers loaded.

    Used after a data-dir wipe (``/reset``) where the LanceDB handle is invalid
    but the loaded llama-cpp/embedder/reranker models are still good. Avoids the
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
