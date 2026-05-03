"""Typed service container: single point of access for all singletons.

All runtime dependencies (provider, store, embedder, reranker, concepts,
clusterer, searcher, worker pool) are created lazily on first call to
``get_services()`` and cached for the process lifetime. Tests call
``reset_services()`` between runs.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

log = logging.getLogger(__name__)

_DEPRECATED_SUBPROCESS_EMBED_LOGGED = False

if TYPE_CHECKING:
    from lilbee.catalog.hf_client import HfClient
    from lilbee.core.config import Config
    from lilbee.data.store import Store
    from lilbee.modelhub.model_manager import ModelManager
    from lilbee.modelhub.registry import ModelRegistry
    from lilbee.providers.base import LLMProvider
    from lilbee.providers.worker.pool import PoolRuntime, WorkerPool
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
    Ctrl+C and the chat-stream cancel action; it bridges pool-mode
    (subprocess abort flag) and fallback-mode (in-process Event).
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

    def cancel_inference(self) -> None:
        """Interrupt any in-flight inference call.

        Routes the cancel to the right destination based on
        ``cfg.worker_pool_enabled``: pool-mode flips the subprocess
        abort flag (``mp.Value``) for every live role; fallback-mode
        sets the in-process ``threading.Event`` honored by
        ``llama_cpp``'s in-process abort callback. Idempotent.
        """
        from lilbee.core.config import cfg
        from lilbee.providers.llama_cpp.abort_signal import request_abort

        if cfg.worker_pool_enabled:
            for role_name in self.worker_pool.registered_roles:
                self.worker_pool.accessor(role_name).cancel()
        request_abort()


_svc: Services | None = None


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
    from lilbee.modelhub.registry import ModelRegistry
    from lilbee.providers.factory import create_provider
    from lilbee.providers.worker.pool import PoolRuntime, WorkerPool
    from lilbee.retrieval.clustering import Clusterer
    from lilbee.retrieval.concepts import ConceptGraph
    from lilbee.retrieval.embedder import Embedder
    from lilbee.retrieval.query import Searcher
    from lilbee.retrieval.reranker import Reranker
    from lilbee.runtime.ingest_lock import IngestLockRegistry

    _log_subprocess_embed_deprecation_once(cfg)
    worker_pool = WorkerPool(
        spawner=_make_pool_spawner(cfg),
        max_idle_s=cfg.worker_pool_max_idle_s,
        restart_attempts=cfg.worker_pool_restart_attempts,
        restart_window_s=cfg.worker_pool_restart_window_s,
        health_timeout_s=cfg.worker_pool_health_timeout_s,
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
    )
    # Eager start is opt-in: pays the per-worker cold-start (1-3s each)
    # at TUI mount instead of on first request. Most users keep it off
    # so `lilbee --help` and the splash screen stay snappy.
    if cfg.worker_pool_eager_start and cfg.worker_pool_enabled:
        from contextlib import suppress

        with suppress(Exception):
            pool_runtime.start()
            pool_runtime.run_sync(worker_pool.start_eager(), timeout=30.0)
    return _svc


def set_services(services: Services | None) -> None:
    """Replace the cached Services singleton (for testing)."""
    global _svc
    _svc = services


def _log_subprocess_embed_deprecation_once(cfg_obj: Config) -> None:
    """Emit a one-time deprecation warning if cfg.subprocess_embed is True.

    The new ``worker_pool_enabled`` (default True) supersedes the
    per-call ``subprocess_embed`` path with a persistent worker per
    role. Users who explicitly opted into subprocess isolation get the
    new pool transparently; the legacy path remains as a fallback when
    the pool fails. This log nudges them to drop the deprecated setting
    from their config so the next major release can remove it.
    """
    global _DEPRECATED_SUBPROCESS_EMBED_LOGGED
    if _DEPRECATED_SUBPROCESS_EMBED_LOGGED:
        return
    if not cfg_obj.subprocess_embed:
        return
    _DEPRECATED_SUBPROCESS_EMBED_LOGGED = True
    log.warning(
        "cfg.subprocess_embed is deprecated. The persistent worker pool "
        "(cfg.worker_pool_enabled, default True) supersedes it. The legacy "
        "per-call subprocess remains as a fallback path; remove "
        "subprocess_embed from your config when convenient."
    )


def reset_services() -> None:
    """Shut down and discard all cached instances."""
    global _svc, _DEPRECATED_SUBPROCESS_EMBED_LOGGED
    if _svc is not None:
        _shutdown_pool(_svc)
        _svc.provider.shutdown()
        _svc.store.close()
    _svc = None
    _DEPRECATED_SUBPROCESS_EMBED_LOGGED = False


def _shutdown_pool(services: Services) -> None:
    """Drain the worker pool and stop its runtime loop. Idempotent."""
    pool = services.worker_pool
    runtime = services.pool_runtime
    try:
        runtime.run_sync(pool.shutdown(), timeout=10.0)
    except (TimeoutError, RuntimeError, OSError) as exc:
        log.warning("Pool shutdown raised %s; forcing runtime stop", exc)
    runtime.shutdown(timeout=5.0)


def _make_pool_spawner(cfg_obj: Config) -> Any:
    """Pick the worker-pool spawner implementation based on cfg.

    Default ``"pipe"`` returns the stdlib :class:`PipeSpawner`. Future
    backends (``"zmq"``) plug in here without touching the pool or any
    consumer. Hot-swap is not supported; the choice is made at Services
    construction.
    """
    from lilbee.providers.worker.transport_pipe import PipeSpawner

    backend = cfg_obj.worker_pool_backend
    if backend == "pipe":
        return PipeSpawner()
    raise ValueError(
        f"Unknown worker_pool_backend {backend!r}. "
        f"Supported backends: 'pipe' (additional backends land in future PRs)."
    )


atexit.register(reset_services)
