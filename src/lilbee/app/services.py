"""Typed service container: single point of access for all singletons.

All runtime dependencies (provider, store, embedder, reranker, concepts,
clusterer, searcher, worker pool) are created lazily on first call to
``get_services()`` and cached for the process lifetime. Tests call
``reset_services()`` between runs.

``build_services(config)`` is the construction seam: it builds a full container
against an arbitrary Config without touching the process-global singleton. The
library API (:class:`lilbee.Lilbee`) builds one per instance and installs it for
the duration of each call via :func:`services_scope`, so ingest code reaching for
``get_services()`` resolves the caller's container. The override is a ContextVar,
so ``reset_services`` / ``set_services`` / ``peek_services`` (which operate only
on the global singleton) never see it.
"""

from __future__ import annotations

import asyncio
import atexit
import signal
import sys
import threading
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from lilbee.catalog.hf_client import HfClient
    from lilbee.core.config import Config
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


_SIGNAL_EXIT_BASE = 128


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

    def reload_role(self, role_name: WorkerRole, *, wait: bool = False) -> None:
        """Respawn only *role_name*'s model server so it picks up changed cfg.

        Other roles' servers and any in-flight stream they own are untouched. Use
        when one role-bound model setting changed (e.g. embedding_model). The
        respawn runs off the caller's thread, so this returns immediately, unless
        ``wait=True`` (the caller is already off the event loop and wants to block
        until the new model has loaded).
        """
        self.provider.reload_role(role_name, wait=wait)

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


class _ServicesState:
    """The cached process-global singleton plus the per-task scoped override.

    ``singleton`` is set on first ``get_services()`` call. Concurrency
    contract: lilbee runs the asyncio loop on a single worker thread +
    Textual's main thread; ``get_services()`` is idempotent (the early-out
    covers re-entry from a background thread), and the Services dataclass is
    logically immutable post-construction, so concurrent reads are safe
    without a lock. Tests that need a custom container call
    ``set_services(make_mock_services(...))``; ``peek_services()`` is the
    read-only inspector for cleanup fixtures.

    ``override`` shadows the singleton for the entering task: set by
    :func:`services_scope` (the library API's per-call binding), read by
    :func:`get_services`, and invisible to ``reset_services`` /
    ``set_services`` / ``peek_services``, which only touch the singleton.
    """

    def __init__(self) -> None:
        self.singleton: Services | None = None
        self.override: ContextVar[Services | None] = ContextVar(
            "lilbee_services_override", default=None
        )


_state = _ServicesState()


def build_services(
    config: Config,
    *,
    provider: LLMProvider | None = None,
    registry: ModelRegistry | None = None,
) -> Services:
    """Build a full Services container bound to *config*, without caching it.

    ``get_services()`` calls this with the process-global cfg; the library API
    calls it per instance. Service modules are imported inside the function to
    keep CLI startup fast (they transitively pull in lancedb / kreuzberg). Pass
    *provider* to reuse a caller-supplied one; otherwise it is built from
    *config* via the provider factory. Pass *registry* to reuse one already built
    (get_services builds it for embedding-dim reconciliation). Embedding-dim
    reconciliation is a global-cfg concern owned by :func:`get_services`, not
    done here.
    """
    from lilbee.catalog.hf_client import HfClient
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

    provider = provider or create_provider(config)
    registry = registry or ModelRegistry(config.models_dir)
    store = Store(config)
    embedder = Embedder(config, provider)
    reranker = Reranker(config)
    concepts = ConceptGraph(config, store)
    clusterer = Clusterer(config, store)
    searcher = Searcher(config, provider, store, embedder, reranker, concepts)
    hf_client = HfClient()
    ingest_lock_registry = IngestLockRegistry()
    model_manager = ModelManager(config.models_dir)
    crawler_semaphore = (
        asyncio.Semaphore(config.crawl_max_concurrent) if config.crawl_max_concurrent > 0 else None
    )
    crawler_sync_state = CrawlerSyncState()
    known_models = KnownModelCache()
    return Services(
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


def get_services() -> Services:
    """Return the active container: a scoped override if set, else the cached singleton.

    Creates the singleton on first call (against the process-global cfg). A
    config-file embedding_model with no embedding_dim would otherwise build the
    store at the stale 768 default, so the width is pinned to the embedder before
    the store is built.
    """
    override = _state.override.get()
    if override is not None:
        return override
    if _state.singleton is not None:
        return _state.singleton

    from lilbee.app.settings import reconcile_embedding_dim
    from lilbee.core.config import cfg
    from lilbee.modelhub.registry import ModelRegistry

    registry = ModelRegistry(cfg.models_dir)
    # Pin the store width to the embedder before Store(); pass the registry so
    # resolution doesn't re-enter this half-built get_services.
    reconcile_embedding_dim(registry)
    _state.singleton = build_services(cfg, registry=registry)
    # Eager start is the default: pay the spawn cost per role server at TUI mount
    # so the first user action lands on a warm fleet. Roles whose model is unset
    # are skipped, so a setup with only chat + embed never spawns rerank or
    # vision. Set ``cfg.worker_pool_eager_start = false`` for headless scripts
    # where mount time matters more than first-call latency.
    if cfg.worker_pool_eager_start:
        from contextlib import suppress

        with suppress(Exception):
            _state.singleton.provider.warm_up_pool()
    return _state.singleton


@contextmanager
def services_scope(services: Services) -> Iterator[None]:
    """Bind *services* as the container ``get_services()`` returns for this block.

    Isolated to the entering task via a ContextVar (it propagates into
    ``to_ingest_thread`` workers), and never affects the global singleton, so
    ``reset_services`` is unnecessary and unused around a scoped call.
    """
    token = _state.override.set(services)
    try:
        yield
    finally:
        _state.override.reset(token)


def set_services(services: Services | None) -> None:
    """Replace the cached Services singleton (for testing)."""
    _state.singleton = services


def peek_services() -> Services | None:
    """Return the cached Services container, or None if not yet initialized.

    Public read-only accessor for test cleanup helpers that need to
    inspect the singleton without forcing initialization.
    """
    return _state.singleton


def reset_services() -> None:
    """Shut down and discard all cached instances.

    Swap the module reference to ``None`` *before* tearing the old instances
    down, so a new caller never observes a half-closed container. On the shared
    HTTP daemon every entry point that would call this mid-flight is refused, so
    it only ever runs single-client (CLI, TUI, stdio MCP).
    """
    old = _state.singleton
    _state.singleton = None
    if old is not None:
        old.provider.shutdown()
        old.store.close()


def reset_store() -> None:
    """Close and rebuild only the Store and its dependents; keep providers loaded.

    Used after a data-dir wipe (``/reset``) where the LanceDB handle is invalid
    but the running provider/embedder/reranker are still good. Avoids the
    multi-second reload cost of ``reset_services()``.
    """
    svc = _state.singleton
    if svc is None:
        return
    from dataclasses import replace

    from lilbee.core.config import cfg
    from lilbee.data.store import Store
    from lilbee.retrieval.clustering import Clusterer
    from lilbee.retrieval.concepts import ConceptGraph
    from lilbee.retrieval.query import Searcher

    # Build the replacement, swap the reference, then close the old store last so
    # a new caller never observes a closed handle mid-swap.
    old_store = svc.store
    store = Store(cfg)
    concepts = ConceptGraph(cfg, store)
    clusterer = Clusterer(cfg, store)
    searcher = Searcher(cfg, svc.provider, store, svc.embedder, svc.reranker, concepts)
    _state.singleton = replace(
        svc,
        store=store,
        concepts=concepts,
        clusterer=clusterer,
        searcher=searcher,
    )
    old_store.close()


class _EngineLifecycle:
    """Owns the hard-exit hooks that stop the engine fleet."""

    def __init__(self) -> None:
        self._installed = False

    @staticmethod
    def _hard_exit_signals() -> tuple[signal.Signals, ...]:
        """Signals whose default disposition kills us without running atexit."""
        if sys.platform == "win32":  # pragma: no cover - Windows has no SIGHUP
            return (signal.SIGTERM,)
        return (signal.SIGTERM, signal.SIGHUP)

    def install(self) -> None:
        """Route hard-exit signals through teardown. Idempotent; no-op off the main thread."""
        if self._installed:
            return
        try:
            for sig in self._hard_exit_signals():
                signal.signal(sig, self._on_hard_exit)
        except ValueError:
            return
        self._installed = True

    def reset(self) -> None:
        """Forget that handlers were installed."""
        self._installed = False

    def _on_hard_exit(self, signum: int, frame: object) -> None:
        """Stop the fleet, then exit with the conventional signal status."""
        del frame
        reset_services()
        raise SystemExit(_SIGNAL_EXIT_BASE + signum)


_lifecycle = _EngineLifecycle()


def install_engine_lifecycle_hooks() -> None:
    """Make a terminal close or ``kill`` stop the engine fleet instead of orphaning it."""
    _lifecycle.install()


atexit.register(reset_services)
