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
import logging
import multiprocessing
import os
import queue
import signal
import sys
import threading
import time
import weakref
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from lilbee.app.engine_choice import EngineBook
    from lilbee.catalog.hf_client import HfClient
    from lilbee.core.config import Config
    from lilbee.core.profile_files import ProfileStore
    from lilbee.data.store import Store
    from lilbee.data.store.canary import CanaryProbe
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
    from lilbee.runtime.progress import DetailedProgressCallback, EventType, ProgressEvent
    from lilbee.sessions import SessionStore


log = logging.getLogger(__name__)

_SIGNAL_EXIT_BASE = 128

SERVICES_STOPPING_MESSAGE = "lilbee is stopping."


class ServicesStoppingError(RuntimeError):
    """The engine stop has begun; no new container is built."""

    def __init__(self) -> None:
        super().__init__(SERVICES_STOPPING_MESSAGE)


def _default_session_store() -> SessionStore:
    """Build the file-backed session store, importing it lazily.

    ``lilbee.sessions`` pulls in the config/catalog import chain, so importing it
    at this module's top would form a cycle during CLI config load.
    """
    from lilbee.sessions import SessionStore

    return SessionStore()


def _default_profile_store() -> ProfileStore:
    """Build the profile store, importing it lazily."""
    # circular: core.config -> catalog (model-ref validator) -> app.services -> core.profile_files
    from lilbee.core.profile_files import ProfileStore

    return ProfileStore()


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
    engines: EngineBook
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
    session_store: SessionStore = field(default_factory=_default_session_store)
    profile_store: ProfileStore = field(default_factory=_default_profile_store)

    def cancel_inference(self) -> None:
        """Interrupt any in-flight generation. Idempotent.

        The fleet engine severs its live chat streams (llama-server stops
        generating when the connection drops); providers with nothing in
        flight treat this as a no-op.
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
    contract: creation is serialized by ``_singleton_create_lock`` (several
    worker threads can first-touch services at once, and a duplicate build
    would collide in xberg's process-global backend registry), and the
    Services dataclass is logically immutable post-construction, so concurrent
    reads are safe without a lock. Tests that need a custom container call
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
        # Whether this process is an interactive session (the TUI). Recorded by
        # the interactive entry point before anything builds the container, and
        # read once at build so the provider it creates holds its fleet resident
        # for the session. Build-time intent only; the state that matters after
        # that lives on the provider itself.
        self.interactive: bool = False
        # Where this process shows a library upgrade; each surface sets its own.
        self.on_upgrade: DetailedProgressCallback | None = None
        # How this process takes config.toml now; set by each surface that follows the file.
        self.config_refresh: Callable[[], None] | None = None
        # Every store this process built, so the exit can wait for any library open
        # (the first, the one /reset makes, a lazy one) without each path telling it.
        self.stores: weakref.WeakSet[Store] = weakref.WeakSet()
        self.stores_lock = threading.Lock()

    def report_upgrade(self, event_type: EventType, event: ProgressEvent) -> None:
        """Pass a library upgrade event to the reporter the surface set last, if any."""
        if self.on_upgrade is not None:
            self.on_upgrade(event_type, event)


_state = _ServicesState()


def _new_store(config: Config, probe: CanaryProbe | None = None) -> Store:
    """Build a Store and record it, so ``wait_for_opens`` sees its open; the one way to make one."""
    from lilbee.data.store import Store

    store = Store(config, on_upgrade=_state.report_upgrade, canary_probe=probe)
    with _state.stores_lock:
        _state.stores.add(store)
    return store


def wait_for_opens(timeout: float) -> bool:
    """Wait up to *timeout* seconds in all, for every library open still running.

    Return False when an open is still running at the deadline.
    """
    deadline = time.monotonic() + timeout
    with _state.stores_lock:
        stores = list(_state.stores)
    done = True
    for store in stores:
        if not store.wait_for_open(max(0.0, deadline - time.monotonic())):
            done = False
    return done


def _run_cleanup(step: Callable[[], None]) -> None:
    """Run one cleanup step of a failed build; log its error so the build's own error stays."""
    try:
        step()
    except Exception:
        log.warning("Cleanup after a failed services build failed", exc_info=True)


def build_services(
    config: Config,
    *,
    provider: LLMProvider | None = None,
    registry: ModelRegistry | None = None,
    interactive: bool = False,
) -> Services:
    """Build a full Services container bound to *config*, without caching it.

    ``get_services()`` calls this with the process-global cfg; the library API
    calls it per instance. Service modules are imported inside the function to
    keep CLI startup fast (they transitively pull in lancedb / xberg). Pass
    *provider* to reuse a caller-supplied one; otherwise it is built from
    *config* via the provider factory. Pass *registry* to reuse one already built
    (get_services builds it for embedding-dim reconciliation). Embedding-dim
    reconciliation is a global-cfg concern owned by :func:`get_services`, not
    done here.

    Side effect: binds *provider* into xberg's process-global OCR and embedding
    backends (the chunker binds the tokenizer on demand). The registry is one
    process-wide slot, so every container (singleton or per-instance library)
    binds its own, not only get_services.
    """
    from lilbee.app.engine_choice import EngineBook
    from lilbee.catalog.hf_client import HfClient
    from lilbee.modelhub.model_manager import ModelManager
    from lilbee.modelhub.model_manager.discovery import KnownModelCache
    from lilbee.modelhub.registry import ModelRegistry
    from lilbee.modelhub.role_validator import warn_unregistered_role_refs
    from lilbee.providers.factory import create_provider
    from lilbee.retrieval.canary_probe import canary_probe
    from lilbee.retrieval.clustering import Clusterer
    from lilbee.retrieval.concepts import ConceptGraph
    from lilbee.retrieval.embedder import Embedder
    from lilbee.retrieval.query import Searcher
    from lilbee.retrieval.reranker import Reranker
    from lilbee.runtime.ingest_lock import IngestLockRegistry

    engines = EngineBook(config)
    owns_provider = provider is None
    provider = provider or create_provider(
        config, hold_warm=interactive, config_refresh=refresh_config
    )
    store: Store | None = None
    try:
        from lilbee.data.extract.backends import sync_xberg_backends

        sync_xberg_backends(provider)
        registry = registry or ModelRegistry(config.models_dir)
        # The first point that sees env vars, --model and config.toml together.
        warn_unregistered_role_refs(config, registry)
        embedder = Embedder(config, provider)
        store = _new_store(config, canary_probe(config, embedder, registry))
        reranker = Reranker(config)
        concepts = ConceptGraph(config, store)
        clusterer = Clusterer(config, store)
        searcher = Searcher(config, provider, store, embedder, reranker, concepts)
        hf_client = HfClient(config)
        ingest_lock_registry = IngestLockRegistry()
        model_manager = ModelManager(config.models_dir, config)
        crawler_semaphore = (
            asyncio.Semaphore(config.crawl_max_concurrent)
            if config.crawl_max_concurrent > 0
            else None
        )
        crawler_sync_state = CrawlerSyncState()
        known_models = KnownModelCache()
        return Services(
            provider=provider,
            engines=engines,
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
    except BaseException:
        # A failed build leaves nothing open: the store it made and the provider it
        # created (a caller-supplied provider stays the caller's).
        if store is not None:
            _run_cleanup(store.close)
        if owns_provider:
            _run_cleanup(provider.shutdown)
        raise


# Serializes first-touch singleton creation: several worker threads (e.g.
# concurrent downloads) can call get_services() before the singleton exists,
# and a duplicate build re-registers xberg's process-global backends mid-flight,
# raising 'already registered' in the losing thread.
_singleton_create_lock = threading.Lock()


def get_services() -> Services:
    """Return the active container: a scoped override if set, else the cached singleton.

    Creates the singleton on first call (against the process-global cfg). A
    config-file embedding_model with no embedding_dim would otherwise build the
    store at the stale 768 default, so the width is pinned to the embedder before
    the store is built. Once the engine stop has begun it builds nothing and
    raises ServicesStoppingError, so no container outlives the stop.
    """
    override = _state.override.get()
    if override is not None:
        return override
    if _state.singleton is not None:
        return _state.singleton

    from lilbee.core.config import cfg

    with _singleton_create_lock:
        if _state.singleton is not None:
            return _state.singleton
        if _lifecycle.stop_requested:
            raise ServicesStoppingError

        from lilbee.app.settings import reconcile_embedding_dim
        from lilbee.modelhub.registry import ModelRegistry

        registry = ModelRegistry(cfg.models_dir)
        # Pin the store width to the embedder before Store(); pass the registry so
        # resolution doesn't re-enter this half-built get_services.
        reconcile_embedding_dim(registry)
        services = build_services(cfg, registry=registry, interactive=_state.interactive)
        # A stop asked for during the build may already have swapped the singleton
        # out, so this container is shut down here instead of outliving the stop.
        with _reset_swap_lock:
            stopping = _lifecycle.stop_requested
            if not stopping:
                _state.singleton = services
        if stopping:
            services.provider.shutdown()
            services.store.close()
            raise ServicesStoppingError
        # Eager start is the default: pay the spawn cost per role server at TUI mount
        # so the first user action lands on a warm fleet. Roles whose model is unset
        # are skipped, so a setup with only chat + embed never spawns rerank or
        # vision. Set ``cfg.worker_pool_eager_start = false`` for headless scripts
        # where mount time matters more than first-call latency.
        if cfg.worker_pool_eager_start:
            from contextlib import suppress

            with suppress(Exception):
                services.provider.warm_up_pool()
        return services


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


def mark_interactive_session() -> None:
    """Record that this process is an interactive session before services build.

    The TUI owns the process for its whole lifetime, so an engine bound to it
    keeps its weights resident rather than idle-unloading under a user who is
    still in the app; closing lilbee releases it. An engine that outlives the
    session (shared with its other users, or kept warm) keeps its idle window
    instead. Called by the interactive entry point before anything touches
    ``get_services``, so the provider is constructed with that intent; a
    one-shot CLI or the MCP server never calls it.
    """
    _state.interactive = True


def set_config_refresh(hook: Callable[[], None] | None) -> None:
    """Register how this process takes config.toml changes now (None clears it).

    serve, the stdio MCP server and the TUI register the check they already run, so
    the local engine can take an owner's new settings before it retries a call.
    """
    _state.config_refresh = hook


def refresh_config() -> None:
    """Take config.toml changes now through the registered check; nothing without one."""
    hook = _state.config_refresh
    if hook is not None:
        hook()


def report_library_upgrade(callback: DetailedProgressCallback | None) -> None:
    """Show this process's library upgrades through *callback* (None: nowhere), for every store."""
    _state.on_upgrade = callback


def set_services(services: Services | None) -> None:
    """Replace the cached Services singleton (for testing)."""
    _state.singleton = services


def chunks_in_store() -> bool:
    """True when the index holds a chunk, asked of the store alone.

    A bound container (scoped or cached) answers with its own store. Without
    one, a short-lived store answers and no provider is built, so the check
    starts no model warm-up.
    """
    from lilbee.core.config import cfg

    bound = _state.override.get() or _state.singleton
    if bound is not None:
        return bound.store.has_chunks()
    store = _new_store(cfg)
    try:
        return store.has_chunks()
    finally:
        store.close()


def peek_services() -> Services | None:
    """Return the cached Services container, or None if not yet initialized.

    Public read-only accessor for test cleanup helpers that need to
    inspect the singleton without forcing initialization.
    """
    return _state.singleton


def answer_provider_waiters() -> None:
    """Mark the server as stopping and answer the callers that wait on the provider."""
    _server_exit.mark_stopping()
    services = peek_services()
    if services is not None:
        services.provider.stop_waiting()


# Serializes the singleton swap in reset_services: the engine stop thread
# and another caller can race, and both tearing down the same container
# would double-close the store.
_reset_swap_lock = threading.Lock()

# Serializes reset_store builds; taken before _reset_swap_lock, never inside it.
_store_rebuild_lock = threading.Lock()


def reset_services() -> None:
    """Shut down and discard all cached instances.

    Swap the module reference to ``None`` *before* tearing the old instances
    down, so a new caller never observes a half-closed container. The swap is
    locked so concurrent callers (the engine stop thread plus another one)
    tear the container down exactly once. On ``lilbee serve``, SIGINT,
    SIGTERM and SIGHUP run it on the engine stop thread while requests
    are still open; the shutdown route runs it after those requests end.
    """
    with _reset_swap_lock:
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
    from dataclasses import replace

    from lilbee.core.config import cfg
    from lilbee.retrieval.canary_probe import canary_probe
    from lilbee.retrieval.clustering import Clusterer
    from lilbee.retrieval.concepts import ConceptGraph
    from lilbee.retrieval.query import Searcher

    # One rebuild at a time, so two resets never build from the same container. This
    # lock is taken before _reset_swap_lock and never inside it; get_services and
    # reset_services never take it, so the stop is not held up by a build.
    with _store_rebuild_lock:
        svc = _state.singleton
        if svc is None:
            return
        # Build the replacement, swap the reference, then close the old store last so
        # a new caller never observes a closed handle mid-swap.
        store = _new_store(cfg, canary_probe(cfg, svc.embedder, svc.registry))
        try:
            concepts = ConceptGraph(cfg, store)
            clusterer = Clusterer(cfg, store)
            searcher = Searcher(cfg, svc.provider, store, svc.embedder, svc.reranker, concepts)
            rebuilt = replace(
                svc, store=store, concepts=concepts, clusterer=clusterer, searcher=searcher
            )
        except BaseException:
            store.close()
            raise
        # Same lock and stop check as get_services: a rebuild must not bring back a
        # container that the stop cleared, nor one that reset_services replaced.
        with _reset_swap_lock:
            stopping = _lifecycle.stop_requested
            swapped = not stopping and _state.singleton is svc
            if swapped:
                _state.singleton = rebuilt
    if swapped:
        svc.store.close()
        return
    store.close()
    if stopping:
        raise ServicesStoppingError


class _ServerExit:
    """Holds the running server's graceful-exit callback while it serves."""

    def __init__(self) -> None:
        self._hook: Callable[[int], None] | None = None
        self.stopping = False

    def set(self, hook: Callable[[int], None] | None) -> None:
        self._hook = hook
        if hook is not None:
            self.stopping = False

    def mark_stopping(self) -> None:
        self.stopping = True

    def request(self, status: int) -> bool:
        """Run the hook with the exit *status*; False when no server registered one."""
        if self._hook is None:
            return False
        self._hook(status)
        return True


_server_exit = _ServerExit()


def set_server_exit_hook(hook: Callable[[int], None] | None) -> None:
    """Register the running server's graceful-exit callback (None clears it)."""
    _server_exit.set(hook)


def server_is_stopping() -> bool:
    """Whether the running server has begun to stop."""
    return _server_exit.stopping


def request_server_exit(status: int = 0) -> bool:
    """Ask the running server to exit with *status*; False when none registered.

    A nonzero *status* is a hard exit: responses in flight get a short wait, then cancel.
    """
    return _server_exit.request(status)


class _StopStep(StrEnum):
    """The teardown step the stop runs."""

    STARTING = "starting the stop"
    ANSWER_WAITERS = "answering the callers that wait on the provider"
    RESET_SERVICES = "stopping the services"


class _EngineLifecycle:
    """Owns the hard-exit hooks and the one thread that stops the engine fleet."""

    def __init__(self) -> None:
        self._installed = False
        self._reset_in_forked_child()

    def _reset_in_forked_child(self) -> None:
        """Drop the stop thread and its queue, which a fork does not copy."""
        self._start_lock = threading.Lock()
        self._requests: queue.SimpleQueue[int] = queue.SimpleQueue()
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None
        self._forget_stop()

    def _forget_stop(self) -> None:
        """Clear every stop request, so the next request runs a new stop."""
        self._stopped.clear()
        self.stop_requested = False
        self._signalled = False
        self.owed_status: int | None = None
        self.stop_step = _StopStep.STARTING

    def _ensure_stop_thread(self) -> None:
        """Start the stop thread once, in ordinary context, before any handler can ask for it."""
        with self._start_lock:
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._serve_stops, name="engine-stop", daemon=True
                )
                self._thread.start()

    def _serve_stops(self) -> None:
        while True:
            self._run_stop(self._requests.get())

    def _run_stop(self, signum: int) -> None:
        try:
            self._stop(signum)
        finally:
            self._stopped.set()

    def _stop(self, signum: int) -> None:
        """Log the signal (0: an exit asked), answer the provider's waiters, stop the services."""
        if signum:
            log.info(
                "Received signal %s; stopping the engine fleet before exit",
                signal.Signals(signum).name,
            )
        self.stop_step = _StopStep.ANSWER_WAITERS
        answer_provider_waiters()
        self.stop_step = _StopStep.RESET_SERVICES
        reset_services()

    def stop_for_exit(self) -> bool:
        """Ask the stop thread for the stop; True when the caller must wait for it.

        With no stop thread (no handler installed: library use, or atexit) the stop
        runs on the calling thread, which starts no thread at interpreter shutdown.
        """
        if self._thread is None:
            self.stop_requested = True
            self._run_stop(0)
            return False
        self.request_stop(0)
        return True

    def request_stop(self, signum: int) -> None:
        """Queue the stop once; a later request joins the stop already asked for."""
        if not self.stop_requested:
            self.stop_requested = True
            self._requests.put(signum)

    def wait_stopped(self, timeout: float) -> bool:
        """Wait for the stop to finish; False when *timeout* ran out first."""
        return self._stopped.wait(timeout)

    @staticmethod
    def _hard_exit_signals() -> tuple[signal.Signals, ...]:  # pragma: no cover - platform split
        """Signals whose default disposition kills us without running atexit."""
        if sys.platform == "win32":  # Windows has no SIGHUP
            return (signal.SIGTERM,)
        return (signal.SIGTERM, signal.SIGHUP)

    @staticmethod
    def _ignored_at_launch(sig: signal.Signals) -> bool:
        """Whether the launch left *sig* ignored (nohup does it for SIGHUP)."""
        return signal.getsignal(sig) == signal.SIG_IGN

    @classmethod
    def _routed_signals(cls, *, interrupt: bool) -> tuple[signal.Signals, ...]:
        """Signals to route: SIGTERM always; SIGHUP, and SIGINT when *interrupt*, unless ignored."""
        wanted = cls._hard_exit_signals()
        if interrupt:
            wanted = (signal.SIGINT, *wanted)
        return tuple(
            sig for sig in wanted if sig == signal.SIGTERM or not cls._ignored_at_launch(sig)
        )

    def install(self) -> None:
        """Route hard-exit signals through teardown. Idempotent; no-op off the main thread."""
        if self._installed:
            return
        self._ensure_stop_thread()
        try:
            for sig in self._routed_signals(interrupt=False):
                signal.signal(sig, self._on_hard_exit)
        except ValueError:
            return
        self._installed = True

    def route_interrupt(self) -> None:
        """Send Ctrl-C through the hard-exit path; a second press ends the process at once.

        An inherited ignore of SIGINT stays in place.
        """
        if signal.SIGINT not in self._routed_signals(interrupt=True):
            return
        self._ensure_stop_thread()
        signal.signal(signal.SIGINT, self._on_hard_exit)

    def routed_in_loop(self) -> tuple[signal.Signals, ...]:
        """Signals a running loop takes over: the ones install and route_interrupt route."""
        return self._routed_signals(interrupt=True)

    def on_loop_signal(self, signum: int) -> None:  # pragma: posix only - no loop signals
        """Run the handler of *signum* as a loop callback, with no interrupted frame."""
        self._on_hard_exit(signum, None)

    def reset(self) -> None:
        """Forget that handlers were installed and every stop request."""
        self._installed = False
        self._forget_stop()

    def _on_hard_exit(self, signum: int, frame: object) -> None:
        """Queue the stop and exit with the signal status; the handler only records.

        It starts no thread, takes no lock and logs nothing: a signal can land
        while the main thread holds one of threading's own locks. A second
        routed signal ends the process at once: the exit after the first one
        joins the executor threads, and a stuck worker never ends that join.
        """
        del frame
        status = _SIGNAL_EXIT_BASE + signum
        if self._signalled:
            _force_quit_from_handler(status)
            return
        server_took = request_server_exit(status)
        joining = server_took or server_is_stopping() or self.stop_requested
        if joining and not server_took:
            self.owed_status = status
        self._signalled = True
        self.request_stop(signum)
        if not joining:
            raise SystemExit(status)


_FORCE_QUIT_EXIT_CODE = 130

_STOPPING_NOTE = "lilbee: stopping the engine. Press Ctrl-C again to force quit.\n"

_STOPPING_NOTE_GRACE_S = 1.0
"""Seconds a teardown may run silently; a fast exit stays as quiet as ever."""

_SIGNAL_CHECK_S = 0.1
"""Seconds between the exit wait's checks for a signal that another thread took.

That thread's C handler only sets a flag, and a lock wait on the main thread does
not wake for it, so the Python handler runs when the main thread next runs code.
"""

_FORCE_QUIT_NOTE = "lilbee: force quit. The engine stops with it.\n"

_FORCE_QUIT_NOTE_BYTES = _FORCE_QUIT_NOTE.encode()

_STRAGGLER_BUDGET_S = 5.0
"""Seconds every leftover non-daemon thread gets, together, before the exit."""

_STRAGGLER_NOTE = "lilbee: a background task would not stop; exiting anyway.\n"

_OPEN_EXIT_NOTE = "lilbee: the library is still opening; exiting now. The next start finishes it.\n"


def _write_exit_note(note: str) -> None:
    """Print an exit-path status line; stderr may already be gone at exit."""
    try:
        sys.stderr.write(note)
        sys.stderr.flush()
    except (OSError, ValueError):
        pass


def _flush_logs() -> None:
    """Flush every log handler and stdio; ``os._exit`` drops what a buffer still holds."""
    # loggerDict also holds PlaceHolder entries for loggers that have no object yet.
    named = [e for e in logging.Logger.manager.loggerDict.values() if isinstance(e, logging.Logger)]
    loggers = [logging.getLogger(), *named]
    for logger in loggers:
        for handler in logger.handlers:
            with suppress(OSError, ValueError):
                handler.flush()
    for stream in (sys.stdout, sys.stderr):
        with suppress(AttributeError, OSError, ValueError):
            stream.flush()


def _exit_now(status: int, note: str) -> None:
    """Kill the worker processes, write *note*, flush the logs, then end with *status*.

    It skips atexit, so it also skips multiprocessing's end of the daemonic workers.
    """
    _kill_worker_processes()
    _write_exit_note(note)
    _flush_logs()
    os._exit(status)


def _kill_worker_processes() -> None:
    """Kill the worker processes: only Linux ends one with its parent, and a forced exit joins none.

    A signal handler may run it: it takes no lock and starts no thread.
    """
    for worker in multiprocessing.active_children():
        # One that another thread ended first must not stop the exit.
        with suppress(OSError, ValueError):
            worker.kill()


def _force_quit_from_handler(status: int) -> None:
    """End the process with *status* from a signal handler: a bare write of the note to fd 2."""
    _kill_worker_processes()
    with suppress(OSError):
        os.write(2, _FORCE_QUIT_NOTE_BYTES)
    os._exit(status)


def force_quit_from_handler() -> None:
    """End the process from a Ctrl-C handler, with the Ctrl-C status."""
    _force_quit_from_handler(_FORCE_QUIT_EXIT_CODE)


def force_quit() -> None:
    """End the process with the Ctrl-C status; a normal exit joins a stuck worker thread."""
    _exit_now(_FORCE_QUIT_EXIT_CODE, _FORCE_QUIT_NOTE)


@contextmanager
def _interrupt_force_quits() -> Iterator[None]:
    """Turn a Ctrl-C during an exit wait into the force quit."""
    try:
        yield
    except KeyboardInterrupt:
        force_quit()


def _all_threads() -> list[threading.Thread]:
    """Seam over threading.enumerate, so tests never patch the stdlib global."""
    return threading.enumerate()


def _wait_for_stop() -> None:
    """Wait for the stop thread's teardown; silent for the grace, then the stopping note.

    An interrupt while waiting force-quits the process: waiting is the only
    thing left, and the engine reaper (or, without kernel file locks, the
    child guard) stops the engine once this was its last user.
    """
    with _interrupt_force_quits():
        if not _lifecycle.wait_stopped(_STOPPING_NOTE_GRACE_S):
            _write_exit_note(_STOPPING_NOTE)
            while not _lifecycle.wait_stopped(_SIGNAL_CHECK_S):
                pass


def wait_for_hard_exit_teardown() -> None:
    """Block until a stop that a signal or an exit asked for finishes; else return.

    Lets ``serve`` hold its OS locks through the fleet stop, so a successor
    cannot acquire them while this server's models still occupy memory.
    """
    if _lifecycle.stop_requested:
        _wait_for_stop()


def reset_services_on_exit() -> None:
    """Run the teardown and wait for it; no-op with nothing to stop.

    With signal handlers installed the stop thread runs it, since a Ctrl-C on the
    main thread would skip the engine release and atexit cannot retry. Without
    them it runs here, and a Ctrl-C during it force-quits.
    """
    if peek_services() is None and not _lifecycle.stop_requested:
        return
    with _interrupt_force_quits():
        if _lifecycle.stop_for_exit():
            _wait_for_stop()


def _straggler_threads() -> list[threading.Thread]:
    """Non-daemon threads the interpreter would join at shutdown, besides us."""
    return [
        thread
        for thread in _all_threads()
        if thread.is_alive()
        and not thread.daemon
        and thread is not threading.main_thread()
        and thread is not threading.current_thread()
    ]


def exit_when_stragglers_would_hang(status: int) -> None:
    """Exit rather than let interpreter shutdown join a wedged thread forever.

    threading joins every non-daemon thread after atexit, so one worker
    blocked in an unbounded network read hangs the process after all real
    work is done. Stragglers share a short budget; whatever remains cannot
    finish, and the child guard reaps the engine when the process dies, with
    *status*. Called from the TUI's own exit path and inert elsewhere: a test process
    or embedding host owns its threads, and an exit would take them with it.
    """
    if not _state.interactive:
        return
    _exit_if_stragglers(status, time.monotonic() + _STRAGGLER_BUDGET_S)


def _exit_if_stragglers(status: int, deadline: float) -> None:
    """Join leftover non-daemon threads before *deadline*, else exit with *status*."""
    with _interrupt_force_quits():
        for thread in _straggler_threads():
            thread.join(max(0.0, deadline - time.monotonic()))
        if _straggler_threads():
            _exit_now(status, _STRAGGLER_NOTE)


class _UnjoinedExecutor(ThreadPoolExecutor):
    """A thread pool whose shutdown never waits for its workers.

    asyncio's runner joins the loop's default executor at close with no bound
    on Python 3.11, so one stuck worker held the exit forever. Its threads stay
    non-daemon, so the straggler exit joins them within its budget instead.
    """

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        del wait
        super().shutdown(wait=False, cancel_futures=cancel_futures)


def new_exit_bounded_loop() -> asyncio.AbstractEventLoop:
    """A new event loop whose close leaves a stuck executor worker to the straggler exit."""
    loop = asyncio.new_event_loop()
    loop.set_default_executor(_UnjoinedExecutor())
    return loop


def exit_if_opens_remain(status: int) -> None:
    """End the process with *status* when a library open still runs; else return."""
    # A daemon thread inside LanceDB at finalization aborts Python 3.11 and 3.12
    # (CPython gh-87135). The exit skips atexit, so the teardown runs first.
    if wait_for_opens(0.0):
        return
    reset_services_on_exit()
    _exit_now(status, _OPEN_EXIT_NOTE)


def stop_on_interrupt() -> None:
    """Make Ctrl-C stop a serving process as SIGTERM does; a second press ends it at once."""
    _lifecycle.route_interrupt()


def exit_after_signal(status: int) -> None:
    """Finish the engine stop, then exit with *status* even past a wedged thread.

    The stdio MCP server reads stdin on a non-daemon thread that no signal
    wakes, so the interpreter would wait on it forever after the main thread
    unwound. The stop and the stragglers share one budget; a cut names its step.
    """
    deadline = time.monotonic() + _STRAGGLER_BUDGET_S
    if _lifecycle.stop_for_exit() and not _lifecycle.wait_stopped(_STRAGGLER_BUDGET_S):
        log.warning("The exit cut the engine stop while %s", _lifecycle.stop_step)
        _exit_now(status, _STRAGGLER_NOTE)
    _exit_if_stragglers(status, deadline)


_lifecycle = _EngineLifecycle()


def install_engine_lifecycle_hooks() -> None:
    """Make a terminal close or ``kill`` stop the engine fleet instead of orphaning it."""
    _lifecycle.install()


@contextmanager
def loop_takes_signals(loop: asyncio.AbstractEventLoop) -> Iterator[None]:
    """Deliver the routed signals as callbacks of *loop* while it runs.

    A plain handler raises at whatever bytecode the main thread is running. That can
    be inside the loop's own bookkeeping or a library's ``except BaseException``, and
    then the exit is lost and the loop idles. A loop callback runs between steps, so
    the SystemExit it raises always leaves ``run_forever``. The handlers that were
    installed before return when the loop ends, so a late signal still reaches the stop.
    """
    routed = _lifecycle.routed_in_loop()
    previous = {sig: signal.getsignal(sig) for sig in routed}
    added: list[signal.Signals] = []
    try:
        for sig in routed:
            loop.add_signal_handler(sig, _lifecycle.on_loop_signal, sig)
            added.append(sig)  # pragma: posix only - Windows loops raise on the add
    except (NotImplementedError, RuntimeError):  # pragma: no cover - Windows, off the main thread
        pass
    try:
        yield
    finally:
        for sig in added:  # pragma: posix only - empty on Windows
            loop.remove_signal_handler(sig)
            signal.signal(sig, previous[sig])


def exit_status_after_stop(status: int) -> int:
    """Return the status of a first signal that joined an exit's stop, else *status*."""
    owed = _lifecycle.owed_status
    return status if owed is None else owed


if hasattr(os, "register_at_fork"):  # pragma: no cover - absent on Windows
    os.register_at_fork(after_in_child=_lifecycle._reset_in_forked_child)

atexit.register(reset_services_on_exit)
