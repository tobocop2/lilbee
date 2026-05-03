"""Typed service container: single point of access for all singletons.

All runtime dependencies (provider, store, embedder, reranker, concepts,
clusterer, searcher) are created lazily on first call to ``get_services()``
and cached for the process lifetime. Tests call ``reset_services()``
between runs.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

log = logging.getLogger(__name__)

_DEPRECATED_SUBPROCESS_EMBED_LOGGED = False

if TYPE_CHECKING:
    from lilbee.catalog.hf_client import HfClient
    from lilbee.data.store import Store
    from lilbee.modelhub.model_manager import ModelManager
    from lilbee.modelhub.registry import ModelRegistry
    from lilbee.providers.base import LLMProvider
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
    """Holds all runtime service instances."""

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
    from lilbee.retrieval.clustering import Clusterer
    from lilbee.retrieval.concepts import ConceptGraph
    from lilbee.retrieval.embedder import Embedder
    from lilbee.retrieval.query import Searcher
    from lilbee.retrieval.reranker import Reranker
    from lilbee.runtime.ingest_lock import IngestLockRegistry

    _log_subprocess_embed_deprecation_once(cfg)
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
    )
    return _svc


def set_services(services: Services | None) -> None:
    """Replace the cached Services singleton (for testing)."""
    global _svc
    _svc = services


def _log_subprocess_embed_deprecation_once(cfg_obj: object) -> None:
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
    if not getattr(cfg_obj, "subprocess_embed", False):
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
        _svc.provider.shutdown()
        _svc.store.close()
    _svc = None
    _DEPRECATED_SUBPROCESS_EMBED_LOGGED = False


atexit.register(reset_services)
