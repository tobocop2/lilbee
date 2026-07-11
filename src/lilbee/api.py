"""Programmatic access to lilbee's retrieval pipeline.

Retrieval only -- no LLM chat. Search your indexed documents from Python.
Optional features (concept graph, reranker) activate automatically when
their dependencies are installed.

Usage::

    from lilbee import Lilbee

    bee = Lilbee("./docs")
    bee.sync()
    results = bee.search("authentication")
    bee.close()

Each instance binds its own Config and Services for the duration of every call
via contextvar scopes, so it runs against its own data root without mutating the
process-global cfg or the shared services singleton. The scope is per task, so
two instances may be driven from different threads or asyncio tasks without
clobbering one another, and the global HTTP daemon's fleet is untouched. An
instance holds a live engine between calls; call :meth:`Lilbee.close` to release
it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

# app.ingest stays at module top: it is a thin wrapper over shutil + the
# config singleton (~50ms cumulative beyond core.config). data.ingest is
# deferred at each callsite below because it transitively imports spaCy via
# the wiki package and adds ~3s on first touch.
from lilbee.app.ingest import copy_files
from lilbee.app.services import build_services, services_scope
from lilbee.core.config import Config, cfg, config_scope
from lilbee.data.store import LOCAL_OWNER, MemoryKind, MemoryRow

if TYPE_CHECKING:
    from lilbee.app.services import Services
    from lilbee.data.ingest import SyncResult
    from lilbee.data.store import SearchChunk, Store
    from lilbee.providers.base import LLMProvider
    from lilbee.retrieval.embedder import Embedder
    from lilbee.retrieval.query import Searcher


class Lilbee:
    """Programmatic access to lilbee's retrieval pipeline.

    Usage::

        from lilbee import Lilbee

        bee = Lilbee("./docs")
        bee.sync()
        results = bee.search("authentication")
    """

    def __init__(
        self,
        documents_dir: str | Path | None = None,
        *,
        config: Config | None = None,
        provider: LLMProvider | None = None,
    ) -> None:
        """Create a lilbee instance.
        Args:
            documents_dir: Path to documents folder. Creates a default Config
                with derived data and lancedb directories.
            config: Full Config instance for complete control.
            provider: LLM provider instance. If not given, creates one from config.

        Pass documents_dir or config, not both. If neither is given, uses
        ``Config()`` (same defaults as the CLI).
        """
        if documents_dir is not None and config is not None:
            raise ValueError("Pass documents_dir or config, not both")

        if config is not None:
            self._config = config
        elif documents_dir is not None:
            root = Path(documents_dir).resolve()
            self._config = cfg.model_copy(
                update={
                    "data_root": root,
                    "documents_dir": root / "documents",
                    "data_dir": root / "data",
                    "lancedb_dir": root / "data" / "lancedb",
                },
            )
        else:
            self._config = Config()

        self._config.documents_dir.mkdir(parents=True, exist_ok=True)
        self._config.data_dir.mkdir(parents=True, exist_ok=True)

        self._services: Services = build_services(self._config, provider=provider)
        self._closed = False

    @property
    def config(self) -> Config:
        """The Config instance backing this Lilbee."""
        return self._config

    @property
    def store(self) -> Store:
        """The Store component."""
        return self._services.store

    @property
    def embedder(self) -> Embedder:
        """The Embedder component."""
        return self._services.embedder

    @property
    def searcher(self) -> Searcher:
        """The Searcher component."""
        return self._services.searcher

    def sync(self, *, quiet: bool = True) -> SyncResult:
        """Sync documents to the vector store. Returns what changed."""
        # heavy: data.ingest transitively imports spaCy via wiki
        from lilbee.data.ingest import sync as _sync

        with config_scope(self._config), services_scope(self._services):
            return asyncio.run(_sync(quiet=quiet))

    def search(self, query: str, *, top_k: int = 0) -> list[SearchChunk]:
        """Search indexed documents. Returns ranked chunks."""
        with config_scope(self._config), services_scope(self._services):
            return self._services.searcher.search(query, top_k=top_k)

    def add(self, paths: list[str | Path]) -> SyncResult:
        """Add files to the knowledge base and sync.
        Copies each path into the documents directory, then syncs.
        """
        # heavy: data.ingest transitively imports spaCy via wiki
        from lilbee.data.ingest import sync as _sync

        resolved = [Path(p).resolve() for p in paths]
        with config_scope(self._config), services_scope(self._services):
            copy_files(resolved, force=True)
            return asyncio.run(_sync(quiet=True))

    def remove(self, name: str) -> None:
        """Remove a document from the index by source name."""
        with config_scope(self._config), services_scope(self._services):
            self._services.store.remove_documents([name], delete_files=True)

    def status(self) -> dict[str, object]:
        """Return index stats (document count, data directory, etc.)."""
        with config_scope(self._config), services_scope(self._services):
            sources = self._services.store.get_sources()
            return {
                "documents_dir": str(self._config.documents_dir),
                "data_dir": str(self._config.data_dir),
                "document_count": len(sources),
                "sources": [s["filename"] for s in sources],
            }

    def rebuild(self) -> SyncResult:
        """Rebuild the entire index from scratch."""
        # heavy: data.ingest transitively imports spaCy via wiki
        from lilbee.data.ingest import sync as _sync

        with config_scope(self._config), services_scope(self._services):
            return asyncio.run(_sync(force_rebuild=True, quiet=True))

    def remember(
        self,
        text: str,
        *,
        kind: MemoryKind = MemoryKind.FACT,
        shared: bool = False,
    ) -> str:
        """Store a fact or preference in long-term memory; returns its id.

        This library primitive does not consult ``memory_enabled``: that flag
        gates the interactive surfaces (TUI/CLI/MCP/REST) and the chat-prompt
        injection, not direct programmatic access. ``remember`` and ``recall``
        operate as a pair regardless of the flag.
        """
        from lilbee.app.memory import make_memory_row

        with config_scope(self._config), services_scope(self._services):
            record = make_memory_row(text, self._services.embedder.embed, kind=kind, shared=shared)
            return self._services.store.add_memory(record)

    def recall(self, query: str, *, top_k: int | None = None) -> list[MemoryRow]:
        """Recall facts relevant to *query* (own memories plus agent-shared)."""
        from lilbee.data.store import human_recall_predicate

        with config_scope(self._config), services_scope(self._services):
            return self._services.store.search_memories(
                self._services.embedder.embed_query(query),
                owner_predicate=human_recall_predicate(),
                top_k=self._config.memory_top_k if top_k is None else top_k,
                max_distance=self._config.memory_max_distance,
            )

    def memories(self) -> list[MemoryRow]:
        """List all stored memories, newest first."""
        from lilbee.data.store import local_owner_predicate

        with config_scope(self._config), services_scope(self._services):
            return self._services.store.get_memories(owner_predicate=local_owner_predicate())

    def forget(self, memory_id: str) -> bool:
        """Delete a local memory by id; True when it existed and was removed."""
        with config_scope(self._config), services_scope(self._services):
            return self._services.store.delete_memory(memory_id, owner=LOCAL_OWNER)

    def close(self) -> None:
        """Release the engine and store this instance holds. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._services.provider.shutdown()
        self._services.store.close()
