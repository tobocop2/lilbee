"""Programmatic access to lilbee's retrieval pipeline.

Retrieval only -- no LLM chat. Search your indexed documents from Python.
Optional features (concept graph, reranker) activate automatically when
their dependencies are installed.

Usage::

    from lilbee import Lilbee

    bee = Lilbee("./docs")
    bee.sync()
    results = bee.search("authentication")
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from lilbee.config import Config, cfg
from lilbee.embedder import Embedder
from lilbee.providers.factory import create_provider
from lilbee.query import Searcher
from lilbee.store import Store

if TYPE_CHECKING:
    from lilbee.ingest import SyncResult
    from lilbee.providers.base import LLMProvider
    from lilbee.store import SearchChunk


@contextmanager
def _swap_config(target: Config) -> Iterator[None]:
    """Temporarily replace the global cfg fields with *target*'s values.

    Not thread-safe -- sequential use only.
    """
    from lilbee.services import reset_services

    snapshot = {name: getattr(cfg, name) for name in type(cfg).model_fields}
    for name in type(target).model_fields:
        setattr(cfg, name, getattr(target, name))
    reset_services()
    try:
        yield
    finally:
        reset_services()
        for name, val in snapshot.items():
            setattr(cfg, name, val)


class Lilbee:
    """Programmatic access to lilbee's retrieval pipeline.

    Composes Store, Embedder, Searcher, and Indexer. Each holds a reference
    to config and its dependencies -- no god class, no global mutation in the
    public API.

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
        ``Config.from_env()`` (same defaults as the CLI).
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
            self._config = Config.from_env()

        self._config.documents_dir.mkdir(parents=True, exist_ok=True)
        self._config.data_dir.mkdir(parents=True, exist_ok=True)

        self._provider = provider or create_provider(self._config)
        self._store = Store(self._config)
        self._embedder = Embedder(self._config, self._provider)
        self._searcher = Searcher(self._config, self._provider, self._store, self._embedder)

    @property
    def config(self) -> Config:
        """The Config instance backing this Lilbee."""
        return self._config

    @property
    def store(self) -> Store:
        """The Store component."""
        return self._store

    @property
    def embedder(self) -> Embedder:
        """The Embedder component."""
        return self._embedder

    @property
    def searcher(self) -> Searcher:
        """The Searcher component."""
        return self._searcher

    def sync(self, *, quiet: bool = True) -> SyncResult:
        """Sync documents to the vector store. Returns what changed."""
        from lilbee.ingest import sync as _sync

        with _swap_config(self._config):
            return asyncio.run(_sync(quiet=quiet))

    def search(self, query: str, *, top_k: int = 0) -> list[SearchChunk]:
        """Search indexed documents. Returns ranked chunks."""
        with _swap_config(self._config):
            return self._searcher.search(query, top_k=top_k)

    def add(self, paths: list[str | Path]) -> SyncResult:
        """Add files to the knowledge base and sync.

        Copies each path into the documents directory, then syncs.
        """
        from lilbee.cli.helpers import copy_files
        from lilbee.ingest import sync as _sync

        resolved = [Path(p).resolve() for p in paths]
        with _swap_config(self._config):
            copy_files(resolved, force=True)
            return asyncio.run(_sync(quiet=True))

    def remove(self, name: str) -> None:
        """Remove a document from the index by source name."""
        with _swap_config(self._config):
            self._store.delete_by_source(name)
            self._store.delete_source(name)
            doc_path = self._config.documents_dir / name
            if not doc_path.resolve().is_relative_to(self._config.documents_dir.resolve()):
                return
            if doc_path.exists():
                doc_path.unlink()

    def status(self) -> dict[str, object]:
        """Return index stats (document count, data directory, etc.)."""
        with _swap_config(self._config):
            sources = self._store.get_sources()
            return {
                "documents_dir": str(self._config.documents_dir),
                "data_dir": str(self._config.data_dir),
                "document_count": len(sources),
                "sources": [s["filename"] for s in sources],
            }

    def rebuild(self) -> SyncResult:
        """Rebuild the entire index from scratch."""
        from lilbee.ingest import sync as _sync

        with _swap_config(self._config):
            return asyncio.run(_sync(force_rebuild=True, quiet=True))
