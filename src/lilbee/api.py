"""Programmatic access to lilbee's knowledge base pipeline.

Usage::

    from lilbee import Lilbee

    bee = Lilbee("./docs")
    bee.sync()
    results = bee.search("authentication")
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lilbee.config import Config, cfg

if TYPE_CHECKING:
    from lilbee.ingest import SyncResult
    from lilbee.providers.base import LLMProvider
    from lilbee.query import AskResult, ChatMessage
    from lilbee.reasoning import StreamToken
    from lilbee.store import RemoveResult, SearchChunk


@contextmanager
def _swap_config(target: Config) -> Iterator[None]:
    """Temporarily replace the global cfg fields with *target*'s values.

    Not thread-safe -- sequential use only.  Used by methods that call
    into sub-modules still reading the global ``cfg`` singleton (e.g.
    ingest, store).  Will be removed once store/ingest accept explicit config.
    """
    snapshot = {name: getattr(cfg, name) for name in type(cfg).model_fields}
    for name in type(target).model_fields:
        setattr(cfg, name, getattr(target, name))
    try:
        yield
    finally:
        for name, val in snapshot.items():
            setattr(cfg, name, val)


class Lilbee:
    """Single entry point for all lilbee pipeline operations.

    Owns a ``Config`` instance and a lazily-created ``LLMProvider``.
    All pipeline methods use dependency injection internally.

    Usage::

        from lilbee import Lilbee

        bee = Lilbee("./docs")
        bee.sync()
        results = bee.search("authentication")
        answer = bee.ask_raw("How does auth work?")
    """

    def __init__(
        self,
        documents_dir: str | Path | None = None,
        *,
        config: Config | None = None,
    ) -> None:
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
        self._provider: LLMProvider | None = None

    @property
    def config(self) -> Config:
        """The Config instance backing this Lilbee."""
        return self._config

    @property
    def provider(self) -> LLMProvider:
        """Lazily-created LLM provider."""
        if self._provider is None:
            from lilbee.providers.factory import create_provider

            self._provider = create_provider(self._config)
        return self._provider

    def search(self, query: str, *, top_k: int = 0) -> list[SearchChunk]:
        """Search indexed documents. Returns ranked chunks."""
        from lilbee.query import _search_context

        with _swap_config(self._config):
            return _search_context(query, top_k=top_k, config=self._config, provider=self.provider)

    def ask_raw(
        self,
        question: str,
        *,
        top_k: int = 0,
        history: list[ChatMessage] | None = None,
        options: dict[str, Any] | None = None,
    ) -> AskResult:
        """One-shot question returning structured answer + raw sources."""
        from lilbee.query import _ask_raw

        with _swap_config(self._config):
            return _ask_raw(
                question,
                top_k=top_k,
                history=history,
                options=options,
                config=self._config,
                provider=self.provider,
            )

    def ask_stream(
        self,
        question: str,
        *,
        top_k: int = 0,
        history: list[ChatMessage] | None = None,
        options: dict[str, Any] | None = None,
    ) -> Generator[StreamToken, None, None]:
        """Streaming question: yields classified tokens, then source citations."""
        from lilbee.query import _ask_stream

        with _swap_config(self._config):
            yield from _ask_stream(
                question,
                top_k=top_k,
                history=history,
                options=options,
                config=self._config,
                provider=self.provider,
            )

    def sync(
        self,
        *,
        quiet: bool = True,
        force_rebuild: bool = False,
        force_vision: bool = False,
        on_progress: Any = None,
    ) -> SyncResult:
        """Sync documents to the vector store. Returns what changed."""
        from lilbee.ingest import sync as _sync
        from lilbee.progress import noop_callback

        callback = on_progress if on_progress is not None else noop_callback
        with _swap_config(self._config):
            return asyncio.run(
                _sync(
                    force_rebuild=force_rebuild,
                    quiet=quiet,
                    force_vision=force_vision,
                    on_progress=callback,
                )
            )

    def add(
        self,
        paths: list[str | Path],
        *,
        force: bool = True,
        force_vision: bool = False,
    ) -> SyncResult:
        """Add files to the knowledge base and sync."""
        from lilbee.cli.helpers import copy_files
        from lilbee.ingest import sync as _sync

        resolved = [Path(p).resolve() for p in paths]
        with _swap_config(self._config):
            copy_files(resolved, force=force)
            return asyncio.run(_sync(quiet=True, force_vision=force_vision))

    def remove(self, name: str) -> None:
        """Remove a document from the index by source name."""
        from lilbee import store

        with _swap_config(self._config):
            store.delete_by_source(name)
            store.delete_source(name)
            doc_path = self._config.documents_dir / name
            if doc_path.exists():
                doc_path.unlink()

    def remove_documents(self, names: list[str], *, delete_files: bool = False) -> RemoveResult:
        """Remove documents from the knowledge base by source name."""
        from lilbee.store import remove_documents

        with _swap_config(self._config):
            return remove_documents(
                names, delete_files=delete_files, documents_dir=self._config.documents_dir
            )

    def status(self) -> dict[str, object]:
        """Return index stats (document count, data directory, etc.)."""
        from lilbee import store

        with _swap_config(self._config):
            sources = store.get_sources()
            return {
                "documents_dir": str(self._config.documents_dir),
                "data_dir": str(self._config.data_dir),
                "document_count": len(sources),
                "sources": [s["filename"] for s in sources],
            }

    def rebuild(self) -> SyncResult:
        """Rebuild the entire index from scratch."""
        return self.sync(force_rebuild=True, quiet=True)
