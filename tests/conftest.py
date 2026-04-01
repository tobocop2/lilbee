"""Shared test helpers."""

import shutil
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

from lilbee.config import cfg

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def make_mock_services(**overrides):
    """Create a mock Services container. Override individual services via kwargs."""
    from lilbee.providers.base import LLMProvider
    from lilbee.query import Searcher
    from lilbee.services import Services

    provider = overrides.pop("provider", None)
    if provider is None:
        provider = MagicMock(spec=LLMProvider)

    store = overrides.pop("store", None)
    if store is None:
        store = MagicMock()
        store.search.return_value = []
        store.bm25_probe.return_value = []
        store.get_sources.return_value = []
        store.add_chunks.side_effect = lambda records: len(records)

    embedder = overrides.pop("embedder", None)
    if embedder is None:
        embedder = MagicMock()
        embedder.embed.return_value = [0.1] * 768
        embedder.embed_batch.side_effect = lambda texts, **kw: [[0.1] * 768 for _ in texts]

    reranker = overrides.pop("reranker", None)
    if reranker is None:
        reranker = MagicMock()
        reranker.rerank.side_effect = lambda q, r, **kw: r

    concepts = overrides.pop("concepts", None)
    if concepts is None:
        concepts = MagicMock()
        concepts.get_graph.return_value = False

    searcher = overrides.pop("searcher", None)
    if searcher is None:
        searcher = Searcher(cfg, provider, store, embedder, reranker, concepts)

    registry = overrides.pop("registry", None)
    if registry is None:
        registry = MagicMock()

    return Services(
        provider=provider,
        store=store,
        embedder=embedder,
        reranker=reranker,
        concepts=concepts,
        searcher=searcher,
        registry=registry,
    )


@contextmanager
def patched_lilbee_dirs(db_dir: Path, documents_dir: Path) -> Generator[None, None, None]:
    """Temporarily patch lilbee config to use the given directories."""
    from lilbee.services import reset_services

    snapshot = cfg.model_copy()
    cfg.lancedb_dir = db_dir
    cfg.documents_dir = documents_dir
    reset_services()
    try:
        yield
    finally:
        reset_services()
        for name in type(cfg).model_fields:
            setattr(cfg, name, getattr(snapshot, name))


def copy_fixtures_to(subdir: str, dest: Path) -> None:
    """Copy all files from FIXTURES_DIR/subdir into dest."""
    src = FIXTURES_DIR / subdir
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dest / item.name)
