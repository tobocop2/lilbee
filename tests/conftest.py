"""Shared helpers for E2E accuracy tests."""

import shutil
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import pytest

from lilbee.config import cfg

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _models_available() -> bool:
    """Check that both embedding and chat models are available."""
    try:
        from lilbee.embedder import embed
        from lilbee.providers import get_provider

        embed("test")  # fastembed, no Ollama needed
        # Just verify the chat model exists — don't run inference.
        installed = set(get_provider().list_models())
        return cfg.chat_model in installed
    except Exception:
        return False


requires_models = pytest.mark.skipif(
    not _models_available(),
    reason="Ollama not running or required models not pulled",
)


@contextmanager
def patched_lilbee_dirs(db_dir: Path, documents_dir: Path) -> Generator[None, None, None]:
    """Temporarily patch lilbee config to use the given directories."""
    snapshot = cfg.model_copy()
    cfg.lancedb_dir = db_dir
    cfg.documents_dir = documents_dir
    try:
        yield
    finally:
        for name in type(cfg).model_fields:
            setattr(cfg, name, getattr(snapshot, name))


def copy_fixtures_to(subdir: str, dest: Path) -> None:
    """Copy all files from FIXTURES_DIR/subdir into dest."""
    src = FIXTURES_DIR / subdir
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dest / item.name)


def batch_search(queries: list[str], top_k: int = 10) -> dict[str, list]:
    """Embed all queries in one batch call, then search for each. Returns {query: results}."""
    from lilbee.embedder import embed_batch
    from lilbee.store import search

    vectors = embed_batch(queries)
    return {q: search(v, top_k=top_k) for q, v in zip(queries, vectors, strict=True)}
