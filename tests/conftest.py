"""Shared helpers for E2E accuracy tests."""

import shutil
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import pytest

import lilbee.config as cfg
import lilbee.store as store_mod

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _models_available() -> bool:
    """Check that both embedding and chat models are available."""
    try:
        import ollama

        from lilbee.embedder import embed

        embed("test")  # fastembed, no Ollama needed
        ollama.chat(model=cfg.CHAT_MODEL, messages=[{"role": "user", "content": "hi"}])
        return True
    except Exception:
        return False


requires_models = pytest.mark.skipif(
    not _models_available(),
    reason="Ollama not running or required models not pulled",
)


@contextmanager
def patched_lilbee_dirs(db_dir: Path, documents_dir: Path) -> Generator[None, None, None]:
    """Temporarily patch lilbee config to use the given directories."""
    original_db = cfg.LANCEDB_DIR
    original_docs = cfg.DOCUMENTS_DIR
    cfg.LANCEDB_DIR = db_dir
    cfg.DOCUMENTS_DIR = documents_dir
    store_mod.LANCEDB_DIR = db_dir
    try:
        yield
    finally:
        cfg.LANCEDB_DIR = original_db
        cfg.DOCUMENTS_DIR = original_docs
        store_mod.LANCEDB_DIR = original_db


def copy_fixtures_to(subdir: str, dest: Path) -> None:
    """Copy all files from FIXTURES_DIR/subdir into dest."""
    src = FIXTURES_DIR / subdir
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dest / item.name)


def batch_search(queries: list[str], top_k: int = 10) -> dict[str, list[dict]]:
    """Embed all queries in one batch call, then search for each. Returns {query: results}."""
    from lilbee.embedder import embed_batch

    vectors = embed_batch(queries)
    return {q: store_mod.search(v, top_k=top_k) for q, v in zip(queries, vectors, strict=True)}
