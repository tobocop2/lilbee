"""Shared test helpers."""

import shutil
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from lilbee.config import cfg

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@contextmanager
def patched_lilbee_dirs(db_dir: Path, documents_dir: Path) -> Generator[None, None, None]:
    """Temporarily patch lilbee config to use the given directories."""
    from lilbee.embedder import reset_embedder
    from lilbee.providers.factory import reset_provider
    from lilbee.store import reset_store

    snapshot = cfg.model_copy()
    cfg.lancedb_dir = db_dir
    cfg.documents_dir = documents_dir
    reset_provider()
    reset_embedder()
    reset_store()
    try:
        yield
    finally:
        reset_provider()
        reset_embedder()
        reset_store()
        for name in type(cfg).model_fields:
            setattr(cfg, name, getattr(snapshot, name))


def copy_fixtures_to(subdir: str, dest: Path) -> None:
    """Copy all files from FIXTURES_DIR/subdir into dest."""
    src = FIXTURES_DIR / subdir
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dest / item.name)
