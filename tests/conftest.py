"""Shared test helpers."""

import shutil
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import pytest

from lilbee.config import cfg

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _reset_bee_singletons() -> None:
    """Reset all cached Lilbee instances and provider singletons."""
    import lilbee.cli.app as cli_app
    import lilbee.mcp as mcp_mod
    import lilbee.providers.factory as factory_mod
    import lilbee.server.handlers as handlers_mod

    cli_app._bee = None
    mcp_mod._bee = None
    handlers_mod._bee = None
    factory_mod._provider = None


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset cached Lilbee singletons before and after every test."""
    _reset_bee_singletons()
    yield
    _reset_bee_singletons()


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
