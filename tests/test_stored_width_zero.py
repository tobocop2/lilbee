"""A store whose ``_meta`` row holds a width of 0 starts like a store with no stored width."""

from __future__ import annotations

from unittest.mock import MagicMock

import lancedb
import pytest

from lilbee.app import services as services_mod
from lilbee.app.settings import reconcile_embedding_dim
from lilbee.core.config import META_TABLE, cfg
from lilbee.data.store import Store
from lilbee.data.store.lance_helpers import persisted_embedding_width
from tests.conftest import make_mock_services

MODEL = "ollama/stored-embed:latest"
CONFIGURED_WIDTH = 8
STORED_WIDTH = 16


@pytest.fixture(autouse=True)
def isolated_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path / "global-data"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    snapshot = cfg.model_copy()
    cfg.documents_dir = tmp_path / "docs"
    cfg.documents_dir.mkdir()
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.concept_graph = False
    cfg.embedding_model = MODEL
    cfg.embedding_dim = STORED_WIDTH
    yield
    services_mod._state.singleton = None
    services_mod.set_services(None)
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _store_with_stored_width(width: int) -> None:
    """Write the ``_meta`` row the way the store writer does, then set its width."""
    Store(cfg).restamp_meta()
    table = lancedb.connect(str(cfg.lancedb_dir)).open_table(META_TABLE)
    table.update(values={"embedding_dim": width})
    assert table.to_arrow()["embedding_dim"].to_pylist() == [width]
    cfg.embedding_dim = CONFIGURED_WIDTH


def test_startup_takes_a_positive_stored_width():
    _store_with_stored_width(STORED_WIDTH)

    reconcile_embedding_dim()

    assert cfg.embedding_dim == STORED_WIDTH


@pytest.mark.parametrize("stored", [0, -1])
def test_startup_keeps_the_configured_width_when_the_stored_width_is_below_1(stored):
    _store_with_stored_width(stored)

    reconcile_embedding_dim()

    assert cfg.embedding_dim == CONFIGURED_WIDTH


@pytest.mark.parametrize(
    ("stored", "read"), [(1, 1), (STORED_WIDTH, STORED_WIDTH), (0, None), (-1, None)]
)
def test_the_stored_width_read_gives_a_width_above_0_only(stored, read):
    _store_with_stored_width(stored)

    assert persisted_embedding_width(cfg.lancedb_dir, MODEL) == read


def test_the_stored_width_read_gives_none_for_another_model():
    _store_with_stored_width(STORED_WIDTH)

    assert persisted_embedding_width(cfg.lancedb_dir, "ollama/other:latest") is None


def test_the_stored_width_read_gives_none_for_a_library_without_a_meta_table():
    cfg.lancedb_dir.mkdir(parents=True)
    lancedb.connect(str(cfg.lancedb_dir))

    assert persisted_embedding_width(cfg.lancedb_dir, MODEL) is None


def _build_recording_width(monkeypatch) -> list[int]:
    built: list[int] = []

    def build(config, provider=None, registry=None, interactive=False):
        built.append(config.embedding_dim)
        return make_mock_services()

    monkeypatch.setattr(services_mod, "build_services", build)
    monkeypatch.setattr(cfg, "worker_pool_eager_start", False)
    services_mod._state.singleton = None
    return built


def _start_cli() -> None:
    from lilbee.cli.helpers import print_source_listing

    print_source_listing()


def _start_mcp() -> None:
    from lilbee import mcp_server

    mcp_server.status()


async def _start_serve() -> None:
    from lilbee.server.app import _lifespan

    async with _lifespan(MagicMock()):
        pass


@pytest.mark.parametrize("surface", ["cli", "mcp"])
def test_a_sync_surface_starts_on_a_stored_width_of_0(monkeypatch, surface):
    _store_with_stored_width(0)
    built = _build_recording_width(monkeypatch)

    {"cli": _start_cli, "mcp": _start_mcp}[surface]()

    assert built == [CONFIGURED_WIDTH]


async def test_serve_starts_on_a_stored_width_of_0(monkeypatch):
    _store_with_stored_width(0)
    built = _build_recording_width(monkeypatch)

    await _start_serve()

    assert built == [CONFIGURED_WIDTH]
