"""A store whose ``_meta`` row holds a width of 0 starts like a store with no stored width."""

from __future__ import annotations

from unittest.mock import MagicMock

import lancedb
import numpy as np
import pytest
from typer.testing import CliRunner

from lilbee.app import services as services_mod
from lilbee.app.settings import reconcile_embedding_dim
from lilbee.core.config import META_TABLE, cfg
from lilbee.data.store import Store
from lilbee.data.store.lance_helpers import persisted_embedding_width
from lilbee.data.store.types import EmbeddingModelMismatchError
from tests.conftest import make_mock_services

MODEL = "ollama/stored-embed:latest"
CONFIGURED_WIDTH = 8
STORED_WIDTH = 16


@pytest.fixture(autouse=True)
def isolated_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path))
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


def _store_with_stored_width(
    width: int | None, chunks: bool = False, chunk_width: int = CONFIGURED_WIDTH
) -> None:
    """Write the ``_meta`` row the way the store writer does, then set its width."""
    store = Store(cfg)
    if chunks:
        cfg.embedding_dim = chunk_width
        store.add_chunks([_record("a.md", 0), _record("b.md", 1)])
    else:
        store.restamp_meta()
    table = lancedb.connect(str(cfg.lancedb_dir)).open_table(META_TABLE)
    if width is None:
        table.update(values_sql={"embedding_dim": "NULL"})
    else:
        table.update(values={"embedding_dim": width})
    assert table.to_arrow()["embedding_dim"].to_pylist() == [width]
    cfg.embedding_dim = CONFIGURED_WIDTH


def _record(source: str, axis: int) -> dict:
    vector = np.zeros(cfg.embedding_dim, dtype=np.float32)
    vector[axis] = 1.0
    return {
        "source": source,
        "content_type": "text",
        "chunk_type": "raw",
        "page_start": 0,
        "page_end": 0,
        "line_start": 0,
        "line_end": 0,
        "chunk": f"{source} body",
        "chunk_index": 0,
        "vector": vector,
    }


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
        return make_mock_services(store=Store(config))

    monkeypatch.setattr(services_mod, "build_services", build)
    monkeypatch.setattr(cfg, "worker_pool_eager_start", False)
    services_mod._state.singleton = None
    return built


def _status_on_cli() -> str:
    from lilbee.cli import app

    result = CliRunner().invoke(app, ["status"])
    assert result.exit_code == 0, result.output
    return result.output


def _status_on_mcp() -> dict | None:
    from lilbee import mcp_server

    return mcp_server.status()["index"]


async def _start_serve() -> None:
    from lilbee.server.app import _lifespan

    async with _lifespan(MagicMock()):
        pass


@pytest.mark.parametrize("stored", [0, -1, None])
def test_cli_status_reports_no_index_width_for_a_stored_width_below_1_or_null(monkeypatch, stored):
    """These cases cannot prove the start-up reconcile call: with no width it changes nothing."""
    _store_with_stored_width(stored, chunks=True)
    built = _build_recording_width(monkeypatch)

    output = _status_on_cli()

    assert built == [CONFIGURED_WIDTH]
    assert "dims" not in output
    assert "Index built with" not in output


@pytest.mark.parametrize("stored", [0, -1, None])
def test_mcp_status_reports_no_index_width_for_a_stored_width_below_1_or_null(monkeypatch, stored):
    """These cases cannot prove the start-up reconcile call: with no width it changes nothing."""
    _store_with_stored_width(stored, chunks=True)
    built = _build_recording_width(monkeypatch)

    index = _status_on_mcp()

    assert built == [CONFIGURED_WIDTH]
    assert index is None


def test_cli_status_reports_the_stored_width_when_it_is_above_0(monkeypatch):
    _store_with_stored_width(STORED_WIDTH, chunks=True)
    built = _build_recording_width(monkeypatch)

    output = _status_on_cli()

    assert built == [STORED_WIDTH]
    assert f"({STORED_WIDTH} dims)" in output


def test_mcp_status_reports_the_stored_width_when_it_is_above_0(monkeypatch):
    _store_with_stored_width(STORED_WIDTH, chunks=True)
    built = _build_recording_width(monkeypatch)

    index = _status_on_mcp()

    assert built == [STORED_WIDTH]
    assert index is not None
    assert index["embedding_dim"] == STORED_WIDTH


@pytest.mark.parametrize(
    ("stored", "ends_with"), [(0, CONFIGURED_WIDTH), (STORED_WIDTH, STORED_WIDTH)]
)
async def test_serve_starts_with_the_width_that_start_up_reconciles(monkeypatch, stored, ends_with):
    """The stored 0 case cannot prove the reconcile call: with no width it changes nothing."""
    _store_with_stored_width(stored, chunks=True)
    built = _build_recording_width(monkeypatch)

    await _start_serve()

    assert built == [ends_with]


@pytest.mark.parametrize("stored", [0, -1, None])
def test_a_search_after_a_stored_width_below_1_or_null_is_not_refused(stored):
    _store_with_stored_width(stored, chunks=True)
    query = _record("q.md", 0)["vector"]

    hits = Store(cfg).search(query, top_k=2, max_distance=0)

    assert {hit.source for hit in hits} == {"a.md", "b.md"}


def test_a_search_after_a_stored_width_that_differs_is_still_refused():
    _store_with_stored_width(STORED_WIDTH, chunks=True)

    with pytest.raises(EmbeddingModelMismatchError, match="dim 16"):
        Store(cfg).search(_record("q.md", 0)["vector"], top_k=2, max_distance=0)


@pytest.mark.parametrize("stored", [0, -1, None])
def test_a_search_with_a_damaged_width_and_chunks_of_another_width_is_refused(stored):
    _store_with_stored_width(stored, chunks=True, chunk_width=STORED_WIDTH)
    query = np.zeros(CONFIGURED_WIDTH, dtype=np.float32)

    with pytest.raises(EmbeddingModelMismatchError, match="dim 16") as refused:
        Store(cfg).search(query, top_k=2, max_distance=0)

    assert "so rebuild the index" in str(refused.value)


def test_a_search_with_no_meta_row_and_chunks_of_another_width_is_refused():
    _store_with_stored_width(STORED_WIDTH, chunks=True, chunk_width=STORED_WIDTH)
    lancedb.connect(str(cfg.lancedb_dir)).drop_table(META_TABLE)
    cfg.embedding_dim = CONFIGURED_WIDTH

    with pytest.raises(EmbeddingModelMismatchError, match="dim 16"):
        Store(cfg).search(np.zeros(CONFIGURED_WIDTH, dtype=np.float32), top_k=2, max_distance=0)


@pytest.mark.parametrize("stored", [0, -1, None])
def test_the_first_ingest_with_a_damaged_width_and_no_chunks_stamps_the_configured_width(stored):
    _store_with_stored_width(stored)

    Store(cfg).add_chunks([_record("a.md", 0)])

    meta = lancedb.connect(str(cfg.lancedb_dir)).open_table(META_TABLE).to_arrow()
    assert meta["embedding_dim"].to_pylist() == [CONFIGURED_WIDTH]
