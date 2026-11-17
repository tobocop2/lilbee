"""A forced rebuild adopts the width the embedder serves, so the rebuild the error names works."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lilbee.app.services import set_services
from lilbee.core.config import cfg
from lilbee.data.ingest import sync
from lilbee.data.store import Store
from lilbee.retrieval.embedder import Embedder
from tests.conftest import make_mock_services

OLD_WIDTH = 8
NEW_WIDTH = 16
OLD_MODEL = "ollama/old-embed:latest"
NEW_MODEL = "ollama/new-embed:latest"
TEXT = "alpha beta gamma delta " * 150


class _Engine:
    """An engine whose model answers at whatever width it is set to."""

    def __init__(self, width: int) -> None:
        self.width = width
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[0.1] * self.width for _ in texts]


@pytest.fixture(autouse=True)
def isolated_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path / "global-data"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr("lilbee.retrieval.embedder.is_model_available", lambda *_: True)
    snapshot = cfg.model_copy()
    cfg.documents_dir = tmp_path / "docs"
    cfg.documents_dir.mkdir()
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.concept_graph = False
    cfg.embedding_model = OLD_MODEL
    cfg.embedding_dim = OLD_WIDTH
    yield
    set_services(None)
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _use(store: Store, engine: _Engine) -> None:
    set_services(
        make_mock_services(
            store=store,
            embedder=Embedder(cfg, engine),
            worker_pool=MagicMock(),
            pool_runtime=MagicMock(),
            pool_health_ticker=MagicMock(),
        )
    )


async def _index_then_swap_to_a_wider_model() -> tuple[Store, _Engine]:
    for name in ("a.txt", "b.txt"):
        (cfg.documents_dir / name).write_text(f"{name} {TEXT}", encoding="utf-8")
    engine = _Engine(OLD_WIDTH)
    store = Store(cfg)
    _use(store, engine)
    await sync(quiet=True)
    assert store.count_chunks() > 0, "setup: the first sync must index the files"
    engine.width = NEW_WIDTH
    cfg.embedding_model = NEW_MODEL
    return store, engine


async def test_a_rebuild_after_a_width_change_re_embeds_at_the_width_served():
    store, _ = await _index_then_swap_to_a_wider_model()

    result = await sync(force_rebuild=True, quiet=True)

    assert result.failed == []
    assert sorted(result.added) == ["a.txt", "b.txt"]
    assert cfg.embedding_dim == NEW_WIDTH
    meta = store.get_meta()
    assert meta is not None
    assert (meta["embedding_model"], meta["embedding_dim"]) == (cfg.embedding_model, NEW_WIDTH)
    stored = store.open_table("chunks").to_arrow()["vector"][0].as_py()
    assert len(stored) == NEW_WIDTH


async def test_a_rebuild_whose_embedder_does_not_answer_stops_with_the_index_unchanged():
    for name in ("a.txt", "b.txt"):
        (cfg.documents_dir / name).write_text(f"{name} {TEXT}", encoding="utf-8")
    engine = _Engine(OLD_WIDTH)
    store = Store(cfg)
    _use(store, engine)
    await sync(quiet=True)
    chunks = store.count_chunks()
    engine.embed = MagicMock(side_effect=RuntimeError("down"))

    with pytest.raises(RuntimeError, match="Rebuild stopped and the index is unchanged"):
        await sync(force_rebuild=True, quiet=True)

    assert cfg.embedding_dim == OLD_WIDTH
    assert store.count_chunks() == chunks


async def test_a_rebuild_after_a_model_switch_is_refused_when_the_new_model_is_missing(monkeypatch):
    store, _ = await _index_then_swap_to_a_wider_model()
    chunks = store.count_chunks()
    monkeypatch.setattr("lilbee.retrieval.embedder.is_model_available", lambda *_: False)

    with pytest.raises(RuntimeError, match="Ingest needs an embedding model"):
        await sync(force_rebuild=True, quiet=True)

    assert store.count_chunks() == chunks
    assert cfg.embedding_dim == OLD_WIDTH


async def test_a_rebuild_adopts_the_width_of_one_embed_call():
    _, engine = await _index_then_swap_to_a_wider_model()
    before = engine.calls

    embedder = Embedder(cfg, engine)
    embedder.adopt_served_width("probe")

    assert engine.calls == before + 1
    assert cfg.embedding_dim == NEW_WIDTH


def test_an_embedder_that_answers_with_no_vector_is_refused():
    class _Silent:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[] for _ in texts]

    with pytest.raises(ValueError, match="no vector"):
        Embedder(cfg, _Silent()).adopt_served_width("probe")

    assert cfg.embedding_dim == OLD_WIDTH


async def test_a_same_name_rebuild_drops_the_store_and_writes_it_at_the_wider_width():
    for name in ("a.txt", "b.txt"):
        (cfg.documents_dir / name).write_text(f"{name} {TEXT}", encoding="utf-8")
    engine = _Engine(OLD_WIDTH)
    store = Store(cfg)
    _use(store, engine)
    await sync(quiet=True)
    engine.width = NEW_WIDTH

    result = await sync(force_rebuild=True, quiet=True)

    assert result.failed == []
    assert cfg.embedding_model == OLD_MODEL
    meta = store.get_meta()
    assert meta is not None
    assert (meta["embedding_model"], meta["embedding_dim"]) == (OLD_MODEL, NEW_WIDTH)
    assert len(store.open_table("chunks").to_arrow()["vector"][0].as_py()) == NEW_WIDTH


async def test_a_rebuild_stopped_at_the_memory_step_puts_the_width_back(monkeypatch):
    store, _ = await _index_then_swap_to_a_wider_model()
    monkeypatch.setattr(
        store, "rebuild_memory_embeddings", MagicMock(side_effect=RuntimeError("down"))
    )

    with pytest.raises(RuntimeError, match="Rebuild stopped and the index is unchanged"):
        await sync(force_rebuild=True, quiet=True)

    assert cfg.embedding_dim == OLD_WIDTH
    meta = store.get_meta()
    assert meta is not None
    assert meta["embedding_dim"] == OLD_WIDTH


async def test_a_new_process_reads_the_width_the_rebuild_wrote():
    from lilbee.app.settings import reconcile_embedding_dim

    store, _ = await _index_then_swap_to_a_wider_model()
    await sync(force_rebuild=True, quiet=True)
    meta = store.get_meta()
    assert meta is not None
    assert meta["embedding_dim"] == NEW_WIDTH
    cfg.embedding_dim = OLD_WIDTH  # a new process starts at the config default

    reconcile_embedding_dim()

    assert cfg.embedding_dim == NEW_WIDTH


async def test_startup_leaves_the_width_when_the_store_holds_another_model():
    from lilbee.app.settings import reconcile_embedding_dim

    store, _ = await _index_then_swap_to_a_wider_model()
    meta = store.get_meta()
    assert meta is not None
    assert meta["embedding_model"] == OLD_MODEL
    assert cfg.embedding_model == NEW_MODEL

    reconcile_embedding_dim()

    assert cfg.embedding_dim == OLD_WIDTH


def test_startup_with_no_library_leaves_the_width():
    from lilbee.app.settings import reconcile_embedding_dim

    assert not cfg.lancedb_dir.exists()

    reconcile_embedding_dim()

    assert cfg.embedding_dim == OLD_WIDTH
