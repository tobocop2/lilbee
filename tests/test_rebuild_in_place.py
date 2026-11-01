"""A forced rebuild never loses the index: it re-ingests in place, or checks the embedder first."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lilbee.app.services import set_services
from lilbee.core.config import cfg
from lilbee.data.ingest import sync
from lilbee.data.store import Store
from lilbee.providers.base import ProviderError
from tests.conftest import make_mock_services

DIM = 8
TEXT = "alpha beta gamma delta " * 150


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
    cfg.embedding_dim = DIM
    yield
    set_services(None)
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.validate_model.return_value = True
    embedder.embedding_available.return_value = True
    embedder.truncated_total = 0
    embedder.embed_batch.side_effect = lambda texts, **kw: [[0.1] * DIM for _ in texts]
    return embedder


def _use(store: Store, embedder: MagicMock) -> None:
    set_services(
        make_mock_services(
            store=store,
            embedder=embedder,
            worker_pool=MagicMock(),
            pool_runtime=MagicMock(),
            pool_health_ticker=MagicMock(),
        )
    )


async def _indexed(embedder: MagicMock, names=("a.txt", "b.txt", "c.txt")) -> Store:
    for name in names:
        (cfg.documents_dir / name).write_text(f"{name} {TEXT}")
    store = Store(cfg)
    _use(store, embedder)
    await sync(quiet=True)
    assert store.count_chunks() > 0, "setup: the first sync must index the files"
    return store


def _chunks_of(store: Store, name: str) -> int:
    table = store.open_table("chunks")
    ids = store.source_ids([name], store_source_type())
    return table.count_rows(f"source_id = '{ids[name]}'")


def store_source_type():
    from lilbee.data.store import SourceType

    return SourceType.DOCUMENT


async def test_refused_rebuild_without_an_embedding_model_keeps_the_index():
    embedder = _embedder()
    store = await _indexed(embedder)
    chunks = store.count_chunks()
    embedder.validate_model.return_value = False
    with pytest.raises(RuntimeError, match="embedding model"):
        await sync(force_rebuild=True, quiet=True)
    assert store.count_chunks() == chunks


async def test_rebuild_where_every_embed_fails_keeps_the_index_and_reports_the_files():
    embedder = _embedder()
    store = await _indexed(embedder)
    chunks = store.count_chunks()
    embedder.embed_batch.side_effect = ProviderError(
        "No embed model server is running.", provider="fleet"
    )
    result = await sync(force_rebuild=True, quiet=True)
    assert store.count_chunks() == chunks
    assert sorted(result.failed) == ["a.txt", "b.txt", "c.txt"]


async def test_rebuild_with_a_new_chunk_size_replaces_the_chunks_in_place():
    embedder = _embedder()
    store = await _indexed(embedder)
    before = _chunks_of(store, "a.txt")
    cfg.chunk_size = 64
    cfg.chunk_overlap = 8
    result = await sync(force_rebuild=True, quiet=True)
    assert sorted(result.added) == ["a.txt", "b.txt", "c.txt"]
    assert _chunks_of(store, "a.txt") > before


async def test_rebuild_removes_the_source_of_a_deleted_file():
    embedder = _embedder()
    store = await _indexed(embedder)
    (cfg.documents_dir / "b.txt").unlink()
    result = await sync(force_rebuild=True, quiet=True)
    assert result.removed == ["b.txt"]
    assert sorted(s["filename"] for s in store.get_sources()) == ["a.txt", "c.txt"]
    assert store.source_ids(["b.txt"], store_source_type()) == {}


async def test_rebuild_does_not_reuse_the_chunks_of_a_renamed_file():
    embedder = _embedder()
    store = await _indexed(embedder, names=("a.txt",))
    cfg.chunk_size = 64
    cfg.chunk_overlap = 8
    (cfg.documents_dir / "a.txt").rename(cfg.documents_dir / "z.txt")
    embedder.embed_batch.reset_mock()
    result = await sync(force_rebuild=True, quiet=True)
    assert result.added == ["z.txt"]
    assert result.relocated == []
    assert embedder.embed_batch.called
    assert [s["filename"] for s in store.get_sources()] == ["z.txt"]


async def test_rebuild_keeps_the_old_chunks_of_the_one_file_that_fails_to_embed():
    embedder = _embedder()
    store = await _indexed(embedder)
    kept = _chunks_of(store, "b.txt")
    cfg.chunk_size = 64
    cfg.chunk_overlap = 8

    def embed(texts, **kw):
        if any(text.startswith("b.txt") for text in texts):
            raise ProviderError("embed failed", provider="fleet")
        return [[0.1] * DIM for _ in texts]

    embedder.embed_batch.side_effect = embed
    result = await sync(force_rebuild=True, quiet=True)
    assert result.failed == ["b.txt"]
    assert _chunks_of(store, "b.txt") == kept
    assert _chunks_of(store, "a.txt") > kept


async def test_rebuild_with_a_changed_embedder_and_a_dead_probe_keeps_the_index():
    embedder = _embedder()
    store = await _indexed(embedder)
    chunks = store.count_chunks()
    cfg.embedding_model = "other/embedder-GGUF"
    cfg.embedding_dim = DIM + 4
    assert store.index_mismatch() is not None
    embedder.embed_batch.side_effect = ProviderError("down", provider="fleet")
    with pytest.raises(RuntimeError, match="embedding model did not answer"):
        await sync(force_rebuild=True, quiet=True)
    assert store.count_chunks() == chunks


async def test_rebuild_with_a_changed_embedder_drops_and_reingests_when_the_probe_answers():
    embedder = _embedder()
    store = await _indexed(embedder)
    cfg.embedding_model = "other/embedder-GGUF"
    cfg.embedding_dim = DIM + 4
    embedder.embed_batch.side_effect = lambda texts, **kw: [[0.2] * (DIM + 4) for _ in texts]
    result = await sync(force_rebuild=True, quiet=True)
    assert sorted(result.added) == ["a.txt", "b.txt", "c.txt"]
    assert store.index_mismatch() is None
    assert store.count_chunks() > 0


async def test_clean_rebuild_stamps_the_store_with_the_current_schema():
    embedder = _embedder()
    store = await _indexed(embedder)
    stamped = MagicMock()
    store.restamp_meta = stamped
    await sync(force_rebuild=True, quiet=True)
    stamped.assert_called_once_with()


async def test_rebuild_with_a_failed_file_leaves_the_schema_stamp_alone():
    embedder = _embedder()
    store = await _indexed(embedder)
    stamped = MagicMock()
    store.restamp_meta = stamped
    embedder.embed_batch.side_effect = ProviderError("down", provider="fleet")
    await sync(force_rebuild=True, quiet=True)
    stamped.assert_not_called()
