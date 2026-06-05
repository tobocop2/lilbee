"""Tests for the HTTP memory routes (list, remember, update flags, forget)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from litestar.testing import TestClient

from lilbee.app import services as svc_mod
from lilbee.core.config import cfg
from lilbee.data.store import LOCAL_OWNER, MemoryKind, MemoryRow, MemorySource
from tests.conftest import make_mock_services


def _row(text: str, *, shared: bool = False) -> MemoryRow:
    return MemoryRow(
        id="abc123",
        owner=LOCAL_OWNER,
        shared=shared,
        kind=MemoryKind.FACT,
        source=MemorySource.MANUAL,
        text=text,
        vector=[0.1],
        created_at="t",
        updated_at="t",
    )


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.memory_enabled = True
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture
def store():
    store = MagicMock()
    store.add_memory.return_value = "newid"
    store.get_memories.return_value = []
    store.update_memory.return_value = True
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 768
    svc_mod.set_services(make_mock_services(store=store, embedder=embedder))
    yield store
    svc_mod.set_services(None)


@pytest.fixture
def client(store):
    import lilbee.server.auth as auth_mod
    from lilbee.server.app import create_app

    auth_mod.session_manager.token = None
    yield TestClient(create_app())
    auth_mod.session_manager.token = None


class TestList:
    def test_lists_memories(self, client, store):
        store.get_memories.return_value = [_row("uses rust", shared=True)]
        resp = client.get("/api/memories")
        assert resp.status_code == 200
        body = resp.json()
        assert body["memories"][0]["text"] == "uses rust"
        assert body["memories"][0]["shared"] is True

    def test_disabled_returns_404(self, client):
        cfg.memory_enabled = False
        assert client.get("/api/memories").status_code == 404


class TestRemember:
    def test_stores_fact(self, client, store):
        resp = client.post("/api/memories", json={"text": "uses rust"})
        assert resp.status_code == 201
        assert resp.json() == {"id": "newid", "kind": "fact"}
        record = store.add_memory.call_args.args[0]
        assert record.kind is MemoryKind.FACT

    def test_stores_preference_shared(self, client, store):
        resp = client.post(
            "/api/memories", json={"text": "be terse", "kind": "preference", "shared": True}
        )
        assert resp.status_code == 201
        record = store.add_memory.call_args.args[0]
        assert record.kind is MemoryKind.PREFERENCE
        assert record.shared is True

    def test_disabled_returns_404(self, client, store):
        cfg.memory_enabled = False
        resp = client.post("/api/memories", json={"text": "x"})
        assert resp.status_code == 404
        store.add_memory.assert_not_called()

    def test_bad_kind_rejected_with_400(self, client):
        resp = client.post("/api/memories", json={"text": "x", "kind": "bogus"})
        assert resp.status_code == 400


class TestUpdateShared:
    def test_patch_sets_shared(self, client, store):
        resp = client.patch("/api/memories/abc123", json={"shared": True})
        assert resp.status_code == 200
        assert resp.json() == {"id": "abc123", "updated": True}
        store.update_memory.assert_called_once_with("abc123", shared=True)

    def test_patch_unknown_id_reports_not_updated(self, client, store):
        store.update_memory.return_value = False
        resp = client.patch("/api/memories/missing", json={"shared": True})
        assert resp.json()["updated"] is False


class TestRemove:
    def test_delete_removes_memory(self, client, store):
        resp = client.delete("/api/memories/abc123")
        assert resp.status_code == 200
        assert resp.json() == {"removed": "abc123"}
        store.delete_memory.assert_called_once_with("abc123")

    def test_disabled_returns_404(self, client, store):
        cfg.memory_enabled = False
        assert client.delete("/api/memories/abc123").status_code == 404
        store.delete_memory.assert_not_called()
