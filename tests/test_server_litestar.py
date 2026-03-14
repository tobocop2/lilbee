"""Tests for the Litestar HTTP adapter."""

from dataclasses import fields, replace
from unittest import mock
from unittest.mock import AsyncMock

import pytest
from litestar.testing import TestClient

from lilbee.config import cfg


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    """Redirect config paths for all adapter tests."""
    snapshot = replace(cfg)
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir()
    cfg.data_dir = tmp_path / "data"
    cfg.data_root = tmp_path
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    yield tmp_path
    for f in fields(cfg):
        setattr(cfg, f.name, getattr(snapshot, f.name))


@pytest.fixture()
def client():
    from lilbee.server.litestar_app import create_app

    app = create_app()
    return TestClient(app)


async def mock_async_gen(*events):
    for e in events:
        yield e


# ── GET routes ───────────────────────────────────────────────────────────────


class TestHealthRoute:
    @mock.patch(
        "lilbee.server.handlers.health",
        new_callable=AsyncMock,
        return_value={"status": "ok", "version": "1.0.0"},
    )
    def test_returns_json(self, _mock, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["version"] == "1.0.0"


class TestStatusRoute:
    @mock.patch(
        "lilbee.server.handlers.status",
        new_callable=AsyncMock,
        return_value={"config": {}, "sources": [], "total_chunks": 0},
    )
    def test_returns_json(self, _mock, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        assert resp.json()["total_chunks"] == 0


class TestSearchRoute:
    @mock.patch("lilbee.server.handlers.search", new_callable=AsyncMock, return_value=[])
    def test_empty_results(self, mock_search, client):
        resp = client.get("/api/search", params={"q": "hello", "top_k": "3"})
        assert resp.status_code == 200
        assert resp.json() == []
        mock_search.assert_awaited_once_with("hello", top_k=3)

    @mock.patch(
        "lilbee.server.handlers.search",
        new_callable=AsyncMock,
        return_value=[{"source": "a.md", "chunks": []}],
    )
    def test_with_results(self, _mock, client):
        resp = client.get("/api/search", params={"q": "test"})
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    @mock.patch("lilbee.server.handlers.search", new_callable=AsyncMock, return_value=[])
    def test_default_top_k(self, mock_search, client):
        client.get("/api/search", params={"q": "x"})
        mock_search.assert_awaited_once_with("x", top_k=5)


# ── POST routes ──────────────────────────────────────────────────────────────


class TestAskRoute:
    @mock.patch(
        "lilbee.server.handlers.ask",
        new_callable=AsyncMock,
        return_value={"answer": "42", "sources": []},
    )
    def test_returns_answer(self, mock_ask, client):
        resp = client.post("/api/ask", json={"question": "meaning?"})
        assert resp.status_code == 201
        assert resp.json()["answer"] == "42"
        mock_ask.assert_awaited_once_with(question="meaning?", top_k=0)

    @mock.patch(
        "lilbee.server.handlers.ask",
        new_callable=AsyncMock,
        return_value={"answer": "yes", "sources": []},
    )
    def test_forwards_top_k(self, mock_ask, client):
        resp = client.post("/api/ask", json={"question": "q", "top_k": 10})
        assert resp.status_code == 201
        mock_ask.assert_awaited_once_with(question="q", top_k=10)


class TestAskStreamRoute:
    @mock.patch("lilbee.server.handlers.ask_stream")
    def test_returns_sse(self, mock_stream, client):
        mock_stream.return_value = mock_async_gen("event: token\ndata: {}\n\n")
        resp = client.post("/api/ask/stream", json={"question": "hi"})
        assert resp.status_code == 201
        assert "text/event-stream" in resp.headers["content-type"]
        assert b"event: token" in resp.content


class TestChatRoute:
    @mock.patch(
        "lilbee.server.handlers.chat",
        new_callable=AsyncMock,
        return_value={"answer": "reply", "sources": []},
    )
    def test_forwards_history(self, mock_chat, client):
        history = [{"role": "user", "content": "hi"}]
        resp = client.post("/api/chat", json={"question": "q", "history": history})
        assert resp.status_code == 201
        mock_chat.assert_awaited_once_with(question="q", history=history, top_k=0)

    @mock.patch(
        "lilbee.server.handlers.chat",
        new_callable=AsyncMock,
        return_value={"answer": "a", "sources": []},
    )
    def test_default_empty_history(self, mock_chat, client):
        client.post("/api/chat", json={"question": "q"})
        mock_chat.assert_awaited_once_with(question="q", history=[], top_k=0)


class TestChatStreamRoute:
    @mock.patch("lilbee.server.handlers.chat_stream")
    def test_returns_sse(self, mock_stream, client):
        mock_stream.return_value = mock_async_gen("event: done\ndata: {}\n\n")
        resp = client.post(
            "/api/chat/stream",
            json={"question": "hi", "history": []},
        )
        assert resp.status_code == 201
        assert b"event: done" in resp.content


class TestSyncRoute:
    @mock.patch("lilbee.server.handlers.sync_stream")
    def test_returns_sse(self, mock_stream, client):
        mock_stream.return_value = mock_async_gen("event: done\ndata: {}\n\n")
        resp = client.post("/api/sync")
        assert resp.status_code == 201
        assert b"event: done" in resp.content
        mock_stream.assert_called_once_with(force_vision=False)

    @mock.patch("lilbee.server.handlers.sync_stream")
    def test_force_vision(self, mock_stream, client):
        mock_stream.return_value = mock_async_gen("event: done\ndata: {}\n\n")
        client.post("/api/sync", json={"force_vision": True})
        mock_stream.assert_called_once_with(force_vision=True)


# ── Models routes ────────────────────────────────────────────────────────────


class TestModelsListRoute:
    @mock.patch(
        "lilbee.server.handlers.list_models",
        new_callable=AsyncMock,
        return_value={"chat": {}, "vision": {}},
    )
    def test_returns_json(self, _mock, client):
        resp = client.get("/api/models")
        assert resp.status_code == 200
        assert "chat" in resp.json()


class TestModelsPullRoute:
    @mock.patch("lilbee.server.handlers.pull_model")
    def test_returns_sse(self, mock_pull, client):
        mock_pull.return_value = mock_async_gen("event: done\ndata: {}\n\n")
        resp = client.post("/api/models/pull", json={"model": "llama3:8b"})
        assert resp.status_code == 201
        assert b"event: done" in resp.content


class TestModelsSetChatRoute:
    @mock.patch(
        "lilbee.server.handlers.set_chat_model",
        new_callable=AsyncMock,
        return_value={"model": "llama3:8b"},
    )
    def test_returns_model(self, mock_set, client):
        resp = client.put("/api/models/chat", json={"model": "llama3:8b"})
        assert resp.status_code == 200
        assert resp.json()["model"] == "llama3:8b"


class TestModelsSetVisionRoute:
    @mock.patch(
        "lilbee.server.handlers.set_vision_model",
        new_callable=AsyncMock,
        return_value={"model": "llava:13b"},
    )
    def test_returns_model(self, mock_set, client):
        resp = client.put("/api/models/vision", json={"model": "llava:13b"})
        assert resp.status_code == 200
        assert resp.json()["model"] == "llava:13b"


# ── CORS ─────────────────────────────────────────────────────────────────────


class TestCors:
    @mock.patch(
        "lilbee.server.handlers.health",
        new_callable=AsyncMock,
        return_value={"status": "ok", "version": "1.0.0"},
    )
    def test_obsidian_origin_allowed(self, _mock, client):
        resp = client.options(
            "/api/health",
            headers={
                "Origin": "app://obsidian.md",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers.get("access-control-allow-origin") == "app://obsidian.md"

    @mock.patch(
        "lilbee.server.handlers.health",
        new_callable=AsyncMock,
        return_value={"status": "ok", "version": "1.0.0"},
    )
    def test_localhost_origin_allowed(self, _mock, client):
        resp = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:7433",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:7433"


# ── __init__.py re-export ────────────────────────────────────────────────────


class TestCreateAppReexport:
    @mock.patch("lilbee.server.litestar_app.create_app")
    def test_lazy_import(self, mock_create):
        from lilbee.server import create_app

        create_app()
        mock_create.assert_called_once()
