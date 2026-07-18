"""Tests for the HTTP session routes (list, get, rename, delete)."""

from __future__ import annotations

import pytest
from litestar.testing import TestClient

from lilbee.app import services as svc_mod
from lilbee.core.config import cfg
from lilbee.server.auth import is_read_only
from lilbee.server.routes.sessions import (
    session_add_message_route,
    session_claim_route,
    session_create_route,
    session_delete_route,
    session_get_route,
    session_rename_route,
    session_set_summary_route,
    sessions_list_route,
)
from lilbee.sessions import MessageRole, SessionMessage, SessionOrigin, TitleSource
from tests.conftest import make_mock_services


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture
def store():
    services = make_mock_services()
    svc_mod.set_services(services)
    yield services.session_store
    svc_mod.set_services(None)


@pytest.fixture
def client(store):
    import lilbee.server.auth as auth_mod
    from lilbee.server.app import create_app

    auth_mod.session_manager.disable()
    yield TestClient(create_app())
    auth_mod.session_manager.cleanup()


def _seed(store, origin: SessionOrigin = SessionOrigin.TUI) -> str:
    session_id = store.create(model_ref="gpt-oss-20b", scope="both", origin=origin)
    store.set_title(session_id, "Torque specs", TitleSource.AUTO)
    store.add_message(session_id, SessionMessage(role=MessageRole.USER, content="what specs?"))
    store.add_message(
        session_id,
        SessionMessage(role=MessageRole.ASSISTANT, content="85 Nm.", sources=("manual.pdf",)),
    )
    return session_id


class TestList:
    def test_empty(self, client):
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        assert resp.json() == {"sessions": []}

    def test_lists_metadata(self, client, store):
        session_id = _seed(store)
        body = client.get("/api/sessions").json()
        entry = next(s for s in body["sessions"] if s["id"] == session_id)
        assert entry["title"] == "Torque specs"
        assert entry["message_count"] == 2
        assert entry["model_ref"] == "gpt-oss-20b"


class TestGet:
    def test_returns_transcript(self, client, store):
        session_id = _seed(store)
        body = client.get(f"/api/sessions/{session_id}").json()
        assert body["meta"]["title"] == "Torque specs"
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][1]["role"] == "assistant"
        assert body["messages"][1]["sources"] == ["manual.pdf"]

    def test_unknown_id_404(self, client):
        assert client.get("/api/sessions/nope").status_code == 404


class TestGetSummary:
    def test_summary_is_on_the_wire(self, client, store):
        """A resumed client needs what compaction folded the old turns into."""
        session_id = _seed(store)
        store.set_summary(session_id, "earlier: torque is 85 Nm")
        body = client.get(f"/api/sessions/{session_id}").json()
        assert body["summary"] == "earlier: torque is 85 Nm"

    def test_summary_defaults_empty(self, client, store):
        session_id = _seed(store)
        assert client.get(f"/api/sessions/{session_id}").json()["summary"] == ""


class TestCreate:
    def test_creates_and_returns_detail(self, client, store):
        resp = client.post("/api/sessions", json={"model_ref": "qwen3-4b", "scope": "both"})
        assert resp.status_code == 201
        body = resp.json()
        assert body["meta"]["model_ref"] == "qwen3-4b"
        assert body["messages"] == []
        # the new session is now listable
        listed = client.get("/api/sessions").json()["sessions"]
        assert any(s["id"] == body["meta"]["id"] for s in listed)


class TestAppendMessage:
    def test_appends_a_turn(self, client, store):
        # HTTP-created, so the HTTP surface owns it (a TUI session would 409;
        # see TestOwnership).
        session_id = client.post("/api/sessions", json={"model_ref": "m", "scope": "both"}).json()[
            "meta"
        ]["id"]
        resp = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"role": "user", "content": "and the head bolts?", "sources": []},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["messages"][-1]["content"] == "and the head bolts?"
        assert body["meta"]["message_count"] == 1

    def test_unknown_id_404(self, client):
        resp = client.post(
            "/api/sessions/nope/messages", json={"role": "user", "content": "x", "sources": []}
        )
        assert resp.status_code == 404


class TestSetSummary:
    def test_sets_summary(self, client, store):
        session_id = _seed(store)
        resp = client.put(f"/api/sessions/{session_id}/summary", json={"summary": "folded: 85 Nm"})
        assert resp.status_code == 200
        assert client.get(f"/api/sessions/{session_id}").json()["summary"] == "folded: 85 Nm"

    def test_unknown_id_404(self, client):
        assert client.put("/api/sessions/nope/summary", json={"summary": "x"}).status_code == 404


class TestRename:
    def test_renames(self, client, store):
        session_id = _seed(store)
        resp = client.patch(f"/api/sessions/{session_id}", json={"title": "Renamed"})
        assert resp.status_code == 200
        assert resp.json() == {"id": session_id, "title": "Renamed"}
        assert client.get(f"/api/sessions/{session_id}").json()["meta"]["title"] == "Renamed"

    def test_unknown_id_404(self, client):
        assert client.patch("/api/sessions/nope", json={"title": "x"}).status_code == 404


class TestDelete:
    def test_deletes(self, client, store):
        session_id = _seed(store)
        resp = client.delete(f"/api/sessions/{session_id}")
        assert resp.status_code == 200
        assert resp.json() == {"id": session_id, "deleted": True}
        assert client.get("/api/sessions").json() == {"sessions": []}

    def test_unknown_id_404(self, client):
        assert client.delete("/api/sessions/nope").status_code == 404


def test_reads_are_read_only_and_writes_are_not():
    assert is_read_only(sessions_list_route.fn)
    assert is_read_only(session_get_route.fn)
    assert not is_read_only(session_rename_route.fn)
    assert not is_read_only(session_delete_route.fn)
    # A read-only token must not be able to create, append, or summarize.
    assert not is_read_only(session_create_route.fn)
    assert not is_read_only(session_add_message_route.fn)
    assert not is_read_only(session_set_summary_route.fn)
    # The takeover operation above all: a read-only token must never claim.
    assert not is_read_only(session_claim_route.fn)


class TestOwnership:
    def test_appending_to_a_tui_session_succeeds(self, client, store):
        """TUI, HTTP, and CLI are one conversation space: no claim dance
        between the plugin and the terminal."""
        session_id = _seed(store)
        resp = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"role": "user", "content": "from obsidian", "sources": []},
        )
        assert resp.status_code == 201
        assert store.get(session_id).meta.origin is SessionOrigin.TUI, "no transfer needed"

    def test_appending_to_an_agent_session_is_409(self, client, store):
        """An HTTP client must not splice into an agent's working state."""
        session_id = _seed(store, origin=SessionOrigin.MCP)
        resp = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"role": "user", "content": "spliced", "sources": []},
        )
        assert resp.status_code == 409
        assert "claim" in resp.json()["detail"].lower()

    def test_claim_brings_an_agent_session_back_to_human_space(self, client, store):
        session_id = _seed(store, origin=SessionOrigin.MCP)
        assert client.post(f"/api/sessions/{session_id}/claim").status_code == 201
        resp = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"role": "user", "content": "mine now", "sources": []},
        )
        assert resp.status_code == 201
        assert any(s["id"] == session_id for s in client.get("/api/sessions").json()["sessions"])

    def test_agent_sessions_are_absent_from_the_list(self, client, store):
        """Agent working state stays out of the human surfaces' lists."""
        mine = _seed(store)
        _seed(store, origin=SessionOrigin.MCP)
        sessions = client.get("/api/sessions").json()["sessions"]
        assert [s["id"] for s in sessions] == [mine]

    def test_http_created_sessions_append_without_claiming(self, client, store):
        created = client.post("/api/sessions", json={"model_ref": "m", "scope": "both"}).json()
        resp = client.post(
            f"/api/sessions/{created['meta']['id']}/messages",
            json={"role": "user", "content": "q", "sources": []},
        )
        assert resp.status_code == 201

    def test_claim_unknown_404(self, client):
        assert client.post("/api/sessions/nope/claim").status_code == 404


class TestSessionsDisabled:
    """Every session route answers 404 when the toggle is off, matching how
    the wiki and memory routes refuse a disabled feature.
    """

    def test_every_route_404s(self, client, store):
        session_id = _seed(store)
        cfg.sessions_enabled = False
        routes = {
            "list": lambda: client.get("/api/sessions"),
            "get": lambda: client.get(f"/api/sessions/{session_id}"),
            "create": lambda: client.post(
                "/api/sessions", json={"model_ref": "m", "scope": "both"}
            ),
            "append": lambda: client.post(
                f"/api/sessions/{session_id}/messages",
                json={"role": "user", "content": "q", "sources": []},
            ),
            "claim": lambda: client.post(f"/api/sessions/{session_id}/claim"),
            "summary": lambda: client.put(
                f"/api/sessions/{session_id}/summary", json={"summary": "s"}
            ),
            "rename": lambda: client.patch(f"/api/sessions/{session_id}", json={"title": "t"}),
            "delete": lambda: client.delete(f"/api/sessions/{session_id}"),
        }
        for name, call in routes.items():
            assert call().status_code == 404, f"{name} did not 404"

    def test_disabled_routes_write_nothing(self, client, store):
        session_id = _seed(store)
        before = len(store.get(session_id).messages)
        cfg.sessions_enabled = False
        client.post(
            f"/api/sessions/{session_id}/messages",
            json={"role": "user", "content": "should not land", "sources": []},
        )
        client.delete(f"/api/sessions/{session_id}")
        cfg.sessions_enabled = True
        assert len(store.get(session_id).messages) == before
