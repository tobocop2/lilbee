"""Tests for the HTTP session routes (list, get, rename, delete)."""

from __future__ import annotations

import re
from urllib.parse import unquote

import pytest
from litestar.testing import TestClient

from lilbee.app import services as svc_mod
from lilbee.app.session_export import default_export_name, session_markdown
from lilbee.core.config import cfg
from lilbee.server.auth import authenticates_itself
from lilbee.server.handlers import sessions as sessions_handlers
from lilbee.server.routes.sessions import (
    session_add_message_route,
    session_claim_route,
    session_create_route,
    session_delete_route,
    session_fork_route,
    session_get_route,
    session_markdown_route,
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


# RFC 6266 quoted filename (printable ASCII, no quote or backslash), then the
# RFC 5987 ext-value, whose attr-chars leave only "%XX" to decode.
_DISPOSITION_RE = re.compile(
    r"attachment; filename=\"([ !#-\[\]-~]*)\"; filename\*=UTF-8''([A-Za-z0-9!#$&+\-.^_`|~%]+)"
)


def _disposition_names(header: str) -> tuple[str, str]:
    """The (ASCII fallback, decoded UTF-8) file names an attachment header carries."""
    match = _DISPOSITION_RE.fullmatch(header)
    assert match, header
    return match[1], unquote(match[2], errors="strict")


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


class TestMarkdown:
    def test_returns_the_session_as_markdown(self, client, store):
        session_id = _seed(store)
        resp = client.get(f"/api/sessions/{session_id}/markdown")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/markdown; charset=utf-8"
        assert resp.text == session_markdown(store.get(session_id))

    def test_unknown_id_404(self, client):
        assert client.get("/api/sessions/nope/markdown").status_code == 404

    def test_names_the_download_with_the_export_file_name(self, client, store):
        session_id = _seed(store)
        resp = client.get(f"/api/sessions/{session_id}/markdown")
        name = f"torque-specs-{session_id[:8]}.md"
        assert resp.headers["content-disposition"] == (
            f"attachment; filename=\"{name}\"; filename*=UTF-8''{name}"
        )

    @pytest.mark.parametrize(
        "title",
        [
            "Bremsbeläge 制动",
            'say "hi" \\ there',
            "tab\tnew\nline\x85next\ufeffbom\x1c\x1d\x1e\x1fend",
            "\x85\ufeff",
        ],
    )
    def test_any_title_gives_a_header_that_names_the_cli_file(self, client, store, title):
        session_id = _seed(store)
        store.set_title(session_id, title, TitleSource.CUSTOM)
        header = client.get(f"/api/sessions/{session_id}/markdown").headers["content-disposition"]
        assert header.isascii()
        assert _disposition_names(header) == (default_export_name(store.get(session_id).meta),) * 2

    def test_a_name_outside_ascii_keeps_an_ascii_fallback(self, client, store, monkeypatch):
        name = 'Bremsbeläge "制动"\\\x85\ufeff\x1c\x7f.md'
        monkeypatch.setattr(sessions_handlers, "default_export_name", lambda meta: name)
        session_id = _seed(store)
        header = client.get(f"/api/sessions/{session_id}/markdown").headers["content-disposition"]
        assert header.isascii()
        fallback, encoded = _disposition_names(header)
        assert encoded == name
        assert fallback.isprintable() and fallback.isascii()
        assert len(fallback) == len(name)

    def test_a_browser_client_can_read_the_file_name(self, client, store):
        session_id = _seed(store)
        resp = client.get(
            f"/api/sessions/{session_id}/markdown", headers={"Origin": "app://obsidian.md"}
        )
        exposed = resp.headers["access-control-expose-headers"].lower().split(", ")
        assert "content-disposition" in exposed

    def test_reads_any_session_the_get_route_reads(self, client, store):
        session_id = _seed(store, origin=SessionOrigin.MCP)
        assert client.get(f"/api/sessions/{session_id}").status_code == 200
        assert client.get(f"/api/sessions/{session_id}/markdown").status_code == 200


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


class TestFork:
    def test_fork_without_a_body_copies_the_whole_conversation(self, client, store):
        session_id = _seed(store)
        store.set_summary(session_id, "notes")
        resp = client.post(f"/api/sessions/{session_id}/fork")
        assert resp.status_code == 201
        body = resp.json()
        assert body["meta"]["id"] != session_id
        assert body["meta"]["forked_from"] == session_id
        assert body["meta"]["title"] == "Torque specs (fork 1)"
        assert body["meta"]["origin"] == "http"
        assert [m["content"] for m in body["messages"]] == ["what specs?", "85 Nm."]
        assert body["summary"] == "notes"

    @pytest.mark.parametrize("message_count", [0, 1])
    def test_fork_copies_the_leading_messages(self, client, store, message_count):
        session_id = _seed(store)
        resp = client.post(
            f"/api/sessions/{session_id}/fork", json={"message_count": message_count}
        )
        assert resp.status_code == 201
        assert resp.json()["meta"]["message_count"] == message_count

    def test_fork_with_a_null_count_copies_everything(self, client, store):
        session_id = _seed(store)
        resp = client.post(f"/api/sessions/{session_id}/fork", json={"message_count": None})
        assert resp.json()["meta"]["message_count"] == 2

    def test_fork_is_listed_first(self, client, store):
        session_id = _seed(store)
        fork_id = client.post(f"/api/sessions/{session_id}/fork").json()["meta"]["id"]
        assert client.get("/api/sessions").json()["sessions"][0]["id"] == fork_id

    @pytest.mark.parametrize("message_count", [-1, 3])
    def test_fork_outside_the_transcript_is_422(self, client, store, message_count):
        session_id = _seed(store)
        resp = client.post(
            f"/api/sessions/{session_id}/fork", json={"message_count": message_count}
        )
        assert resp.status_code == 422
        assert "0 to 2" in resp.json()["detail"]
        assert len(store.list()) == 1

    @pytest.mark.parametrize("message_count", [True, "2", 1.0])
    def test_fork_count_must_be_an_integer(self, client, store, message_count):
        session_id = _seed(store)
        resp = client.post(
            f"/api/sessions/{session_id}/fork", json={"message_count": message_count}
        )
        assert resp.status_code == 400
        assert len(store.list()) == 1

    def test_fork_unknown_id_404(self, client):
        assert client.post("/api/sessions/nope/fork").status_code == 404

    def test_forking_an_agent_session_is_409(self, client, store):
        session_id = _seed(store, origin=SessionOrigin.MCP)
        assert client.post(f"/api/sessions/{session_id}/fork").status_code == 409
        assert len(store.list()) == 1

    def test_appending_to_the_fork_leaves_the_source_unchanged(self, client, store):
        session_id = _seed(store)
        fork_id = client.post(f"/api/sessions/{session_id}/fork").json()["meta"]["id"]
        client.post(
            f"/api/sessions/{fork_id}/messages",
            json={"role": "user", "content": "Q3", "sources": []},
        )
        assert store.get(session_id).meta.message_count == 2
        assert store.get(fork_id).meta.message_count == 3

    def test_every_session_carries_forked_from(self, client, store):
        _seed(store)
        assert client.get("/api/sessions").json()["sessions"][0]["forked_from"] == ""


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


def test_every_session_route_requires_the_token():
    """Reads included: the two GETs used to serve full chat transcripts to any
    caller that could reach the port."""
    assert not authenticates_itself(sessions_list_route.fn)
    assert not authenticates_itself(session_get_route.fn)
    assert not authenticates_itself(session_rename_route.fn)
    assert not authenticates_itself(session_delete_route.fn)
    # A read-only token must not be able to create, append, or summarize.
    assert not authenticates_itself(session_create_route.fn)
    assert not authenticates_itself(session_add_message_route.fn)
    assert not authenticates_itself(session_set_summary_route.fn)
    # The takeover operation above all: a read-only token must never claim.
    assert not authenticates_itself(session_claim_route.fn)
    assert not authenticates_itself(session_fork_route.fn)
    assert not authenticates_itself(session_markdown_route.fn)


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


_DISABLED_ROUTES = {
    "list": lambda client, sid: client.get("/api/sessions"),
    "get": lambda client, sid: client.get(f"/api/sessions/{sid}"),
    "create": lambda client, sid: client.post(
        "/api/sessions", json={"model_ref": "m", "scope": "both"}
    ),
    "append": lambda client, sid: client.post(
        f"/api/sessions/{sid}/messages",
        json={"role": "user", "content": "q", "sources": []},
    ),
    "claim": lambda client, sid: client.post(f"/api/sessions/{sid}/claim"),
    "summary": lambda client, sid: client.put(
        f"/api/sessions/{sid}/summary", json={"summary": "s"}
    ),
    "rename": lambda client, sid: client.patch(f"/api/sessions/{sid}", json={"title": "t"}),
    "delete": lambda client, sid: client.delete(f"/api/sessions/{sid}"),
    "fork": lambda client, sid: client.post(f"/api/sessions/{sid}/fork"),
    "markdown": lambda client, sid: client.get(f"/api/sessions/{sid}/markdown"),
}


class TestSessionsDisabled:
    """Every session route answers 404 when the toggle is off, matching how
    the wiki and memory routes refuse a disabled feature.
    """

    @pytest.mark.parametrize("route", sorted(_DISABLED_ROUTES), ids=sorted(_DISABLED_ROUTES))
    def test_route_404s(self, client, store, route):
        session_id = _seed(store)
        cfg.sessions_enabled = False
        assert _DISABLED_ROUTES[route](client, session_id).status_code == 404

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


class TestSessionVanishesMidRequest:
    """Handlers mutate then re-read to build the response, and the TUI and
    HTTP surfaces share one store, so a delete landing between the two used to
    escape as a 500."""

    def test_a_delete_between_the_mutation_and_the_read_is_a_404(self, client, store):
        session_id = store.create(model_ref=None, scope=None, origin=SessionOrigin.HTTP)
        real_add = store.add_message

        def add_then_vanish(*args, **kwargs):
            real_add(*args, **kwargs)
            store.delete(session_id)

        store.add_message = add_then_vanish
        resp = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"role": "user", "content": "hi"},
        )
        assert resp.status_code == 404
