"""A POST route answers 201 only when it adds one item whose id is in the response."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from litestar.testing import TestClient

from lilbee.app import services as svc_mod
from lilbee.core.config import cfg
from lilbee.server import auth as auth_mod
from lilbee.server.app import create_app
from lilbee.sessions import MessageRole, SessionMessage, SessionOrigin, TitleSource
from tests.conftest import make_mock_services

CREATED = 201
OK = 200
ACCEPTED = 202

# Routes that add one item and answer with its id.
CREATES_AN_ITEM = {
    "/api/memories",
    "/api/presets",
    "/api/sessions",
    "/api/sessions/{session_id:str}/fork",
}

# Routes whose status code is still an open decision; they stay at the base value.
UNDECIDED = {
    "/api/sessions/{session_id:str}/messages",
    "/api/wiki/build",
}

# Routes that set their code on the decorator and answer something other than 201.
EXPLICIT = {
    "/api/add/cancel": OK,
    "/api/analyze": OK,
    "/api/analyze/dismiss": OK,
    "/api/config/reset": OK,
    "/api/engines/install": OK,
    "/api/placement/preview": OK,
    "/api/profiles": OK,
    "/api/profiles/discard": OK,
    "/api/profiles/import": OK,
    "/api/profiles/new": OK,
    "/api/profiles/validate": OK,
    "/api/profiles/{name:str}/apply": OK,
    "/api/profiles/{name:str}/duplicate": OK,
    "/api/shutdown": ACCEPTED,
    "/v1/chat/completions": OK,
    "/v1/messages": OK,
    "/v1/messages/count_tokens": OK,
}

# Routes that stream, run a query, or act on an existing item.
ANSWERS_OK = {
    "/api/add",
    "/api/add/upload",
    "/api/ask",
    "/api/ask/stream",
    "/api/chat",
    "/api/chat/stream",
    "/api/crawl",
    "/api/documents/remove",
    "/api/import",
    "/api/models/pull",
    "/api/models/show",
    "/api/sessions/{session_id:str}/claim",
    "/api/sync",
    "/api/wiki/drafts/accept/{slug:path}",
    "/api/wiki/generate/{slug:path}",
    "/api/wiki/index",
    "/api/wiki/lint",
    "/api/wiki/prune",
    "/api/wiki/synthesize",
    "/setup/crawler",
}

EXPECTED = (
    dict.fromkeys(CREATES_AN_ITEM | UNDECIDED, CREATED) | dict.fromkeys(ANSWERS_OK, OK) | EXPLICIT
)


def _declared_post_codes(app) -> dict[str, int]:
    """The status code each POST route declares, keyed by route path."""
    found: dict[str, int] = {}
    for route in app.routes:
        for handler in getattr(route, "route_handlers", []):
            if "POST" in handler.http_methods:
                found[route.path] = handler.status_code
    return found


async def _events(*frames: str) -> AsyncGenerator[str, None]:
    for frame in frames:
        yield frame


@pytest.fixture()
def client(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir()
    cfg.wiki = True
    cfg.memory_enabled = True
    store = MagicMock()
    store.add_memory.return_value = "new-id"
    embedder = MagicMock()
    embedder.embed.return_value = np.full(768, 0.1, dtype=np.float32)
    svc_mod.set_services(make_mock_services(store=store, embedder=embedder))
    with TestClient(app=create_app()) as c:
        c.headers["Authorization"] = f"Bearer {auth_mod.session_manager.token}"
        yield c
    auth_mod.session_manager.cleanup()
    svc_mod.set_services(None)
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


class TestCensus:
    def test_every_post_route_declares_the_ruled_status(self, client):
        declared = _declared_post_codes(client.app)
        assert declared, "the census found no POST route"
        assert declared == EXPECTED

    def test_only_the_routes_that_add_an_item_answer_201(self, client):
        declared = _declared_post_codes(client.app)
        created = {path for path, code in declared.items() if code == CREATED}
        assert created == CREATES_AN_ITEM | UNDECIDED

    def test_openapi_lists_the_ruled_status_for_each_post(self, client):
        schema = client.app.openapi_schema.to_schema()
        checked = 0
        for path, code in EXPECTED.items():
            if path in UNDECIDED:
                continue
            key = path.replace(":str", "").replace(":path", "")
            responses = schema["paths"][key]["post"]["responses"]
            assert sorted(c for c in responses if c.startswith("2")) == [str(code)], path
            checked += 1
        assert checked == len(EXPECTED) - len(UNDECIDED)


class TestStreamingRoutesAnswer200:
    @pytest.mark.parametrize(
        ("path", "body", "target"),
        [
            ("/api/sync", None, "lilbee.server.handlers.sync_stream"),
            ("/api/ask/stream", {"question": "hi"}, "lilbee.server.handlers.ask_stream"),
            (
                "/api/chat/stream",
                {"question": "hi", "history": []},
                "lilbee.server.handlers.chat_stream",
            ),
            ("/api/models/pull", {"model": "m"}, "lilbee.server.handlers.models_pull"),
            ("/api/wiki/synthesize", None, "lilbee.server.handlers.wiki_synthesize_stream"),
        ],
    )
    def test_a_stream_answers_200(self, client, path, body, target):
        with (
            mock.patch(target, return_value=_events("event: done\ndata: {}\n\n")),
            mock.patch("lilbee.server.handlers.enforce_pull_arch_compat", new_callable=AsyncMock),
        ):
            resp = client.post(path, json=body) if body else client.post(path)
        assert resp.status_code == OK
        assert b"event: done" in resp.content

    def test_add_answers_200_when_the_stream_opens(self, client):
        with (
            mock.patch(
                "lilbee.server.handlers.add_files_stream",
                return_value=_events("event: done\ndata: {}\n\n"),
            ),
            mock.patch(
                "lilbee.server.handlers.validate_add_paths",
                return_value=([], False, None, None),
            ),
        ):
            resp = client.post("/api/add", json={"paths": ["a.md"]})
        assert resp.status_code == OK
        assert b"event: done" in resp.content

    def test_import_answers_200_when_the_stream_opens(self, client):
        with mock.patch(
            "lilbee.server.handlers.import_stream",
            return_value=_events("event: done\ndata: {}\n\n"),
        ):
            resp = client.post("/api/import", params={"format": "jsonl"}, content=b"{}")
        assert resp.status_code == OK
        assert b"event: done" in resp.content

    def test_setup_crawler_answers_200(self, client):
        with mock.patch(
            "lilbee.server.routes.setup._bootstrap_crawler_stream",
            return_value=_events("event: done\ndata: {}\n\n"),
        ):
            resp = client.post("/setup/crawler")
        assert resp.status_code == OK

    def test_crawl_answers_200(self, client):
        with (
            mock.patch("lilbee.crawler.require_valid_crawl_url"),
            mock.patch(
                "lilbee.server.handlers.crawl_stream",
                return_value=_events("event: done\ndata: {}\n\n"),
            ),
        ):
            resp = client.post("/api/crawl", json={"url": "https://example.com"})
        assert resp.status_code == OK

    def test_wiki_generate_answers_200(self, client):
        with (
            mock.patch("lilbee.wiki.lazy.resolve_stub", return_value=MagicMock()),
            mock.patch(
                "lilbee.server.handlers.wiki_generate_stream",
                return_value=_events("event: done\ndata: {}\n\n"),
            ),
        ):
            resp = client.post("/api/wiki/generate/ford")
        assert resp.status_code == OK


class TestQueryAndActionRoutesAnswer200:
    def test_ask_and_chat_answer_200(self, client):
        answer = {"answer": "42", "sources": []}
        with (
            mock.patch("lilbee.server.handlers.ask", new_callable=AsyncMock, return_value=answer),
            mock.patch("lilbee.server.handlers.chat", new_callable=AsyncMock, return_value=answer),
        ):
            asked = client.post("/api/ask", json={"question": "q"})
            chatted = client.post("/api/chat", json={"question": "q", "history": []})
        assert (asked.status_code, chatted.status_code) == (OK, OK)

    def test_models_show_answers_200(self, client):
        with mock.patch(
            "lilbee.server.handlers.models_show",
            new_callable=AsyncMock,
            return_value={"parameters": "temp 0.7"},
        ):
            resp = client.post("/api/models/show", json={"model": "test"})
        assert resp.status_code == OK

    def test_documents_remove_answers_200(self, client):
        with mock.patch(
            "lilbee.server.handlers.delete_documents",
            new_callable=AsyncMock,
            return_value={"removed": ["a.md"], "not_found": []},
        ):
            resp = client.post("/api/documents/remove", json={"names": ["a.md"]})
        assert resp.status_code == OK

    def test_wiki_index_answers_200(self, client):
        with mock.patch("lilbee.wiki.stubs.refresh_stub_index", return_value=[]):
            resp = client.post("/api/wiki/index")
        assert resp.status_code == OK
        assert resp.json() == {"entries": 0}

    def test_wiki_lint_answers_200(self, client):
        report = MagicMock(issues=[], error_count=0, warning_count=0)
        with mock.patch("lilbee.server.wiki._lint_report", return_value=report):
            resp = client.post("/api/wiki/lint")
        assert resp.status_code == OK

    def test_wiki_prune_answers_200(self, client):
        report = MagicMock(records=[], archived_count=0, flagged_count=0, reconciled_count=0)
        with mock.patch("lilbee.server.wiki.prune_mod.prune_wiki", return_value=report):
            resp = client.post("/api/wiki/prune")
        assert resp.status_code == OK

    def test_wiki_draft_accept_answers_200(self, client, tmp_path):
        from lilbee.wiki.drafts import AcceptResult

        def accept(slug, root, store):
            return AcceptResult(
                slug=slug, requested_slug=slug, moved_to=root / "x.md", reindexed_chunks=1
            )

        with mock.patch("lilbee.server.wiki.accept_draft", accept):
            resp = client.post("/api/wiki/drafts/accept/cv-manual")
        assert resp.status_code == OK

    def test_claim_answers_200(self, client):
        sid = svc_mod.get_services().session_store.create(
            model_ref="m", scope="both", origin=SessionOrigin.TUI
        )
        resp = client.post(f"/api/sessions/{sid}/claim")
        assert resp.status_code == OK


class TestCreatingRoutesAnswer201:
    def test_memories_sessions_and_fork_answer_201(self, client):
        store = svc_mod.get_services().session_store
        sid = store.create(model_ref="m", scope="both", origin=SessionOrigin.TUI)
        store.set_title(sid, "T", TitleSource.AUTO)
        store.add_message(sid, SessionMessage(role=MessageRole.USER, content="q"))
        memory = client.post("/api/memories", json={"text": "uses rust"})
        session = client.post("/api/sessions", json={"model_ref": "m", "scope": "both"})
        fork = client.post(f"/api/sessions/{sid}/fork")
        assert (memory.status_code, session.status_code, fork.status_code) == (
            CREATED,
            CREATED,
            CREATED,
        )

    def test_error_replies_keep_their_status(self, client):
        store = svc_mod.get_services().session_store
        sid = store.create(model_ref="m", scope="both", origin=SessionOrigin.TUI)
        assert client.post("/api/memories", json={"text": "x", "kind": "bogus"}).status_code == 400
        assert client.post("/api/sessions/nope/fork").status_code == 404
        assert client.post("/api/sessions/nope/claim").status_code == 404
        assert (
            client.post(f"/api/sessions/{sid}/fork", json={"message_count": 99}).status_code == 422
        )
