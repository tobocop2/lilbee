"""The Sources list of a saved chat turn on every session surface: HTTP, MCP, CLI, export."""

from __future__ import annotations

import json

import pytest
from litestar.testing import TestClient
from typer.testing import CliRunner

from lilbee.app import services as svc_mod
from lilbee.app.session_export import session_markdown
from lilbee.app.session_sources import message_payload, message_sources
from lilbee.core.config import cfg
from lilbee.data.store import SearchChunk
from lilbee.retrieval.query.formatting import (
    FILE_LINK_RE,
    SourceRef,
    format_sources_block,
    source_label,
)
from lilbee.sessions import MessageRole, SessionMessage, SessionOrigin, StoredSourceRef
from tests._crafted_titles import CRAFTED_SOURCE, CRAFTED_TITLES, crafted_label
from tests.conftest import make_mock_services

_PAGE = "_web/example.com/guide.md"
_PAGE_LABEL = "example.com · The Guide"
# The attack titles of obsidian-lilbee #370, then the crafted titles every label surface checks.
_ATTACK_TITLES = tuple(
    dict.fromkeys(
        (
            "[click](https://evil.example/x)",
            "![px](https://evil.example/p.png)",
            "<img src=x onerror=alert(1)>",
            "*Q3*",
            "a_b",
            "a ] b",
            *CRAFTED_TITLES,
        )
    )
)
# A plugin saves the cited subset [1], [3] of an answer with its true numbers.
_POSTED_REFS = [
    {"source": "first.md", "citation_number": 1},
    {"source": _PAGE, "citation_number": 3},
]
_SHOWN_REFS = [
    {"source": "first.md", "label": "first.md", "citation_number": 1, "locator": ""},
    {"source": _PAGE, "label": _PAGE_LABEL, "citation_number": 3, "locator": ""},
]
# A plugin saves the page and the line span each source was read at.
_POSTED_LOCATED_REFS = [
    {"source": "manual.pdf", "citation_number": 1, "locator": "page 3"},
    {"source": "tool.py", "citation_number": 2, "locator": "lines 10-20"},
]
_SHOWN_LOCATED_REFS = [
    {"source": "manual.pdf", "label": "manual.pdf", "citation_number": 1, "locator": "page 3"},
    {"source": "tool.py", "label": "tool.py", "citation_number": 2, "locator": "lines 10-20"},
]
# An answer whose own text keeps a "Sources:" heading before prose, saved with refs.
_HEADED_PROSE = "Answer [1].\n\nSources:\nI used the documents provided."


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.mcp_sessions_enabled = True
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture
def services():
    services = make_mock_services()
    services.store.web_titles.side_effect = lambda names: {
        n: "The Guide" for n in names if n == _PAGE
    }
    svc_mod.set_services(services)
    yield services
    svc_mod.set_services(None)


@pytest.fixture
def store(services):
    return services.session_store


@pytest.fixture
def client(services):
    import lilbee.server.auth as auth_mod
    from lilbee.server.app import create_app

    auth_mod.session_manager.disable()
    yield TestClient(create_app())
    auth_mod.session_manager.cleanup()


def _chunk(
    source: str, content_type: str = "text", page: int = 0, lines: tuple[int, int] = (0, 0)
) -> SearchChunk:
    return SearchChunk(
        source=source,
        content_type=content_type,
        chunk="text",
        page_start=page,
        page_end=page,
        line_start=lines[0],
        line_end=lines[1],
        chunk_index=0,
        vector=[0.1],
    )


def _legacy_tui_content(answer: str, sources: list[str], titles: dict[str, str]) -> str:
    """An answer as the TUI saved it before ``source_refs``: its Sources list in the text."""
    return answer + format_sources_block([_chunk(s) for s in sources], titles)


def _located_chunks() -> list[SearchChunk]:
    return [_chunk("manual.pdf", "pdf", page=3), _chunk("tool.py", "code", lines=(10, 20))]


def _located_refs() -> tuple[StoredSourceRef, ...]:
    return (
        StoredSourceRef("manual.pdf", 1, "page 3"),
        StoredSourceRef("tool.py", 2, "lines 10-20"),
    )


def _session(store, *messages: SessionMessage, origin: SessionOrigin = SessionOrigin.HTTP) -> str:
    session_id = store.create(model_ref="m", scope="both", origin=origin)
    store.add_message(
        session_id, SessionMessage(role=MessageRole.USER, content="q"), surface=origin
    )
    for message in messages:
        store.add_message(session_id, message, surface=origin)
    return session_id


def _refs_of(message: dict) -> list[dict]:
    return message["source_refs"]


def _export(*messages: SessionMessage) -> str:
    from lilbee.sessions import Session, SessionMeta

    meta = SessionMeta(
        id="3f2a1b2c-0000-4000-8000-000000000000",
        title="T",
        created_at="2026-10-20T10:00:00+00:00",
        updated_at="2026-10-20T10:05:00+00:00",
        model_ref="m",
        scope="both",
        message_count=len(messages),
    )
    return session_markdown(Session(meta=meta, messages=messages))


class TestMessageSources:
    def test_stored_refs_are_labelled_from_the_current_stored_title(self, services):
        message = SessionMessage(
            MessageRole.ASSISTANT,
            "See [3].",
            sources=(_PAGE,),
            source_refs=(StoredSourceRef(_PAGE, 3),),
        )
        assert message_sources(message) == ("See [3].", [SourceRef(_PAGE, _PAGE_LABEL, 3)])

    def test_a_legacy_turn_with_only_sources_has_no_numbers(self, services):
        message = SessionMessage(MessageRole.ASSISTANT, "See [3].", sources=("first.md", _PAGE))
        assert message_sources(message)[1] == [
            SourceRef("first.md", "first.md", None),
            SourceRef(_PAGE, _PAGE_LABEL, None),
        ]

    def test_a_legacy_tui_turn_gives_its_text_and_the_list_it_ends_in(self, services):
        content = _legacy_tui_content("Oil [2].", ["notes/a.md", "b.md"], {})
        text, refs = message_sources(SessionMessage(MessageRole.ASSISTANT, content))
        assert text == "Oil [2]."
        assert refs == [SourceRef("notes/a.md", "notes/a.md", 1), SourceRef("b.md", "b.md", 2)]

    def test_a_legacy_tui_turn_keeps_the_locator_of_each_list_line(self, services):
        content = "See [1], [2]." + format_sources_block(_located_chunks(), {})
        assert ", page 3\n" in content and content.endswith(", lines 10-20")
        text, refs = message_sources(SessionMessage(MessageRole.ASSISTANT, content))
        assert text == "See [1], [2]."
        assert refs == [
            SourceRef("manual.pdf", "manual.pdf", 1, "page 3"),
            SourceRef("tool.py", "tool.py", 2, "lines 10-20"),
        ]

    def test_stored_refs_keep_their_locators(self, services):
        message = SessionMessage(
            MessageRole.ASSISTANT, "See [1], [2].", source_refs=_located_refs()
        )
        assert [r.locator for r in message_sources(message)[1]] == ["page 3", "lines 10-20"]

    def test_a_turn_with_refs_keeps_the_prose_after_its_own_sources_heading(self, services):
        """Only a turn with no refs is split at a "Sources:" marker; the TUI restores
        this one through the same call."""
        message = SessionMessage(
            MessageRole.ASSISTANT, _HEADED_PROSE, source_refs=(StoredSourceRef("first.md", 1),)
        )
        assert message_sources(message) == (
            _HEADED_PROSE,
            [SourceRef("first.md", "first.md", 1)],
        )

    def test_a_legacy_turn_keeps_the_prose_after_its_own_sources_heading(self, services):
        """With no lilbee list after the marker nothing is split off."""
        bare = SessionMessage(MessageRole.ASSISTANT, _HEADED_PROSE)
        assert message_sources(bare) == (_HEADED_PROSE, [])
        posted = SessionMessage(MessageRole.ASSISTANT, _HEADED_PROSE, sources=("first.md",))
        assert message_sources(posted) == (_HEADED_PROSE, [SourceRef("first.md", "first.md", None)])

    @pytest.mark.parametrize("title", _ATTACK_TITLES)
    def test_a_legacy_tui_turn_returns_a_stored_attack_title_raw(self, services, title):
        content = _legacy_tui_content("x [1].", [CRAFTED_SOURCE], {CRAFTED_SOURCE: title})
        assert FILE_LINK_RE.search(content) is not None, "the stored list links the page"
        text, refs = message_sources(SessionMessage(MessageRole.ASSISTANT, content))
        assert text == "x [1]."
        assert refs == [SourceRef(CRAFTED_SOURCE, crafted_label(title), 1)]

    def test_a_legacy_list_line_with_no_link_keeps_a_memory_row(self, services):
        content = "x [1] [2].\n\nSources:\n\n1. memory:m1\n2. gone.md\nnot a line"
        assert message_sources(SessionMessage(MessageRole.ASSISTANT, content))[1] == [
            SourceRef("memory:m1", "memory:m1", 1),
            SourceRef("", "gone.md", 2),
        ]

    def test_a_link_outside_every_known_root_has_no_source(self, services):
        content = "x [1].\n\nSources:\n\n1. [elsewhere.md](file:///nowhere/elsewhere.md)"
        assert message_sources(SessionMessage(MessageRole.ASSISTANT, content))[1] == [
            SourceRef("", "elsewhere.md", 1)
        ]

    def test_a_link_under_a_linked_root_is_keyed_by_that_root(self, services, tmp_path):
        root = tmp_path / "vault"
        (root / "sub").mkdir(parents=True)
        cfg.linked_roots = {"vault": str(root)}
        url = (root / "sub" / "n.md").as_uri()
        content = f"x [1].\n\nSources:\n\n1. [n.md]({url})"
        assert message_sources(SessionMessage(MessageRole.ASSISTANT, content))[1] == [
            SourceRef("vault/sub/n.md", "n.md", 1)
        ]


class TestHttp:
    def test_posted_refs_come_back_with_labels_and_their_numbers(self, client, store):
        session_id = _session(store)
        posted = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"role": "assistant", "content": "See [1], [3].", "source_refs": _POSTED_REFS},
        )
        assert posted.status_code == 201
        message = client.get(f"/api/sessions/{session_id}").json()["messages"][-1]
        assert message["content"] == "See [1], [3]."
        assert _refs_of(message) == _SHOWN_REFS
        assert message["sources"] == ["first.md", _PAGE], "derived for older readers"

    def test_posted_locators_are_stored_and_come_back(self, client, store):
        session_id = _session(store)
        client.post(
            f"/api/sessions/{session_id}/messages",
            json={
                "role": "assistant",
                "content": "See [1], [2].",
                "source_refs": _POSTED_LOCATED_REFS,
            },
        )
        assert store.get(session_id).messages[-1].source_refs == _located_refs()
        message = client.get(f"/api/sessions/{session_id}").json()["messages"][-1]
        assert _refs_of(message) == _SHOWN_LOCATED_REFS

    def test_a_posted_text_with_a_sources_heading_comes_back_whole(self, client, store):
        session_id = _session(store)
        client.post(
            f"/api/sessions/{session_id}/messages",
            json={
                "role": "assistant",
                "content": _HEADED_PROSE,
                "source_refs": [{"source": "first.md", "citation_number": 1}],
            },
        )
        message = client.get(f"/api/sessions/{session_id}").json()["messages"][-1]
        assert message["content"] == _HEADED_PROSE
        assert _refs_of(message) == [_SHOWN_REFS[0]]

    def test_a_legacy_turn_with_a_sources_heading_and_prose_comes_back_whole(self, client, store):
        session_id = _session(
            store,
            SessionMessage(MessageRole.ASSISTANT, _HEADED_PROSE),
            origin=SessionOrigin.TUI,
        )
        message = client.get(f"/api/sessions/{session_id}").json()["messages"][-1]
        assert message["content"] == _HEADED_PROSE
        assert _refs_of(message) == []

    def test_a_legacy_tui_turn_sends_its_locators(self, client, store):
        content = "See [1], [2]." + format_sources_block(_located_chunks(), {})
        session_id = _session(
            store, SessionMessage(MessageRole.ASSISTANT, content), origin=SessionOrigin.TUI
        )
        message = client.get(f"/api/sessions/{session_id}").json()["messages"][-1]
        assert message["content"] == "See [1], [2]."
        assert _refs_of(message) == _SHOWN_LOCATED_REFS

    def test_a_post_without_refs_keeps_its_sources(self, client, store):
        session_id = _session(store)
        client.post(
            f"/api/sessions/{session_id}/messages",
            json={"role": "assistant", "content": "a", "sources": ["first.md"]},
        )
        message = client.get(f"/api/sessions/{session_id}").json()["messages"][-1]
        assert message["sources"] == ["first.md"]
        assert _refs_of(message) == [
            {"source": "first.md", "label": "first.md", "citation_number": None, "locator": ""}
        ]

    def test_a_ref_with_no_number_is_stored_with_none(self, client, store):
        session_id = _session(store)
        client.post(
            f"/api/sessions/{session_id}/messages",
            json={"role": "assistant", "content": "a", "source_refs": [{"source": "first.md"}]},
        )
        assert store.get(session_id).messages[-1].source_refs == (StoredSourceRef("first.md"),)

    def test_a_legacy_event_with_only_sources_gives_null_numbers(self, client, store):
        session_id = _session(
            store, SessionMessage(MessageRole.ASSISTANT, "See [3].", sources=(_PAGE,))
        )
        message = client.get(f"/api/sessions/{session_id}").json()["messages"][-1]
        assert _refs_of(message) == [
            {"source": _PAGE, "label": _PAGE_LABEL, "citation_number": None, "locator": ""}
        ]

    @pytest.mark.parametrize("title", _ATTACK_TITLES)
    def test_a_legacy_tui_turn_sends_its_text_without_the_list(self, client, store, title):
        content = _legacy_tui_content("x [1].", [CRAFTED_SOURCE], {CRAFTED_SOURCE: title})
        session_id = _session(
            store, SessionMessage(MessageRole.ASSISTANT, content), origin=SessionOrigin.TUI
        )
        message = client.get(f"/api/sessions/{session_id}").json()["messages"][-1]
        assert message["content"] == "x [1]."
        assert _refs_of(message) == [
            {
                "source": CRAFTED_SOURCE,
                "label": crafted_label(title),
                "citation_number": 1,
                "locator": "",
            }
        ]

    def test_a_fork_keeps_the_refs(self, client, store):
        session_id = _session(
            store,
            SessionMessage(
                MessageRole.ASSISTANT,
                "See [3].",
                sources=(_PAGE,),
                source_refs=(StoredSourceRef(_PAGE, 3),),
            ),
            origin=SessionOrigin.TUI,
        )
        fork = client.post(f"/api/sessions/{session_id}/fork").json()
        assert _refs_of(fork["messages"][-1]) == [
            {"source": _PAGE, "label": _PAGE_LABEL, "citation_number": 3, "locator": ""}
        ]
        assert store.get(fork["meta"]["id"]).messages[-1].source_refs == (
            StoredSourceRef(_PAGE, 3),
        )


class TestMcp:
    def test_append_and_get_match_the_http_pair(self, client, store):
        from lilbee.mcp_server import session_add_message, session_create, session_get

        session_id = session_create("m")["id"]
        appended = session_add_message(
            session_id, "assistant", "See [1], [3].", source_refs=list(_POSTED_REFS)
        )
        assert "error" not in appended
        mcp_message = session_get(session_id)["messages"][-1]
        assert _refs_of(mcp_message) == _SHOWN_REFS
        http_message = client.get(f"/api/sessions/{session_id}").json()["messages"][-1]
        assert mcp_message == http_message

    def test_append_and_get_keep_the_locators(self, client, store):
        from lilbee.mcp_server import session_add_message, session_create, session_get

        session_id = session_create("m")["id"]
        appended = session_add_message(
            session_id, "assistant", "See [1], [2].", source_refs=list(_POSTED_LOCATED_REFS)
        )
        assert "error" not in appended
        assert store.get(session_id).messages[-1].source_refs == _located_refs()
        mcp_message = session_get(session_id)["messages"][-1]
        assert _refs_of(mcp_message) == _SHOWN_LOCATED_REFS
        assert mcp_message == client.get(f"/api/sessions/{session_id}").json()["messages"][-1]

    def test_get_sends_a_text_with_a_sources_heading_whole(self, client, store):
        from lilbee.mcp_server import session_add_message, session_create, session_get

        session_id = session_create("m")["id"]
        appended = session_add_message(
            session_id, "assistant", _HEADED_PROSE, source_refs=[dict(_POSTED_REFS[0])]
        )
        assert "error" not in appended
        mcp_message = session_get(session_id)["messages"][-1]
        assert mcp_message["content"] == _HEADED_PROSE
        assert mcp_message == client.get(f"/api/sessions/{session_id}").json()["messages"][-1]

    def test_append_without_refs_keeps_its_sources(self, services):
        from lilbee.mcp_server import session_add_message, session_create, session_get

        session_id = session_create("m")["id"]
        session_add_message(session_id, "assistant", "a", sources=["first.md"])
        message = session_get(session_id)["messages"][-1]
        assert message["sources"] == ["first.md"]
        assert message["source_refs"][0]["citation_number"] is None

    async def test_the_tool_schema_takes_refs_on_the_wire(self, services):
        """The MCP server accepts ``source_refs`` as its schema declares them."""
        from lilbee.mcp_server import build_mcp_server, session_create, session_get

        server = build_mcp_server()
        tools = {t.name: t for t in await server.list_tools()}
        schema = tools["session_add_message"].input_schema
        assert "source_refs" in schema["properties"]
        session_id = session_create("m")["id"]
        result = await server.call_tool(
            "session_add_message",
            {
                "session_id": session_id,
                "role": "assistant",
                "content": "See [1], [3].",
                "source_refs": [_POSTED_REFS[0], {"source": _PAGE, "citation_number": 3}],
            },
        )
        assert not result.is_error, result
        assert _refs_of(session_get(session_id)["messages"][-1]) == _SHOWN_REFS


class TestCli:
    def test_show_json_sends_the_refs(self, services, tmp_path, monkeypatch):
        from lilbee.cli import app
        from lilbee.cli.app import apply_overrides
        from lilbee.sessions import SessionStore

        monkeypatch.setenv("LILBEE_DATA", str(tmp_path))
        apply_overrides()
        store = SessionStore()
        session_id = _session(
            store,
            SessionMessage(
                MessageRole.ASSISTANT,
                "See [3].",
                sources=(_PAGE,),
                source_refs=(StoredSourceRef(_PAGE, 3),),
            ),
        )
        result = CliRunner().invoke(app, ["--json", "sessions", "show", session_id])
        assert result.exit_code == 0, result.output
        message = json.loads(result.output)["messages"][-1]
        assert message == message_payload(store.get(session_id).messages[-1])
        assert _refs_of(message) == [
            {"source": _PAGE, "label": _PAGE_LABEL, "citation_number": 3, "locator": ""}
        ]


class TestExport:
    def test_numbers_come_from_the_refs(self, services):
        message = SessionMessage(
            MessageRole.ASSISTANT,
            "See [1], [3].",
            sources=("first.md", _PAGE),
            source_refs=(StoredSourceRef("first.md", 1), StoredSourceRef(_PAGE, 3)),
        )
        markdown = _export(message)
        assert "1\\. first.md\\\n3\\. example.com · The Guide" in markdown

    def test_each_line_keeps_its_locator(self, services):
        message = SessionMessage(
            MessageRole.ASSISTANT, "See [1], [2].", source_refs=_located_refs()
        )
        assert _export(message).endswith("1. manual.pdf, page 3\n2. tool.py, lines 10-20\n")

    def test_a_text_with_a_sources_heading_keeps_its_prose_and_lists_the_refs(self, services):
        message = SessionMessage(
            MessageRole.ASSISTANT, _HEADED_PROSE, source_refs=(StoredSourceRef("first.md", 1),)
        )
        markdown = _export(message)
        assert markdown.endswith(f"## Assistant\n\n{_HEADED_PROSE}\n\nSources:\n\n1. first.md\n")
        assert markdown.count("\n1. ") == 1

    @pytest.mark.parametrize("title", _ATTACK_TITLES)
    def test_an_attack_label_is_escaped(self, services, title):
        from markdown_it import MarkdownIt

        services.store.web_titles.side_effect = lambda names: {n: title for n in names}
        message = SessionMessage(
            MessageRole.ASSISTANT,
            "x [1].",
            sources=(CRAFTED_SOURCE,),
            source_refs=(StoredSourceRef(CRAFTED_SOURCE, 1),),
        )
        markdown = _export(message)
        line = markdown.rpartition("\n1. ")[2].strip()
        tokens = MarkdownIt("commonmark").parseInline(line)[0].children or []
        assert {t.type for t in tokens} == {"text"}, line
        assert "".join(t.content for t in tokens) == source_label(CRAFTED_SOURCE, title)
