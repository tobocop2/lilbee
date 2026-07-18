"""Tests for the MCP session tools (list, get, rename, delete)."""

from __future__ import annotations

import pytest

from lilbee.app import services as svc_mod
from lilbee.core.config import cfg
from lilbee.mcp_server import (
    session_add_message,
    session_create,
    session_delete,
    session_get,
    session_rename,
    session_set_summary,
    sessions_list,
)
from lilbee.sessions import MessageRole, SessionMessage, SessionOrigin, TitleSource
from tests.conftest import make_mock_services


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_dir = tmp_path / "data"
    # Agent sessions are off by default; this file exercises the tools, so it
    # turns them on and the toggle tests below switch them back off.
    cfg.mcp_sessions_enabled = True
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture
def store():
    services = make_mock_services()
    svc_mod.set_services(services)
    yield services.session_store
    svc_mod.set_services(None)


def _seed(store, origin: SessionOrigin = SessionOrigin.MCP) -> str:
    session_id = store.create(model_ref="gpt-oss-20b", scope="both", origin=origin)
    store.set_title(session_id, "Torque specs", TitleSource.AUTO)
    store.add_message(session_id, SessionMessage(role=MessageRole.USER, content="what specs?"))
    store.add_message(
        session_id,
        SessionMessage(role=MessageRole.ASSISTANT, content="85 Nm.", sources=("manual.pdf",)),
    )
    return session_id


def test_sessions_list(store):
    session_id = _seed(store)
    result = sessions_list()
    assert result["total"] == 1
    assert result["sessions"][0]["id"] == session_id
    assert result["sessions"][0]["message_count"] == 2


def test_sessions_list_empty(store):
    assert sessions_list() == {"sessions": [], "total": 0}


def test_session_get_returns_transcript(store):
    session_id = _seed(store)
    result = session_get(session_id)
    assert result["meta"]["title"] == "Torque specs"
    assert result["messages"][1]["role"] == "assistant"
    assert result["messages"][1]["sources"] == ["manual.pdf"]


def test_session_get_unknown_errors(store):
    assert "error" in session_get("nope")


def test_session_get_carries_the_summary(store):
    """An agent resuming a compacted conversation needs the summary."""
    session_id = _seed(store)
    store.set_summary(session_id, "earlier: torque is 85 Nm")
    assert session_get(session_id)["summary"] == "earlier: torque is 85 Nm"


def test_session_create_then_append_and_read_back(store):
    """An agent can own a conversation end to end over MCP."""
    created = session_create(model_ref="qwen3-4b", scope="both")
    session_id = created["id"]
    session_add_message(session_id, "user", "what torque?", ["manual.pdf"])
    session_add_message(session_id, "assistant", "85 Nm.")
    got = session_get(session_id)
    assert got["meta"]["model_ref"] == "qwen3-4b"
    assert [m["content"] for m in got["messages"]] == ["what torque?", "85 Nm."]
    assert got["messages"][0]["sources"] == ["manual.pdf"]


def test_session_add_message_bad_role_errors(store):
    session_id = _seed(store)
    assert "error" in session_add_message(session_id, "wizard", "hi")


def test_session_add_message_unknown_errors(store):
    assert "error" in session_add_message("nope", "user", "hi")


def test_session_set_summary_round_trips(store):
    session_id = _seed(store)
    session_set_summary(session_id, "folded: 85 Nm")
    assert session_get(session_id)["summary"] == "folded: 85 Nm"


def test_session_set_summary_unknown_errors(store):
    assert "error" in session_set_summary("nope", "x")


def test_session_rename(store):
    session_id = _seed(store)
    assert session_rename(session_id, "Renamed") == {"id": session_id, "title": "Renamed"}
    assert session_get(session_id)["meta"]["title"] == "Renamed"


def test_session_rename_unknown_errors(store):
    assert "error" in session_rename("nope", "x")


def test_session_delete(store):
    session_id = _seed(store)
    assert session_delete(session_id) == {"id": session_id, "deleted": True}
    assert sessions_list()["total"] == 0


def test_session_delete_unknown_errors(store):
    assert "error" in session_delete("nope")


def test_appending_to_a_foreign_session_errors_with_claim_hint(store):
    """An agent must not splice into a TUI conversation; the error names the fix."""
    session_id = _seed(store, origin=SessionOrigin.TUI)
    result = session_add_message(session_id, "user", "spliced")
    assert "claim" in result["error"].lower()
    assert store.get(session_id).meta.message_count == 2, "nothing landed"


def test_human_sessions_are_invisible_to_agents(store):
    """sessions_list scopes to agent-owned sessions: your conversations are
    not readable by any connected MCP client."""
    _seed(store, origin=SessionOrigin.TUI)
    _seed(store, origin=SessionOrigin.HTTP)
    mine = _seed(store)
    result = sessions_list()
    assert [s["id"] for s in result["sessions"]] == [mine]
    assert result["total"] == 1


def test_human_sessions_answer_not_found_so_agents_cannot_probe(store):
    """Reads and mutations of a human session look identical to a missing id:
    an agent cannot learn which conversations exist."""
    session_id = _seed(store, origin=SessionOrigin.TUI)
    missing = session_get("nope")["error"].replace("'nope'", f"'{session_id}'")
    assert session_get(session_id)["error"] == missing
    assert session_rename(session_id, "x")["error"] == missing
    assert session_set_summary(session_id, "x")["error"] == missing
    assert session_delete(session_id)["error"] == missing
    assert store.get(session_id).meta.title == "Torque specs", "nothing changed"


def test_claim_makes_a_session_readable_and_removes_it_from_human_space(store):
    """claim=True is the one bridge: the session becomes the agent's, so it
    reads over MCP and leaves the human surfaces' lists."""
    session_id = _seed(store, origin=SessionOrigin.TUI)
    assert session_add_message(session_id, "user", "mine now", claim=True)["added"] is True
    assert session_get(session_id)["meta"]["message_count"] == 3
    assert store.list(origins=frozenset({SessionOrigin.MCP})) and not store.list(
        origins=frozenset({SessionOrigin.TUI})
    )


def test_claim_flag_transfers_and_appends_atomically(store):
    session_id = _seed(store)
    assert session_add_message(session_id, "user", "mine now", claim=True) == {
        "id": session_id,
        "added": True,
    }
    # claimed: further appends need no flag
    assert session_add_message(session_id, "user", "again")["added"] is True


def test_claim_flag_on_unknown_session_errors(store):
    assert "error" in session_add_message("nope", "user", "x", claim=True)


# --- the mcp_sessions_enabled toggle -----------------------------------------


_DISABLED_CALLS = {
    "sessions_list": lambda session_id: sessions_list(),
    "session_get": lambda session_id: session_get(session_id),
    "session_create": lambda session_id: session_create("m"),
    "session_add_message": lambda session_id: session_add_message(session_id, "user", "x"),
    "session_set_summary": lambda session_id: session_set_summary(session_id, "s"),
    "session_rename": lambda session_id: session_rename(session_id, "t"),
    "session_delete": lambda session_id: session_delete(session_id),
}


@pytest.mark.parametrize("tool", sorted(_DISABLED_CALLS), ids=sorted(_DISABLED_CALLS))
def test_session_tool_refuses_when_sessions_disabled(store, tool):
    """Every session tool refuses once the toggle goes off mid-process.

    ``_tool_if`` keeps them off the wire when sessions are off at import, but
    ``mcp_sessions_enabled`` is writable at runtime, so a tool registered at start
    can still be called after the user turns sessions off. Without this each
    one would keep writing to disk.
    """
    session_id = _seed(store)
    cfg.mcp_sessions_enabled = False
    result = _DISABLED_CALLS[tool](session_id)
    assert "error" in result
    assert "agent sessions are off" in result["error"].lower()


def test_disabled_session_tools_write_nothing(store):
    """The refusal is real: the transcript and the session list are untouched."""
    session_id = _seed(store)
    before = len(store.get(session_id).messages)
    cfg.mcp_sessions_enabled = False

    session_add_message(session_id, "user", "should not land")
    session_rename(session_id, "should not rename")
    session_delete(session_id)
    session_create("m")

    cfg.mcp_sessions_enabled = True
    assert len(store.get(session_id).messages) == before
    assert [meta.id for meta in store.list(origins=(SessionOrigin.MCP,))] == [session_id]


def test_session_tools_are_off_the_wire_by_default(monkeypatch):
    """A default install offers no session tools over MCP.

    ``isolated_env`` turns them on for this file, so this asserts against the
    declared default rather than the fixture's override. Registration is
    import-time, so the flag decides whether the tools reach the wire at all;
    the runtime refusals above cover a toggle flipped mid-process.
    """
    from lilbee.core.config import Config

    assert Config.model_fields["mcp_sessions_enabled"].default is False
    assert Config.model_fields["sessions_enabled"].default is True, (
        "the human surfaces stay on by default; only the agent half is opt-in"
    )
