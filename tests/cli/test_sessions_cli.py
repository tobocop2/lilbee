"""lilbee sessions CLI: list, show, fork, rename, delete."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from lilbee.cli import app
from lilbee.core.config import cfg
from lilbee.sessions import MessageRole, SessionMessage, SessionOrigin, SessionStore, TitleSource

runner = CliRunner()


@pytest.fixture
def seeded(tmp_path):
    (tmp_path / "data").mkdir(exist_ok=True)
    cfg.data_dir = tmp_path / "data"
    store = SessionStore()
    session_id = store.create(model_ref="gpt-oss-20b", scope="both")
    store.set_title(session_id, "Torque specs", TitleSource.AUTO)
    store.add_message(session_id, SessionMessage(role=MessageRole.USER, content="what specs?"))
    store.add_message(
        session_id,
        SessionMessage(role=MessageRole.ASSISTANT, content="85 Nm.", sources=("manual.pdf",)),
    )
    return tmp_path, session_id


def _args(tmp_path, *rest, json_mode=False):
    base = ["--data-dir", str(tmp_path)]
    if json_mode:
        base.insert(0, "--json")
    return [*base, "sessions", *rest]


def test_list_human(seeded):
    tmp_path, _ = seeded
    result = runner.invoke(app, _args(tmp_path, "list"))
    assert result.exit_code == 0
    assert "Torque specs" in result.output


def test_list_is_the_admin_view_and_labels_agent_sessions(seeded):
    """The CLI lists every origin (it is where stray agent sessions get
    cleaned up), so each row says whose it is."""
    from lilbee.sessions import SessionOrigin

    tmp_path, _ = seeded
    SessionStore().create(model_ref="m", scope="both", origin=SessionOrigin.MCP)
    result = runner.invoke(app, _args(tmp_path, "list"))
    assert result.exit_code == 0
    assert "mcp" in result.output
    assert "tui" in result.output


def test_list_empty(tmp_path):
    (tmp_path / "data").mkdir()
    result = runner.invoke(app, _args(tmp_path, "list"))
    assert result.exit_code == 0
    assert "No saved sessions" in result.output


def test_list_json(seeded):
    tmp_path, session_id = seeded
    result = runner.invoke(app, _args(tmp_path, "list", json_mode=True))
    assert result.exit_code == 0
    body = json.loads(result.output)
    assert body["sessions"][0]["id"] == session_id
    assert body["sessions"][0]["message_count"] == 2


def test_show_by_prefix(seeded):
    tmp_path, session_id = seeded
    result = runner.invoke(app, _args(tmp_path, "show", session_id[:8]))
    assert result.exit_code == 0
    assert "Torque specs" in result.output
    assert "85 Nm." in result.output


def test_show_json(seeded):
    tmp_path, session_id = seeded
    result = runner.invoke(app, _args(tmp_path, "show", session_id, json_mode=True))
    body = json.loads(result.output)
    assert body["messages"][1]["role"] == "assistant"
    assert body["messages"][1]["sources"] == ["manual.pdf"]
    assert body["summary"] == "", "an uncompacted session reports an empty summary"


def test_show_json_carries_the_summary(seeded):
    """A script resuming from CLI JSON needs what compaction produced."""
    tmp_path, session_id = seeded
    SessionStore().set_summary(session_id, "earlier: torque is 85 Nm")
    result = runner.invoke(app, _args(tmp_path, "show", session_id, json_mode=True))
    assert json.loads(result.output)["summary"] == "earlier: torque is 85 Nm"


def test_rename(seeded):
    tmp_path, session_id = seeded
    result = runner.invoke(app, _args(tmp_path, "rename", session_id[:8], "Renamed"))
    assert result.exit_code == 0
    cfg.data_dir = tmp_path / "data"
    assert SessionStore().get(session_id).meta.title == "Renamed"


def test_rename_json(seeded):
    tmp_path, session_id = seeded
    result = runner.invoke(app, _args(tmp_path, "rename", session_id, "New", json_mode=True))
    assert json.loads(result.output) == {"id": session_id, "title": "New"}


def test_delete_with_yes(seeded):
    tmp_path, session_id = seeded
    result = runner.invoke(app, _args(tmp_path, "delete", session_id[:8], "--yes"))
    assert result.exit_code == 0
    cfg.data_dir = tmp_path / "data"
    assert SessionStore().list() == []


def test_delete_confirm_declined(seeded):
    tmp_path, session_id = seeded
    result = runner.invoke(app, _args(tmp_path, "delete", session_id[:8]), input="n\n")
    assert result.exit_code != 0
    cfg.data_dir = tmp_path / "data"
    assert len(SessionStore().list()) == 1


def test_delete_json(seeded):
    tmp_path, session_id = seeded
    result = runner.invoke(app, _args(tmp_path, "delete", session_id, json_mode=True))
    assert json.loads(result.output) == {"id": session_id, "deleted": True}


def test_unknown_prefix_errors(tmp_path):
    (tmp_path / "data").mkdir()
    result = runner.invoke(app, _args(tmp_path, "show", "deadbeef"))
    assert result.exit_code == 1
    assert "No session matching" in result.output


def test_ambiguous_prefix_errors(tmp_path):
    (tmp_path / "data").mkdir()
    cfg.data_dir = tmp_path / "data"
    store = SessionStore()
    # Two sessions; the empty prefix matches both.
    store.create(model_ref="m", scope="both")
    store.create(model_ref="m", scope="both")
    result = runner.invoke(app, _args(tmp_path, "rename", "", "x"))
    assert result.exit_code == 1
    assert "ambiguous" in result.output.lower()


def test_unknown_prefix_json(tmp_path):
    (tmp_path / "data").mkdir()
    result = runner.invoke(app, _args(tmp_path, "delete", "nope", json_mode=True))
    assert result.exit_code == 1
    assert json.loads(result.output) == {"error": "No session matching 'nope'."}


def test_fork_whole_conversation(seeded):
    tmp_path, session_id = seeded
    result = runner.invoke(app, _args(tmp_path, "fork", session_id[:8]))
    assert result.exit_code == 0, result.output
    assert "Torque specs (fork 1)" in result.output
    newest = SessionStore().list()[0]
    assert newest.forked_from == session_id
    assert newest.message_count == 2


def test_fork_json_with_a_count(seeded):
    tmp_path, session_id = seeded
    result = runner.invoke(
        app, _args(tmp_path, "fork", session_id, "--messages", "1", json_mode=True)
    )
    assert result.exit_code == 0, result.output
    meta = json.loads(result.output)["meta"]
    assert meta["forked_from"] == session_id
    assert meta["message_count"] == 1
    assert meta["title"] == "Torque specs (fork 1)"
    assert meta["origin"] == SessionOrigin.CLI


def test_fork_lists_first_with_its_title(seeded):
    tmp_path, session_id = seeded
    runner.invoke(app, _args(tmp_path, "fork", session_id))
    result = runner.invoke(app, _args(tmp_path, "list"))
    rows = [line for line in result.output.splitlines() if "Torque specs" in line]
    assert "(fork 1)" in rows[0]


@pytest.mark.parametrize("count", ["-1", "3"])
def test_fork_outside_the_transcript_exits_1(seeded, count):
    tmp_path, session_id = seeded
    result = runner.invoke(
        app, _args(tmp_path, "fork", session_id, "--messages", count, json_mode=True)
    )
    assert result.exit_code == 1
    assert "0 to 2" in json.loads(result.output)["error"]
    assert len(SessionStore().list()) == 1


def test_fork_of_an_agent_session_exits_1(tmp_path):
    (tmp_path / "data").mkdir()
    cfg.data_dir = tmp_path / "data"
    session_id = SessionStore().create(model_ref="m", scope="both", origin=SessionOrigin.MCP)
    result = runner.invoke(app, _args(tmp_path, "fork", session_id))
    assert result.exit_code == 1
    assert "belongs to the mcp surface" in result.output
    assert len(SessionStore().list()) == 1


def test_fork_unknown_prefix_errors(tmp_path):
    (tmp_path / "data").mkdir()
    result = runner.invoke(app, _args(tmp_path, "fork", "deadbeef"))
    assert result.exit_code == 1
    assert "No session matching" in result.output
