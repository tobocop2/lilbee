"""lilbee sessions CLI: list, show, fork, export, rename, delete."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lilbee.app.session_export import session_markdown, write_session_markdown
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


@pytest.mark.parametrize("blocker", ["[red]", "a[b"], ids=["style-tag", "unbalanced-bracket"])
def test_show_prints_a_bracketed_title_and_message_as_written(seeded, blocker):
    """A bracketed title or message must not go through markup."""
    tmp_path, session_id = seeded
    store = SessionStore()
    store.set_title(session_id, blocker, TitleSource.CUSTOM)
    store.add_message(session_id, SessionMessage(role=MessageRole.USER, content=blocker))
    result = runner.invoke(app, _args(tmp_path, "show", session_id))
    assert result.exit_code == 0, result.output
    assert blocker in result.output
    assert result.output.count(blocker) >= 2


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


@pytest.mark.parametrize(
    "title", ["[red]Renamed", "C:\\notes\\[draft]"], ids=["style-tag", "windows-backslash"]
)
def test_rename_prints_a_bracketed_title_as_written(seeded, title):
    tmp_path, session_id = seeded
    result = runner.invoke(app, _args(tmp_path, "rename", session_id, title))
    assert result.exit_code == 0, result.output
    assert f"Renamed to {title}." in result.output


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


def test_fork_prints_a_bracketed_title_as_written(seeded):
    tmp_path, session_id = seeded
    store = SessionStore()
    store.set_title(session_id, "[red]Torque", TitleSource.CUSTOM)
    result = runner.invoke(app, _args(tmp_path, "fork", session_id))
    assert result.exit_code == 0, result.output
    assert "Forked to [red]Torque (fork 1) (" in result.output


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


def test_export_prints_the_markdown_to_stdout(seeded):
    tmp_path, session_id = seeded
    result = runner.invoke(app, _args(tmp_path, "export", session_id[:8]))
    assert result.exit_code == 0, result.output
    assert result.output == session_markdown(SessionStore().get(session_id))


def test_export_stdout_is_utf8_under_a_legacy_code_page(seeded):
    """A Windows redirect encodes text with the locale code page, which cannot
    hold CJK or emoji; the export writes UTF-8 bytes instead."""
    tmp_path, session_id = seeded
    SessionStore().set_title(session_id, "制动 🙂", TitleSource.CUSTOM)
    cp1252_runner = CliRunner(charset="cp1252")
    result = cp1252_runner.invoke(app, _args(tmp_path, "export", session_id))
    assert result.exit_code == 0, result.output
    assert "# 制动 🙂" in result.stdout_bytes.decode("utf-8")


def test_export_json_carries_the_markdown(seeded):
    tmp_path, session_id = seeded
    result = runner.invoke(app, _args(tmp_path, "export", session_id, json_mode=True))
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "id": session_id,
        "markdown": session_markdown(SessionStore().get(session_id)),
    }


def test_export_to_a_file_writes_it_and_prints_the_path(seeded):
    tmp_path, session_id = seeded
    target = tmp_path / "out.md"
    result = runner.invoke(app, _args(tmp_path, "export", session_id, "-o", str(target)))
    assert result.exit_code == 0, result.output
    assert target.read_text(encoding="utf-8") == session_markdown(SessionStore().get(session_id))
    assert f"Exported to {target.resolve()}." in result.output


@pytest.mark.parametrize(
    "name",
    ["[/x].md", "a\\[\\x].md"],
    ids=["closing-tag", "backslash-before-bracket"],
)
def test_export_prints_a_bracketed_path_as_written(seeded, name):
    """A backslash before a bracket is a markup escape to Rich, and on Windows
    every separator is a backslash, so the path must not go through markup."""
    tmp_path, session_id = seeded
    target = tmp_path / "[draft]" / name
    result = runner.invoke(app, _args(tmp_path, "export", session_id, "-o", str(target)))
    assert result.exit_code == 0, result.output
    printed = result.output.strip().removeprefix("Exported to ").removesuffix(".")
    assert printed == str(target.resolve())
    assert Path(printed).is_file()


@pytest.mark.parametrize(
    "blocker", ["[red]", "b\\[\\x]"], ids=["style-tag", "backslash-before-bracket"]
)
def test_export_error_prints_a_bracketed_path_as_written(seeded, blocker):
    """The error names the path the user typed; markup would eat its brackets."""
    tmp_path, session_id = seeded
    blocking_file = tmp_path / blocker
    blocking_file.parent.mkdir(parents=True, exist_ok=True)
    blocking_file.write_text("x", encoding="utf-8")
    target = str(blocking_file / "out.md")
    with pytest.raises(OSError) as raised:
        write_session_markdown(SessionStore().get(session_id), target)
    result = runner.invoke(app, _args(tmp_path, "export", session_id, "-o", target))
    assert result.exit_code == 1
    assert result.output.strip() == f"Could not write the export: {raised.value}"


def test_export_into_a_directory_uses_the_default_name(seeded):
    tmp_path, session_id = seeded
    out_dir = tmp_path / "notes"
    out_dir.mkdir()
    result = runner.invoke(
        app, _args(tmp_path, "export", session_id, "--output", str(out_dir), json_mode=True)
    )
    assert result.exit_code == 0, result.output
    expected = (out_dir / f"torque-specs-{session_id[:8]}.md").resolve()
    assert json.loads(result.output) == {"id": session_id, "path": str(expected)}
    assert expected.is_file()


def test_export_that_cannot_write_exits_1(seeded):
    tmp_path, session_id = seeded
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    result = runner.invoke(
        app, _args(tmp_path, "export", session_id, "-o", str(blocker / "out.md"))
    )
    assert result.exit_code == 1
    assert "Could not write the export" in result.output


def test_export_unknown_prefix_errors(tmp_path):
    (tmp_path / "data").mkdir()
    result = runner.invoke(app, _args(tmp_path, "export", "deadbeef"))
    assert result.exit_code == 1
    assert "No session matching" in result.output
