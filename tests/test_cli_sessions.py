"""Tests for the ``lilbee sessions`` commands under the sessions_enabled toggle."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from lilbee.cli import app
from lilbee.core.config import cfg
from tests._crafted_titles import CONTROL_TITLES, TERMINAL_CONTROL_RE

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


class TestSessionsDisabled:
    """With the toggle off the commands report it and touch nothing, matching
    how ``lilbee memory`` refuses when the memory subsystem is off.
    """

    @pytest.mark.parametrize(
        "argv",
        [
            ["sessions", "list"],
            ["sessions", "show", "abc"],
            ["sessions", "rename", "abc", "new title"],
            ["sessions", "delete", "abc", "--yes"],
            ["sessions", "fork", "abc"],
            ["sessions", "export", "abc"],
        ],
        ids=["list", "show", "rename", "delete", "fork", "export"],
    )
    def test_command_reports_sessions_are_off(self, argv):
        cfg.sessions_enabled = False
        result = runner.invoke(app, argv)
        assert result.exit_code == 0, result.output
        assert "sessions are off" in result.output.lower()

    def test_json_mode_reports_the_error_envelope(self):
        cfg.sessions_enabled = False
        result = runner.invoke(app, ["--json", "sessions", "list"])
        assert result.exit_code == 0, result.output
        assert "sessions are off" in json.loads(result.output)["error"].lower()

    def test_store_is_never_constructed(self, monkeypatch):
        """The refusal happens before the store is opened, so a disabled run
        cannot create the sessions directory or take the append lock.
        """
        cfg.sessions_enabled = False
        constructed: list[object] = []
        monkeypatch.setattr(
            "lilbee.cli.sessions.SessionStore",
            lambda *args, **kwargs: constructed.append(object()),
        )
        result = runner.invoke(app, ["sessions", "list"])
        assert result.exit_code == 0, result.output
        assert constructed == []


@pytest.fixture
def stored_session(tmp_path, monkeypatch):
    """Seed one session through the real store, and return a function that titles it."""
    from lilbee.cli.app import apply_overrides
    from lilbee.sessions import MessageRole, SessionMessage, SessionStore, TitleSource

    monkeypatch.setenv("LILBEE_DATA", str(tmp_path))
    apply_overrides()
    cfg.sessions_enabled = True
    store = SessionStore()
    session_id = store.create(model_ref="m", scope="both")

    def seed(title: str, content: str = "hello") -> str:
        store.set_title(session_id, title, TitleSource.CUSTOM)
        store.add_message(session_id, SessionMessage(role=MessageRole.USER, content=content))
        return session_id

    return seed


class TestTerminalSafeStoredText:
    """A stored session title or message reaches the terminal without a control character."""

    @pytest.mark.parametrize("title", CONTROL_TITLES)
    def test_list_writes_no_control_character_of_a_title(self, stored_session, title):
        stored_session(title)
        result = runner.invoke(app, ["sessions", "list"])
        assert result.exit_code == 0, result.output
        assert TERMINAL_CONTROL_RE.search(result.output) is None

    @pytest.mark.parametrize("title", CONTROL_TITLES)
    def test_list_json_keeps_the_title_as_stored(self, stored_session, title):
        stored_session(title)
        result = runner.invoke(app, ["--json", "sessions", "list"])
        assert json.loads(result.output)["sessions"][0]["title"] == title

    @pytest.mark.parametrize("text", CONTROL_TITLES)
    def test_show_writes_no_control_character_of_a_title_or_message(self, stored_session, text):
        session_id = stored_session(text, content=text)
        result = runner.invoke(app, ["sessions", "show", session_id])
        assert result.exit_code == 0, result.output
        assert TERMINAL_CONTROL_RE.search(result.output) is None

    @pytest.mark.parametrize("text", CONTROL_TITLES)
    def test_show_json_keeps_the_message_as_stored(self, stored_session, text):
        session_id = stored_session("t", content=text)
        result = runner.invoke(app, ["--json", "sessions", "show", session_id])
        assert json.loads(result.output)["messages"][0]["content"] == text

    @pytest.mark.parametrize("title", CONTROL_TITLES)
    def test_fork_writes_no_control_character_of_the_new_title(self, stored_session, title):
        session_id = stored_session(title)
        result = runner.invoke(app, ["sessions", "fork", session_id])
        assert result.exit_code == 0, result.output
        assert "Forked to" in result.output
        assert TERMINAL_CONTROL_RE.search(result.output) is None


class TestMultiLineBodiesKeepTheirLines:
    """A message body is shown line for line; only the escapes go."""

    def test_show_keeps_the_lines_of_a_message(self, stored_session):
        body = "# Title\n\n- a\n- b\n\n```py\nx = 1\n```\n\x1b[31m"
        session_id = stored_session("t", content=body)
        result = runner.invoke(app, ["sessions", "show", session_id])
        assert result.exit_code == 0, result.output
        lines = [line.strip() for line in result.output.splitlines()]
        assert "user: # Title" in lines
        assert lines[lines.index("user: # Title") + 2 :][:3] == ["- a", "- b", ""]
        assert "x = 1" in lines
        assert "\x1b" not in result.output
