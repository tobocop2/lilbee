"""Tests for the ``lilbee sessions`` commands under the sessions_enabled toggle."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from lilbee.cli import app
from lilbee.core.config import cfg

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
        ],
        ids=["list", "show", "rename", "delete"],
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
