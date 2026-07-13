"""lilbee engine stop kills a warm fleet without the TUI."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from lilbee.cli import app
from lilbee.providers.fleet import swap_manager as sm
from lilbee.providers.fleet.groups import SwapGroup

runner = CliRunner()


def _data_args(tmp_path) -> list[str]:
    """--data-dir wins the CLI's data-root resolution, unlike a cfg monkeypatch."""
    (tmp_path / "data").mkdir(exist_ok=True)
    return ["--data-dir", str(tmp_path)]


def _write_detached(tmp_path) -> object:
    (tmp_path / "data").mkdir(exist_ok=True)
    path = tmp_path / "data" / sm._state_filename(999_999, SwapGroup.CHAT.value)
    path.write_text(json.dumps({"pid": 999_998, "owner_pid": 999_999, "detached": True}))
    return path


def test_stop_reaps_a_detached_fleet(tmp_path):
    path = _write_detached(tmp_path)
    result = runner.invoke(app, [*_data_args(tmp_path), "engine", "stop"])
    assert result.exit_code == 0
    assert "Stopped the warm engine" in result.output
    assert not path.exists()


def test_stop_reports_when_nothing_is_running(tmp_path):
    result = runner.invoke(app, [*_data_args(tmp_path), "engine", "stop"])
    assert result.exit_code == 0
    assert "No warm engine is running" in result.output


def test_stop_emits_json_in_json_mode(tmp_path):
    _write_detached(tmp_path)
    result = runner.invoke(app, [*_data_args(tmp_path), "--json", "engine", "stop"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["command"] == "engine stop"
    assert payload["stopped"] == ["chat"]
