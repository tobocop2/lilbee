"""lilbee engine stop kills the shared engine without the TUI."""

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


def _env(tmp_path) -> dict[str, str]:
    """Point the machine slot at a per-test dir so tests never touch the real one."""
    return {"LILBEE_ENGINE_DIR": str(tmp_path / "machine-slot")}


def _write_engine_state(tmp_path, *, where: str) -> object:
    engine_dir = tmp_path / "machine-slot" if where == "machine" else tmp_path / "data" / "engine"
    engine_dir.mkdir(parents=True, exist_ok=True)
    path = engine_dir / sm._state_filename(999_999, SwapGroup.CHAT.value)
    path.write_text(json.dumps({"pid": 999_998, "member_ports": [4000]}))
    return path


def test_stop_kills_the_machine_slot_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "_is_live_llama_swap", lambda _state: True)  # a live engine
    path = _write_engine_state(tmp_path, where="machine")
    result = runner.invoke(app, [*_data_args(tmp_path), "engine", "stop"], env=_env(tmp_path))
    assert result.exit_code == 0
    assert "Stopped the engine" in result.output
    assert not path.exists()


def test_stop_kills_a_private_overflow_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "_is_live_llama_swap", lambda _state: True)  # a live engine
    path = _write_engine_state(tmp_path, where="private")
    result = runner.invoke(app, [*_data_args(tmp_path), "engine", "stop"], env=_env(tmp_path))
    assert result.exit_code == 0
    assert "Stopped the engine" in result.output
    assert not path.exists()


def test_stop_reports_when_nothing_is_running(tmp_path):
    result = runner.invoke(app, [*_data_args(tmp_path), "engine", "stop"], env=_env(tmp_path))
    assert result.exit_code == 0
    assert "No engine is running" in result.output


def test_stop_cleans_a_stale_record_but_reports_nothing_running(tmp_path):
    # A dead swap with no live orphan servers: the record is stale, so the off
    # switch honestly reports nothing running (it did not kill anything) while
    # still unlinking the leftover record.
    path = _write_engine_state(tmp_path, where="machine")  # pid 999_998 is dead
    result = runner.invoke(app, [*_data_args(tmp_path), "engine", "stop"], env=_env(tmp_path))
    assert result.exit_code == 0
    assert "No engine is running" in result.output
    assert not path.exists()  # stale record cleaned up regardless


def test_stop_emits_json_in_json_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "_is_live_llama_swap", lambda _state: True)  # a live engine
    _write_engine_state(tmp_path, where="machine")
    result = runner.invoke(
        app, [*_data_args(tmp_path), "--json", "engine", "stop"], env=_env(tmp_path)
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["command"] == "engine stop"
    assert payload["stopped"] == ["chat"]
