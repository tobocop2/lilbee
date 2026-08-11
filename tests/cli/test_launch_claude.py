"""Tests for ``lilbee launch claude``."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from lilbee.catalog import agent_model_id
from lilbee.catalog.types import ModelTask
from lilbee.cli import app
from lilbee.core.config import cfg
from lilbee.server.auth import server_json_path

runner = CliRunner()

_CHAT_REF = "Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf"
_TOKEN = "test-token-xyz"
_PORT = 8888


def _write_server_session() -> None:
    server_json_path().write_text(json.dumps({"token": _TOKEN}), encoding="utf-8")
    (cfg.data_dir / "server.port").write_text(str(_PORT), encoding="utf-8")


@pytest.fixture(autouse=True)
def _stub_registry() -> None:
    manifest = MagicMock()
    manifest.ref = _CHAT_REF
    manifest.task = ModelTask.CHAT
    registry = MagicMock()
    registry.list_installed.return_value = [manifest]
    services = MagicMock()
    services.registry = registry
    with patch("lilbee.cli.launchers.server.get_services", return_value=services):
        yield


@pytest.fixture(autouse=True)
def _healthy_by_default(monkeypatch):
    monkeypatch.setattr("lilbee.cli.launchers.server.health_ok", lambda _port: True)


@pytest.fixture(autouse=True)
def _warm_by_default(monkeypatch):
    monkeypatch.setattr("lilbee.cli.launchers.launcher.wait_for_chat_warm", lambda _port: True)
    # prepare() reads the served window over HTTP; default it to unknown so
    # tests do not hit the network. The auto-compact test overrides.
    monkeypatch.setattr("lilbee.cli.launchers.claude.client_chat_ctx", lambda _port: None)


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch) -> Path:
    monkeypatch.delenv("LILBEE_DATA", raising=False)
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir(exist_ok=True)
    cfg.data_dir = tmp_path / "data"
    cfg.data_dir.mkdir(exist_ok=True)
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.chat_model = _CHAT_REF
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    yield tmp_path
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _launch(*args: str):
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.launchers.claude.shutil.which", return_value="/usr/local/bin/claude"),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed) as run,
    ):
        result = runner.invoke(app, ["launch", "claude", *args])
    return result, run


def test_launch_claude_without_binary_exits_1(tmp_path):
    _write_server_session()
    with patch("lilbee.cli.launchers.claude.shutil.which", return_value=None):
        result = runner.invoke(app, ["launch", "claude"])
    assert result.exit_code == 1
    assert "claude binary not found" in result.stderr


def test_find_binary_falls_back_to_claude_local(tmp_path):
    from lilbee.cli.launchers.claude import _find_claude_binary

    local = tmp_path / ".claude" / "local" / "claude"
    local.parent.mkdir(parents=True)
    local.write_text("#!/bin/sh\n", encoding="utf-8")
    local.chmod(0o755)
    with patch("lilbee.cli.launchers.claude.shutil.which", return_value=None):
        assert _find_claude_binary() == str(local)


def test_launch_claude_wires_anthropic_env(tmp_path):
    _write_server_session()
    result, run = _launch()
    assert result.exit_code == 0
    run.assert_called_once()
    argv = run.call_args.args[0]
    env = run.call_args.kwargs["env"]
    model_id = agent_model_id(_CHAT_REF)
    assert argv[0] == "/usr/local/bin/claude"
    assert argv[1:3] == ["--model", model_id]
    assert env["ANTHROPIC_BASE_URL"] == f"http://127.0.0.1:{_PORT}"
    assert env["ANTHROPIC_AUTH_TOKEN"] == _TOKEN
    assert env["ANTHROPIC_API_KEY"] == ""  # a real key must not leak through
    for tier in ("OPUS", "SONNET", "HAIKU"):
        assert env[f"ANTHROPIC_DEFAULT_{tier}_MODEL"] == model_id
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == model_id
    assert env["LILBEE_TOKEN"] == _TOKEN
    assert env["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"
    assert env["DISABLE_ERROR_REPORTING"] == "1"
    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW" not in env  # window unknown here


def test_launch_claude_sets_auto_compact_window_when_ctx_known(tmp_path, monkeypatch):
    _write_server_session()
    monkeypatch.setattr("lilbee.cli.launchers.claude.client_chat_ctx", lambda _port: 65536)
    result, run = _launch()
    assert result.exit_code == 0
    env = run.call_args.kwargs["env"]
    assert env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "65536"


def test_launch_claude_writes_mcp_config_with_env_ref(tmp_path):
    _write_server_session()
    result, run = _launch()
    assert result.exit_code == 0
    argv = run.call_args.args[0]
    mcp_path = cfg.data_dir / "launchers" / "claude-mcp.json"
    assert "--mcp-config" in argv
    assert argv[argv.index("--mcp-config") + 1] == str(mcp_path)
    config = json.loads(mcp_path.read_text(encoding="utf-8"))
    server = config["mcpServers"]["lilbee"]
    assert server["type"] == "http"
    assert server["url"] == f"http://127.0.0.1:{_PORT}/mcp"
    # The env reference, never the literal token
    assert server["headers"]["Authorization"] == "Bearer ${LILBEE_TOKEN}"
    assert _TOKEN not in mcp_path.read_text(encoding="utf-8")


def test_launch_claude_installs_skill_and_records_marker(tmp_path):
    _write_server_session()
    result, _run = _launch()
    assert result.exit_code == 0
    assert (tmp_path / ".claude" / "skills" / "lilbee-mcp" / "SKILL.md").exists()
    assert (cfg.data_dir / "launchers" / "claude-setup.json").exists()
    assert "First-time Claude Code setup will write" in result.stdout


def test_launch_claude_no_mcp_skips_config_skill_and_gate(tmp_path):
    _write_server_session()
    result, run = _launch("--no-mcp")
    assert result.exit_code == 0
    argv = run.call_args.args[0]
    assert "--mcp-config" not in argv
    assert not (cfg.data_dir / "launchers" / "claude-mcp.json").exists()
    assert not (tmp_path / ".claude" / "skills" / "lilbee-mcp").exists()
    assert "First-time Claude Code setup" not in result.stdout


def test_launch_claude_interactive_decline_skips_launch(tmp_path):
    _write_server_session()
    with (
        patch("lilbee.cli.launchers.claude.shutil.which", return_value="/usr/local/bin/claude"),
        patch("lilbee.cli.launchers.setup_gate._is_interactive", return_value=True),
        patch("lilbee.cli.launchers.setup_gate.typer.confirm", return_value=False),
        patch("lilbee.cli.launchers.launcher.subprocess.run") as run,
    ):
        result = runner.invoke(app, ["launch", "claude"])
    assert result.exit_code == 0
    run.assert_not_called()
    assert not (cfg.data_dir / "launchers" / "claude-setup.json").exists()


def test_launch_claude_propagates_exit_code(tmp_path):
    _write_server_session()
    completed = MagicMock(returncode=42)
    with (
        patch("lilbee.cli.launchers.claude.shutil.which", return_value="/usr/local/bin/claude"),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
    ):
        result = runner.invoke(app, ["launch", "claude"])
    assert result.exit_code == 42
