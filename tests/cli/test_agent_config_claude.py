"""Tests for `lilbee agent-config claude`."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from lilbee.cli import app
from lilbee.core.config import cfg
from lilbee.server.auth import server_json_path

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.delenv("LILBEE_DATA", raising=False)
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir(exist_ok=True)
    cfg.data_dir = tmp_path / "data"
    cfg.data_dir.mkdir(exist_ok=True)
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    yield tmp_path
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _write_server_session(token: str, port: int) -> None:
    server_json_path().write_text(json.dumps({"token": token}))
    (cfg.data_dir / "server.port").write_text(str(port))


def test_claude_config_prints_the_http_registration_with_the_live_session():
    _write_server_session("test-token-abc", 8765)
    result = runner.invoke(app, ["agent-config", "claude"])

    assert result.exit_code == 0, result.stderr
    server = json.loads(result.stdout)["mcpServers"]["lilbee"]
    assert server["type"] == "http"
    assert server["url"] == "http://127.0.0.1:8765/mcp"
    assert server["headers"]["Authorization"] == "Bearer test-token-abc"


def test_claude_config_offers_the_stdio_registration_on_stderr():
    """stdout stays a single pasteable block; the alternative goes to stderr."""
    _write_server_session("tok", 9000)
    result = runner.invoke(app, ["agent-config", "claude"])

    assert result.exit_code == 0, result.stderr
    assert '"command": "lilbee"' in result.stderr
    assert '"mcp"' in result.stderr
    assert "stdio" not in result.stdout


def test_claude_config_does_not_read_the_model_registry():
    """Claude Code takes the MCP tools only, so no provider or model list is built."""
    _write_server_session("tok", 9000)
    registry = MagicMock()
    with patch("lilbee.app.models.get_services", return_value=MagicMock(registry=registry)):
        result = runner.invoke(app, ["agent-config", "claude"])

    assert result.exit_code == 0, result.stderr
    registry.list_installed.assert_not_called()
    assert "provider" not in result.stdout


def test_claude_config_without_running_server_exits_1():
    result = runner.invoke(app, ["agent-config", "claude"])
    assert result.exit_code == 1
    assert "lilbee serve" in result.stderr


def test_claude_config_applies_data_dir_override(tmp_path):
    alt = tmp_path / "alt"
    alt_data = alt / "data"
    alt_data.mkdir(parents=True)
    (alt_data / "server.json").write_text(json.dumps({"token": "alt-token"}))
    (alt_data / "server.port").write_text("8799")

    result = runner.invoke(app, ["agent-config", "claude", "--data-dir", str(alt)])

    assert "No such option" not in result.output
    assert result.exit_code == 0, result.stderr
    server = json.loads(result.stdout)["mcpServers"]["lilbee"]
    assert server["url"] == "http://127.0.0.1:8799/mcp"
    assert server["headers"]["Authorization"] == "Bearer alt-token"
