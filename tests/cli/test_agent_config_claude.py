"""Tests for ``lilbee agent-config claude``."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from lilbee.catalog.types import ModelTask
from lilbee.cli import app

runner = CliRunner()

_TOKEN = "test-token-abc"
_PORT = 8123


@pytest.fixture(autouse=True)
def _stub_registry() -> None:
    manifest = MagicMock()
    manifest.ref = "Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf"
    manifest.task = ModelTask.CHAT
    registry = MagicMock()
    registry.list_installed.return_value = [manifest]
    services = MagicMock()
    services.registry = registry
    with patch("lilbee.cli.launchers.server.get_services", return_value=services):
        yield


def test_agent_config_claude_requires_running_server():
    with patch(
        "lilbee.cli.commands.agent_config.running_server_session", return_value=None
    ):
        result = runner.invoke(app, ["agent-config", "claude"])
    assert result.exit_code == 1
    assert "lilbee serve" in result.stderr


def test_agent_config_claude_prints_mcp_fragment():
    with patch(
        "lilbee.cli.commands.agent_config.running_server_session",
        return_value=(_TOKEN, _PORT),
    ):
        result = runner.invoke(app, ["agent-config", "claude"])
    assert result.exit_code == 0
    block = json.loads(result.stdout)
    server = block["mcpServers"]["lilbee"]
    assert server["type"] == "http"
    assert server["url"] == f"http://127.0.0.1:{_PORT}/mcp"
    # The paste path embeds the live token, matching the sibling commands.
    assert server["headers"]["Authorization"] == f"Bearer {_TOKEN}"
