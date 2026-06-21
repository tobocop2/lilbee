"""Tests for `lilbee agent-config opencode`."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from lilbee.cli import app
from lilbee.core.config import cfg
from lilbee.server.auth import server_json_path

runner = CliRunner()

_CHAT_REF_A = "Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf"
_CHAT_REF_B = "bartowski/SmolLM-135M-Instruct-GGUF/SmolLM-135M-Instruct.Q8_0.gguf"
_EMBED_REF = "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf"


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


def _fake_registry(refs_and_tasks: list[tuple[str, str]]) -> MagicMock:
    manifests = []
    for ref, task in refs_and_tasks:
        m = MagicMock()
        m.ref = ref
        m.task = task
        manifests.append(m)
    registry = MagicMock()
    registry.list_installed.return_value = manifests
    return registry


def test_opencode_config_prints_provider_with_real_port_and_token():
    _write_server_session("test-token-abc", 8765)
    registry = _fake_registry([(_CHAT_REF_A, "chat"), (_EMBED_REF, "embedding")])
    with patch(
        "lilbee.cli.launchers.server.get_services",
        return_value=MagicMock(registry=registry),
    ):
        result = runner.invoke(app, ["agent-config", "opencode"])

    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    options = payload["provider"]["lilbee"]["options"]
    assert options["baseURL"] == "http://127.0.0.1:8765/v1"
    assert options["apiKey"] == "test-token-abc"
    assert set(payload["provider"]["lilbee"]["models"].keys()) == {_CHAT_REF_A}
    # opencode picker shows a cleaned label rather than the full /-/-/.gguf ref.
    assert payload["provider"]["lilbee"]["models"][_CHAT_REF_A] == {"name": "Qwen3-0.6B Q4_K_M"}
    mcp_block = payload["mcp"]["lilbee"]
    assert mcp_block["type"] == "remote"
    assert mcp_block["url"] == "http://127.0.0.1:8765/mcp"
    assert mcp_block["headers"]["Authorization"] == "Bearer test-token-abc"
    assert mcp_block["enabled"] is True
    assert payload["$schema"] == "https://opencode.ai/config.json"


def test_opencode_config_lists_all_chat_models_sorted():
    _write_server_session("tok", 9000)
    registry = _fake_registry(
        [(_CHAT_REF_B, "chat"), (_CHAT_REF_A, "chat"), (_EMBED_REF, "embedding")]
    )
    with patch(
        "lilbee.cli.launchers.server.get_services",
        return_value=MagicMock(registry=registry),
    ):
        result = runner.invoke(app, ["agent-config", "opencode"])

    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    model_keys = list(payload["provider"]["lilbee"]["models"].keys())
    assert model_keys == sorted([_CHAT_REF_A, _CHAT_REF_B])


def test_opencode_config_without_running_server_exits_1():
    result = runner.invoke(app, ["agent-config", "opencode"])
    assert result.exit_code == 1
    assert "lilbee serve" in result.stderr


def test_opencode_config_accepts_data_root_flags(tmp_path):
    """Entry-point parity (bb-7jg1.21): agent-config accepts --data-dir/--global like
    its siblings, applying the override before resolving the server session."""
    alt = tmp_path / "alt"
    result = runner.invoke(app, ["agent-config", "opencode", "--data-dir", str(alt)])
    # Flags are recognized (not a usage error); no running server -> serve hint.
    assert "No such option" not in result.output
    assert result.exit_code == 1
    assert "lilbee serve" in result.stderr


def test_opencode_config_without_port_file_exits_1():
    server_json_path().write_text(json.dumps({"token": "t"}))
    # no server.port written
    result = runner.invoke(app, ["agent-config", "opencode"])
    assert result.exit_code == 1
    assert "lilbee serve" in result.stderr


def test_opencode_config_with_corrupt_server_json_exits_1():
    server_json_path().write_text("not json{{{")
    (cfg.data_dir / "server.port").write_text("8765")
    result = runner.invoke(app, ["agent-config", "opencode"])
    assert result.exit_code == 1
    assert "lilbee serve" in result.stderr


def test_opencode_config_with_non_string_token_exits_1():
    """server.json present but ``token`` field is not a string."""
    server_json_path().write_text(json.dumps({"token": 12345}))
    (cfg.data_dir / "server.port").write_text("8765")
    result = runner.invoke(app, ["agent-config", "opencode"])
    assert result.exit_code == 1
    assert "lilbee serve" in result.stderr
