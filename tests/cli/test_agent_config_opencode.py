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
    # Keyed and routed by the clean agent id (not the full ref); lilbee's /v1
    # resolves it back, so no surface leaks the shard path.
    assert set(payload["provider"]["lilbee"]["models"].keys()) == {"Qwen3-0.6B"}
    assert payload["provider"]["lilbee"]["models"]["Qwen3-0.6B"] == {"name": "Qwen3 0.6B"}
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
    # Keys are the clean agent ids, in the same (ref-sorted) order the entries emit.
    model_keys = list(payload["provider"]["lilbee"]["models"].keys())
    assert model_keys == ["Qwen3-0.6B", "SmolLM-135M"]


def test_opencode_config_without_running_server_exits_1():
    result = runner.invoke(app, ["agent-config", "opencode"])
    assert result.exit_code == 1
    assert "lilbee serve" in result.stderr


def test_opencode_config_applies_data_dir_override(tmp_path):
    """Entry-point parity (bb-7jg1.21): --data-dir is applied before the server
    session is resolved, so the session is read from the alt root. A running
    session exists ONLY under the alt root, so exit 0 + the alt port/token prove
    the override took effect (not merely that the flag parsed)."""
    alt = tmp_path / "alt"
    alt_data = alt / "data"
    alt_data.mkdir(parents=True)
    (alt_data / "server.json").write_text(json.dumps({"token": "alt-token"}))
    (alt_data / "server.port").write_text("8799")
    registry = _fake_registry([(_CHAT_REF_A, "chat")])
    with patch(
        "lilbee.cli.launchers.server.get_services",
        return_value=MagicMock(registry=registry),
    ):
        result = runner.invoke(app, ["agent-config", "opencode", "--data-dir", str(alt)])
    assert "No such option" not in result.output
    assert result.exit_code == 0, result.stderr
    options = json.loads(result.stdout)["provider"]["lilbee"]["options"]
    assert options["baseURL"] == "http://127.0.0.1:8799/v1"
    assert options["apiKey"] == "alt-token"


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
