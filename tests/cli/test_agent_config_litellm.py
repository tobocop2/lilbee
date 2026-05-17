"""Tests for `lilbee agent-config litellm`."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import yaml
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


def test_litellm_config_emits_yaml_with_one_entry_per_chat_model():
    _write_server_session("litellm-token", 8765)
    registry = _fake_registry(
        [(_CHAT_REF_A, "chat"), (_CHAT_REF_B, "chat"), (_EMBED_REF, "embedding")]
    )
    with patch(
        "lilbee.cli.commands.agent_config.get_services",
        return_value=MagicMock(registry=registry),
    ):
        result = runner.invoke(app, ["agent-config", "litellm"])

    assert result.exit_code == 0, result.stderr
    payload = yaml.safe_load(result.stdout)
    assert isinstance(payload, dict)
    entries = payload["model_list"]
    assert len(entries) == 2
    model_names = {entry["model_name"] for entry in entries}
    assert model_names == {f"lilbee/{_CHAT_REF_A}", f"lilbee/{_CHAT_REF_B}"}
    for entry in entries:
        params = entry["litellm_params"]
        assert params["api_base"] == "http://127.0.0.1:8765/v1"
        assert params["api_key"] == "litellm-token"
        # `model` must use the openai/<ref> prefix so LiteLLM routes via openai
        assert params["model"].startswith("openai/")


def test_litellm_config_skips_non_chat_models():
    _write_server_session("tok", 9000)
    registry = _fake_registry([(_EMBED_REF, "embedding")])
    with patch(
        "lilbee.cli.commands.agent_config.get_services",
        return_value=MagicMock(registry=registry),
    ):
        result = runner.invoke(app, ["agent-config", "litellm"])

    assert result.exit_code == 0, result.stderr
    payload = yaml.safe_load(result.stdout)
    assert payload == {"model_list": []}


def test_litellm_config_entries_are_sorted():
    _write_server_session("tok", 9000)
    registry = _fake_registry([(_CHAT_REF_A, "chat"), (_CHAT_REF_B, "chat")])
    with patch(
        "lilbee.cli.commands.agent_config.get_services",
        return_value=MagicMock(registry=registry),
    ):
        result = runner.invoke(app, ["agent-config", "litellm"])

    assert result.exit_code == 0, result.stderr
    payload = yaml.safe_load(result.stdout)
    names = [e["model_name"] for e in payload["model_list"]]
    assert names == sorted(names)


def test_litellm_config_without_running_server_exits_1():
    result = runner.invoke(app, ["agent-config", "litellm"])
    assert result.exit_code == 1
    assert "lilbee serve" in result.stderr
