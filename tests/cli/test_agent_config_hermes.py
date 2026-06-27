"""Tests for the hermes config-fragment builder and `lilbee agent-config hermes`."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from lilbee.catalog.types import ModelTask
from lilbee.cli import app
from lilbee.cli.agent_configs.hermes import hermes_config
from lilbee.core.config import cfg
from lilbee.server.auth import server_json_path

runner = CliRunner()

_REF = "Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf"
_TOKEN = "tok-abc"
_PORT = 8123


def test_provider_block_shape():
    cfg_frag = hermes_config(
        base_url="http://127.0.0.1:8080",
        api_key="${LILBEE_TOKEN}",
        model_refs=[_REF],
        default_ref=_REF,
        chat_ctx=8192,
    )
    prov = cfg_frag["providers"]["lilbee"]
    assert prov["api"] == "http://127.0.0.1:8080/v1"
    assert prov["api_key"] == "${LILBEE_TOKEN}"
    assert prov["default_model"] == _REF
    assert prov["context_length"] == 8192
    assert cfg_frag["model"] == {"default": _REF, "provider": "lilbee"}


def test_mcp_block_streamable_http_with_bearer():
    cfg_frag = hermes_config(
        base_url="http://127.0.0.1:8080", api_key="${LILBEE_TOKEN}", model_refs=[_REF]
    )
    mcp = cfg_frag["mcp_servers"]["lilbee"]
    assert mcp["url"] == "http://127.0.0.1:8080/mcp"
    assert mcp["transport"] == "streamable-http"
    assert mcp["headers"]["Authorization"] == "Bearer ${LILBEE_TOKEN}"
    assert mcp["timeout"] == 120


def test_no_mcp_omits_block():
    cfg_frag = hermes_config(
        base_url="http://127.0.0.1:8080",
        api_key="${LILBEE_TOKEN}",
        model_refs=[_REF],
        include_mcp=False,
    )
    assert "mcp_servers" not in cfg_frag


def test_literal_api_key_inline_for_paste():
    cfg_frag = hermes_config(
        base_url="http://127.0.0.1:8080", api_key="literal-tok", model_refs=[_REF]
    )
    assert cfg_frag["providers"]["lilbee"]["api_key"] == "literal-tok"
    assert cfg_frag["mcp_servers"]["lilbee"]["headers"]["Authorization"] == "Bearer literal-tok"


def test_default_falls_back_to_first_model_ref():
    cfg_frag = hermes_config(base_url="http://127.0.0.1:8080", api_key="k", model_refs=[_REF])
    assert cfg_frag["model"] == {"default": _REF, "provider": "lilbee"}
    assert cfg_frag["providers"]["lilbee"]["default_model"] == _REF


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    monkeypatch.delenv("LILBEE_DATA", raising=False)
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.data_dir.mkdir(exist_ok=True)
    yield tmp_path
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture(autouse=True)
def _stub_registry():
    manifest = MagicMock()
    manifest.ref = _REF
    manifest.task = ModelTask.CHAT
    registry = MagicMock()
    registry.list_installed.return_value = [manifest]
    services = MagicMock()
    services.registry = registry
    with patch("lilbee.cli.launchers.server.get_services", return_value=services):
        yield


def _write_server_session() -> None:
    server_json_path().write_text(json.dumps({"token": _TOKEN}))
    (cfg.data_dir / "server.port").write_text(str(_PORT))


def test_agent_config_hermes_prints_yaml_block(monkeypatch):
    _write_server_session()
    monkeypatch.setattr("lilbee.cli.commands.agent_config.served_chat_ctx", lambda _p: None)
    result = runner.invoke(app, ["agent-config", "hermes"])
    assert result.exit_code == 0
    block = yaml.safe_load(result.stdout)
    assert "lilbee" in block["providers"]
    assert block["providers"]["lilbee"]["api"] == f"http://127.0.0.1:{_PORT}/v1"
    # Paste path embeds the literal token (explicit copy), not an env ref.
    assert block["providers"]["lilbee"]["api_key"] == _TOKEN


def test_agent_config_hermes_requires_running_server():
    result = runner.invoke(app, ["agent-config", "hermes"])
    assert result.exit_code == 1
    assert "lilbee serve" in result.stderr
