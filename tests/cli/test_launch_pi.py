"""Tests for ``lilbee launch pi``."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from lilbee.app.agent_configs.pi import pi_models
from lilbee.catalog import agent_model_id
from lilbee.catalog.types import ModelTask
from lilbee.cli import app
from lilbee.cli.launchers import launch_app
from lilbee.cli.launchers.skill_install import bundled_pi_extension, bundled_skill_dir
from lilbee.core.config import cfg
from lilbee.server.auth import server_json_path

runner = CliRunner()

_CHAT_REF = "Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf"
_TOKEN = "pi-token-xyz"
_PORT = 8890
_PI = "/usr/local/bin/pi"
_SERVED_CTX = 65536


def _write_server_session() -> None:
    server_json_path().write_text(json.dumps({"token": _TOKEN}), encoding="utf-8")
    (cfg.data_dir / "server.port").write_text(str(_PORT), encoding="utf-8")


@pytest.fixture(autouse=True)
def _stub_registry():
    manifest = MagicMock()
    manifest.ref = _CHAT_REF
    manifest.task = ModelTask.CHAT
    registry = MagicMock()
    registry.list_installed.return_value = [manifest]
    services = MagicMock()
    services.registry = registry
    services.known_models.resolve.return_value = None
    with patch("lilbee.app.models.get_services", return_value=services):
        yield


@pytest.fixture(autouse=True)
def _healthy_and_warm(monkeypatch):
    monkeypatch.setattr("lilbee.cli.launchers.server.health_ok", lambda _port: True)
    monkeypatch.setattr("lilbee.cli.launchers.launcher.wait_for_chat_warm", lambda _port: True)
    monkeypatch.setattr("lilbee.cli.launchers.pi.client_chat_ctx", lambda _port: _SERVED_CTX)


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch) -> Path:
    monkeypatch.delenv("LILBEE_DATA", raising=False)
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


def _launch(*flags: str, agent_rc: int = 0):
    """Run ``lilbee launch pi``; return (result, the one pi call)."""
    _write_server_session()
    calls: list = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return MagicMock(returncode=agent_rc)

    with (
        patch("lilbee.cli.launchers.pi.shutil.which", return_value=_PI),
        patch("lilbee.cli.launchers.launcher.subprocess.run", side_effect=fake_run),
        patch("lilbee.cli.launchers.server.spawn_server") as spawn,
    ):
        result = runner.invoke(app, ["launch", "pi", *flags])
    spawn.assert_not_called()
    assert len(calls) <= 1
    return result, (calls[0] if calls else None)


def _files_under(root: Path) -> set[Path]:
    return {p for p in root.rglob("*") if p.is_file()}


def test_launch_pi_without_binary_exits_1():
    _write_server_session()
    with patch("lilbee.cli.launchers.pi.shutil.which", return_value=None):
        result = runner.invoke(app, ["launch", "pi"])
    assert result.exit_code == 1
    assert "pi binary not found on PATH" in result.stderr


def test_launch_pi_loads_the_extension_and_skill_and_pins_the_model(monkeypatch):
    served = 2 * _SERVED_CTX
    monkeypatch.setattr("lilbee.cli.launchers.pi.client_chat_ctx", lambda _port: served)
    result, agent = _launch()
    assert result.exit_code == 0, result.output
    cmd, kwargs = agent
    assert cmd == [
        _PI,
        "-e",
        str(bundled_pi_extension()),
        "--skill",
        str(bundled_skill_dir()),
        "--model",
        f"lilbee/{agent_model_id(_CHAT_REF)}",
    ]
    env = kwargs["env"]
    assert env["LILBEE_TOKEN"] == _TOKEN
    assert env["LILBEE_OPENAI_URL"] == f"http://127.0.0.1:{_PORT}/v1"
    assert env["LILBEE_MCP_URL"] == f"http://127.0.0.1:{_PORT}/mcp"
    models = json.loads(env["LILBEE_PI_MODELS"])
    assert models == pi_models([_CHAT_REF], served)
    assert [m["contextWindow"] for m in models] == [served]


def test_launch_pi_writes_no_file_under_home(tmp_path):
    _write_server_session()
    before = _files_under(tmp_path)
    assert before
    result, agent = _launch()
    assert result.exit_code == 0, result.output
    assert agent is not None
    assert _files_under(tmp_path) == before
    assert not (tmp_path / ".pi").exists()


def test_launch_pi_passes_agent_exit_code():
    result, _ = _launch(agent_rc=3)
    assert result.exit_code == 3


def test_launch_pi_no_pin_when_chat_model_not_installed(monkeypatch):
    monkeypatch.setattr(cfg, "chat_model", "owner/Absent-GGUF/absent.Q4_K_M.gguf")
    other = "owner/Other-GGUF/other.Q4_K_M.gguf"
    with patch("lilbee.cli.launchers.launcher.installed_chat_model_refs", return_value=[other]):
        result, agent = _launch()
    assert result.exit_code == 0
    assert "--model" not in agent[0]
    assert [m["id"] for m in json.loads(agent[1]["env"]["LILBEE_PI_MODELS"])] == [
        agent_model_id(other)
    ]


def test_launch_pi_uses_the_target_window_when_the_served_one_is_unknown(monkeypatch):
    monkeypatch.setattr("lilbee.cli.launchers.pi.client_chat_ctx", lambda _port: None)
    result, agent = _launch()
    assert result.exit_code == 0
    models = json.loads(agent[1]["env"]["LILBEE_PI_MODELS"])
    assert [m["contextWindow"] for m in models] == [cfg.chat_n_ctx_target]


def test_launch_pi_no_mcp_drops_the_server_and_skill():
    result, agent = _launch("--no-mcp")
    assert result.exit_code == 0
    cmd, kwargs = agent
    assert cmd[:3] == [_PI, "-e", str(bundled_pi_extension())]
    assert "--skill" not in cmd
    assert "LILBEE_MCP_URL" not in kwargs["env"]
    assert "LILBEE_OPENAI_URL" in kwargs["env"]


def test_launch_pi_config_default_off_and_mcp_override(monkeypatch):
    monkeypatch.setattr(cfg, "agent_mcp_enabled", False)
    _, agent = _launch()
    assert "LILBEE_MCP_URL" not in agent[1]["env"]
    _, agent = _launch("--mcp")
    assert "LILBEE_MCP_URL" in agent[1]["env"]


def test_launch_app_lists_pi():
    names = {cmd.name for cmd in launch_app.registered_commands}
    assert names
    assert {"opencode", "hermes", "claude", "pi"} <= names
