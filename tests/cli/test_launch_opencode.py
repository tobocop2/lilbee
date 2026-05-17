"""Tests for `lilbee opencode` (launcher)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from lilbee.catalog.types import ModelTask
from lilbee.cli import app
from lilbee.cli.agent_configs.opencode import LILBEE_PRIMING
from lilbee.core.config import cfg
from lilbee.server.auth import server_json_path

runner = CliRunner()

_CHAT_REF = "Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf"
_TOKEN = "test-token-xyz"
_PORT = 8888


def _write_server_session() -> None:
    server_json_path().write_text(json.dumps({"token": _TOKEN}))
    (cfg.data_dir / "server.port").write_text(str(_PORT))


@pytest.fixture(autouse=True)
def _stub_registry() -> None:
    manifest = MagicMock()
    manifest.ref = _CHAT_REF
    manifest.task = ModelTask.CHAT
    registry = MagicMock()
    registry.list_installed.return_value = [manifest]
    services = MagicMock()
    services.registry = registry
    with patch("lilbee.cli.commands.agent_config.get_services", return_value=services):
        yield


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch) -> Path:
    """Isolate cfg.data_dir and redirect Path.home so launcher writes land in tmp."""
    monkeypatch.delenv("LILBEE_DATA", raising=False)
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir(exist_ok=True)
    cfg.data_dir = tmp_path / "data"
    cfg.data_dir.mkdir(exist_ok=True)
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    yield tmp_path
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def test_opencode_launch_without_opencode_binary_exits_1():
    _write_server_session()
    with patch("lilbee.cli.commands.launch.shutil.which", return_value=None):
        result = runner.invoke(app, ["launch", "opencode"])
    assert result.exit_code == 1
    assert "opencode binary not found" in result.stderr


def test_opencode_launch_reuses_running_server_and_invokes_opencode(tmp_path):
    _write_server_session()
    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.commands.launch.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.commands.launch.subprocess.run", return_value=completed) as run,
        patch("lilbee.cli.commands.launch._spawn_server") as spawn,
    ):
        result = runner.invoke(app, ["launch", "opencode"])

    assert result.exit_code == 0
    spawn.assert_not_called()
    run.assert_called_once()
    call_args, call_kwargs = run.call_args
    assert call_args[0] == [fake_opencode]
    config_path = Path(call_kwargs["env"]["OPENCODE_CONFIG"])
    assert config_path.exists()
    payload = json.loads(config_path.read_text())
    assert payload["provider"]["lilbee"]["options"]["baseURL"] == f"http://127.0.0.1:{_PORT}/v1"
    assert payload["provider"]["lilbee"]["options"]["apiKey"] == _TOKEN
    assert _CHAT_REF in payload["provider"]["lilbee"]["models"]
    instructions = payload.get("instructions") or []
    assert any(Path(p).name == "AGENTS.md" for p in instructions)
    priming = (config_path.parent / "AGENTS.md").read_text()
    assert priming == LILBEE_PRIMING


def test_opencode_launch_spawns_server_when_none_running():
    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=0)
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None

    def _spawn_and_write(port: int):
        # Simulate the server writing its session files mid-boot.
        _write_server_session()
        return fake_proc

    with (
        patch("lilbee.cli.commands.launch.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.commands.launch._spawn_server", side_effect=_spawn_and_write),
        patch("lilbee.cli.commands.launch._wait_for_health", return_value=True),
        patch("lilbee.cli.commands.launch._free_port", return_value=_PORT),
        patch("lilbee.cli.commands.launch.subprocess.run", return_value=completed),
    ):
        result = runner.invoke(app, ["launch", "opencode"])

    assert result.exit_code == 0
    fake_proc.terminate.assert_called_once()


def test_opencode_launch_keep_serving_skips_terminate():
    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=0)
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None

    def _spawn_and_write(port: int):
        _write_server_session()
        return fake_proc

    with (
        patch("lilbee.cli.commands.launch.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.commands.launch._spawn_server", side_effect=_spawn_and_write),
        patch("lilbee.cli.commands.launch._wait_for_health", return_value=True),
        patch("lilbee.cli.commands.launch._free_port", return_value=_PORT),
        patch("lilbee.cli.commands.launch.subprocess.run", return_value=completed),
    ):
        result = runner.invoke(app, ["launch", "opencode", "--keep-serving"])

    assert result.exit_code == 0
    fake_proc.terminate.assert_not_called()


def test_opencode_launch_health_timeout_terminates_spawn_and_exits_1():
    fake_opencode = "/usr/local/bin/opencode"
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    with (
        patch("lilbee.cli.commands.launch.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.commands.launch._spawn_server", return_value=fake_proc),
        patch("lilbee.cli.commands.launch._wait_for_health", return_value=False),
        patch("lilbee.cli.commands.launch._free_port", return_value=_PORT),
    ):
        result = runner.invoke(app, ["launch", "opencode"])
    assert result.exit_code == 1
    assert "failed to start" in result.stderr
    fake_proc.terminate.assert_called_once()


def test_opencode_launch_health_ok_but_session_missing_exits_1():
    """Health endpoint replies but server.json never lands (rare race)."""
    fake_opencode = "/usr/local/bin/opencode"
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    with (
        patch("lilbee.cli.commands.launch.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.commands.launch._spawn_server", return_value=fake_proc),
        patch("lilbee.cli.commands.launch._wait_for_health", return_value=True),
        patch("lilbee.cli.commands.launch._free_port", return_value=_PORT),
    ):
        result = runner.invoke(app, ["launch", "opencode"])
    assert result.exit_code == 1
    assert "did not write a session file" in result.stderr
    fake_proc.terminate.assert_called_once()


def test_free_port_returns_open_port():
    from lilbee.cli.commands import launch as launch_mod

    port = launch_mod._free_port()
    assert 1024 < port < 65536


def test_wait_for_health_returns_true_on_200(monkeypatch):
    from lilbee.cli.commands import launch as launch_mod

    resp = MagicMock()
    resp.status_code = 200
    monkeypatch.setattr(launch_mod.httpx, "get", lambda url, timeout: resp)
    assert launch_mod._wait_for_health(8765, timeout_s=1.0) is True


def test_wait_for_health_swallows_http_errors_until_timeout(monkeypatch):
    from lilbee.cli.commands import launch as launch_mod

    def _boom(url, timeout):
        raise launch_mod.httpx.HTTPError("connection refused")

    monkeypatch.setattr(launch_mod.httpx, "get", _boom)
    monkeypatch.setattr(launch_mod.time, "sleep", lambda _seconds: None)
    assert launch_mod._wait_for_health(8765, timeout_s=0.05) is False


def test_wait_for_health_returns_false_on_non_200(monkeypatch):
    from lilbee.cli.commands import launch as launch_mod

    resp = MagicMock()
    resp.status_code = 503
    monkeypatch.setattr(launch_mod.httpx, "get", lambda url, timeout: resp)
    monkeypatch.setattr(launch_mod.time, "sleep", lambda _seconds: None)
    assert launch_mod._wait_for_health(8765, timeout_s=0.05) is False


def test_spawn_server_returns_popen(monkeypatch):
    from lilbee.cli.commands import launch as launch_mod

    fake = MagicMock()
    monkeypatch.setattr(launch_mod.subprocess, "Popen", lambda *a, **k: fake)
    out = launch_mod._spawn_server(8765)
    assert out is fake


def test_opencode_launch_kills_when_terminate_times_out():
    import subprocess as _subprocess

    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=0)
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    fake_proc.wait.side_effect = [_subprocess.TimeoutExpired(cmd="lilbee serve", timeout=10), None]

    def _spawn_and_write(port: int):
        _write_server_session()
        return fake_proc

    with (
        patch("lilbee.cli.commands.launch.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.commands.launch._spawn_server", side_effect=_spawn_and_write),
        patch("lilbee.cli.commands.launch._wait_for_health", return_value=True),
        patch("lilbee.cli.commands.launch._free_port", return_value=_PORT),
        patch("lilbee.cli.commands.launch.subprocess.run", return_value=completed),
    ):
        result = runner.invoke(app, ["launch", "opencode"])

    assert result.exit_code == 0
    fake_proc.terminate.assert_called_once()
    fake_proc.kill.assert_called_once()


def test_opencode_launch_propagates_opencode_exit_code():
    _write_server_session()
    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=42)
    with (
        patch("lilbee.cli.commands.launch.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.commands.launch.subprocess.run", return_value=completed),
    ):
        result = runner.invoke(app, ["launch", "opencode"])
    assert result.exit_code == 42
