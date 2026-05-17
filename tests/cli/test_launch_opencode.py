"""Tests for ``lilbee launch opencode``."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from lilbee.catalog.types import ModelTask
from lilbee.cli import app
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


def test_launch_opencode_without_binary_exits_1():
    _write_server_session()
    with patch("lilbee.cli.commands.launch.shutil.which", return_value=None):
        result = runner.invoke(app, ["launch", "opencode"])
    assert result.exit_code == 1
    assert "opencode binary not found" in result.stderr


def test_launch_opencode_with_running_server_emits_inline_config_env(tmp_path):
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
    _, call_kwargs = run.call_args
    raw = call_kwargs["env"]["OPENCODE_CONFIG_CONTENT"]
    payload = json.loads(raw)
    options = payload["provider"]["lilbee"]["options"]
    assert options["baseURL"] == f"http://127.0.0.1:{_PORT}/v1"
    assert options["apiKey"] == _TOKEN
    assert _CHAT_REF in payload["provider"]["lilbee"]["models"]
    assert "mcp" not in payload, "launcher inline config must not bundle mcp"


def test_launch_opencode_installs_skill_into_global_skills_dir(tmp_path):
    _write_server_session()
    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.commands.launch.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.commands.launch.subprocess.run", return_value=completed),
    ):
        runner.invoke(app, ["launch", "opencode"])

    skill_path = tmp_path / ".config" / "opencode" / "skills" / "lilbee-mcp" / "SKILL.md"
    assert skill_path.exists()
    assert "lilbee-mcp" in skill_path.read_text()


def test_launch_opencode_skips_skill_install_when_already_present(tmp_path):
    _write_server_session()
    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=0)
    existing_dir = tmp_path / ".config" / "opencode" / "skills" / "lilbee-mcp"
    existing_dir.mkdir(parents=True)
    custom = existing_dir / "SKILL.md"
    custom.write_text("user customization")
    with (
        patch("lilbee.cli.commands.launch.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.commands.launch.subprocess.run", return_value=completed),
    ):
        runner.invoke(app, ["launch", "opencode"])

    assert custom.read_text() == "user customization"


def test_launch_opencode_updates_picker_state_on_unix(tmp_path):
    _write_server_session()
    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.commands.launch.sys.platform", "darwin"),
        patch("lilbee.cli.commands.launch.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.commands.launch.subprocess.run", return_value=completed),
    ):
        runner.invoke(app, ["launch", "opencode"])

    state_path = tmp_path / ".local" / "state" / "opencode" / "model.json"
    assert state_path.exists()
    state = json.loads(state_path.read_text())
    assert state["recent"][0] == {"providerID": "lilbee", "modelID": _CHAT_REF}


def test_launch_opencode_picker_state_dedupes_prior_lilbee_entries(tmp_path):
    _write_server_session()
    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=0)
    state_path = tmp_path / ".local" / "state" / "opencode" / "model.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "recent": [
                    {"providerID": "lilbee", "modelID": _CHAT_REF},
                    {"providerID": "anthropic", "modelID": "claude-3-5-sonnet"},
                ],
                "favorite": [],
                "variant": {},
            }
        )
    )
    with (
        patch("lilbee.cli.commands.launch.sys.platform", "linux"),
        patch("lilbee.cli.commands.launch.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.commands.launch.subprocess.run", return_value=completed),
    ):
        runner.invoke(app, ["launch", "opencode"])

    state = json.loads(state_path.read_text())
    lilbee_entries = [e for e in state["recent"] if e.get("providerID") == "lilbee"]
    assert len(lilbee_entries) == 1
    assert state["recent"][1] == {"providerID": "anthropic", "modelID": "claude-3-5-sonnet"}


def test_launch_opencode_skips_picker_state_on_windows(tmp_path):
    _write_server_session()
    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.commands.launch.sys.platform", "win32"),
        patch("lilbee.cli.commands.launch.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.commands.launch.subprocess.run", return_value=completed),
    ):
        runner.invoke(app, ["launch", "opencode"])

    state_path = tmp_path / ".local" / "state" / "opencode" / "model.json"
    assert not state_path.exists()


def test_launch_opencode_picker_state_recovers_from_corrupt_file(tmp_path):
    _write_server_session()
    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=0)
    state_path = tmp_path / ".local" / "state" / "opencode" / "model.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text("not json{{")
    with (
        patch("lilbee.cli.commands.launch.sys.platform", "darwin"),
        patch("lilbee.cli.commands.launch.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.commands.launch.subprocess.run", return_value=completed),
    ):
        runner.invoke(app, ["launch", "opencode"])

    state = json.loads(state_path.read_text())
    assert state["recent"][0]["modelID"] == _CHAT_REF


def test_launch_opencode_picker_state_ignores_non_dict_root(tmp_path):
    _write_server_session()
    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=0)
    state_path = tmp_path / ".local" / "state" / "opencode" / "model.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps(["unexpected", "shape"]))
    with (
        patch("lilbee.cli.commands.launch.sys.platform", "darwin"),
        patch("lilbee.cli.commands.launch.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.commands.launch.subprocess.run", return_value=completed),
    ):
        runner.invoke(app, ["launch", "opencode"])

    state = json.loads(state_path.read_text())
    assert isinstance(state, dict)
    assert state["recent"][0]["modelID"] == _CHAT_REF


def test_launch_opencode_picker_state_skips_when_no_models(tmp_path):
    _write_server_session()
    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.commands.launch.sys.platform", "darwin"),
        patch("lilbee.cli.commands.launch.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.commands.launch.subprocess.run", return_value=completed),
        patch("lilbee.cli.commands.launch.installed_chat_model_refs", return_value=[]),
    ):
        runner.invoke(app, ["launch", "opencode"])

    state_path = tmp_path / ".local" / "state" / "opencode" / "model.json"
    assert not state_path.exists()


def test_launch_opencode_spawns_server_when_none_running():
    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=0)
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None

    def _spawn_and_write(_port: int):
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


def test_launch_opencode_keep_serving_skips_terminate():
    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=0)
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None

    def _spawn_and_write(_port: int):
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


def test_launch_opencode_health_timeout_terminates_spawn_and_exits_1():
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


def test_launch_opencode_health_ok_but_session_missing_exits_1():
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


def test_stop_spawned_server_kills_on_terminate_timeout():
    import subprocess as _subprocess

    from lilbee.cli.commands import launch as launch_mod

    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    fake_proc.wait.side_effect = [_subprocess.TimeoutExpired(cmd="x", timeout=10), None]

    launch_mod._stop_spawned_server(fake_proc)

    fake_proc.terminate.assert_called_once()
    fake_proc.kill.assert_called_once()


def test_stop_spawned_server_noop_when_process_already_exited():
    from lilbee.cli.commands import launch as launch_mod

    fake_proc = MagicMock()
    fake_proc.poll.return_value = 0
    launch_mod._stop_spawned_server(fake_proc)
    fake_proc.terminate.assert_not_called()


def test_launch_opencode_propagates_opencode_exit_code():
    _write_server_session()
    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=42)
    with (
        patch("lilbee.cli.commands.launch.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.commands.launch.subprocess.run", return_value=completed),
    ):
        result = runner.invoke(app, ["launch", "opencode"])
    assert result.exit_code == 42
