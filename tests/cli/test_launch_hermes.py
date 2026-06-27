"""Tests for ``lilbee launch hermes``."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from lilbee.catalog.types import ModelTask
from lilbee.cli import app
from lilbee.core.config import cfg
from lilbee.server.auth import server_json_path

runner = CliRunner()

_CHAT_REF = "Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf"
_TOKEN = "hermes-token-xyz"
_PORT = 8771


def _write_server_session() -> None:
    server_json_path().write_text(json.dumps({"token": _TOKEN}))
    (cfg.data_dir / "server.port").write_text(str(_PORT))


def _hermes_config(home: Path) -> dict:
    return yaml.safe_load((home / ".hermes" / "config.yaml").read_text())


@pytest.fixture(autouse=True)
def _stub_registry():
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
def _healthy_and_warm(monkeypatch):
    monkeypatch.setattr("lilbee.cli.launchers.server.health_ok", lambda _port: True)
    monkeypatch.setattr("lilbee.cli.launchers.launcher.wait_for_chat_warm", lambda _port: True)
    monkeypatch.setattr("lilbee.cli.launchers.hermes.served_chat_ctx", lambda _port: None)


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
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    yield tmp_path
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def test_find_binary_miss_exits_1():
    _write_server_session()
    with patch("lilbee.cli.launchers.hermes.shutil.which", return_value=None):
        result = runner.invoke(app, ["launch", "hermes"])
    assert result.exit_code == 1
    assert "hermes binary not found" in result.stderr


def test_launch_hermes_writes_provider_token_not_literal(tmp_path):
    _write_server_session()
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.launchers.hermes.shutil.which", return_value="/usr/local/bin/hermes"),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed) as run,
        patch("lilbee.cli.launchers.server.spawn_server"),
    ):
        result = runner.invoke(app, ["launch", "hermes"])

    assert result.exit_code == 0
    config = _hermes_config(tmp_path)
    prov = config["providers"]["lilbee"]
    assert prov["api"] == f"http://127.0.0.1:{_PORT}/v1"
    assert prov["api_key"] == "${LILBEE_TOKEN}"
    config_text = (tmp_path / ".hermes" / "config.yaml").read_text()
    assert _TOKEN not in config_text  # token never in the config file
    env_text = (tmp_path / ".hermes" / ".env").read_text()
    assert f"LILBEE_TOKEN={_TOKEN}" in env_text  # only in the secret store
    assert run.call_args.kwargs["env"]["LILBEE_TOKEN"] == _TOKEN


def test_launch_hermes_preserves_existing_config(tmp_path):
    _write_server_session()
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        yaml.safe_dump({"providers": {"openrouter": {"api": "x"}}, "memory": {"k": 1}})
    )
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.launchers.hermes.shutil.which", return_value="/usr/local/bin/hermes"),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
        patch("lilbee.cli.launchers.server.spawn_server"),
    ):
        runner.invoke(app, ["launch", "hermes"])

    config = _hermes_config(tmp_path)
    assert config["providers"]["openrouter"] == {"api": "x"}
    assert config["memory"] == {"k": 1}
    assert "lilbee" in config["providers"]


def test_launch_hermes_refuses_corrupt_config(tmp_path):
    _write_server_session()
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("{ this: is: not: yaml")
    with (
        patch("lilbee.cli.launchers.hermes.shutil.which", return_value="/usr/local/bin/hermes"),
        patch("lilbee.cli.launchers.launcher.subprocess.run") as run,
        patch("lilbee.cli.launchers.server.spawn_server"),
    ):
        result = runner.invoke(app, ["launch", "hermes"])
    assert result.exit_code == 1
    assert "did not parse" in result.stderr
    assert (home / "config.yaml").read_text() == "{ this: is: not: yaml"
    run.assert_not_called()


def test_launch_hermes_no_mcp_prunes_stale_entry(tmp_path):
    _write_server_session()
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({"mcp_servers": {"lilbee": {"url": "old"}}}))
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.launchers.hermes.shutil.which", return_value="/usr/local/bin/hermes"),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
        patch("lilbee.cli.launchers.server.spawn_server"),
    ):
        runner.invoke(app, ["launch", "hermes", "--no-mcp"])
    config = _hermes_config(tmp_path)
    assert "lilbee" not in config.get("mcp_servers", {})
    assert "lilbee" in config["providers"]  # provider stays; only MCP removed


def test_launch_hermes_installs_skill(tmp_path):
    _write_server_session()
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.launchers.hermes.shutil.which", return_value="/usr/local/bin/hermes"),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
        patch("lilbee.cli.launchers.server.spawn_server"),
    ):
        runner.invoke(app, ["launch", "hermes"])
    assert (tmp_path / ".hermes" / "skills" / "lilbee-mcp" / "SKILL.md").exists()


def test_launch_hermes_env_upsert_preserves_other_lines(tmp_path):
    _write_server_session()
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / ".env").write_text("OTHER=1\nLILBEE_TOKEN=stale\n")
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.launchers.hermes.shutil.which", return_value="/usr/local/bin/hermes"),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
        patch("lilbee.cli.launchers.server.spawn_server"),
    ):
        runner.invoke(app, ["launch", "hermes"])
    env_text = (home / ".env").read_text()
    assert "OTHER=1" in env_text
    assert f"LILBEE_TOKEN={_TOKEN}" in env_text
    assert "LILBEE_TOKEN=stale" not in env_text
