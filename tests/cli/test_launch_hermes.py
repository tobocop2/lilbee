"""Tests for ``lilbee launch hermes``."""

from __future__ import annotations

import json
import os
import sys
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
    with patch("lilbee.app.models.get_services", return_value=services):
        yield


@pytest.fixture(autouse=True)
def _healthy_and_warm(monkeypatch):
    monkeypatch.setattr("lilbee.cli.launchers.server.health_ok", lambda _port: True)
    monkeypatch.setattr("lilbee.cli.launchers.launcher.wait_for_chat_warm", lambda _port: True)
    monkeypatch.setattr("lilbee.cli.launchers.hermes.client_chat_ctx", lambda _port: None)


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
        result = runner.invoke(app, ["launch", "hermes", "--no-mcp"])
    assert (tmp_path / ".hermes" / "skills" / "lilbee-mcp" / "SKILL.md").is_file()
    config = _hermes_config(tmp_path)
    assert "lilbee" not in config.get("mcp_servers", {})
    assert "lilbee" in config["providers"]  # provider stays; only MCP removed
    # --no-mcp is a deliberate opt-out, so no ungrounded warning.
    assert "without grounding" not in result.stderr.lower()


def test_launch_hermes_warns_when_mcp_cannot_connect(tmp_path):
    """A hermes whose MCP extra will not connect must not launch silently ungrounded."""
    _write_server_session()
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.launchers.hermes.shutil.which", return_value="/usr/local/bin/hermes"),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
        patch("lilbee.cli.launchers.server.spawn_server"),
        patch("lilbee.cli.launchers.hermes.ensure_hermes_http_mcp", return_value=False),
    ):
        result = runner.invoke(app, ["launch", "hermes"])
    assert result.exit_code == 0  # the launch still proceeds
    err = result.stderr.lower()
    assert "without grounding" in err  # names the consequence, not just an install hint
    assert "lilbee_search" in result.stderr


def test_launch_hermes_quiet_when_mcp_connects(tmp_path):
    _write_server_session()
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.launchers.hermes.shutil.which", return_value="/usr/local/bin/hermes"),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
        patch("lilbee.cli.launchers.server.spawn_server"),
        patch("lilbee.cli.launchers.hermes.ensure_hermes_http_mcp", return_value=True),
    ):
        result = runner.invoke(app, ["launch", "hermes"])
    assert result.exit_code == 0
    assert "without grounding" not in result.stderr.lower()


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


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits only")
def test_upsert_env_token_writes_the_env_owner_only(tmp_path):
    """The .env carries the bearer token, so it must be 0600 the whole time.

    Asserts the resulting mode rather than that a chmod was called: the write
    creates the file owner-only and keeps that mode across the atomic replace,
    so there is no chmod to observe and no window where it is world-readable.
    """
    import stat

    from lilbee.cli.launchers import hermes

    previous = os.umask(0)
    try:
        env = tmp_path / ".env"
        hermes._upsert_env_token(env, "tok-xyz")
    finally:
        os.umask(previous)

    assert "LILBEE_TOKEN=tok-xyz" in env.read_text()
    assert stat.S_IMODE(env.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    ("security", "expected"),
    [
        (None, True),
        ({}, True),
        ({"allow_lazy_installs": True}, True),
        ({"allow_lazy_installs": False}, False),
    ],
    ids=["no-security-section", "empty-section", "allowed", "refused"],
)
def test_launch_hermes_hands_the_lazy_install_setting_to_the_mcp_setup(
    tmp_path, security, expected
):
    _write_server_session()
    if security is not None:
        home = tmp_path / ".hermes"
        home.mkdir()
        (home / "config.yaml").write_text(yaml.safe_dump({"security": security}), encoding="utf-8")
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.launchers.hermes.shutil.which", return_value="/usr/local/bin/hermes"),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
        patch("lilbee.cli.launchers.server.spawn_server"),
        patch("lilbee.cli.launchers.hermes.ensure_hermes_http_mcp", return_value=True) as ensure,
    ):
        result = runner.invoke(app, ["launch", "hermes"])
    assert result.exit_code == 0
    ensure.assert_called_once()
    assert ensure.call_args.args == ("/usr/local/bin/hermes",)
    assert ensure.call_args.kwargs["allow_lazy_installs"] is expected


def _launch_on_a_terminal(*args: str, answer: bool):
    """Run ``lilbee launch hermes`` on a terminal; return (result, confirm, ensure)."""
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.launchers.hermes.shutil.which", return_value="/usr/local/bin/hermes"),
        patch("lilbee.cli.launchers.setup_gate._is_interactive", return_value=True),
        patch("lilbee.cli.launchers.setup_gate.typer.confirm", return_value=answer) as confirm,
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
        patch("lilbee.cli.launchers.server.spawn_server"),
        patch("lilbee.cli.launchers.hermes.ensure_hermes_http_mcp", return_value=True) as ensure,
    ):
        result = runner.invoke(app, ["launch", "hermes", *args])
    return result, confirm, ensure


@pytest.mark.parametrize("mode", ["--mcp", "--no-mcp"])
def test_a_declined_hermes_setup_writes_and_installs_nothing(tmp_path, mode):
    _write_server_session()
    before = set(tmp_path.rglob("*"))
    result, confirm, ensure = _launch_on_a_terminal(mode, answer=False)
    assert result.exit_code == 0
    confirm.assert_called_once_with("Proceed with hermes setup?", default=True)
    ensure.assert_not_called()
    assert set(tmp_path.rglob("*")) == before


def test_an_accepted_hermes_setup_is_not_asked_again(tmp_path):
    _write_server_session()
    _launch_on_a_terminal(answer=True)
    result, confirm, _ensure = _launch_on_a_terminal(answer=False)
    assert result.exit_code == 0
    confirm.assert_not_called()
    assert (cfg.data_dir / "launchers" / "hermes-setup.json").is_file()
    assert "First-time hermes setup" not in result.stdout


def test_launch_hermes_yes_flag_skips_the_question_on_a_terminal(tmp_path):
    _write_server_session()
    result, confirm, _ensure = _launch_on_a_terminal("--yes", answer=False)
    assert result.exit_code == 0
    confirm.assert_not_called()
    assert (tmp_path / ".hermes" / "config.yaml").is_file()
    assert (cfg.data_dir / "launchers" / "hermes-setup.json").is_file()


def test_hermes_setup_plan_with_mcp_names_every_file_and_the_extra_install(capsys):
    from lilbee.cli.launchers.hermes import _print_setup_plan

    _print_setup_plan(include_mcp=True)
    home = Path.home() / ".hermes"
    assert capsys.readouterr().out == (
        "First-time hermes setup will write:\n"
        f"  - lilbee provider entry -> {home / 'config.yaml'}\n"
        f"  - lilbee token, owner-only -> {home / '.env'}\n"
        f"  - lilbee MCP skill -> {home / 'skills' / 'lilbee-mcp'}\n"
        "  - lilbee MCP entry -> the same config.yaml\n"
        "  - hermes's mcp extra, installed into hermes's own environment when it is "
        "missing and hermes allows lazy installs\n"
        "Only the `lilbee` keys and the LILBEE_TOKEN line are written; your other hermes "
        "settings are preserved. To undo, remove those entries and the skill dir.\n"
    )


def test_hermes_setup_plan_without_mcp_leaves_out_the_mcp_lines(capsys):
    from lilbee.cli.launchers.hermes import _print_setup_plan

    _print_setup_plan(include_mcp=False)
    out = capsys.readouterr().out
    assert f"lilbee MCP skill -> {Path.home() / '.hermes' / 'skills' / 'lilbee-mcp'}" in out
    assert "MCP entry" not in out
    assert "mcp extra" not in out
