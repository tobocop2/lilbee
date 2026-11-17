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


_PINNED_REF = "Acme/Pinned-GGUF/Pinned-Q4_K_M.gguf"
_SERVED_CTX = 70_000


def _serve_ctx_on_port(monkeypatch, ctx: int | None) -> None:
    """Report *ctx* only when asked about the launch port, so a wrong port reads as unknown."""
    monkeypatch.setattr(
        "lilbee.cli.launchers.hermes.client_chat_ctx",
        lambda port: ctx if port == _PORT else None,
    )


def test_hermes_prepare_writes_the_exact_config(tmp_path, monkeypatch):
    """The whole file is pinned: provider, context sizes, MCP entry, model pin and key order."""
    from lilbee.catalog import agent_model_id
    from lilbee.cli.launchers.hermes import HermesLauncher

    _serve_ctx_on_port(monkeypatch, _SERVED_CTX)
    monkeypatch.setattr(cfg, "chat_model", _PINNED_REF)
    base = f"http://127.0.0.1:{_PORT}"

    HermesLauncher(include_mcp=True).prepare(
        token=_TOKEN, port=_PORT, model_refs=[_CHAT_REF, _PINNED_REF]
    )

    text = (tmp_path / ".hermes" / "config.yaml").read_text(encoding="utf-8")
    pinned_id = agent_model_id(_PINNED_REF)
    assert yaml.safe_load(text) == {
        "providers": {
            "lilbee": {
                "api": f"{base}/v1",
                "api_key": "${LILBEE_TOKEN}",
                "max_tokens": 8192,
                "default_model": pinned_id,
                "context_length": _SERVED_CTX,
            }
        },
        "context_file_max_chars": _SERVED_CTX,
        "mcp_servers": {
            "lilbee": {
                "url": f"{base}/mcp",
                "headers": {"Authorization": "Bearer ${LILBEE_TOKEN}"},
            }
        },
        "model": {"default": pinned_id, "provider": "lilbee", "max_tokens": 8192},
    }
    # sort_keys=False keeps the builder's order; a sorted dump starts with context_file_max_chars.
    assert list(yaml.safe_load(text)) == [
        "providers",
        "context_file_max_chars",
        "mcp_servers",
        "model",
    ]


def test_hermes_prepare_without_mcp_writes_no_mcp_entry(tmp_path, monkeypatch):
    from lilbee.cli.launchers.hermes import HermesLauncher

    _serve_ctx_on_port(monkeypatch, _SERVED_CTX)
    HermesLauncher(include_mcp=False).prepare(token=_TOKEN, port=_PORT, model_refs=[_CHAT_REF])

    assert "mcp_servers" not in _hermes_config(tmp_path)


def test_hermes_prepare_warns_with_the_served_window_below_the_minimum(monkeypatch, capsys):
    from lilbee.cli.launchers.hermes import HermesLauncher

    _serve_ctx_on_port(monkeypatch, 8_192)
    HermesLauncher(include_mcp=False).prepare(token=_TOKEN, port=_PORT, model_refs=[_CHAT_REF])

    assert capsys.readouterr().err == (
        "Warning: hermes requires at least a 64,000-token context and lilbee serves 8,192, "
        "so hermes will refuse to start. Chat with a longer-context model, a smaller "
        "quantization, or a higher gpu_memory_fraction.\n"
    )


def test_hermes_prepare_accepts_a_served_window_of_exactly_the_minimum(
    tmp_path, monkeypatch, capsys
):
    from lilbee.cli.launchers.hermes import HermesLauncher

    _serve_ctx_on_port(monkeypatch, 64_000)
    HermesLauncher(include_mcp=False).prepare(token=_TOKEN, port=_PORT, model_refs=[_CHAT_REF])

    assert capsys.readouterr().err == ""
    assert _hermes_config(tmp_path)["providers"]["lilbee"]["context_length"] == 64_000


def test_hermes_prepare_pins_the_first_model_ref_when_chat_model_is_unset(tmp_path, monkeypatch):
    from lilbee.catalog import agent_model_id
    from lilbee.cli.launchers.hermes import HermesLauncher

    monkeypatch.setattr(cfg, "chat_model", "")

    HermesLauncher(include_mcp=False).prepare(
        token=_TOKEN, port=_PORT, model_refs=[_PINNED_REF, _CHAT_REF]
    )

    written = _hermes_config(tmp_path)
    pinned_id = agent_model_id(_PINNED_REF)
    assert written["providers"]["lilbee"]["default_model"] == pinned_id
    assert written["model"] == {"default": pinned_id, "provider": "lilbee", "max_tokens": 8192}


def test_hermes_prepare_names_the_config_in_the_parse_error(tmp_path, capsys):
    import typer

    from lilbee.cli.launchers.hermes import HermesLauncher

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("{ this: is: not: yaml", encoding="utf-8")

    with pytest.raises(typer.Exit) as exit_info:
        HermesLauncher(include_mcp=True).prepare(token=_TOKEN, port=_PORT, model_refs=[_CHAT_REF])

    assert exit_info.value.exit_code == 1
    assert capsys.readouterr().err == (
        "Your hermes config (config.yaml) did not parse, so lilbee will not overwrite it. "
        "Fix or remove it, then retry.\n"
    )
