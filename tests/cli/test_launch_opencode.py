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
    with patch("lilbee.cli.launchers.server.get_services", return_value=services):
        yield


@pytest.fixture(autouse=True)
def _healthy_by_default(request, monkeypatch):
    """Most tests pre-write a session and expect reuse; default the health
    probe to True so the launcher does not try to spawn fresh.

    Tests marked ``no_health_default`` (the helper unit tests for ``_health_ok``
    itself, and the stale-session test) skip this patch.
    """
    if "no_health_default" in request.keywords:
        return
    monkeypatch.setattr("lilbee.cli.launchers.server.health_ok", lambda _port: True)


@pytest.fixture(autouse=True)
def _warm_by_default(request, monkeypatch):
    """Default the chat warm-wait to an instant success so launch tests do not
    poll the (absent) server for the full warm timeout.

    Tests marked ``no_warm_default`` (the warm-gate's own unit tests) skip this.
    """
    if "no_warm_default" in request.keywords:
        return
    # run_launcher imports the name into its own module, so patch it there.
    monkeypatch.setattr("lilbee.cli.launchers.launcher.wait_for_chat_warm", lambda _port: True)
    # opencode's prepare() reads the served window over HTTP; default it to
    # unknown so tests do not hit the network. The limit.context test overrides.
    monkeypatch.setattr("lilbee.cli.launchers.opencode.served_chat_ctx", lambda _port: None)


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
    with patch("lilbee.cli.launchers.opencode.shutil.which", return_value=None):
        result = runner.invoke(app, ["launch", "opencode"])
    assert result.exit_code == 1
    assert "opencode binary not found" in result.stderr


def test_launch_opencode_with_running_server_emits_inline_config_env(tmp_path):
    _write_server_session()
    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.launchers.opencode.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed) as run,
        patch("lilbee.cli.launchers.server.spawn_server") as spawn,
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
    mcp_entry = payload["mcp"]["lilbee"]
    assert mcp_entry["type"] == "remote"
    assert mcp_entry["enabled"] is True
    assert mcp_entry["url"] == f"http://127.0.0.1:{_PORT}/mcp"
    assert mcp_entry["headers"]["Authorization"] == f"Bearer {_TOKEN}"


@pytest.mark.no_health_default
def test_launch_opencode_spawns_fresh_server_when_session_files_are_stale(tmp_path):
    """A leftover server.port from a crashed server must not poison reuse."""
    _write_server_session()
    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=0)
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None

    def _spawn_and_rewrite_session(_port: int):
        # Real server would write fresh session files on boot; simulate that
        # so running_server_session() returns the spawned (fresh) port/token.
        _write_server_session()
        return fake_proc

    with (
        patch("lilbee.cli.launchers.opencode.shutil.which", return_value=fake_opencode),
        # Override the autouse "healthy by default" fixture for this test.
        patch("lilbee.cli.launchers.server.health_ok", return_value=False),
        patch("lilbee.cli.launchers.server.spawn_server", side_effect=_spawn_and_rewrite_session),
        patch("lilbee.cli.launchers.server.wait_for_health", return_value=True),
        patch("lilbee.cli.launchers.server.free_port", return_value=_PORT),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
    ):
        result = runner.invoke(app, ["launch", "opencode"])

    assert result.exit_code == 0
    fake_proc.terminate.assert_called_once()


@pytest.mark.no_health_default
def test_health_ok_returns_false_on_connection_error(monkeypatch):
    from lilbee.cli.launchers import server as launch_mod

    def _boom(url, timeout):
        raise launch_mod.httpx.HTTPError("refused")

    monkeypatch.setattr(launch_mod.httpx, "get", _boom)
    assert launch_mod.health_ok(8765) is False


@pytest.mark.no_health_default
def test_health_ok_returns_true_on_200(monkeypatch):
    from lilbee.cli.launchers import server as launch_mod

    resp = MagicMock()
    resp.status_code = 200
    monkeypatch.setattr(launch_mod.httpx, "get", lambda url, timeout: resp)
    assert launch_mod.health_ok(8765) is True


@pytest.mark.no_health_default
def test_health_ok_returns_false_on_non_200(monkeypatch):
    from lilbee.cli.launchers import server as launch_mod

    resp = MagicMock()
    resp.status_code = 503
    monkeypatch.setattr(launch_mod.httpx, "get", lambda url, timeout: resp)
    assert launch_mod.health_ok(8765) is False


@pytest.mark.no_warm_default
def test_chat_ready_true_when_health_reports_warm(monkeypatch):
    from lilbee.cli.launchers import server as launch_mod

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"chat_ready": True}
    monkeypatch.setattr(launch_mod.httpx, "get", lambda url, timeout: resp)
    assert launch_mod.chat_ready(8765) is True


@pytest.mark.no_warm_default
def test_chat_ready_false_when_health_reports_cold(monkeypatch):
    from lilbee.cli.launchers import server as launch_mod

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"chat_ready": False}
    monkeypatch.setattr(launch_mod.httpx, "get", lambda url, timeout: resp)
    assert launch_mod.chat_ready(8765) is False


@pytest.mark.no_warm_default
def test_chat_ready_false_on_connection_error(monkeypatch):
    from lilbee.cli.launchers import server as launch_mod

    def _boom(url, timeout):
        raise launch_mod.httpx.HTTPError("refused")

    monkeypatch.setattr(launch_mod.httpx, "get", _boom)
    assert launch_mod.chat_ready(8765) is False


@pytest.mark.no_warm_default
def test_wait_for_chat_warm_returns_true_once_ready(monkeypatch):
    from lilbee.cli.launchers import server as launch_mod

    calls = {"n": 0}

    def _ready(_port):
        calls["n"] += 1
        return calls["n"] >= 2  # cold on the first probe, warm on the second

    monkeypatch.setattr(launch_mod, "chat_ready", _ready)
    monkeypatch.setattr(launch_mod.time, "sleep", lambda _s: None)
    assert launch_mod.wait_for_chat_warm(8765) is True
    assert calls["n"] >= 2


@pytest.mark.no_warm_default
def test_wait_for_chat_warm_returns_false_on_timeout(monkeypatch):
    from lilbee.cli.launchers import server as launch_mod

    monkeypatch.setattr(launch_mod, "chat_ready", lambda _port: False)
    monkeypatch.setattr(launch_mod.time, "sleep", lambda _s: None)
    assert launch_mod.wait_for_chat_warm(8765, timeout_s=0.0) is False


@pytest.mark.no_warm_default
def test_served_chat_ctx_reads_health_window(monkeypatch):
    from lilbee.cli.launchers import server as launch_mod

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"chat_ready": True, "chat_ctx": 40960}
    monkeypatch.setattr(launch_mod.httpx, "get", lambda url, timeout: resp)
    assert launch_mod.served_chat_ctx(8765) == 40960


@pytest.mark.no_warm_default
def test_served_chat_ctx_none_when_window_absent(monkeypatch):
    from lilbee.cli.launchers import server as launch_mod

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"chat_ready": True}
    monkeypatch.setattr(launch_mod.httpx, "get", lambda url, timeout: resp)
    assert launch_mod.served_chat_ctx(8765) is None


def test_opencode_config_sets_limit_context_when_known():
    from lilbee.cli.agent_configs.opencode import opencode_config

    block = opencode_config(
        base_url="http://127.0.0.1:9", api_key="k", model_refs=["a/M/m.gguf"], chat_ctx=32768
    )
    entry = block["provider"]["lilbee"]["models"]["a/M/m.gguf"]
    # Both keys required: opencode rejects a limit with only context (bb-c4t).
    assert entry["limit"] == {"context": 32768, "output": 8192}


def test_opencode_config_omits_limit_when_ctx_unknown():
    from lilbee.cli.agent_configs.opencode import opencode_config

    block = opencode_config(
        base_url="http://127.0.0.1:9", api_key="k", model_refs=["a/M/m.gguf"], chat_ctx=None
    )
    assert "limit" not in block["provider"]["lilbee"]["models"]["a/M/m.gguf"]


def test_launch_opencode_waits_for_chat_warm_before_handoff(tmp_path):
    """The launcher must block on the warm-gate before exec'ing the client, so
    the client never opens onto a cold, silent stream."""
    _write_server_session()
    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=0)
    order: list[str] = []
    with (
        patch("lilbee.cli.launchers.opencode.shutil.which", return_value=fake_opencode),
        patch(
            "lilbee.cli.launchers.launcher.wait_for_chat_warm",
            side_effect=lambda _port: order.append("warm") or True,
        ) as warm,
        patch(
            "lilbee.cli.launchers.launcher.subprocess.run",
            side_effect=lambda *a, **k: order.append("run") or completed,
        ),
        patch("lilbee.cli.launchers.server.spawn_server"),
    ):
        result = runner.invoke(app, ["launch", "opencode"])

    assert result.exit_code == 0
    warm.assert_called_once_with(_PORT)
    assert order == ["warm", "run"]  # warmed before the client launched


def test_launch_opencode_installs_skill_into_global_skills_dir(tmp_path):
    _write_server_session()
    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.launchers.opencode.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
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
        patch("lilbee.cli.launchers.opencode.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
    ):
        runner.invoke(app, ["launch", "opencode"])

    assert custom.read_text() == "user customization"


def test_launch_opencode_updates_picker_state_on_unix(tmp_path):
    _write_server_session()
    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.launchers.opencode.sys.platform", "darwin"),
        patch("lilbee.cli.launchers.opencode.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
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
        patch("lilbee.cli.launchers.opencode.sys.platform", "linux"),
        patch("lilbee.cli.launchers.opencode.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
    ):
        runner.invoke(app, ["launch", "opencode"])

    state = json.loads(state_path.read_text())
    lilbee_entries = [e for e in state["recent"] if e.get("providerID") == "lilbee"]
    assert len(lilbee_entries) == 1
    assert state["recent"][1] == {"providerID": "anthropic", "modelID": "claude-3-5-sonnet"}


def test_launch_opencode_updates_picker_state_on_windows(tmp_path):
    # opencode uses the same XDG-style state path on every platform, so the
    # picker state is written on Windows too (no platform skip).
    _write_server_session()
    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.launchers.opencode.sys.platform", "win32"),
        patch("lilbee.cli.launchers.opencode.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
    ):
        runner.invoke(app, ["launch", "opencode"])

    state_path = tmp_path / ".local" / "state" / "opencode" / "model.json"
    assert state_path.exists()
    state = json.loads(state_path.read_text())
    assert state["recent"][0] == {"providerID": "lilbee", "modelID": _CHAT_REF}


def test_launch_opencode_picker_state_recovers_from_corrupt_file(tmp_path):
    _write_server_session()
    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=0)
    state_path = tmp_path / ".local" / "state" / "opencode" / "model.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text("not json{{")
    with (
        patch("lilbee.cli.launchers.opencode.sys.platform", "darwin"),
        patch("lilbee.cli.launchers.opencode.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
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
        patch("lilbee.cli.launchers.opencode.sys.platform", "darwin"),
        patch("lilbee.cli.launchers.opencode.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
    ):
        runner.invoke(app, ["launch", "opencode"])

    state = json.loads(state_path.read_text())
    assert isinstance(state, dict)
    assert state["recent"][0]["modelID"] == _CHAT_REF


def test_launch_opencode_writes_lilbee_provider_to_persistent_config(tmp_path):
    """Picker rendering needs the provider in opencode's on-disk config file,
    not just the env var. The launcher must merge our provider into
    ``~/.config/opencode/opencode.json`` so a "lilbee" section appears
    alongside the user's other configured providers (ollama, etc.).
    """
    _write_server_session()
    config_path = tmp_path / ".config" / "opencode" / "opencode.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "$schema": "https://opencode.ai/config.json",
                "plugin": ["user-custom-plugin"],
                "provider": {
                    "ollama": {
                        "npm": "@ai-sdk/openai-compatible",
                        "name": "Ollama (local)",
                        "options": {"baseURL": "http://localhost:11434/v1"},
                        "models": {"qwen3-coder:30b": {"name": "qwen3-coder:30b"}},
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.launchers.opencode.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
    ):
        runner.invoke(app, ["launch", "opencode"])

    merged = json.loads(config_path.read_text(encoding="utf-8"))
    # User's other providers and top-level settings survive.
    assert merged["plugin"] == ["user-custom-plugin"]
    assert merged["provider"]["ollama"]["name"] == "Ollama (local)"
    # Lilbee section now appears with the current launch's port + token.
    lilbee = merged["provider"]["lilbee"]
    assert lilbee["options"]["baseURL"] == f"http://127.0.0.1:{_PORT}/v1"
    assert lilbee["options"]["apiKey"] == _TOKEN
    assert _CHAT_REF in lilbee["models"]


def test_launch_opencode_creates_config_file_when_absent(tmp_path):
    """When the user has no existing opencode.json, the launcher creates it."""
    _write_server_session()
    config_path = tmp_path / ".config" / "opencode" / "opencode.json"
    assert not config_path.exists()

    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.launchers.opencode.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
    ):
        runner.invoke(app, ["launch", "opencode"])

    assert config_path.exists()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert "lilbee" in payload["provider"]


def test_launch_opencode_recovers_from_corrupt_existing_config(tmp_path):
    """A garbled existing opencode.json is overwritten with a minimal valid
    config carrying the lilbee provider; the launcher doesn't crash.
    """
    _write_server_session()
    config_path = tmp_path / ".config" / "opencode" / "opencode.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("not json", encoding="utf-8")

    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.launchers.opencode.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
    ):
        result = runner.invoke(app, ["launch", "opencode"])
    assert result.exit_code == 0
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert "lilbee" in payload["provider"]


def test_launch_opencode_overwrites_non_dict_provider_field(tmp_path):
    """If the existing config has ``provider`` as the wrong shape (string,
    list, null), the merge resets it to a dict rather than silently dropping
    the lilbee entry.
    """
    _write_server_session()
    config_path = tmp_path / ".config" / "opencode" / "opencode.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "$schema": "https://opencode.ai/config.json",
                "plugin": [],
                "provider": "this should be a dict but isn't",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.launchers.opencode.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
    ):
        result = runner.invoke(app, ["launch", "opencode"])

    assert result.exit_code == 0
    merged = json.loads(config_path.read_text(encoding="utf-8"))
    assert isinstance(merged["provider"], dict)
    assert "lilbee" in merged["provider"]
    # Other top-level keys still survive the rewrite.
    assert merged["plugin"] == []


def test_launch_opencode_picker_state_skips_when_no_models(tmp_path):
    _write_server_session()
    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.launchers.opencode.sys.platform", "darwin"),
        patch("lilbee.cli.launchers.opencode.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
        patch("lilbee.cli.launchers.launcher.installed_chat_model_refs", return_value=[]),
    ):
        result = runner.invoke(app, ["launch", "opencode"])

    state_path = tmp_path / ".local" / "state" / "opencode" / "model.json"
    assert not state_path.exists()
    # No chat models -> the provider would be unusable; warn loudly (bb-c4t).
    assert "no chat models are installed" in result.output


def test_launch_opencode_spawns_server_when_none_running():
    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=0)
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None

    def _spawn_and_write(_port: int):
        _write_server_session()
        return fake_proc

    with (
        patch("lilbee.cli.launchers.opencode.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.launchers.server.spawn_server", side_effect=_spawn_and_write),
        patch("lilbee.cli.launchers.server.wait_for_health", return_value=True),
        patch("lilbee.cli.launchers.server.free_port", return_value=_PORT),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
    ):
        result = runner.invoke(app, ["launch", "opencode"])

    assert result.exit_code == 0
    fake_proc.terminate.assert_called_once()


def test_launch_opencode_health_timeout_terminates_spawn_and_exits_1():
    fake_opencode = "/usr/local/bin/opencode"
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    with (
        patch("lilbee.cli.launchers.opencode.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.launchers.server.spawn_server", return_value=fake_proc),
        patch("lilbee.cli.launchers.server.wait_for_health", return_value=False),
        patch("lilbee.cli.launchers.server.free_port", return_value=_PORT),
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
        patch("lilbee.cli.launchers.opencode.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.launchers.server.spawn_server", return_value=fake_proc),
        patch("lilbee.cli.launchers.server.wait_for_health", return_value=True),
        patch("lilbee.cli.launchers.server.free_port", return_value=_PORT),
    ):
        result = runner.invoke(app, ["launch", "opencode"])
    assert result.exit_code == 1
    assert "did not write a session file" in result.stderr
    fake_proc.terminate.assert_called_once()


def test_free_port_returns_open_port():
    from lilbee.cli.launchers import server as launch_mod

    port = launch_mod.free_port()
    assert 1024 < port < 65536


@pytest.mark.no_health_default
def test_wait_for_health_returns_true_on_200(monkeypatch):
    from lilbee.cli.launchers import server as launch_mod

    resp = MagicMock()
    resp.status_code = 200
    monkeypatch.setattr(launch_mod.httpx, "get", lambda url, timeout: resp)
    assert launch_mod.wait_for_health(8765, timeout_s=1.0) is True


@pytest.mark.no_health_default
def test_wait_for_health_swallows_http_errors_until_timeout(monkeypatch):
    from lilbee.cli.launchers import server as launch_mod

    def _boom(url, timeout):
        raise launch_mod.httpx.HTTPError("connection refused")

    monkeypatch.setattr(launch_mod.httpx, "get", _boom)
    monkeypatch.setattr(launch_mod.time, "sleep", lambda _seconds: None)
    assert launch_mod.wait_for_health(8765, timeout_s=0.05) is False


@pytest.mark.no_health_default
def test_wait_for_health_returns_false_on_non_200(monkeypatch):
    from lilbee.cli.launchers import server as launch_mod

    resp = MagicMock()
    resp.status_code = 503
    monkeypatch.setattr(launch_mod.httpx, "get", lambda url, timeout: resp)
    monkeypatch.setattr(launch_mod.time, "sleep", lambda _seconds: None)
    assert launch_mod.wait_for_health(8765, timeout_s=0.05) is False


def test_spawn_server_returns_popen(monkeypatch):
    from lilbee.cli.launchers import server as launch_mod

    fake = MagicMock()
    monkeypatch.setattr(launch_mod.subprocess, "Popen", lambda *a, **k: fake)
    out = launch_mod.spawn_server(8765)
    assert out is fake


def test_stop_spawned_server_kills_on_terminate_timeout():
    import subprocess as _subprocess

    from lilbee.cli.launchers import server as launch_mod

    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    fake_proc.wait.side_effect = [_subprocess.TimeoutExpired(cmd="x", timeout=10), None]

    launch_mod.stop_spawned_server(fake_proc)

    fake_proc.terminate.assert_called_once()
    fake_proc.kill.assert_called_once()


def test_stop_spawned_server_noop_when_process_already_exited():
    from lilbee.cli.launchers import server as launch_mod

    fake_proc = MagicMock()
    fake_proc.poll.return_value = 0
    launch_mod.stop_spawned_server(fake_proc)
    fake_proc.terminate.assert_not_called()


def _setup_marker(tmp_path: Path) -> Path:
    return tmp_path / "data" / "launchers" / "opencode-setup.json"


def test_launch_opencode_first_run_records_setup_marker(tmp_path):
    # Non-TTY (CliRunner): invoking launch is consent; the marker is recorded.
    _write_server_session()
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.launchers.opencode.shutil.which", return_value="/usr/local/bin/opencode"),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
    ):
        result = runner.invoke(app, ["launch", "opencode"])
    assert result.exit_code == 0
    assert _setup_marker(tmp_path).exists()
    assert "First-time opencode setup will write" in result.stdout


def test_launch_opencode_skips_prompt_when_marker_present(tmp_path):
    # A recorded marker means later launches do not re-print the setup plan.
    _write_server_session()
    marker = _setup_marker(tmp_path)
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({"accepted": True}))
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.launchers.opencode.shutil.which", return_value="/usr/local/bin/opencode"),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
    ):
        result = runner.invoke(app, ["launch", "opencode"])
    assert result.exit_code == 0
    assert "First-time opencode setup will write" not in result.stdout


def test_launch_opencode_interactive_accept_runs_setup(tmp_path):
    _write_server_session()
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.launchers.opencode.shutil.which", return_value="/usr/local/bin/opencode"),
        patch("lilbee.cli.launchers.opencode._is_interactive", return_value=True),
        patch("lilbee.cli.launchers.opencode.typer.confirm", return_value=True),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed) as run,
    ):
        result = runner.invoke(app, ["launch", "opencode"])
    assert result.exit_code == 0
    run.assert_called_once()
    assert _setup_marker(tmp_path).exists()


def test_launch_opencode_interactive_decline_skips_setup_and_launch(tmp_path):
    _write_server_session()
    config_path = tmp_path / ".config" / "opencode" / "opencode.json"
    with (
        patch("lilbee.cli.launchers.opencode.shutil.which", return_value="/usr/local/bin/opencode"),
        patch("lilbee.cli.launchers.opencode._is_interactive", return_value=True),
        patch("lilbee.cli.launchers.opencode.typer.confirm", return_value=False),
        patch("lilbee.cli.launchers.launcher.subprocess.run") as run,
    ):
        result = runner.invoke(app, ["launch", "opencode"])
    assert result.exit_code == 0
    run.assert_not_called()  # declined: opencode is not launched
    assert not config_path.exists()  # no config written
    assert not _setup_marker(tmp_path).exists()  # decline is not remembered


def test_launch_opencode_yes_flag_skips_prompt_when_interactive(tmp_path):
    _write_server_session()
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.launchers.opencode.shutil.which", return_value="/usr/local/bin/opencode"),
        patch("lilbee.cli.launchers.opencode._is_interactive", return_value=True),
        patch("lilbee.cli.launchers.opencode.typer.confirm") as confirm,
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
    ):
        result = runner.invoke(app, ["launch", "opencode", "--yes"])
    assert result.exit_code == 0
    confirm.assert_not_called()  # --yes bypasses the prompt
    assert _setup_marker(tmp_path).exists()


def test_launch_opencode_propagates_opencode_exit_code():
    _write_server_session()
    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=42)
    with (
        patch("lilbee.cli.launchers.opencode.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
    ):
        result = runner.invoke(app, ["launch", "opencode"])
    assert result.exit_code == 42
