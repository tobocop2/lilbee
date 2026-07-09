"""Tests for ``lilbee launch opencode``."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from lilbee.catalog import agent_model_id
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
    # On Windows the launcher writes under %APPDATA%\opencode; point it at the
    # same tmp .config layout the POSIX path uses so expectations are uniform.
    monkeypatch.setenv("APPDATA", str(tmp_path / ".config"))
    yield tmp_path
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def test_launch_opencode_without_binary_exits_1():
    _write_server_session()
    with patch("lilbee.cli.launchers.opencode.shutil.which", return_value=None):
        result = runner.invoke(app, ["launch", "opencode"])
    assert result.exit_code == 1
    assert "opencode binary not found" in result.stderr


def _written_opencode_config(tmp_path: Path) -> dict:
    return json.loads((tmp_path / ".config" / "opencode" / "opencode.json").read_text())


def test_launch_opencode_with_running_server_writes_provider_into_config(tmp_path):
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
    env = run.call_args.kwargs["env"]
    assert env["LILBEE_TOKEN"] == _TOKEN  # token reaches the child via env
    assert "OPENCODE_CONFIG_CONTENT" not in env  # ephemeral injection is gone
    payload = _written_opencode_config(tmp_path)
    options = payload["provider"]["lilbee"]["options"]
    assert options["baseURL"] == f"http://127.0.0.1:{_PORT}/v1"
    assert options["apiKey"] == "{env:LILBEE_TOKEN}"  # reference, not the literal
    assert _TOKEN not in json.dumps(payload)  # no literal token on disk
    assert agent_model_id(_CHAT_REF) in payload["provider"]["lilbee"]["models"]
    mcp_entry = payload["mcp"]["lilbee"]
    assert mcp_entry["type"] == "remote"
    assert mcp_entry["enabled"] is True
    assert mcp_entry["url"] == f"http://127.0.0.1:{_PORT}/mcp"
    assert mcp_entry["headers"]["Authorization"] == "Bearer {env:LILBEE_TOKEN}"
    # Startup pin: without the top-level model key opencode boots on its own
    # default provider instead of the lilbee-served chat model.
    assert payload["model"] == f"lilbee/{agent_model_id(cfg.chat_model)}"


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
def test_wait_for_chat_warm_returns_streamed_result(monkeypatch):
    # When the progress stream runs, its verdict is returned directly without a
    # second readiness poll (don't double-spend the budget).
    from lilbee.cli.launchers import server as launch_mod

    monkeypatch.setattr(launch_mod, "chat_ready", lambda _port: False)
    monkeypatch.setattr(launch_mod, "render_warm", lambda _url, _timeout: True)
    assert launch_mod.wait_for_chat_warm(8765, timeout_s=5.0) is True

    monkeypatch.setattr(launch_mod, "render_warm", lambda _url, _timeout: False)
    assert launch_mod.wait_for_chat_warm(8765, timeout_s=5.0) is False


@pytest.mark.no_warm_default
def test_wait_for_chat_warm_falls_back_to_poll_when_stream_unavailable(monkeypatch):
    from lilbee.cli.launchers import server as launch_mod

    calls = {"n": 0}

    def _ready(_port):
        calls["n"] += 1
        return calls["n"] >= 2  # cold past the early check, warm on the poll

    monkeypatch.setattr(launch_mod, "render_warm", lambda _url, _timeout: None)
    monkeypatch.setattr(launch_mod, "chat_ready", _ready)
    monkeypatch.setattr(launch_mod.time, "sleep", lambda _s: None)
    assert launch_mod.wait_for_chat_warm(8765, timeout_s=5.0) is True
    assert calls["n"] >= 2  # confirms the poll fallback actually ran


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
    entry = block["provider"]["lilbee"]["models"]["M"]
    # Both keys required: opencode rejects a limit with only context (bb-c4t).
    assert entry["limit"] == {"context": 32768, "output": 8192}


def test_opencode_config_omits_limit_when_ctx_unknown():
    from lilbee.cli.agent_configs.opencode import opencode_config

    block = opencode_config(
        base_url="http://127.0.0.1:9", api_key="k", model_refs=["a/M/m.gguf"], chat_ctx=None
    )
    assert "limit" not in block["provider"]["lilbee"]["models"]["M"]


def test_opencode_config_pins_default_model_when_ref_given():
    from lilbee.cli.agent_configs.opencode import opencode_config

    block = opencode_config(
        base_url="http://127.0.0.1:9",
        api_key="k",
        model_refs=["a/M/m.gguf"],
        default_ref="a/M/m.gguf",
    )
    assert block["model"] == "lilbee/M"


def test_opencode_config_includes_mcp_by_default():
    from lilbee.cli.agent_configs.opencode import opencode_config

    block = opencode_config(base_url="http://127.0.0.1:9", api_key="k", model_refs=["a/M/m.gguf"])
    assert block["mcp"]["lilbee"]["url"] == "http://127.0.0.1:9/mcp"


def test_opencode_config_omits_mcp_block_when_disabled():
    """include_mcp=False drops the mcp block entirely but keeps lilbee as provider."""
    from lilbee.cli.agent_configs.opencode import opencode_config

    block = opencode_config(
        base_url="http://127.0.0.1:9",
        api_key="k",
        model_refs=["a/M/m.gguf"],
        include_mcp=False,
    )
    assert "mcp" not in block
    assert "lilbee" in block["provider"]


def test_opencode_config_omits_model_key_without_default_ref():
    from lilbee.cli.agent_configs.opencode import opencode_config

    block = opencode_config(base_url="http://127.0.0.1:9", api_key="k", model_refs=["a/M/m.gguf"])
    assert "model" not in block


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


def _config_block_from_launch(args: list[str], tmp_path: Path) -> dict:
    """Invoke ``launch opencode`` (with given extra args) and return the
    opencode config block prepare() merged into the user's opencode.json."""
    _write_server_session()
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.launchers.opencode.shutil.which", return_value="/usr/local/bin/opencode"),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
    ):
        runner.invoke(app, ["launch", *args])
    return _written_opencode_config(tmp_path)


def test_launch_opencode_no_mcp_removes_block_and_skips_skill(tmp_path):
    """--no-mcp drops the mcp block and does not install the lilbee-mcp skill."""
    block = _config_block_from_launch(["opencode", "--no-mcp"], tmp_path)
    assert "mcp" not in block
    assert "lilbee" in block["provider"]  # still the model provider
    skill_path = tmp_path / ".config" / "opencode" / "skills" / "lilbee-mcp"
    assert not skill_path.exists()


def test_launch_opencode_no_mcp_prunes_stale_lilbee_entry(tmp_path):
    """--no-mcp actively removes a previously-registered lilbee MCP entry."""
    _write_server_session()
    config_path = tmp_path / ".config" / "opencode" / "opencode.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps({"mcp": {"lilbee": {"url": "old"}, "other": {"url": "y"}}}))
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.launchers.opencode.shutil.which", return_value="/usr/local/bin/opencode"),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
    ):
        runner.invoke(app, ["launch", "opencode", "--no-mcp"])
    payload = json.loads(config_path.read_text())
    assert "lilbee" not in payload["mcp"]  # stale lilbee entry removed
    assert payload["mcp"]["other"] == {"url": "y"}  # other servers preserved
    assert "lilbee" in payload["provider"]  # provider still registered


def test_launch_opencode_mcp_flag_overrides_disabled_config(tmp_path, monkeypatch):
    """--mcp forces the block on even when agent_mcp_enabled is False."""
    from lilbee.core.config import cfg

    monkeypatch.setattr(cfg, "agent_mcp_enabled", False)
    block = _config_block_from_launch(["opencode", "--mcp"], tmp_path)
    assert "mcp" in block


def test_launch_opencode_defaults_to_config_when_no_flag(tmp_path, monkeypatch):
    """With no flag, the config field decides; agent_mcp_enabled=False omits mcp."""
    from lilbee.core.config import cfg

    monkeypatch.setattr(cfg, "agent_mcp_enabled", False)
    block = _config_block_from_launch(["opencode"], tmp_path)
    assert "mcp" not in block


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


def test_default_first_leads_with_configured_model():
    from lilbee.providers.model_ref import default_first

    refs = ["aaa/x.gguf", "zzz/y.gguf", "mmm/z.gguf"]
    assert default_first(refs, "mmm/z.gguf") == ["mmm/z.gguf", "aaa/x.gguf", "zzz/y.gguf"]
    # A default that isn't installed leaves the order untouched.
    assert default_first(refs, "not/installed.gguf") == refs


def test_launch_opencode_warns_when_configured_chat_model_not_installed(tmp_path):
    _write_server_session()
    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=0)
    other_ref = "bartowski/SmolLM-135M-Instruct-GGUF/SmolLM-135M-Instruct.Q8_0.gguf"
    with (
        patch("lilbee.cli.launchers.opencode.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
        patch("lilbee.cli.launchers.launcher.installed_chat_model_refs", return_value=[other_ref]),
    ):
        result = runner.invoke(app, ["launch", "opencode"])

    # cfg.chat_model (the default Qwen ref) is absent from the served refs, so the
    # startup pin would dangle and the client would open on its own provider.
    assert "is not installed" in result.output


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
    from lilbee.cli.launchers.server import _SPAWN_ATTEMPTS

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
    # Every bounded attempt spawned a server and then stopped it.
    assert fake_proc.terminate.call_count == _SPAWN_ATTEMPTS


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


def test_run_launcher_serves_remote_configured_chat_model():
    """A remote-configured chat model reaches the client picker first, with no
    false 'no chat models' warning and no local warm wait."""
    import typer

    from lilbee.cli.launchers.launcher import run_launcher

    launcher = MagicMock()
    launcher.find_binary.return_value = "/usr/local/bin/client"
    launcher.prepare.return_value = ([], {})
    with (
        patch.object(cfg, "chat_model", "ollama/qwen3:8b"),
        patch(
            "lilbee.cli.launchers.launcher.ensure_server_running",
            return_value=(("tok", 1234), None),
        ),
        patch("lilbee.cli.launchers.launcher.installed_chat_model_refs", return_value=[]),
        patch("lilbee.cli.launchers.launcher.wait_for_chat_warm") as warm,
        patch(
            "lilbee.cli.launchers.launcher.subprocess.run",
            return_value=MagicMock(returncode=0),
        ),
        pytest.raises(typer.Exit),
    ):
        run_launcher(launcher)
    assert launcher.prepare.call_args.kwargs["model_refs"] == ["ollama/qwen3:8b"]
    # No native model is installed, so there is no local load to wait out.
    warm.assert_not_called()


def test_run_launcher_stops_spawned_server_when_prepare_raises():
    """A raise from prepare() (e.g. declining setup) must still stop a freshly
    spawned server, not leak it."""
    import typer

    from lilbee.cli.launchers.launcher import run_launcher

    launcher = MagicMock()
    launcher.find_binary.return_value = "/usr/local/bin/client"
    launcher.prepare.side_effect = typer.Exit(0)
    fake_proc = MagicMock()
    with (
        patch.object(cfg, "chat_model", "ollama/qwen3:8b"),
        patch(
            "lilbee.cli.launchers.launcher.ensure_server_running",
            return_value=(("tok", 1234), fake_proc),
        ),
        patch("lilbee.cli.launchers.launcher.installed_chat_model_refs", return_value=[]),
        patch("lilbee.cli.launchers.launcher.stop_spawned_server") as stop,
        pytest.raises(typer.Exit),
    ):
        run_launcher(launcher)
    stop.assert_called_once_with(fake_proc)


def test_run_launcher_warns_and_skips_warm_when_no_models_at_all(capsys):
    """No native models and a native-configured ref: warn, don't warm."""
    import typer

    from lilbee.cli.launchers.launcher import run_launcher

    launcher = MagicMock()
    launcher.find_binary.return_value = "/usr/local/bin/client"
    launcher.prepare.return_value = ([], {})
    with (
        patch.object(cfg, "chat_model", _CHAT_REF),
        patch(
            "lilbee.cli.launchers.launcher.ensure_server_running",
            return_value=(("tok", 1234), None),
        ),
        patch("lilbee.cli.launchers.launcher.installed_chat_model_refs", return_value=[]),
        patch("lilbee.cli.launchers.launcher.wait_for_chat_warm") as warm,
        patch(
            "lilbee.cli.launchers.launcher.subprocess.run",
            return_value=MagicMock(returncode=0),
        ),
        pytest.raises(typer.Exit),
    ):
        run_launcher(launcher)
    assert launcher.prepare.call_args.kwargs["model_refs"] == []
    assert "no chat models are installed" in capsys.readouterr().err
    warm.assert_not_called()


def test_run_launcher_warms_native_chat_models():
    """Installed native chat models still gate the launch on the warm wait."""
    import typer

    from lilbee.cli.launchers.launcher import run_launcher

    launcher = MagicMock()
    launcher.find_binary.return_value = "/usr/local/bin/client"
    launcher.prepare.return_value = ([], {})
    with (
        patch.object(cfg, "chat_model", _CHAT_REF),
        patch(
            "lilbee.cli.launchers.launcher.ensure_server_running",
            return_value=(("tok", 1234), None),
        ),
        patch(
            "lilbee.cli.launchers.launcher.installed_chat_model_refs",
            return_value=[_CHAT_REF],
        ),
        patch("lilbee.cli.launchers.launcher.wait_for_chat_warm") as warm,
        patch(
            "lilbee.cli.launchers.launcher.subprocess.run",
            return_value=MagicMock(returncode=0),
        ),
        pytest.raises(typer.Exit),
    ):
        run_launcher(launcher)
    assert launcher.prepare.call_args.kwargs["model_refs"] == [_CHAT_REF]
    warm.assert_called_once_with(1234)


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


def test_record_setup_writes_marker_atomically(tmp_path, monkeypatch):
    from lilbee.cli.launchers import opencode

    marker = tmp_path / "opencode-setup.json"
    monkeypatch.setattr(opencode, "_setup_marker_path", lambda: marker)
    opencode._record_setup()
    assert json.loads(marker.read_text()) == {"accepted": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_record_setup_leaves_existing_marker_intact_on_failure(tmp_path, monkeypatch):
    """A failed rewrite must not truncate or litter: the prior marker survives."""
    from lilbee.cli.launchers import opencode

    marker = tmp_path / "opencode-setup.json"
    marker.write_text(json.dumps({"accepted": True}))

    def _boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(opencode, "_setup_marker_path", lambda: marker)
    monkeypatch.setattr(opencode.os, "replace", _boom)
    with pytest.raises(OSError):
        opencode._record_setup()
    assert json.loads(marker.read_text()) == {"accepted": True}
    assert not list(tmp_path.glob("*.tmp"))


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


def test_run_launcher_disables_eager_warm_in_launcher_process():
    """The launcher delegates inference to the spawned `lilbee serve`, so it must
    turn off the eager fleet warm in its own process. Otherwise get_services()
    starts a second llama-swap that races the server's for the model's port and the
    loser gets connection-refused. Regression for the opencode double-spawn."""
    import typer

    from lilbee.cli.launchers.launcher import run_launcher
    from lilbee.core.config import cfg

    launcher = MagicMock()
    launcher.find_binary.return_value = "/usr/local/bin/client"
    launcher.prepare.return_value = ([], {})
    old = cfg.worker_pool_eager_start
    cfg.worker_pool_eager_start = True
    try:
        with (
            patch(
                "lilbee.cli.launchers.launcher.ensure_server_running",
                return_value=(("tok", 1234), None),
            ),
            patch("lilbee.cli.launchers.launcher.installed_chat_model_refs", return_value=[]),
            patch(
                "lilbee.cli.launchers.launcher.subprocess.run",
                return_value=MagicMock(returncode=0),
            ),
            pytest.raises(typer.Exit),
        ):
            run_launcher(launcher)
        assert cfg.worker_pool_eager_start is False
    finally:
        cfg.worker_pool_eager_start = old


def test_launch_opencode_merges_into_real_config_preserving_others(tmp_path):
    """The launcher registers lilbee into the real opencode.json, leaving the
    user's other providers and settings intact and never writing a literal token."""
    _write_server_session()
    config_path = tmp_path / ".config" / "opencode" / "opencode.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps({"provider": {"anthropic": {"name": "anthropic"}}, "theme": "x"})
    )
    fake_opencode = "/usr/local/bin/opencode"
    completed = MagicMock(returncode=0)
    with (
        patch("lilbee.cli.launchers.opencode.shutil.which", return_value=fake_opencode),
        patch("lilbee.cli.launchers.launcher.subprocess.run", return_value=completed),
        patch("lilbee.cli.launchers.server.spawn_server"),
    ):
        result = runner.invoke(app, ["launch", "opencode"])

    assert result.exit_code == 0
    payload = json.loads(config_path.read_text())
    assert payload["provider"]["anthropic"] == {"name": "anthropic"}  # preserved
    assert payload["theme"] == "x"  # preserved
    assert "lilbee" in payload["provider"]  # registered
    assert _TOKEN not in config_path.read_text()  # no literal token on disk


def test_launch_opencode_refuses_to_overwrite_corrupt_config(tmp_path):
    """A corrupt opencode.json is never clobbered; the launcher exits non-zero."""
    _write_server_session()
    config_path = tmp_path / ".config" / "opencode" / "opencode.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{ not: valid json")
    with (
        patch("lilbee.cli.launchers.opencode.shutil.which", return_value="/usr/local/bin/opencode"),
        patch("lilbee.cli.launchers.launcher.subprocess.run") as run,
    ):
        result = runner.invoke(app, ["launch", "opencode"])
    assert result.exit_code == 1
    assert "did not parse" in result.stderr
    assert config_path.read_text() == "{ not: valid json"  # untouched
    run.assert_not_called()
    # Load-before-side-effects: a corrupt config aborts before the skill install too.
    assert not (tmp_path / ".config" / "opencode" / "skills" / "lilbee-mcp").exists()


def test_opencode_config_sets_generous_mcp_timeout():
    from lilbee.cli.agent_configs.opencode import _MCP_TIMEOUT_MS, opencode_config

    block = opencode_config(base_url="http://127.0.0.1:9", api_key="k", model_refs=["a/M/m.gguf"])
    # opencode defaults remote-MCP requests to 5000 ms, which the first
    # lilbee_search can exceed while the embedding model cold-loads.
    assert block["mcp"]["lilbee"]["timeout"] == _MCP_TIMEOUT_MS


def test_chat_warm_budget_falls_back_to_floor_when_model_unresolvable():
    from lilbee.cli.launchers import server as launch_mod

    # cfg.chat_model points at an uninstalled ref in the isolated env.
    assert launch_mod.chat_warm_budget_s() == launch_mod._WARM_TIMEOUT_S


def test_chat_warm_budget_scales_with_split_giant_weights(tmp_path):
    """A split giant's warm wait must cover the fleet's own cold-load budget.

    Only split models exercise the shard-sum path: a single-file ref resolves
    to its blob (no co-located siblings) and floors, which is correct because
    single files load well under the floor.
    """
    import hashlib

    from lilbee.cli.launchers import server as launch_mod
    from lilbee.providers.fleet.swap_config import cold_load_timeout_s

    models_dir = tmp_path / "models"
    cache = models_dir / "models--org--Big-GGUF"
    (cache / "blobs").mkdir(parents=True)
    snap = cache / "snapshots" / "rev"
    snap.mkdir(parents=True)
    shards = [f"Big-Q4-0000{i}-of-00002.gguf" for i in (1, 2)]
    total = 0
    for shard in shards:
        payload = shard.encode() * 64
        total += len(payload)
        digest = hashlib.sha256(payload).hexdigest()
        blob = cache / "blobs" / digest
        blob.write_bytes(payload)
        (snap / shard).symlink_to(blob)
    (cache / "refs").mkdir()
    (cache / "refs" / "main").write_text("rev")
    # The autouse _isolated_env fixture snapshots and restores every cfg field.
    cfg.models_dir = models_dir
    cfg.chat_model = f"org/Big-GGUF/{shards[0]}"

    budget = launch_mod.chat_warm_budget_s()
    assert budget == max(launch_mod._WARM_TIMEOUT_S, float(cold_load_timeout_s(total)))


class TestOpencodeConfigDir:
    """_opencode_config_dir() returns the platform-correct directory."""

    def test_posix_returns_home_config_opencode(self, monkeypatch) -> None:
        from lilbee.cli.launchers.opencode import _opencode_config_dir

        monkeypatch.setattr("lilbee.cli.launchers.opencode.sys.platform", "linux")
        result = _opencode_config_dir()
        assert result == Path.home() / ".config" / "opencode"

    def test_win32_uses_appdata_env(self, monkeypatch, tmp_path) -> None:
        from lilbee.cli.launchers.opencode import _opencode_config_dir

        monkeypatch.setenv("APPDATA", str(tmp_path))
        monkeypatch.setattr("lilbee.cli.launchers.opencode.sys.platform", "win32")
        result = _opencode_config_dir()
        assert result == tmp_path / "opencode"

    def test_win32_fallback_when_appdata_unset(self, monkeypatch) -> None:
        from lilbee.cli.launchers.opencode import _opencode_config_dir

        monkeypatch.delenv("APPDATA", raising=False)
        monkeypatch.setattr("lilbee.cli.launchers.opencode.sys.platform", "win32")
        result = _opencode_config_dir()
        assert result.name == "opencode"
        assert "AppData" in str(result) or "opencode" in str(result)

    def test_config_path_nests_under_config_dir(self, monkeypatch, tmp_path) -> None:
        from lilbee.cli.launchers.opencode import _opencode_config_path

        monkeypatch.setenv("APPDATA", str(tmp_path))
        monkeypatch.setattr("lilbee.cli.launchers.opencode.sys.platform", "win32")
        result = _opencode_config_path()
        assert result == tmp_path / "opencode" / "opencode.json"

    def test_skill_dest_nests_under_config_dir(self, monkeypatch, tmp_path) -> None:
        from lilbee.cli.launchers.opencode import _opencode_skill_dest

        monkeypatch.setenv("APPDATA", str(tmp_path))
        monkeypatch.setattr("lilbee.cli.launchers.opencode.sys.platform", "win32")
        result = _opencode_skill_dest()
        assert result == tmp_path / "opencode" / "skills" / "lilbee-mcp"
