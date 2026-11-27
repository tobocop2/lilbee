"""A config.toml that cannot be read shows one clean line on every surface.

Each test locks a real config.toml, then goes through the surface a user touches. The lock is a
chmod 000 on POSIX, and on Windows an open handle that shares nothing.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

import pytest
import typer
from typer.testing import CliRunner

from lilbee import mcp_server
from lilbee.cli import app
from lilbee.core import settings
from lilbee.core.config import cfg
from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.server_registry import ConfigReadError, loopback_reserve_bytes, route_for
from tests._fleet_engines import engines_for_all_roles
from tests.conftest import clean_env

READ_TEXT = "Could not read config.toml"
GB = 1024**3
_PRESET = {"name": "wired", "url": "http://127.0.0.1:8000/v1", "roles": ["chat"]}
_SERVICES = "lilbee.server.handlers.models.get_services"

runner = CliRunner()


@pytest.fixture
def locked_config(tmp_path: Path) -> Iterator[Path]:
    """A data root whose config.toml holds a preset and refuses every read."""
    previous = cfg.data_root
    cfg.data_root = tmp_path
    settings.set_value(tmp_path, "server_presets", json.dumps([_PRESET]))
    config = tmp_path / "config.toml"
    if sys.platform == "win32":
        yield from _held_open_without_sharing(config, previous)
        return
    config.chmod(0o000)
    if os.access(config, os.R_OK):
        config.chmod(0o600)
        cfg.data_root = previous
        pytest.skip("this user can read a chmod 000 file")
    yield tmp_path
    config.chmod(0o600)
    cfg.data_root = previous


def _held_open_without_sharing(config: Path, previous: Path) -> Iterator[Path]:
    """Hold *config* open with no sharing, as another Windows program can; every read fails.

    Windows keeps no POSIX mode bits, so a chmod 000 file stays readable there.
    """
    import _winapi

    no_sharing = 0
    handle = _winapi.CreateFile(
        str(config), _winapi.GENERIC_READ, no_sharing, 0, _winapi.OPEN_EXISTING, 0, 0
    )
    try:
        yield config.parent
    finally:
        _winapi.CloseHandle(handle)
        cfg.data_root = previous


def _services() -> mock.Mock:
    services = mock.Mock()
    services.provider.list_models.return_value = []
    return services


class TestCli:
    def test_model_use_prints_one_clean_line(self, locked_config: Path) -> None:
        with mock.patch(_SERVICES, return_value=_services()):
            result = runner.invoke(
                app, ["--data-dir", str(locked_config), "model", "use", "chat", "wired/m"]
            )
        assert result.exit_code == 1
        assert READ_TEXT in result.output
        assert "Traceback" not in result.output
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_a_command_that_routes_a_preset_ref_stops_with_one_line(
        self, locked_config: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from lilbee.runtime import launcher

        with pytest.raises(SystemExit) as exit_info:
            launcher._run_cli(lambda: route_for("wired"))
        assert exit_info.value.code == 1
        err = capsys.readouterr().err
        assert err.startswith(f"Error: {READ_TEXT}")
        assert err.count("\n") == 1

    def test_a_real_command_with_an_env_preset_prints_no_traceback(
        self, locked_config: Path
    ) -> None:
        env = {
            **clean_env(locked_config),
            "LILBEE_EMBEDDING_MODEL": "wired/e",
            "LILBEE_NO_SPLASH": "1",
            "NO_COLOR": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        done = subprocess.run(
            [sys.executable, "-m", "lilbee", "status"],
            env=env,
            capture_output=True,
            encoding="utf-8",
            stdin=subprocess.DEVNULL,
            timeout=120,
        )
        assert done.returncode == 1
        assert "Traceback" not in done.stderr
        shown = [line for line in done.stderr.splitlines() if not line.startswith("Failed to read")]
        assert len(shown) == 1
        assert shown[0].startswith(f"Error: {READ_TEXT}")

    def test_a_write_failure_prints_the_caller_message(self, capsys: pytest.CaptureFixture) -> None:
        from lilbee.cli.commands._shared import run_or_fail

        def denied() -> None:
            raise PermissionError(13, "denied")

        with pytest.raises(typer.Exit):
            run_or_fail(denied, lambda exc: f"write failed with code {exc.errno}")
        assert "write failed with code 13" in capsys.readouterr().out


class TestRest:
    def test_set_chat_model_answers_503_with_the_read_text(self, locked_config: Path) -> None:
        from litestar.testing import TestClient

        import lilbee.server.auth as auth_mod
        from lilbee.server.app import create_app

        auth_mod.session_manager.disable()
        client = TestClient(create_app())
        try:
            with mock.patch(_SERVICES, return_value=_services()):
                resp = client.put("/api/models/chat", json={"model": "wired/m"})
        finally:
            auth_mod.session_manager.cleanup()
        assert resp.status_code == 503
        assert READ_TEXT in resp.json()["detail"]

    @pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
    @pytest.mark.parametrize("stream", [False, True])
    def test_a_chat_on_the_v1_routes_answers_503_with_the_read_text(
        self, locked_config: Path, path: str, stream: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from litestar.testing import TestClient

        import lilbee.server.auth as auth_mod
        from lilbee.server.app import create_app

        monkeypatch.setattr(cfg, "chat_model", "wired/m")
        body = {
            "model": "wired/m",
            "max_tokens": 16,
            "stream": stream,
            "messages": [{"role": "user", "content": "hi"}],
        }
        auth_mod.session_manager.disable()
        client = TestClient(create_app())
        try:
            resp = client.post(path, json=body)
        finally:
            auth_mod.session_manager.cleanup()
        assert resp.status_code == 503
        assert resp.json()["error"]["message"].startswith(READ_TEXT)


class TestMcp:
    def test_settings_set_returns_the_read_text_as_an_error(self, locked_config: Path) -> None:
        result = mcp_server.settings_set({"chat_model": "wired/m"})
        assert READ_TEXT in result["error"]


class TestTui:
    def test_assigning_a_model_toasts_the_read_text(self, locked_config: Path) -> None:
        from lilbee.cli.tui.app import LilbeeApp

        tui = LilbeeApp()
        with mock.patch.object(tui, "notify") as notify:
            assert tui.set_active_model("chat_model", "wired/m") is None
        notify.assert_called_once()
        assert READ_TEXT in notify.call_args.args[0]
        assert notify.call_args.kwargs["severity"] == "error"

    def test_the_startup_swap_logs_the_read_text_and_boots(
        self, locked_config: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.modelhub.model_manager import CanonicalRef, ValidationResult

        stale = CanonicalRef("gone/model", "wired/m", ValidationResult.UNKNOWN)
        fine = CanonicalRef("ok/model", "ok/model", ValidationResult.OK)
        with (
            mock.patch("lilbee.modelhub.model_manager.canonicalize_chat_model", return_value=fine),
            mock.patch(
                "lilbee.modelhub.model_manager.canonicalize_embedding_model", return_value=stale
            ),
            mock.patch("lilbee.cli.tui.app.call_from_thread"),
            caplog.at_level(logging.WARNING, logger="lilbee.cli.tui.app"),
        ):
            LilbeeApp().canonicalize_persisted_models()
        assert any(
            "gone/model" in r.getMessage() and READ_TEXT in str(r.exc_info[1])
            for r in caplog.records
            if r.exc_info
        )

    def test_the_settings_screen_toasts_the_read_text(self, locked_config: Path) -> None:
        from lilbee.cli.tui.app import LilbeeApp

        tui = LilbeeApp()
        with mock.patch.object(tui, "notify") as notify:
            assert tui.set_setting("chat_model", "wired/m") is False
        notify.assert_called_once()
        assert READ_TEXT in notify.call_args.args[0]
        assert notify.call_args.kwargs["severity"] == "error"


class TestPlanner:
    def test_planning_the_fleet_refuses_with_the_read_text(
        self, locked_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "lilbee.providers.fleet.engines.llama.resolve_llama_server",
            lambda: Path("/bin/llama-server"),
        )
        monkeypatch.setattr(planning_mod, "_plan_devices", lambda _binary, _engine: [])
        monkeypatch.setattr("lilbee.providers.model_cache.total_system_memory", lambda: 64 * GB)
        monkeypatch.setattr("lilbee.providers.model_cache.free_system_memory", lambda: 40 * GB)
        with pytest.raises(ConfigReadError, match=READ_TEXT):
            planning_mod.plan_all_launches(engines=engines_for_all_roles(LLAMA))

    def test_the_reserve_never_reads_as_zero(self, locked_config: Path) -> None:
        with pytest.raises(ConfigReadError, match=READ_TEXT):
            loopback_reserve_bytes()
