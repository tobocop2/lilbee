"""A config.toml that cannot be read shows one clean line on every surface.

Each test locks a real config.toml with chmod 000, then goes through the surface a user touches.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

import pytest
from typer.testing import CliRunner

from lilbee import mcp_server
from lilbee.cli import app
from lilbee.core import settings
from lilbee.core.config import cfg
from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.server_registry import ConfigReadError, loopback_reserve_bytes
from tests._fleet_engines import engines_for_all_roles

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
    config.chmod(0o000)
    if os.access(config, os.R_OK):
        config.chmod(0o600)
        cfg.data_root = previous
        pytest.skip("this user can read a chmod 000 file")
    yield tmp_path
    config.chmod(0o600)
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
