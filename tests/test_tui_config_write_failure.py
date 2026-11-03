"""A config.toml write that fails shows one error toast and leaves the TUI running."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from textual.widgets import Input

from lilbee.cli.tui.screens.settings import SettingsScreen
from lilbee.cli.tui.widgets.model_bar import ChatModeToggle
from lilbee.core import settings as persistent_settings
from lilbee.core.config import cfg
from tests._lilbee_app_test_host import LilbeeAppHost, pump_until
from tests.test_tui_screens import ChatTestApp
from tests.test_tui_settings_external import _App as SettingsApp
from tests.test_tui_settings_external import _open_retrieval

SIZE = (120, 40)


@pytest.fixture
def read_only_config():
    """Make config.toml under the test's data dir read-only; return the expected toast text."""
    persistent_settings.save(cfg.data_root, {"top_k": 7})
    path = cfg.data_root / "config.toml"
    path.chmod(0o444)
    try:
        yield f"{path} is read-only; make it writable and try again."
    finally:
        path.chmod(0o644)


def _errors(app, text: str) -> list[str]:
    return [n.message for n in app._notifications if n.message == text and n.severity == "error"]


async def _one_error_toast(app, pilot, text: str) -> None:
    """Wait for the toast (a posted message), then check it shows once."""
    assert await pump_until(pilot, lambda: bool(_errors(app, text))), "no error toast"
    assert _errors(app, text) == [text]


async def test_set_setting_toasts_the_write_error_and_returns_false(read_only_config):
    app = LilbeeAppHost()
    async with app.run_test(size=SIZE) as pilot:
        before = cfg.top_k
        assert app.set_setting("top_k", before + 1) is False
        await _one_error_toast(app, pilot, read_only_config)
        assert cfg.top_k == before
        assert app.is_running


async def test_the_lilbee_path_toggle_survives_a_failed_write(read_only_config):
    app = LilbeeAppHost()
    async with app.run_test(size=SIZE) as pilot:
        before = cfg.show_lilbee_path
        app.action_toggle_lilbee_path()
        await _one_error_toast(app, pilot, read_only_config)
        assert cfg.show_lilbee_path == before
        assert app.is_running


async def test_cycling_the_theme_survives_a_failed_write(read_only_config):
    app = LilbeeAppHost()
    async with app.run_test(size=SIZE) as pilot:
        app.action_cycle_theme()
        await _one_error_toast(app, pilot, read_only_config)
        assert not [n for n in app._notifications if n.message.startswith("Theme")]
        assert app.is_running


async def test_the_theme_command_survives_a_failed_write(read_only_config):
    app = LilbeeAppHost()
    async with app.run_test(size=SIZE) as pilot:
        name = next(t for t in app.available_themes if t != app.theme)
        app.set_theme(name)
        await _one_error_toast(app, pilot, read_only_config)
        assert app.is_running


async def test_the_settings_editor_keeps_an_unsaved_value_after_a_failed_write(read_only_config):
    app = SettingsApp()
    async with app.run_test(size=SIZE) as pilot:
        screen, field = await _open_retrieval(app, pilot)
        assert isinstance(screen, SettingsScreen)
        baseline = screen._mount_display["top_k"]
        field.focus()
        assert await pump_until(pilot, lambda: app.focused is field)
        field.value = str(cfg.top_k + 5)
        await pilot.press("enter")
        await _one_error_toast(app, pilot, read_only_config)
        assert isinstance(screen.query_one("#ed-top_k"), Input)
        assert screen._mount_display["top_k"] == baseline
        assert app.is_running


async def test_chat_set_shows_the_write_error_and_no_success_echo(read_only_config):
    app = ChatTestApp()
    async with app.run_test(size=SIZE) as pilot:
        before = cfg.top_k
        app.screen._handle_slash(f"/set top_k {before + 1}")
        await _one_error_toast(app, pilot, read_only_config)
        assert not [n for n in app._notifications if n.message.startswith("top_k = ")]
        assert app.is_running


async def test_the_mode_toggle_reports_no_flip_when_the_write_fails(read_only_config):
    app = ChatTestApp()
    async with app.run_test(size=SIZE) as pilot:
        toggle = app.screen.query_one(ChatModeToggle)
        before = cfg.chat_mode
        target = "chat" if before != "chat" else "search"
        with patch.object(toggle, "_embedding_ready", return_value=True):
            assert toggle.set_mode(target) is False
        await _one_error_toast(app, pilot, read_only_config)
        assert cfg.chat_mode == before
        assert app.is_running
