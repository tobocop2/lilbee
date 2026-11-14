"""Every TUI write path shows a plain toast when the config lock cannot be taken."""

from __future__ import annotations

from collections.abc import Iterator
from unittest import mock

import pytest
from filelock import FileLock
from textual.app import ComposeResult

from lilbee.app.settings_map import SETTINGS_MAP
from lilbee.cli.tui.app import LilbeeApp
from lilbee.core import settings
from lilbee.core.config import cfg
from tests._lilbee_app_test_host import LilbeeAppHost
from tests.conftest import TEST_LOCAL_REF
from tests.test_tui_screens import ChatTestApp, SettingsTestApp

PLAIN_MESSAGE_START = "Could not write config.toml: another lilbee process held the lock"


@pytest.fixture
def held_config_lock(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Hold the config lock from outside the writer, so the real lock refuses it."""
    monkeypatch.setattr(settings, "_CONFIG_LOCK_TIMEOUT_S", 0.01)
    held = FileLock(str(cfg.data_root / "config.toml") + ".lock")
    held.acquire()
    try:
        yield
    finally:
        held.release()


def _toasts(notify: mock.MagicMock) -> list[tuple[str, str | None]]:
    return [(call.args[0], call.kwargs.get("severity")) for call in notify.call_args_list]


def _assert_one_plain_error_toast(notify: mock.MagicMock) -> None:
    toasts = _toasts(notify)
    assert len(toasts) == 1
    text, severity = toasts[0]
    assert text.startswith(PLAIN_MESSAGE_START)
    assert "Nothing was changed." in text
    assert severity == "error"


@pytest.mark.usefixtures("held_config_lock")
class TestToastsWhenTheConfigLockIsHeld:
    def test_picking_a_model_toasts_and_keeps_the_old_model(self) -> None:
        before = cfg.chat_model
        app = LilbeeApp()
        with mock.patch.object(app, "notify") as notify:
            app.set_active_model("chat_model", TEST_LOCAL_REF)
        _assert_one_plain_error_toast(notify)
        assert cfg.chat_model == before

    async def test_picking_a_theme_toasts(self) -> None:
        async with LilbeeAppHost().run_test(size=(80, 24)) as pilot:
            app = pilot.app
            assert isinstance(app, LilbeeApp)
            name = next(t for t in app.available_themes if t != app.theme)
            with mock.patch.object(app, "notify") as notify:
                app.set_theme(name)
            _assert_one_plain_error_toast(notify)

    def test_toggling_the_path_pill_toasts_and_keeps_the_setting(self) -> None:
        before = cfg.show_lilbee_path
        app = LilbeeApp()
        with mock.patch.object(app, "notify") as notify:
            app.action_toggle_lilbee_path()
        _assert_one_plain_error_toast(notify)
        assert cfg.show_lilbee_path == before

    async def test_the_chat_set_command_toasts(self) -> None:
        before = cfg.top_k
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)):
            with mock.patch.object(app, "notify") as notify:
                app.screen._cmd_set(f"top_k {before + 1}")
        _assert_one_plain_error_toast(notify)
        assert cfg.top_k == before

    async def test_a_settings_screen_edit_toasts(self) -> None:
        before = cfg.top_k
        app = SettingsTestApp()
        async with app.run_test(size=(120, 40)):
            with mock.patch.object(app, "notify") as notify:
                app.screen._persist_value("top_k", SETTINGS_MAP["top_k"], str(before + 1))
        _assert_one_plain_error_toast(notify)
        assert cfg.top_k == before

    async def test_flipping_the_chat_mode_toggle_toasts_and_keeps_the_mode(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import ChatModeToggle

        class _Host(LilbeeAppHost):
            def compose(self) -> ComposeResult:
                yield ChatModeToggle()

        cfg.chat_mode = "chat"
        async with _Host().run_test(size=(80, 24)) as pilot:
            toggle = pilot.app.query_one(ChatModeToggle)
            with (
                mock.patch.object(pilot.app, "notify") as notify,
                mock.patch.object(toggle, "_embedding_ready", return_value=True),
            ):
                changed = toggle.set_mode("search")
        assert changed is False
        _assert_one_plain_error_toast(notify)
        assert cfg.chat_mode == "chat"


class TestAnotherOsErrorStillSurfaces:
    def test_a_permission_error_is_not_swallowed_by_the_toggle(self) -> None:
        app = LilbeeApp()
        with (
            mock.patch.object(app, "set_setting", side_effect=PermissionError("denied")),
            pytest.raises(PermissionError, match="denied"),
        ):
            app.action_toggle_lilbee_path()
