"""Settings screen: reset buttons, ctrl+r, reset all, editor refresh and flash attention."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lilbee.app.settings_map import SettingGroup
from lilbee.core import settings as persistent_settings
from lilbee.core.config import cfg
from lilbee.core.config.enums import OcrMode
from lilbee.core.config.resolve import builtin_value
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import await_chat, pump_until, saved_setting
from tests._settings_test_app import LazySettingsTestApp, _blur_after_focus, populate_pane_of
from tests._tui_screen_fixtures import (  # noqa: F401
    _isolated_cfg,
    _patch_chat_setup,
    _settings_gates_open,
    mock_svc,
)


async def test_reset_button_resets_scalar():
    """Pressing the reset button restores a scalar setting to its cfg default."""
    from textual.widgets import Button, Input

    cfg.wiki_clusterer_k = 99
    app = LazySettingsTestApp("wiki_clusterer_k")
    async with app.run_test(size=(120, 40)):
        button = app.screen.query_one("#reset-wiki_clusterer_k", Button)
        async with saved_setting(app, "wiki_clusterer_k"):
            button.press()
        target = builtin_value("wiki_clusterer_k")
        assert cfg.wiki_clusterer_k == target
        editor = app.screen.query_one("#ed-wiki_clusterer_k", Input)
        assert editor.value == str(target)


async def test_reset_button_absent_for_readonly():
    """Read-only rows (e.g. chat_model) do not render a reset button."""
    app = LazySettingsTestApp("chat_model")
    async with app.run_test(size=(120, 40)) as _pilot:
        chat_row = app.screen.query_one("#row-chat_model")
        buttons = chat_row.query(".setting-reset-button")
        assert len(buttons) == 0


async def test_ctrl_r_resets_focused_row():
    """Ctrl+R walks up from the focused editor to reset its row."""
    from textual.widgets import Input

    cfg.top_k = 99
    app = LazySettingsTestApp("top_k")
    async with app.run_test(size=(120, 40)) as pilot:
        editor = app.screen.query_one("#ed-top_k", Input)
        editor.focus()
        assert await wait_until(pilot, lambda: app.focused is editor), "the editor never got focus"
        await pilot.press("ctrl+r")
        await pilot.pause()
        assert cfg.top_k == builtin_value("top_k")


async def test_ctrl_r_with_no_focus_is_noop():
    """action_reset_focused is a no-op when nothing is focused."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        with patch.object(screen, "_reset_to_default") as mock_reset:
            screen.focused = None  # type: ignore[misc]
            screen.action_reset_focused()
            mock_reset.assert_not_called()


async def test_ctrl_r_on_non_row_focus_is_noop():
    """action_reset_focused ignores focus that isn't inside a setting row.

    AUTO_FOCUS targets ``#settings-tabs Tabs`` -- the Tabs strip widget,
    which sits outside any setting row -- so the screen lands on a
    non-row widget by construction. ``Tab`` would move into the first
    row's editor (which IS a row), so do NOT press Tab here; assert
    directly from the auto-focus state.
    """
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        await pilot.pause()
        with patch.object(screen, "_reset_to_default") as mock_reset:
            screen.action_reset_focused()
            mock_reset.assert_not_called()


async def test_refresh_editor_updates_checkbox():
    """Resetting a boolean setting syncs the Checkbox widget to the default."""
    from textual.widgets import Checkbox

    default = bool(builtin_value("show_reasoning"))
    cfg.show_reasoning = not default
    app = LazySettingsTestApp("show_reasoning")
    async with app.run_test(size=(120, 40)) as pilot:
        checkbox = app.screen.query_one("#ed-show_reasoning", Checkbox)
        checkbox.value = not default
        await pilot.pause()
        from textual.widgets import Button

        button = app.screen.query_one("#reset-show_reasoning", Button)
        async with saved_setting(app, "show_reasoning"):
            button.press()
        assert cfg.show_reasoning == default
        assert checkbox.value == default


async def test_reset_nullable_to_none():
    """Resetting a nullable scalar clears cfg and empties the Input widget."""
    from textual.widgets import Button, Input

    cfg.seed = 42
    app = LazySettingsTestApp("seed")
    async with app.run_test(size=(120, 40)):
        button = app.screen.query_one("#reset-seed", Button)
        async with saved_setting(app, "seed"):
            button.press()
        assert cfg.seed is None
        editor = app.screen.query_one("#ed-seed", Input)
        assert editor.value == ""


async def test_reset_select_clears_when_default_none():
    """Resetting a Select-backed field to a None default clears the widget."""
    import dataclasses

    from textual.widgets import Select

    from lilbee.app.settings_map import SETTINGS_MAP, SettingDef
    from lilbee.cli.tui.screens.settings import SettingsScreen

    # The blank row exists only on a Select built for a nullable setting.
    nullable = dataclasses.replace(SETTINGS_MAP["wiki_clusterer"], nullable=True)
    app = LazySettingsTestApp("wiki_clusterer")
    with patch.dict(SETTINGS_MAP, {"wiki_clusterer": nullable}):
        async with app.run_test(size=(120, 40)) as _pilot:
            screen = app.screen
            assert isinstance(screen, SettingsScreen)
            select_widget = screen.query_one("#ed-wiki_clusterer", Select)
            # Seed a non-default choice so we can detect the clear.
            select_widget.value = "embedding"
            defn = SettingDef(
                type=str, nullable=True, group="Wiki", choices=("embedding", "concepts")
            )
            screen._refresh_editor("wiki_clusterer", defn, None)
            # clear() yields the "no selection" sentinel, not the seeded choice.
            assert select_widget.value is Select.NULL


async def test_reset_select_sets_value_when_default_provided():
    """Resetting a Select to a concrete string sets widget.value accordingly."""
    from textual.widgets import Select

    from lilbee.app.settings_map import SettingDef
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = LazySettingsTestApp("wiki_clusterer")
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        select_widget = screen.query_one("#ed-wiki_clusterer", Select)
        select_widget.value = "embedding"
        defn = SettingDef(type=str, nullable=False, group="Wiki", choices=("embedding", "concepts"))
        screen._refresh_editor("wiki_clusterer", defn, "concepts")
        assert select_widget.value == "concepts"


async def test_refresh_editor_missing_widget_is_logged(caplog):
    """Missing editor widget on refresh logs a debug message instead of raising."""
    from lilbee.app.settings_map import SettingDef
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        defn = SettingDef(type=str, nullable=True, group=SettingGroup.SYSTEM)
        with caplog.at_level("DEBUG", logger="lilbee.cli.tui.screens.settings"):
            screen._refresh_editor("nonexistent_key_xyz", defn, "irrelevant")
        assert any("nonexistent_key_xyz" in record.message for record in caplog.records)


async def test_refresh_editor_updates_textarea_list(monkeypatch):
    """Future-proofing: _refresh_editor loads list values into a TextArea."""
    from textual.widgets import TextArea

    from lilbee.app.settings_map import SettingDef
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        fake = MagicMock(spec=TextArea)
        monkeypatch.setattr(screen, "query_one", lambda *a, **kw: fake)
        defn = SettingDef(type=list, nullable=False, group="Crawling")
        screen._refresh_editor("fake_list", defn, ["a", "b"])
        fake.load_text.assert_called_once_with("a\nb")


async def test_refresh_editor_updates_textarea_scalar(monkeypatch):
    """TextArea non-list scalar values go through load_text as plain str."""
    from textual.widgets import TextArea

    from lilbee.app.settings_map import SettingDef
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        fake = MagicMock(spec=TextArea)
        monkeypatch.setattr(screen, "query_one", lambda *a, **kw: fake)
        defn = SettingDef(type=str, nullable=False, group=SettingGroup.SYSTEM)
        screen._refresh_editor("fake_str", defn, "hello")
        fake.load_text.assert_called_once_with("hello")


async def test_refresh_editor_updates_textarea_none(monkeypatch):
    """TextArea with None value loads an empty string."""
    from textual.widgets import TextArea

    from lilbee.app.settings_map import SettingDef
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        fake = MagicMock(spec=TextArea)
        monkeypatch.setattr(screen, "query_one", lambda *a, **kw: fake)
        defn = SettingDef(type=str, nullable=True, group=SettingGroup.SYSTEM)
        screen._refresh_editor("fake_str", defn, None)
        fake.load_text.assert_called_once_with("")


async def test_refresh_editor_guard_skips_untracked_key():
    """_refresh_editor must not seed a baseline for a key not already tracked.

    Every writable key with a real editor is seeded into ``_mount_display``
    at row construction, so the ``key in self._mount_display`` guard here is
    always satisfied today; it exists to stop a future editor kind from
    gaining a baseline through a plain refresh before its own construction
    path starts tracking it. Clearing an already-tracked key's entry stands
    in for that "never tracked" case.
    """
    from lilbee.app.settings_map import SETTINGS_MAP
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = LazySettingsTestApp("top_k")
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        assert "top_k" in screen._mount_display
        del screen._mount_display["top_k"]
        screen._refresh_editor("top_k", SETTINGS_MAP["top_k"], cfg.top_k)
        assert "top_k" not in screen._mount_display


async def test_reset_to_default_ignores_readonly_keys():
    """_reset_to_default is a no-op for read-only settings."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        with patch.object(app, "reset_and_publish") as mock_reset:
            screen._reset_to_default("chat_model")
            mock_reset.assert_not_called()


@pytest.mark.parametrize("error", [RuntimeError, OSError, ValueError])
async def test_a_store_error_at_reset_shows_a_message_and_keeps_the_tui(error):
    from lilbee.cli.tui import messages as msg
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        with (
            patch.object(app, "reset_and_publish", side_effect=error("lance: table corrupt")),
            patch.object(screen, "notify") as notify,
        ):
            screen._reset_to_default("top_k")
        await pilot.pause()
        assert app.is_running
        notify.assert_called_once_with(
            msg.SETTINGS_INVALID_VALUE.format(error="lance: table corrupt"), severity="error"
        )


async def test_reset_to_default_ignores_unknown_keys():
    """_reset_to_default is a no-op when the key is not in SETTINGS_MAP."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        with patch.object(app, "reset_and_publish") as mock_reset:
            screen._reset_to_default("nonexistent_key_xyz")
            mock_reset.assert_not_called()


async def test_reset_button_with_malformed_id_is_noop():
    """Button press events with unexpected ids are gracefully ignored."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        event = MagicMock()
        event.button.id = "not-a-reset-button"
        with patch.object(screen, "_reset_to_default") as mock_reset:
            screen._on_reset_pressed(event)
            mock_reset.assert_not_called()
        event.button.id = None
        with patch.object(screen, "_reset_to_default") as mock_reset:
            screen._on_reset_pressed(event)
            mock_reset.assert_not_called()


async def test_list_restore_removes_the_user_key(tmp_path):
    """Restore defaults deletes the list key from config.toml and shows the resolved list."""
    from textual.widgets import Button

    from lilbee.cli.tui.widgets.list_text_area import ListTextArea

    persistent_settings.update_values(tmp_path, {"crawl_exclude_patterns": "mine"})
    cfg.crawl_exclude_patterns = ["mine"]
    expected = builtin_value("crawl_exclude_patterns")
    app = LazySettingsTestApp("crawl_exclude_patterns")
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        screen.query_one("#list-restore-crawl_exclude_patterns", Button).press()
        await wait_until(pilot, lambda: cfg.crawl_exclude_patterns == expected)
        assert cfg.crawl_exclude_patterns == expected
        assert "crawl_exclude_patterns" not in persistent_settings.load(tmp_path)
        ta = screen.query_one("#ed-crawl_exclude_patterns", ListTextArea)
        assert ta.text == "\n".join(expected)


async def test_ctrl_r_then_blur_writes_nothing(tmp_path):
    """Ctrl+R deletes the key; blurring the reset field afterwards does not write it back."""
    from textual.widgets import Input

    persistent_settings.update_values(tmp_path, {"top_k": 99, "chunk_size": 1024})
    cfg.top_k = 99
    default = builtin_value("top_k")
    app = LazySettingsTestApp("top_k", "max_distance")
    async with app.run_test(size=(120, 40)) as pilot:
        editor = app.screen.query_one("#ed-top_k", Input)
        editor.focus()
        await wait_until(pilot, lambda: app.screen.focused is editor)
        await pilot.press("ctrl+r")
        await wait_until(pilot, lambda: cfg.top_k == default)
        assert editor.value == str(default)
        assert persistent_settings.load(tmp_path) == {"chunk_size": 1024}
        app.screen.query_one("#ed-max_distance", Input).focus()
        await wait_until(pilot, lambda: not editor.has_focus)
        for _ in range(5):
            await pilot.pause()
        assert persistent_settings.load(tmp_path) == {"chunk_size": 1024}


@pytest.mark.parametrize(
    ("stored", "shown"),
    [(None, ""), (True, "True"), (False, "False")],
)
async def test_flash_attention_shows_three_distinct_states(tmp_path, stored, shown):
    """Auto, on and off each look different: a Select, not a Checkbox that hides auto as off."""
    from textual.widgets import Select

    cfg.data_root = tmp_path
    cfg.flash_attention = stored
    app = LazySettingsTestApp("flash_attention")
    async with app.run_test(size=(120, 40)):
        select = app.screen.query_one("#ed-flash_attention", Select)
        assert select.is_blank() if shown == "" else (select.value == shown)


async def test_flash_attention_picks_save_and_auto_clears(tmp_path):
    """Picking On or Off stores a bool; clearing the Select stores auto (None)."""
    from textual.widgets import Select

    cfg.data_root = tmp_path
    cfg.flash_attention = None
    app = LazySettingsTestApp("flash_attention")
    async with app.run_test(size=(120, 40)) as pilot:
        select = app.screen.query_one("#ed-flash_attention", Select)
        select.value = "False"
        await pump_until(pilot, lambda: cfg.flash_attention is False)
        assert cfg.flash_attention is False
        select.value = "True"
        await pump_until(pilot, lambda: cfg.flash_attention is True)
        assert cfg.flash_attention is True
        select.clear()
        await pump_until(pilot, lambda: cfg.flash_attention is None)
        assert cfg.flash_attention is None


async def test_a_late_editor_sync_does_not_save_the_old_select_value_again(tmp_path):
    """The first pick's editor sync runs after the second pick; the second pick stays saved."""
    from textual.widgets import Select

    cfg.data_root = tmp_path
    cfg.flash_attention = None
    app = LazySettingsTestApp("flash_attention")
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        select = screen.query_one("#ed-flash_attention", Select)
        held = []
        real_after_refresh = screen.call_after_refresh
        screen.call_after_refresh = lambda cb, *a, **k: held.append((cb, a, k))
        select.value = "False"
        await pump_until(pilot, lambda: cfg.flash_attention is False)
        screen.call_after_refresh = real_after_refresh
        assert held
        select.value = "True"
        for callback, args, kwargs in held:
            callback(*args, **kwargs)
        await pump_until(pilot, lambda: cfg.flash_attention is True)
        await pilot.pause()
        assert cfg.flash_attention is True


async def test_select_reset_then_blur_writes_nothing(tmp_path):
    """A reset Select shows the resolved value and its blur does not write it back."""
    from textual.widgets import Button, Select

    persistent_settings.update_values(tmp_path, {"ocr_strategy": "scanned_pages", "top_k": 5})
    cfg.ocr_strategy = "scanned_pages"
    default = str(builtin_value("ocr_strategy"))
    app = LazySettingsTestApp("ocr_strategy", "max_distance")
    async with app.run_test(size=(120, 40)) as pilot:
        select = app.screen.query_one("#ed-ocr_strategy", Select)
        app.screen.query_one("#reset-ocr_strategy", Button).press()
        await wait_until(pilot, lambda: str(select.value) == default)
        assert persistent_settings.load(tmp_path) == {"top_k": 5}
        await _blur_after_focus(pilot, app, select)
        assert persistent_settings.load(tmp_path) == {"top_k": 5}


async def test_multiline_reset_then_blur_writes_nothing(tmp_path):
    """A reset multiline editor shows the resolved text and its blur does not write it back."""
    from textual.widgets import Button

    from lilbee.cli.tui.widgets.list_text_area import ListTextArea

    persistent_settings.update_values(tmp_path, {"general_system_prompt": "mine", "top_k": 5})
    cfg.general_system_prompt = "mine"
    default = str(builtin_value("general_system_prompt"))
    app = LazySettingsTestApp("general_system_prompt", "max_distance")
    async with app.run_test(size=(120, 40)) as pilot:
        editor = app.screen.query_one("#ed-general_system_prompt", ListTextArea)
        app.screen.query_one("#reset-general_system_prompt", Button).press()
        await wait_until(pilot, lambda: editor.text == default)
        assert persistent_settings.load(tmp_path) == {"top_k": 5}
        await _blur_after_focus(pilot, app, editor)
        assert persistent_settings.load(tmp_path) == {"top_k": 5}


async def test_list_restore_then_blur_writes_nothing(tmp_path):
    """Restoring a list's defaults and then leaving the editor writes nothing back."""
    from textual.widgets import Button

    from lilbee.cli.tui.widgets.list_text_area import ListTextArea

    persistent_settings.update_values(tmp_path, {"crawl_exclude_patterns": "mine", "top_k": 5})
    cfg.crawl_exclude_patterns = ["mine"]
    expected = builtin_value("crawl_exclude_patterns")
    app = LazySettingsTestApp("crawl_exclude_patterns", "max_distance")
    async with app.run_test(size=(120, 40)) as pilot:
        editor = app.screen.query_one("#ed-crawl_exclude_patterns", ListTextArea)
        app.screen.query_one("#list-restore-crawl_exclude_patterns", Button).press()
        await wait_until(pilot, lambda: editor.text == "\n".join(expected))
        assert persistent_settings.load(tmp_path) == {"top_k": 5}
        await _blur_after_focus(pilot, app, editor)
        assert persistent_settings.load(tmp_path) == {"top_k": 5}


async def test_list_restore_updates_the_title_and_clears_the_error(tmp_path):
    """Restore defaults shows the resolved line count and hides a visible list error."""
    from textual.widgets import Button, Collapsible, Static

    from lilbee.cli.tui import messages as msg
    from lilbee.cli.tui.screens.settings_widgets import LIST_ERROR_VISIBLE_CLASS

    key = "crawl_exclude_patterns"
    persistent_settings.update_values(tmp_path, {key: "a\nb\nc\nd\ne"})
    cfg.crawl_exclude_patterns = ["a", "b", "c", "d", "e"]
    expected = builtin_value("crawl_exclude_patterns")
    assert len(expected) != 5
    app = LazySettingsTestApp("crawl_exclude_patterns")
    async with app.run_test(size=(120, 40)) as pilot:
        collapsible = app.screen.query_one("#collapsible-crawl_exclude_patterns", Collapsible)
        assert str(collapsible.title) == msg.SETTINGS_LIST_EDITOR_TITLE.format(key=key, count=5)
        error = app.screen.query_one("#err-crawl_exclude_patterns", Static)
        error.add_class(LIST_ERROR_VISIBLE_CLASS)
        app.screen.query_one("#list-restore-crawl_exclude_patterns", Button).press()
        await wait_until(pilot, lambda: cfg.crawl_exclude_patterns == expected)
        assert str(collapsible.title) == msg.SETTINGS_LIST_EDITOR_TITLE.format(
            key=key, count=len(expected)
        )
        assert not error.has_class(LIST_ERROR_VISIBLE_CLASS)


async def test_checkbox_reset_does_not_write_the_value_back(tmp_path):
    """Resetting a checkbox moves its value without the change handler saving it."""
    from textual.widgets import Button, Checkbox

    default = bool(builtin_value("show_reasoning"))
    persistent_settings.update_values(tmp_path, {"show_reasoning": not default, "top_k": 7})
    cfg.show_reasoning = not default
    app = LazySettingsTestApp("show_reasoning")
    async with app.run_test(size=(120, 40)) as pilot:
        checkbox = app.screen.query_one("#ed-show_reasoning", Checkbox)
        app.screen.query_one("#reset-show_reasoning", Button).press()
        await wait_until(pilot, lambda: checkbox.value == default)
        for _ in range(5):
            await pilot.pause()
        assert checkbox.value == default
        assert cfg.show_reasoning == default
        assert persistent_settings.load(tmp_path) == {"top_k": 7}


async def test_checkbox_toggle_queued_before_a_reset_does_not_resave(tmp_path):
    """A Changed message still queued when a reset runs must not undo it.

    The toggle is applied directly (not via pilot.press) so its Changed
    message is still sitting in the widget's queue, unprocessed, when the
    reset call runs synchronously right after it in the same tick.
    """
    from textual.widgets import Checkbox

    default = bool(builtin_value("show_reasoning"))
    persistent_settings.update_values(tmp_path, {"show_reasoning": not default, "top_k": 5})
    cfg.show_reasoning = not default
    app = LazySettingsTestApp("show_reasoning")
    async with app.run_test(size=(120, 40)) as pilot:
        checkbox = app.screen.query_one("#ed-show_reasoning", Checkbox)
        checkbox.toggle()
        app.screen._reset_to_default("show_reasoning")
        for _ in range(5):
            await pilot.pause()
        assert checkbox.value == default
        assert cfg.show_reasoning == default
        assert persistent_settings.load(tmp_path) == {"top_k": 5}


async def test_reset_all_removes_every_user_key(tmp_path):
    """Reset-all leaves config.toml with no writable key, even after the checkboxes settle."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    persistent_settings.update_values(tmp_path, {"top_k": 99, "show_reasoning": True})
    cfg.top_k = 99
    cfg.show_reasoning = True
    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        # Build only the pane that holds the checkbox this test watches settle.
        populate_pane_of(screen, "show_reasoning")
        assert await wait_until(pilot, lambda: bool(screen.query("#ed-show_reasoning")))
        screen._on_reset_all_confirmed(True)
        for _ in range(5):
            await pilot.pause()
        assert persistent_settings.load(tmp_path) == {}
        assert cfg.top_k == builtin_value("top_k")


async def test_a_refresh_of_every_key_walks_the_screen_once():
    """The refresh after a reset of N keys costs one walk of the screen, not N walks."""
    from textual.css import query as query_module

    from lilbee.app.settings_map import SETTINGS_MAP

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)):
        screen = app.screen
        widgets = len(list(screen.walk_children()))
        screen._stale_source_keys = {key for key, defn in SETTINGS_MAP.items() if defn.writable}
        with patch.object(query_module, "match", wraps=query_module.match) as matched:
            screen._refresh_changed_rows()
        assert matched.call_count <= 3 * widgets


async def test_reset_failure_toasts_and_changes_nothing(tmp_path):
    """A boundary refusal on reset toasts an error and leaves the field and file alone."""
    from textual.widgets import Input

    from lilbee.cli.tui.screens.settings import SettingsScreen

    persistent_settings.update_values(tmp_path, {"top_k": 99})
    cfg.top_k = 99
    app = LazySettingsTestApp("top_k")
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        with (
            patch(
                "lilbee.app.settings.persistent_settings.delete_values",
                side_effect=OSError("disk full"),
            ),
            patch.object(screen, "notify") as mock_notify,
        ):
            screen._reset_to_default("top_k")
        assert mock_notify.call_args.kwargs.get("severity") == "error"
        assert cfg.top_k == 99
        assert screen.query_one("#ed-top_k", Input).value == "99"
        assert persistent_settings.load(tmp_path) == {"top_k": 99}


async def test_reset_that_turns_the_wiki_off_offers_the_wipe(tmp_path):
    """A reset that switches the wiki off makes the same wipe offer /set does."""
    persistent_settings.update_values(tmp_path, {"wiki": True})
    cfg.wiki = True
    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch.object(app, "_offer_wiki_wipe") as offer:
            assert app.reset_and_publish(["wiki"]) == ["wiki"]
        offer.assert_called_once()
        assert cfg.wiki is False
        assert "wiki" not in persistent_settings.load(tmp_path)


async def test_reset_with_the_wiki_already_off_offers_nothing(tmp_path):
    """Reset only offers the wipe when it is what turned the wiki off."""
    cfg.wiki = False
    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch.object(app, "_offer_wiki_wipe") as offer:
            assert app.reset_and_publish(["wiki", "top_k"]) == ["top_k", "wiki"]
        offer.assert_not_called()


async def test_a_reset_toasts_the_warning_of_its_update(tmp_path, monkeypatch):
    """A reset that brings an env var's vision model back under ocr = off toasts the notice."""
    ref = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
    monkeypatch.setenv("LILBEE_VISION_MODEL", ref)
    (tmp_path / "config.toml").write_text("ocr = 'off'\n", encoding="utf-8")
    cfg.ocr = OcrMode.OFF
    cfg.vision_model = ""
    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with (
            patch("lilbee.app.settings._invalidate_caches"),
            patch.object(app, "notify") as notify,
        ):
            assert app.reset_and_publish(["vision_model"]) == ["vision_model"]
        notify.assert_called_once()
        assert ref in notify.call_args.args[0]
        assert notify.call_args.kwargs.get("severity") == "warning"
        assert cfg.ocr is OcrMode.OFF


async def test_reset_theme_applies_the_resolved_theme(tmp_path):
    """Resetting the theme repaints with the resolved theme, not only cfg."""
    default = builtin_value("theme")
    other = next(name for name in ("gruvbox", "nord", "dracula") if name != default)
    persistent_settings.update_values(tmp_path, {"theme": other})
    cfg.theme = other
    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.theme = other
        app.reset_and_publish(["theme"])
        assert cfg.theme == default
        assert app.theme == default


async def test_reset_all_uses_footer_binding_not_button():
    """Reset-all is now a Ctrl+Shift+R footer binding, not a button widget.

    Per user feedback the button was awkwardly placed; replaced with a
    keyboard binding shown in the Footer so the destructive action is
    deliberate (modal confirms before mutating cfg).
    """
    from textual.widgets import Button

    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        # The button widget must be gone.
        assert not list(app.screen.query("#reset-all-defaults").results(Button))
        # The footer-visible binding must be present.
        names = [b.action for b in screen.BINDINGS]  # type: ignore[attr-defined]
        assert "reset_all" in names


async def test_reset_all_cancel_does_nothing():
    """Dismissing the confirm dialog with False leaves cfg untouched."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        with patch.object(screen, "_reset_to_default") as mock_reset:
            screen._on_reset_all_confirmed(False)
            mock_reset.assert_not_called()
        with patch.object(screen, "_reset_to_default") as mock_reset:
            screen._on_reset_all_confirmed(None)
            mock_reset.assert_not_called()


async def test_reset_all_confirm_deletes_in_one_batch():
    """Confirming the dialog issues one batched delete and writes no value."""
    from lilbee.app.settings import reset_settings
    from lilbee.app.settings_map import SETTINGS_MAP
    from lilbee.cli.tui.screens.settings import SettingsScreen

    # The screen resets whatever the boundary accepts under skip_unresettable=True;
    # query that set so this test stays agnostic to which keys the boundary refuses.
    writable_keys = set(
        reset_settings(
            [k for k, d in SETTINGS_MAP.items() if d.writable], skip_unresettable=True
        ).updated
    )
    assert writable_keys
    readonly_keys = {k for k, d in SETTINGS_MAP.items() if not d.writable}
    assert readonly_keys, "test invariant: SETTINGS_MAP must contain a readonly field"

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        with (
            patch("lilbee.app.settings.persistent_settings.update_values") as mock_update,
            patch("lilbee.app.settings.persistent_settings.delete_values") as mock_delete,
        ):
            screen._on_reset_all_confirmed(True)
        mock_update.assert_not_called()
        mock_delete.assert_called_once()
        deleted = set(mock_delete.call_args.args[1])
        assert deleted == writable_keys
        assert not deleted & readonly_keys


async def test_reset_all_suppresses_per_field_toasts():
    """Reset-all fires exactly one summary toast; no per-field CMD_SET_SUCCESS spam."""
    from lilbee.cli.tui import messages as msg
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        with (
            patch("lilbee.app.settings.persistent_settings.delete_values"),
            patch.object(screen, "notify") as mock_notify,
        ):
            screen._on_reset_all_confirmed(True)
        assert mock_notify.call_count == 1
        assert mock_notify.call_args.args[0] == msg.SETTINGS_RESET_ALL_SUCCESS


async def test_reset_all_actually_mutates_cfg():
    """Batch reset restores cfg values to pydantic defaults (not just the TOML write)."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    default_top_k = builtin_value("top_k")
    cfg.top_k = 999
    assert cfg.top_k == 999

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        with patch("lilbee.app.settings.persistent_settings.delete_values"):
            screen._on_reset_all_confirmed(True)
        assert cfg.top_k == default_top_k


async def test_reset_all_leaves_cfg_on_disk_write_failure():
    """OSError from the TOML write leaves cfg as it was so UI and disk stay in sync."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    cfg.top_k = 999
    cfg.wiki_clusterer_k = 77

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        with (
            patch(
                "lilbee.app.settings.persistent_settings.delete_values",
                side_effect=OSError("disk full"),
            ),
            patch.object(screen, "notify") as mock_notify,
        ):
            screen._on_reset_all_confirmed(True)
        assert cfg.top_k == 999
        assert cfg.wiki_clusterer_k == 77
        assert mock_notify.called
        notify_kwargs = mock_notify.call_args.kwargs
        assert notify_kwargs.get("severity") == "error"


async def test_reset_all_publishes_signals_on_lilbee_app():
    """When the screen runs under LilbeeApp, the reset loop fans out a signal per key."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        await pilot.pause()
        app.push_screen(SettingsScreen())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        with (
            patch("lilbee.app.settings.persistent_settings.delete_values"),
            patch.object(app.settings_changed_signal, "publish") as mock_pub,
        ):
            screen._on_reset_all_confirmed(True)
        # One publish per writable setting the boundary actually resets.
        from lilbee.app.settings import reset_settings
        from lilbee.app.settings_map import SETTINGS_MAP

        expected = len(
            reset_settings(
                [k for k, d in SETTINGS_MAP.items() if d.writable], skip_unresettable=True
            ).updated
        )
        assert mock_pub.call_count == expected


async def test_reset_all_action_opens_confirm_dialog():
    """Triggering action_reset_all (Ctrl+Shift+R) pushes the ConfirmDialog."""
    from lilbee.cli.tui.screens.settings import SettingsScreen
    from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog

    app = LazySettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        with patch.object(screen.app, "push_screen") as mock_push:
            screen.action_reset_all()
        mock_push.assert_called_once()
        pushed_screen = mock_push.call_args.args[0]
        assert isinstance(pushed_screen, ConfirmDialog)
