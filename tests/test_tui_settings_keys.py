"""Settings screen keys: select-on-focus, Escape in a field, the auto pill, the Advanced title."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from textual.app import ComposeResult
from textual.widgets import Checkbox, Collapsible, Input, Static, TabbedContent
from textual.widgets import Footer as _Footer
from typer.testing import CliRunner

from lilbee.app.settings import setting_sources
from lilbee.app.settings_map import SETTINGS_MAP, SettingGroup
from lilbee.cli.app import app as cli_app
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.screens.settings import SettingsScreen
from lilbee.cli.tui.screens.settings_widgets import SettingRow
from lilbee.cli.tui.widgets.stable_footer import StableFooter
from lilbee.core.config import cfg
from lilbee.core.config.enums import SettingSource
from tests._lilbee_app_test_host import LilbeeAppHost, pump_until
from tests.test_tui_settings_perf import _make_eager_settings_screen

SIZES = [(80, 24), (120, 40)]
RETRIEVAL = "settings-tab-retrieval"
OCR = "settings-tab-ocr_tuning"


class _App(LilbeeAppHost):
    CSS = ""

    def compose(self) -> ComposeResult:
        yield _Footer()

    def on_mount(self) -> None:
        self.push_screen(_make_eager_settings_screen()())


async def _open(app: _App, pilot, pane: str) -> SettingsScreen:
    assert await pump_until(pilot, lambda: isinstance(app.screen, SettingsScreen))
    screen = app.screen
    assert isinstance(screen, SettingsScreen)
    tabs = screen.query_one("#settings-tabs", TabbedContent)
    assert await pump_until(pilot, lambda: bool(screen.query(f"#{pane} Input, #{pane} Checkbox")))
    tabs.active = pane
    assert await pump_until(pilot, lambda: tabs.active == pane)
    return screen


def _active_pane(screen: SettingsScreen) -> str:
    return screen.query_one("#settings-tabs", TabbedContent).active


@pytest.mark.parametrize("size", SIZES)
async def test_key_focus_selects_the_number_so_typing_replaces_it(size):
    app = _App()
    async with app.run_test(size=size) as pilot:
        screen = await _open(app, pilot, RETRIEVAL)
        field = screen.query_one("#ed-top_k", Input)
        field.value = "12"
        field.focus()
        assert await pump_until(pilot, lambda: field.has_focus)
        await pilot.press("2", "5")
        assert await pump_until(pilot, lambda: field.value == "25"), field.value


@pytest.mark.parametrize("size", SIZES)
async def test_click_focus_selects_the_number_and_a_second_click_places_the_cursor(size):
    app = _App()
    async with app.run_test(size=size) as pilot:
        screen = await _open(app, pilot, RETRIEVAL)
        field = screen.query_one("#ed-top_k", Input)
        field.value = "12"
        await pilot.click("#ed-top_k")
        assert await pump_until(pilot, lambda: field.has_focus)
        assert await pump_until(pilot, lambda: field.selected_text == "12"), field.selection
        await pilot.press("2", "5")
        assert await pump_until(pilot, lambda: field.value == "25"), field.value
        await pilot.click("#ed-top_k")
        assert await pump_until(pilot, lambda: field.selected_text == "")
        await pilot.press("9")
        assert await pump_until(pilot, lambda: "9" in field.value and len(field.value) == 3)


@pytest.mark.parametrize("size", SIZES)
async def test_a_click_on_a_row_outside_its_field_leaves_focus_in_the_field(size):
    app = _App()
    async with app.run_test(size=size) as pilot:
        screen = await _open(app, pilot, RETRIEVAL)
        field = screen.query_one("#ed-top_k", Input)
        row = field.parent.parent
        assert isinstance(row, SettingRow)
        field.focus()
        assert await pump_until(pilot, lambda: field.has_focus)
        await pilot.click(row, offset=(0, 0))
        await pilot.press("7")
        assert await pump_until(pilot, lambda: "7" in field.value), field.value
        assert app.focused is field


@pytest.mark.parametrize("size", SIZES)
async def test_escape_in_a_field_leaves_the_field_and_a_second_escape_goes_back(size):
    app = _App()
    async with app.run_test(size=size) as pilot:
        screen = await _open(app, pilot, RETRIEVAL)
        field = screen.query_one("#ed-top_k", Input)
        field.focus()
        assert await pump_until(pilot, lambda: field.has_focus)
        # The host app has no chat screen to go back to, so go_back is the observed close.
        with patch.object(app, "go_back") as go_back:
            await pilot.press("escape")
            assert await pump_until(pilot, lambda: isinstance(app.focused, SettingRow))
            assert app.focused is field.parent.parent
            assert app.screen is screen
            assert go_back.call_count == 0
            await pilot.press("escape")
            assert await pump_until(pilot, lambda: go_back.call_count == 1)


@pytest.mark.parametrize("size", SIZES)
async def test_field_keeps_angle_brackets_and_tab_keys_work_after_escape(size):
    app = _App()
    async with app.run_test(size=size) as pilot:
        screen = await _open(app, pilot, RETRIEVAL)
        field = screen.query_one("#ed-top_k", Input)
        field.value = ""
        field.focus()
        assert await pump_until(pilot, lambda: field.has_focus)
        await pilot.press("<", ">", "[", "]")
        assert await pump_until(pilot, lambda: field.value == "<>[]"), field.value
        assert _active_pane(screen) == RETRIEVAL
        await pilot.press("escape")
        assert await pump_until(pilot, lambda: isinstance(app.focused, SettingRow))
        await pilot.press(">")
        assert await pump_until(pilot, lambda: _active_pane(screen) != RETRIEVAL)
        assert field.value == "<>[]"


@pytest.mark.parametrize("size", SIZES)
async def test_tab_keys_work_from_a_checkbox_in_the_body(size):
    app = _App()
    async with app.run_test(size=size) as pilot:
        screen = await _open(app, pilot, "settings-tab-ingest")
        box = screen.query("#settings-tab-ingest Checkbox").first(Checkbox)
        box.focus()
        assert await pump_until(pilot, lambda: box.has_focus)
        await pilot.press(">")
        assert await pump_until(pilot, lambda: _active_pane(screen) != "settings-tab-ingest")


@pytest.mark.parametrize("size", SIZES)
async def test_footer_names_leave_field_only_while_a_field_has_focus(size):
    app = _App()
    async with app.run_test(size=size) as pilot:
        screen = await _open(app, pilot, RETRIEVAL)
        footer = screen.query_one(StableFooter)
        field = screen.query_one("#ed-top_k", Input)

        def shown() -> dict[str, str]:
            return {
                key: binding.description
                for key, (_ns, binding, enabled, _tip) in screen.active_bindings.items()
                if binding.show and enabled
            }

        assert footer.is_attached
        field.focus()
        assert await pump_until(pilot, lambda: shown().get("escape") == msg.SETTINGS_LEAVE_FIELD)
        assert "greater_than_sign" not in shown()
        await pilot.press("escape")
        assert await pump_until(pilot, lambda: "escape" not in shown())
        assert "greater_than_sign" in shown()


@pytest.mark.parametrize("size", SIZES)
async def test_auto_source_shows_the_pill_the_cli_prints(size, tmp_path, monkeypatch):
    # A live value that differs from the built-in, with no stored value, is an auto key.
    monkeypatch.setattr(cfg, "num_ctx", 4096)
    command = ["--json", "settings", "list", "--group", "Generation", "--data-dir", str(tmp_path)]
    listed = CliRunner().invoke(cli_app, command)
    assert listed.exit_code == 0, listed.output
    cli_source = {s["key"]: s["source"] for s in json.loads(listed.output)["settings"]}["num_ctx"]
    assert cli_source == "auto"
    app = _App()
    async with app.run_test(size=size) as pilot:
        screen = await _open(app, pilot, "settings-tab-generation")
        title = screen.query_one("#row-num_ctx .setting-title", Static)
        assert await pump_until(pilot, lambda: f" {cli_source} " in str(title.render()))
        built_in = screen.query_one("#row-temperature .setting-title", Static)
        assert f" {cli_source} " not in str(built_in.render())


def _advanced_keys(group: SettingGroup) -> list[str]:
    return [k for k, d in SETTINGS_MAP.items() if d.group is group and d.advanced and not d.hidden]


@pytest.mark.parametrize("size", SIZES)
async def test_collapsed_advanced_title_counts_profile_set_keys(size):
    keys = _advanced_keys(SettingGroup.OCR_TUNING)
    assert keys
    sources = {**setting_sources(), keys[0]: SettingSource.PROFILE}
    app = _App()
    with patch("lilbee.cli.tui.screens.settings.setting_sources", return_value=sources):
        async with app.run_test(size=size) as pilot:
            screen = await _open(app, pilot, OCR)
            section = screen.query_one(f"#settings-advanced-{OCR}", Collapsible)
            assert section.collapsed
            want = f"Advanced ({len(keys)}, 1 set by profile)"
            assert await pump_until(pilot, lambda: section.title == want), section.title
            assert section.collapsed


@pytest.mark.parametrize("size", SIZES)
async def test_collapsed_advanced_title_is_plain_without_a_profile_key(size):
    keys = _advanced_keys(SettingGroup.OCR_TUNING)
    sources = {k: v for k, v in setting_sources().items()}
    sources.update({k: SettingSource.BUILT_IN for k in keys})
    app = _App()
    with patch("lilbee.cli.tui.screens.settings.setting_sources", return_value=sources):
        async with app.run_test(size=size) as pilot:
            screen = await _open(app, pilot, OCR)
            section = screen.query_one(f"#settings-advanced-{OCR}", Collapsible)
            assert section.title == msg.SETTINGS_ADVANCED_TITLE.format(count=len(keys))


@pytest.mark.parametrize("size", SIZES)
async def test_advanced_title_follows_a_profile_change_while_the_screen_is_open(size):
    keys = _advanced_keys(SettingGroup.OCR_TUNING)
    sources = {**setting_sources(), **{k: SettingSource.BUILT_IN for k in keys}}
    app = _App()
    with patch("lilbee.cli.tui.screens.settings.setting_sources", return_value=sources):
        async with app.run_test(size=size) as pilot:
            screen = await _open(app, pilot, OCR)
            section = screen.query_one(f"#settings-advanced-{OCR}", Collapsible)
            assert section.title == msg.SETTINGS_ADVANCED_TITLE.format(count=len(keys))
            sources[keys[0]] = SettingSource.PROFILE
            screen._on_setting_changed((keys[0], None))
            want = f"Advanced ({len(keys)}, 1 set by profile)"
            assert await pump_until(pilot, lambda: section.title == want), section.title
