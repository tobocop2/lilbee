"""The Profile tab, the profile line, the Apply and Save-as dialogs, and /profile."""

from __future__ import annotations

import tomllib
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from unittest import mock

import pytest
from textual import events
from textual.app import ComposeResult
from textual.geometry import Region
from textual.pilot import Pilot
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Checkbox, DataTable, Footer, Input, Select, Static, TabbedContent

from lilbee.app import profiles
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.command_registry import get_command
from lilbee.cli.tui.commands import LilbeeCommandProvider
from lilbee.cli.tui.screens.profile_dialogs import (
    ApplyProfileDialog,
    ProfilePathDialog,
    SaveProfileDialog,
    value_text,
)
from lilbee.cli.tui.screens.profile_tab import ProfileTab
from lilbee.cli.tui.screens.settings import PROFILE_PANE_ID, SettingsScreen
from lilbee.cli.tui.widgets.autocomplete import get_completions
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog, ConfirmPill
from lilbee.cli.tui.widgets.model_bar import ModelBar
from lilbee.cli.tui.widgets.profile_line import ProfileLinePill
from lilbee.cli.tui.widgets.slash_command_catalog import CATALOG_GROUPS
from lilbee.cli.tui.widgets.suggester import SlashSuggester
from lilbee.core import settings as persistent_settings
from lilbee.core.config import cfg
from lilbee.core.profile_files import PROFILES_DIRNAME, ProfileFolder, ProfileStore
from lilbee.core.system import default_data_dir
from tests._async_wait import press_widget, wait_until
from tests._lilbee_app_test_host import LilbeeAppHost, await_chat
from tests._profile_fixtures import (
    gate_releases_at_once,  # noqa: F401 -- autouse fixture, applied by import
    isolated_cfg,  # noqa: F401 -- autouse fixture, applied by import
    sources_totaling,
)


class _Pick(StrEnum):
    ONE = "one"


class _PillApp(LilbeeAppHost):
    def __init__(self) -> None:
        super().__init__()
        self.picked: list[object] = []

    def compose(self) -> ComposeResult:
        yield ConfirmPill("One", pill_id="pick-one", answer=_Pick.ONE)

    def on_confirm_pill_picked(self, event: ConfirmPill.Picked) -> None:
        self.picked.append(event.answer)


async def test_a_pill_carries_its_own_enum_answer() -> None:
    app = _PillApp()
    async with app.run_test() as pilot:
        await _press(pilot, app.query_one("#pick-one", ConfirmPill))
        assert await _until(pilot, lambda: bool(app.picked))
        assert app.picked == [_Pick.ONE]


async def test_publish_settings_tells_subscribers_each_key_and_its_value() -> None:
    app = LilbeeAppHost()
    seen: list[tuple[str, object]] = []
    async with app.run_test() as pilot:
        app.settings_changed_signal.subscribe(app, seen.append)
        cfg.chunk_size = 700
        cfg.top_k = 9
        app.publish_settings(["chunk_size", "top_k"])
        assert await _until(pilot, lambda: len(seen) == 2)
    assert seen == [("chunk_size", 700), ("top_k", 9)]


def test_start_rebuild_queues_a_forced_rebuild_on_the_chat_screen() -> None:
    app = LilbeeAppHost()
    chat = mock.MagicMock()
    with mock.patch.object(LilbeeAppHost, "chat_screen", return_value=chat):
        app.start_rebuild()
    chat.run_sync.assert_called_once_with(force_rebuild=True)


def test_start_rebuild_without_a_chat_screen_does_nothing() -> None:
    app = LilbeeAppHost()
    with mock.patch.object(LilbeeAppHost, "chat_screen", return_value=None) as chat_screen:
        app.start_rebuild()
    chat_screen.assert_called_once_with()


_LEGAL = """[profile]
name = "legal-discovery"
description = "Scanned legal exhibits."
authors = [{ name = "Jane Doe", github = "janedoe" }]
tested_on = "4,000 county filings"

[values]
chunk_size = 512
layout_detection = true
max_chunks_per_file = 3000
"""
_PAUSES = 300
_YOURS = "chunk_size = 384\nlayout_detection = false\nmax_chunks_per_file = 2000\n"


def _global_dir() -> Path:
    return default_data_dir() / PROFILES_DIRNAME


def _write_global(stem: str, text: str) -> Path:
    _global_dir().mkdir(parents=True, exist_ok=True)
    path = _global_dir() / f"{stem}.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _config_path() -> Path:
    return cfg.data_root / "config.toml"


def _set_yours(text: str) -> None:
    """Put top-level keys above the ``[profile]`` table and load them into cfg."""
    existing = _config_path().read_text(encoding="utf-8") if _config_path().exists() else ""
    _config_path().write_text(text + existing, encoding="utf-8")
    persistent_settings.overlay_persisted_settings(cfg.data_root)


def _on_legal_with_three_changes() -> None:
    _write_global("legal-discovery", _LEGAL)
    profiles.apply(ProfileStore(), "legal-discovery")
    _set_yours(_YOURS)


class _SettingsApp(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        yield Footer()

    def on_mount(self) -> None:
        self.push_screen(SettingsScreen())


def _settings(app: LilbeeAppHost) -> SettingsScreen:
    return next(s for s in app.screen_stack if isinstance(s, SettingsScreen))


def _text(widget: Static) -> str:
    return str(widget.render())


async def _loaded(app: LilbeeAppHost, pilot: Pilot, name: str) -> SettingsScreen:
    """Wait until the Profile tab and the line show *name*."""

    def _ready() -> bool:
        screens = [s for s in app.screen_stack if isinstance(s, SettingsScreen)]
        if not screens or not screens[0].query(ProfileTab):
            return False
        select = screens[0].query_one("#profile-select", Select)
        line = screens[0].query_one("#profile-line-name", Static)
        return select.value == name and _text(line) == name

    assert await _until(pilot, _ready), f"the tab never showed {name}"
    return _settings(app)


def _table_rows(screen: SettingsScreen, table_id: str) -> list[list[str]]:
    table = screen.query_one(table_id, DataTable)
    return [[str(cell) for cell in table.get_row_at(i)] for i in range(table.row_count)]


async def _until(pilot: Pilot, predicate: Callable[[], bool]) -> bool:
    """Wait on *predicate* with room for a loaded runner."""
    return await wait_until(pilot, predicate, max_pauses=_PAUSES)


async def _press(pilot: Pilot, widget: Widget) -> None:
    """Focus *widget*, wait until it holds focus, then press Enter on it."""
    await press_widget(pilot, widget, max_pauses=_PAUSES)


async def _dialog(app: LilbeeAppHost, pilot: Pilot, kind: type[Screen]) -> Screen:
    """Wait until a *kind* dialog is on top and has put focus inside itself."""
    assert await _until(
        pilot,
        lambda: (
            isinstance(app.screen, kind)
            and app.focused is not None
            and app.focused.screen is app.screen
        ),
    ), kind
    return app.screen


def _inside(region: Region, outer: Region) -> bool:
    return outer.contains_region(region)


def _pick(screen: SettingsScreen, name: str) -> None:
    screen.query_one("#profile-select", Select).value = name


@pytest.fixture
def sources():
    with sources_totaling(412) as services:
        yield services


async def test_profile_line_is_exactly_one_row_tall() -> None:
    app = _SettingsApp()
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await _loaded(app, pilot, "Default")
        line = screen.query_one("#profile-line")
        assert line.size.height == 1
        assert screen.query_one("#settings-tabs").size.height > 1


async def test_line_shows_the_profile_and_your_count_on_any_tab_and_jumps_to_the_tab() -> None:
    _on_legal_with_three_changes()
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _loaded(app, pilot, "legal-discovery")
        tabs = screen.query_one("#settings-tabs", TabbedContent)
        tabs.active = "settings-tab-ingest"
        await _until(pilot, lambda: tabs.active == "settings-tab-ingest")
        assert _text(screen.query_one("#profile-line-count", Static)) == "3 values set by you"
        assert not screen.query_one("#profile-line-status", Static).display
        await _press(pilot, screen.query_one("#profile-line-name", ProfileLinePill))
        assert await _until(pilot, lambda: tabs.active == PROFILE_PANE_ID)
        select = screen.query_one("#profile-select", Select)
        assert await _until(pilot, lambda: select.has_focus)


async def test_clicking_the_line_pill_jumps_to_the_profile_tab() -> None:
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _loaded(app, pilot, "Default")
        tabs = screen.query_one("#settings-tabs", TabbedContent)
        tabs.active = "settings-tab-models"
        await _until(pilot, lambda: tabs.active == "settings-tab-models")
        assert _text(screen.query_one("#profile-line-count", Static)) == "0 values set by you"
        await pilot.click("#profile-line-name")
        assert await _until(pilot, lambda: tabs.active == PROFILE_PANE_ID)


async def test_profile_tab_is_first_and_lists_your_changes_with_their_cost() -> None:
    _on_legal_with_three_changes()
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _loaded(app, pilot, "legal-discovery")
        assert screen._pane_ids[0] == PROFILE_PANE_ID
        assert screen.query_one("#settings-tabs", TabbedContent).active == PROFILE_PANE_ID
        assert _table_rows(screen, "#profile-changes") == [
            ["chunk_size", "384", "512", " reindex "],
            ["max_chunks_per_file", "2000", "3000", "new files only"],
            ["layout_detection", "off", "on", " reindex "],
        ]
        table = screen.query_one("#profile-changes", DataTable)
        cost = table.ordered_columns[-1]
        assert cost.get_render_width(table) >= len("new files only")
        assert _text(screen.query_one("#profile-description", Static)) == "Scanned legal exhibits."
        assert _text(screen.query_one("#profile-credit", Static)) == (
            "by Jane Doe (@janedoe). Tested on: 4,000 county filings"
        )
        assert _text(screen.query_one("#profile-folder", Static)) == "Saved for all projects."
        assert not screen.query_one("#profile-status-note", Static).display
        shown = {p.id for p in screen.query("#profile-actions ConfirmPill") if p.display}
        assert shown == {"profile-update", "profile-save_as", "profile-discard"}
        assert _text(screen.query_one("#profile-update", ConfirmPill)) == "Update legal-discovery"


_LONG_NAME = "A" * 40


async def test_action_row_fits_80_columns_with_a_long_profile_name() -> None:
    assert len(_LONG_NAME) == 40
    _write_global("long-name", f'[profile]\nname = "{_LONG_NAME}"\n\n[values]\nchunk_size = 900\n')
    profiles.apply(ProfileStore(), _LONG_NAME)
    _set_yours("chunk_size = 123\n")
    app = _SettingsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _loaded(app, pilot, _LONG_NAME)
        shown = [p for p in screen.query("#profile-actions ConfirmPill") if p.display]
        assert {p.id for p in shown} == {"profile-update", "profile-save_as", "profile-discard"}
        for pill in shown:
            assert pill.region.x + pill.region.width <= 80, pill.id


async def test_a_builtin_profile_offers_no_update() -> None:
    profiles.apply(ProfileStore(), "Scanned archive")
    _set_yours("chunk_size = 700\n")
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _loaded(app, pilot, "Scanned archive")
        shown = {p.id for p in screen.query("#profile-actions ConfirmPill") if p.display}
        assert shown == {"profile-save_as", "profile-discard"}
        assert _text(screen.query_one("#profile-folder", Static)) == (
            "A built-in profile that ships with lilbee."
        )
        assert not screen.query_one("#profile-credit", Static).display


async def test_no_changes_hides_the_table_and_says_so() -> None:
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _loaded(app, pilot, "Default")
        assert not screen.query_one("#profile-changes", DataTable).display
        assert _text(screen.query_one("#profile-changes-help", Static)) == (
            "No values of yours override Default."
        )


async def test_picking_a_profile_asks_first_and_keeps_your_values(sources) -> None:
    _on_legal_with_three_changes()
    app = _SettingsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _loaded(app, pilot, "legal-discovery")
        _pick(screen, "Notes and markdown")
        await _dialog(app, pilot, ApplyProfileDialog)
        dialog = app.screen
        changes = [row[0] for row in _table_rows(dialog, "#apply-changes")]
        assert "chunk_size" not in changes
        assert changes == ["chunk_overlap", "enable_ocr"]
        assert _text(dialog.query_one("#apply-kept", Static)).startswith("chunk_size 384")
        pills = [p.id for p in dialog.query("#apply-actions ConfirmPill")]
        assert pills == ["apply-reindex", "apply-apply", "apply-cancel"]
        assert _text(dialog.query_one("#apply-summary", Static)) == (
            "One change rebuilds the index of 412 files."
        )
        screen_region = app.screen.region
        assert _inside(dialog.query_one("#apply-body").region, screen_region)
        for pill in dialog.query("#apply-actions ConfirmPill"):
            assert _inside(pill.region, screen_region), pill.id
        await pilot.press("escape")
        assert await _until(pilot, lambda: app.screen is screen)
        select = screen.query_one("#profile-select", Select)
        assert await _until(pilot, lambda: select.value == "legal-discovery")
        assert cfg.chunk_overlap == 100
        assert profiles.active(ProfileStore()).name == "legal-discovery"


async def test_apply_sets_cfg_and_refreshes_the_ingest_editor_and_the_line(sources) -> None:
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _loaded(app, pilot, "Default")
        screen.populate_all_panes()
        assert await _until(pilot, lambda: bool(screen.query("#ed-table_extraction")))
        editor = screen.query_one("#ed-table_extraction", Checkbox)
        assert editor.value is False
        _pick(screen, "Research papers")
        await _dialog(app, pilot, ApplyProfileDialog)
        await _press(pilot, app.screen.query_one("#apply-apply", ConfirmPill))
        await _loaded(app, pilot, "Research papers")
        assert cfg.table_extraction is True
        assert await _until(pilot, lambda: editor.value is True)
        assert profiles.active(ProfileStore()).name == "Research papers"


async def test_apply_and_reindex_starts_a_rebuild(sources) -> None:
    app = _SettingsApp()
    with mock.patch.object(LilbeeAppHost, "start_rebuild") as rebuild:
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _loaded(app, pilot, "Default")
            _pick(screen, "Research papers")
            await _dialog(app, pilot, ApplyProfileDialog)
            await _press(pilot, app.screen.query_one("#apply-reindex", ConfirmPill))
            await _loaded(app, pilot, "Research papers")
    rebuild.assert_called_once_with()


async def test_plain_apply_starts_no_rebuild(sources) -> None:
    app = _SettingsApp()
    with mock.patch.object(LilbeeAppHost, "start_rebuild") as rebuild:
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _loaded(app, pilot, "Default")
            _pick(screen, "Research papers")
            await _dialog(app, pilot, ApplyProfileDialog)
            await _press(pilot, app.screen.query_one("#apply-apply", ConfirmPill))
            await _loaded(app, pilot, "Research papers")
    rebuild.assert_not_called()
    assert cfg.table_extraction is True


async def test_a_switch_with_nothing_to_change_says_so(sources) -> None:
    _write_global("same", "[values]\n")
    app = _SettingsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _loaded(app, pilot, "Default")
        _pick(screen, "same")
        await _dialog(app, pilot, ApplyProfileDialog)
        dialog = app.screen
        assert not dialog.query("#apply-changes")
        assert _text(dialog.query_one("#apply-summary", Static)) == (
            "Nothing changes; lilbee records the profile's name."
        )
        assert [p.id for p in dialog.query("#apply-actions ConfirmPill")] == [
            "apply-apply",
            "apply-cancel",
        ]
        await _press(pilot, dialog.query_one("#apply-apply", ConfirmPill))
        await _loaded(app, pilot, "same")
    sources.store.get_sources.assert_not_called()


_BIG = """[profile]
name = "big-profile"
description = "A profile tuned for the ten-thousand file corpus."
authors = [{ name = "Jane Doe", github = "janedoe" }]
tested_on = "10,000 mixed documents"

[values]
chunk_size = 900
chunk_overlap = 50
max_chunks_per_file = 1500
top_k = 20
max_distance = 0.5
enable_ocr = true
entity_extraction = true
semantic_chunking = true
table_extraction = true
layout_detection = true
min_relevance_score = 0.3
"""


async def test_apply_dialog_scrolls_to_fit_a_big_profile_at_80x24(sources) -> None:
    _write_global("big-profile", _BIG)
    _set_yours("min_relevance_score = 0.1\n")
    app = _SettingsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _loaded(app, pilot, "Default")
        _pick(screen, "big-profile")
        await _dialog(app, pilot, ApplyProfileDialog)
        dialog = app.screen
        scroll = dialog.query_one("#apply-scroll")
        # content overflows the visible viewport: this is what makes scrolling necessary
        assert scroll.virtual_size.height > scroll.size.height
        screen_region = app.screen.region
        for pill in dialog.query("#apply-actions ConfirmPill"):
            assert _inside(pill.region, screen_region), pill.id
        scroll.scroll_end(animate=False)
        assert await _until(pilot, lambda: scroll.scroll_offset.y > 0)
        await pilot.pause()
        for widget_id in ("apply-kept", "apply-untouched", "apply-summary"):
            widget = dialog.query_one(f"#{widget_id}", Static)
            assert _inside(widget.region, screen_region), widget_id
        assert _text(dialog.query_one("#apply-kept", Static)).startswith("min_relevance_score")
        assert len(_table_rows(dialog, "#apply-changes")) == 10
        for pill in dialog.query("#apply-actions ConfirmPill"):
            assert _inside(pill.region, screen_region), pill.id


async def test_a_change_that_needs_no_reindex_shows_no_summary(sources) -> None:
    _write_global("fewer", "[values]\nmax_chunks_per_file = 10\n")
    app = _SettingsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _loaded(app, pilot, "Default")
        _pick(screen, "fewer")
        await _dialog(app, pilot, ApplyProfileDialog)
        dialog = app.screen
        assert _table_rows(dialog, "#apply-changes") == [
            ["max_chunks_per_file", "3000", "10", "new files only"]
        ]
        assert not dialog.query("#apply-summary")
        assert not dialog.query("#apply-kept")
        await _press(pilot, dialog.query_one("#apply-cancel", ConfirmPill))
        assert await _until(pilot, lambda: app.screen is screen)
    assert cfg.max_chunks_per_file == 3000


async def test_a_failed_write_on_apply_toasts_and_changes_nothing(sources) -> None:
    app = _SettingsApp()
    with mock.patch("lilbee.core.settings.write_profile_table", side_effect=OSError("disk full")):
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _loaded(app, pilot, "Default")
            _pick(screen, "Research papers")
            await _dialog(app, pilot, ApplyProfileDialog)
            await _press(pilot, app.screen.query_one("#apply-apply", ConfirmPill))
            assert await _until(
                pilot, lambda: any("disk full" in n.message for n in app._notifications)
            )
            assert await _until(pilot, lambda: app.screen is screen)
    assert cfg.table_extraction is False
    assert profiles.active(ProfileStore()).name == "Default"


async def test_a_failed_diff_reloads_the_tab_and_reverts_the_select() -> None:
    fragile = _write_global("fragile", "[values]\nchunk_size = 900\n")
    app = _SettingsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _loaded(app, pilot, "Default")
        await app.workers.wait_for_complete()
        fragile.write_text("not valid toml [[[", encoding="utf-8")
        select = screen.query_one("#profile-select", Select)
        _pick(screen, "fragile")
        assert await _until(pilot, lambda: any("fragile" in n.message for n in app._notifications))
        await app.workers.wait_for_complete()
        assert app.screen is screen
        assert await _until(pilot, lambda: select.value == "Default")
    assert profiles.active(ProfileStore()).name == "Default"


async def test_a_failed_diff_toasts_an_os_error_and_reverts_the_select(sources) -> None:
    sources.store.get_sources.side_effect = OSError("disk full")
    app = _SettingsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _loaded(app, pilot, "Default")
        await app.workers.wait_for_complete()
        select = screen.query_one("#profile-select", Select)
        _pick(screen, "Research papers")
        assert await _until(
            pilot, lambda: any("disk full" in n.message for n in app._notifications)
        )
        await app.workers.wait_for_complete()
        assert app.screen is screen
        assert await _until(pilot, lambda: select.value == "Default")
    assert profiles.active(ProfileStore()).name == "Default"


async def test_save_as_refuses_a_builtin_name_then_saves_to_this_project() -> None:
    _on_legal_with_three_changes()
    seen: list[str] = []
    app = _SettingsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _loaded(app, pilot, "legal-discovery")
        app.settings_changed_signal.subscribe(app, lambda change: seen.append(change[0]))
        await _press(pilot, screen.query_one("#profile-save_as", ConfirmPill))
        await _dialog(app, pilot, SaveProfileDialog)
        dialog = app.screen
        error = dialog.query_one("#save-error", Static)
        assert _text(error) == "Type a name for the profile."
        await pilot.press(*"Research papers")
        assert await _until(
            pilot, lambda: _text(error) == "A built-in profile already has this name."
        )
        assert dialog.query_one("#save-save", ConfirmPill).has_class("-disabled")
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen is dialog
        dialog.query_one("#save-name", Input).value = "My filings"
        dialog.query_one("#save-folder", Select).value = ProfileFolder.PROJECT
        assert await _until(pilot, lambda: _text(error) == "")
        screen_region = app.screen.region
        assert _inside(dialog.query_one("#save-body").region, screen_region)
        for pill in dialog.query("#save-actions ConfirmPill"):
            assert _inside(pill.region, screen_region), pill.id
        await _press(pilot, dialog.query_one("#save-save", ConfirmPill))
        await _loaded(app, pilot, "My filings")
        assert await _until(pilot, lambda: not screen.query_one("#profile-changes").display)
    saved = cfg.data_root / PROFILES_DIRNAME / "my-filings.toml"
    assert tomllib.loads(saved.read_text(encoding="utf-8"))["values"]["chunk_size"] == 384
    assert profiles.your_changes() == ()
    assert set(seen) == {"chunk_size", "layout_detection", "max_chunks_per_file"}


async def test_save_as_refuses_a_reserved_route_or_device_name() -> None:
    reason = "This name already names a lilbee page or a Windows device."
    app = _SettingsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _loaded(app, pilot, "Default")
        await _press(pilot, screen.query_one("#profile-save_as", ConfirmPill))
        await _dialog(app, pilot, SaveProfileDialog)
        dialog = app.screen
        error = dialog.query_one("#save-error", Static)
        await pilot.press(*"active")
        assert await _until(pilot, lambda: _text(error) == reason)
        assert dialog.query_one("#save-save", ConfirmPill).has_class("-disabled")
        dialog.query_one("#save-name", Input).value = "CON"
        assert await _until(pilot, lambda: _text(error) == reason)
        assert dialog.query_one("#save-save", ConfirmPill).has_class("-disabled")


async def test_save_as_takes_a_name_on_enter_and_cancels_on_escape() -> None:
    app = _SettingsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _loaded(app, pilot, "Default")
        await _press(pilot, screen.query_one("#profile-save_as", ConfirmPill))
        await _dialog(app, pilot, SaveProfileDialog)
        await pilot.press("escape")
        assert await _until(pilot, lambda: app.screen is screen)
        await _press(pilot, screen.query_one("#profile-save_as", ConfirmPill))
        await _dialog(app, pilot, SaveProfileDialog)
        await _press(pilot, app.screen.query_one("#save-cancel", ConfirmPill))
        assert await _until(pilot, lambda: app.screen is screen)
        await _press(pilot, screen.query_one("#profile-save_as", ConfirmPill))
        await _dialog(app, pilot, SaveProfileDialog)
        await pilot.press(*"Mine", "enter")
        await _loaded(app, pilot, "Mine")
    assert (_global_dir() / "mine.toml").is_file()


async def test_save_as_offers_no_project_folder_on_the_global_root() -> None:
    cfg.data_root = default_data_dir()
    app = _SettingsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _loaded(app, pilot, "Default")
        await _press(pilot, screen.query_one("#profile-save_as", ConfirmPill))
        await _dialog(app, pilot, SaveProfileDialog)
        assert not app.screen.query("#save-folder")
        await pilot.press(*"Mine", "enter")
        await _loaded(app, pilot, "Mine")
    assert (_global_dir() / "mine.toml").is_file()


async def test_update_writes_your_values_into_the_profile_file() -> None:
    _on_legal_with_three_changes()
    seen: list[str] = []
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _loaded(app, pilot, "legal-discovery")
        app.settings_changed_signal.subscribe(app, lambda change: seen.append(change[0]))
        await _press(pilot, screen.query_one("#profile-update", ConfirmPill))
        assert await _until(pilot, lambda: not screen.query_one("#profile-changes").display)
    written = tomllib.loads((_global_dir() / "legal-discovery.toml").read_text(encoding="utf-8"))
    assert written["values"]["chunk_size"] == 384
    assert set(seen) >= {"chunk_size", "layout_detection", "max_chunks_per_file"}


async def test_discard_empties_the_table_and_publishes_the_keys() -> None:
    _on_legal_with_three_changes()
    seen: list[str] = []
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _loaded(app, pilot, "legal-discovery")
        app.settings_changed_signal.subscribe(app, lambda change: seen.append(change[0]))
        await _press(pilot, screen.query_one("#profile-discard", ConfirmPill))
        assert await _until(pilot, lambda: not screen.query_one("#profile-changes").display)
        count = screen.query_one("#profile-line-count", Static)
        assert await _until(pilot, lambda: _text(count) == "0 values set by you")
    assert cfg.chunk_size == 512
    assert set(seen) == {"chunk_size", "layout_detection", "max_chunks_per_file"}


@pytest.mark.parametrize(("pill_id", "rebuilds"), [("confirm-yes", 1), ("confirm-no", 0)])
async def test_discarding_a_reindex_setting_offers_a_rebuild(pill_id: str, rebuilds: int) -> None:
    _on_legal_with_three_changes()
    app = _SettingsApp()
    with mock.patch.object(LilbeeAppHost, "start_rebuild") as rebuild:
        async with app.run_test(size=(80, 24)) as pilot:
            screen = await _loaded(app, pilot, "legal-discovery")
            await _press(pilot, screen.query_one("#profile-discard", ConfirmPill))
            dialog = await _dialog(app, pilot, ConfirmDialog)
            assert _text(dialog.query_one("#confirm-title", Static)) == (
                msg.CMD_REBUILD_CONFIRM_TITLE
            )
            assert await _until(pilot, lambda: not screen.query_one("#profile-changes").display)
            await _press(pilot, dialog.query_one(f"#{pill_id}", ConfirmPill))
            assert await _until(pilot, lambda: app.screen is screen)
    assert rebuild.call_count == rebuilds
    assert cfg.chunk_size == 512


async def test_discarding_settings_that_need_no_rebuild_offers_none() -> None:
    _write_global("legal-discovery", _LEGAL)
    profiles.apply(ProfileStore(), "legal-discovery")
    _set_yours("max_chunks_per_file = 2000\n")
    app = _SettingsApp()
    with mock.patch.object(LilbeeAppHost, "start_rebuild") as rebuild:
        async with app.run_test(size=(80, 24)) as pilot:
            screen = await _loaded(app, pilot, "legal-discovery")
            await _press(pilot, screen.query_one("#profile-discard", ConfirmPill))
            assert await _until(pilot, lambda: not screen.query_one("#profile-changes").display)
            assert app.screen is screen
    rebuild.assert_not_called()


async def test_discard_toasts_the_warning_it_leaves_behind() -> None:
    _write_global("ocr-off", "[values]\nenable_ocr = false\n")
    profiles.apply(ProfileStore(), "ocr-off")
    vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
    _set_yours(f'enable_ocr = true\nvision_model = "{vision_model}"\n')
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _loaded(app, pilot, "ocr-off")
        with mock.patch.object(app, "notify") as notify:
            await _press(pilot, screen.query_one("#profile-discard", ConfirmPill))
            assert await _until(pilot, lambda: not screen.query_one("#profile-changes").display)
    warnings = [c for c in notify.call_args_list if c.kwargs.get("severity") == "warning"]
    assert len(warnings) == 1
    assert "enable_ocr" in warnings[0].args[0]
    assert cfg.enable_ocr is False


async def test_update_toasts_the_warning_it_leaves_behind() -> None:
    _write_global("ocr-off", "[values]\nenable_ocr = false\n")
    profiles.apply(ProfileStore(), "ocr-off")
    vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
    _set_yours(f'vision_model = "{vision_model}"\n')
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _loaded(app, pilot, "ocr-off")
        with mock.patch.object(app, "notify") as notify:
            await _press(pilot, screen.query_one("#profile-update", ConfirmPill))
            assert await _until(pilot, lambda: not screen.query_one("#profile-changes").display)
    warnings = [c for c in notify.call_args_list if c.kwargs.get("severity") == "warning"]
    assert len(warnings) == 1
    assert "enable_ocr" in warnings[0].args[0]
    assert cfg.enable_ocr is False


async def test_save_as_toasts_the_warning_it_leaves_behind() -> None:
    _write_global("ocr-off", "[values]\nenable_ocr = false\n")
    profiles.apply(ProfileStore(), "ocr-off")
    vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
    _set_yours(f'vision_model = "{vision_model}"\n')
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _loaded(app, pilot, "ocr-off")
        with mock.patch.object(app, "notify") as notify:
            await _press(pilot, screen.query_one("#profile-save_as", ConfirmPill))
            await _dialog(app, pilot, SaveProfileDialog)
            await pilot.press(*"Mine", "enter")
            await _loaded(app, pilot, "Mine")
    warnings = [c for c in notify.call_args_list if c.kwargs.get("severity") == "warning"]
    assert len(warnings) == 1
    assert "enable_ocr" in warnings[0].args[0]
    assert cfg.enable_ocr is False


async def test_a_refused_operation_toasts_its_reason() -> None:
    _on_legal_with_three_changes()
    app = _SettingsApp()
    with mock.patch.object(profiles, "discard", side_effect=ValueError("nothing to discard")):
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _loaded(app, pilot, "legal-discovery")
            await _press(pilot, screen.query_one("#profile-discard", ConfirmPill))
            assert await _until(
                pilot, lambda: any(n.message == "nothing to discard" for n in app._notifications)
            )
    assert cfg.chunk_size == 384


async def test_a_changed_file_shows_the_note_the_pill_and_reapply(sources) -> None:
    _on_legal_with_three_changes()
    _write_global("legal-discovery", _LEGAL.replace("chunk_size = 512", "chunk_size = 600"))
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _loaded(app, pilot, "legal-discovery")
        note = screen.query_one("#profile-status-note", Static)
        assert _text(note) == profiles.status_note(profiles.ProfileStatus.CHANGED)
        assert _text(screen.query_one("#profile-line-status", Static)) == "file changed"
        reapply = screen.query_one("#profile-reapply", ConfirmPill)
        assert reapply.display and _text(reapply) == "Re-apply legal-discovery"
        assert not screen.query_one("#profile-update", ConfirmPill).display
        await _press(pilot, reapply)
        await _dialog(app, pilot, ApplyProfileDialog)
        await _press(pilot, app.screen.query_one("#apply-apply", ConfirmPill))
        assert await _until(pilot, lambda: not note.display)
    assert profiles.active(ProfileStore()).status is profiles.ProfileStatus.CURRENT


async def test_a_broken_file_shows_the_note_and_the_pill() -> None:
    _on_legal_with_three_changes()
    _write_global("legal-discovery", "not toml [")
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _loaded(app, pilot, "legal-discovery")
        assert _text(screen.query_one("#profile-status-note", Static)) == profiles.status_note(
            profiles.ProfileStatus.BROKEN
        )
        assert _text(screen.query_one("#profile-line-status", Static)) == "file broken"
        select = screen.query_one("#profile-select", Select)
        options = [value for _label, value in select._options if value != Select.NULL]
        assert options[0] == "legal-discovery"
        assert "Default" in options


async def test_a_file_dropped_in_the_global_folder_shows_on_returning_to_the_tab() -> None:
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _loaded(app, pilot, "Default")
        await app.workers.wait_for_complete()
        await pilot.pause()
        select = screen.query_one("#profile-select", Select)
        assert "dropped" not in [value for _label, value in select._options]
        _write_global("dropped", "[values]\nchunk_size = 900\n")
        tabs = screen.query_one("#settings-tabs", TabbedContent)
        tabs.active = "settings-tab-models"
        await _until(pilot, lambda: tabs.active == "settings-tab-models")
        tabs.active = PROFILE_PANE_ID
        assert await _until(
            pilot, lambda: "dropped" in [value for _label, value in select._options]
        )


async def test_returning_to_settings_rescans_the_profiles() -> None:
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _loaded(app, pilot, "Default")
        profiles.apply(ProfileStore(), "Scanned archive")
        screen.post_message(events.ScreenResume())
        await _loaded(app, pilot, "Scanned archive")


async def test_a_value_you_set_elsewhere_reaches_an_unfocused_editor_but_not_a_focused_one() -> (
    None
):
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _loaded(app, pilot, "Default")
        screen.populate_all_panes()
        assert await _until(pilot, lambda: bool(screen.query("#ed-chunk_overlap")))
        chunk = screen.query_one("#ed-chunk_size", Input)
        overlap = screen.query_one("#ed-chunk_overlap", Input)
        overlap.focus()
        await _until(pilot, lambda: overlap.has_focus)
        overlap.value = "7"
        cfg.chunk_size = 900
        cfg.chunk_overlap = 50
        app.publish_settings(["chunk_size", "chunk_overlap"])
        assert await _until(pilot, lambda: chunk.value == "900")
        assert overlap.value == "7"


async def test_a_value_you_set_elsewhere_reaches_the_line_and_the_table() -> None:
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _loaded(app, pilot, "Default")
        await app.workers.wait_for_complete()
        app.set_setting("chunk_size", 900)
        count = screen.query_one("#profile-line-count", Static)
        assert await _until(pilot, lambda: _text(count) == "1 value set by you")
        assert _table_rows(screen, "#profile-changes") == [
            ["chunk_size", "900", "512", " reindex "]
        ]


async def test_a_profile_tab_mounted_after_the_state_loaded_still_fills() -> None:
    _on_legal_with_three_changes()
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _loaded(app, pilot, "legal-discovery")
        await app.workers.wait_for_complete()
        body = screen.query_one(f"#{PROFILE_PANE_ID}-body")
        await body.remove_children()
        await body.mount(ProfileTab())
        tab = screen.query_one(ProfileTab)
        select = tab.query_one("#profile-select", Select)
        assert await _until(pilot, lambda: select.value == "legal-discovery")


def test_profile_dialogs_bind_no_printable_single_key() -> None:
    for dialog in (ApplyProfileDialog, SaveProfileDialog, ProfilePathDialog):
        keys = [binding.key for binding in dialog.BINDINGS]
        assert keys, dialog.__name__
        assert all(len(key) > 1 for key in keys), (dialog.__name__, keys)


@pytest.mark.parametrize(
    ("value", "text"),
    [(True, "on"), (False, "off"), (None, "none"), (["eng", "deu"], "eng, deu"), ([], "none")],
)
def test_value_text_reads_like_a_setting(value, text) -> None:
    assert value_text(value) == text


async def test_picking_the_blank_entry_shows_the_active_profile_again() -> None:
    _on_legal_with_three_changes()
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _loaded(app, pilot, "legal-discovery")
        await app.workers.wait_for_complete()
        await pilot.pause()
        select = screen.query_one("#profile-select", Select)
        select.value = Select.NULL
        assert await _until(pilot, lambda: select.value == "legal-discovery")
        assert app.screen is screen


@pytest.fixture
def chat_app():
    """The real app with a ready chat screen and no model scan."""
    with (
        mock.patch("lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready", return_value=True),
        mock.patch.object(ModelBar, "_scan_models"),
    ):
        yield LilbeeApp()


async def test_slash_profile_alone_opens_the_profile_tab(chat_app) -> None:
    async with chat_app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(chat_app, pilot)
        chat.run_command("/profile")
        assert await _until(pilot, lambda: isinstance(chat_app.screen, SettingsScreen))
        screen = await _loaded(chat_app, pilot, "Default")
        tabs = screen.query_one("#settings-tabs", TabbedContent)
        assert tabs.active == PROFILE_PANE_ID
        select = screen.query_one("#profile-select", Select)
        assert await _until(pilot, lambda: select.has_focus)


async def test_slash_profile_from_another_tab_returns_to_the_profile_tab(chat_app) -> None:
    async with chat_app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(chat_app, pilot)
        chat_app.switch_view("Settings")
        assert await _until(pilot, lambda: isinstance(chat_app.screen, SettingsScreen))
        screen = await _loaded(chat_app, pilot, "Default")
        tabs = screen.query_one("#settings-tabs", TabbedContent)
        tabs.active = "settings-tab-ingest"
        await _until(pilot, lambda: tabs.active == "settings-tab-ingest")
        chat_app.switch_view("Chat")
        # switch_view drops a request while an earlier switch is still settling
        assert await _until(pilot, lambda: chat_app.screen is chat and not chat_app._switching)
        chat.run_command("/profile")
        assert await _until(pilot, lambda: tabs.active == PROFILE_PANE_ID)


async def test_slash_profile_with_a_name_asks_then_applies(chat_app, sources) -> None:
    async with chat_app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(chat_app, pilot)
        chat.run_command("/profile research papers")
        await _dialog(chat_app, pilot, ApplyProfileDialog)
        await _press(pilot, chat_app.screen.query_one("#apply-apply", ConfirmPill))
        assert await _until(pilot, lambda: chat_app.screen is chat)
        assert await _until(pilot, lambda: cfg.table_extraction is True)
    assert profiles.active(ProfileStore()).name == "Research papers"


async def test_slash_profile_with_an_unknown_name_toasts(chat_app) -> None:
    async with chat_app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(chat_app, pilot)
        chat.run_command("/profile nope")
        assert await _until(
            pilot,
            lambda: any("No profile named 'nope'" in n.message for n in chat_app._notifications),
        )
        assert chat_app.screen is chat


def test_profile_completion_lists_usable_names_only() -> None:
    _write_global("mine", "[values]\nchunk_size = 900\n")
    _write_global("busted", "not toml [")
    options = get_completions("/profile ")
    assert "Research papers" in options
    assert "mine" in options
    assert "busted" not in options
    assert get_completions("/profile Scan") == ["Scanned archive"]


async def test_profile_suggestion_completes_a_name_inline() -> None:
    assert await SlashSuggester(use_cache=False).get_suggestion("/profile Sca") == (
        "/profile Scanned archive"
    )


async def test_palette_choose_profile_opens_the_profile_tab(chat_app) -> None:
    async with chat_app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(chat_app, pilot)
        provider = LilbeeCommandProvider(chat, match_style=None)
        entry = next(c for c in provider._get_commands() if c[0] == "Choose profile")
        entry[2]()
        assert await _until(pilot, lambda: isinstance(chat_app.screen, SettingsScreen))
        screen = await _loaded(chat_app, pilot, "Default")
        select = screen.query_one("#profile-select", Select)
        assert await _until(pilot, lambda: select.has_focus)


def test_profile_is_listed_in_the_help_catalog() -> None:
    names = [name for group in CATALOG_GROUPS for name in group.members]
    assert "/profile" in names
    assert get_command("/profile").args_hint == "[name]"
