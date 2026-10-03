"""The Manage profiles library: the list, its detail, and the file operations on each profile."""

from __future__ import annotations

import tomllib
from pathlib import Path
from unittest import mock

import pytest
from textual.pilot import Pilot
from textual.widgets import Input, OptionList, Select, Static

from lilbee.app import profiles
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.screens.profile_dialogs import (
    ApplyProfileDialog,
    ProfilePathDialog,
    SaveProfileDialog,
)
from lilbee.cli.tui.screens.profile_library import ProfileLibrary, entry_label
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog, ConfirmPill
from lilbee.core.config import cfg
from lilbee.core.profile_files import PROFILES_DIRNAME, ProfileFolder, ProfileStore
from tests._lilbee_app_test_host import LilbeeAppHost
from tests._profile_fixtures import (
    gate_releases_at_once,  # noqa: F401 -- autouse fixture, applied by import
    isolated_cfg,  # noqa: F401 -- autouse fixture, applied by import
    sources_totaling,
)
from tests.test_tui_profile_tab import (
    _LEGAL,
    _dialog,
    _global_dir,
    _inside,
    _loaded,
    _press,
    _settings,
    _SettingsApp,
    _table_rows,
    _text,
    _until,
    _write_global,
)


@pytest.fixture
def sources():
    with sources_totaling(1) as services:
        yield services


_PLAIN = "[values]\nchunk_size = 700\n"
_BROKEN = "[values]\ntop_kk = 3\n"


def _project_dir() -> Path:
    return cfg.data_root / PROFILES_DIRNAME


def _write_project(stem: str, text: str) -> Path:
    _project_dir().mkdir(parents=True, exist_ok=True)
    path = _project_dir() / f"{stem}.toml"
    path.write_text(text, encoding="utf-8")
    return path


async def _open(app: LilbeeAppHost, pilot: Pilot, active: str = "Default") -> ProfileLibrary:
    """Open the library from the Profile tab's Manage profiles pill."""
    screen = await _loaded(app, pilot, active)
    await _press(pilot, screen.query_one("#profile-manage", ConfirmPill))
    library = await _dialog(app, pilot, ProfileLibrary)
    assert isinstance(library, ProfileLibrary)
    listing = library.query_one("#library-list", OptionList)
    assert await _until(pilot, lambda: listing.option_count > 0)
    return library


def _rows(library: ProfileLibrary) -> list[str]:
    listing = library.query_one("#library-list", OptionList)
    return [str(option.prompt) for option in listing.options]


async def _highlight(pilot: Pilot, library: ProfileLibrary, row: str) -> None:
    """Highlight the list row that reads *row*, and wait until the detail shows it."""
    listing = library.query_one("#library-list", OptionList)
    listing.highlighted = _rows(library).index(row)
    name_text, tags = row.split("  ")
    tag = tags.split(",")[0]
    folder = next(f for f, text in msg.PROFILE_FOLDER_TAG.items() if text == tag)
    name = library.query_one("#library-name", Static)
    where = library.query_one("#library-folder", Static)
    assert await _until(
        pilot,
        lambda: _text(name) == name_text and _text(where) == msg.PROFILE_FOLDER_TEXT[folder],
    )


async def _toasted(app: LilbeeAppHost, pilot: Pilot, text: str) -> None:
    assert await _until(pilot, lambda: any(text in n.message for n in app._notifications)), text


def _select_names(app: LilbeeAppHost) -> list[str]:
    select = _settings(app).query_one("#profile-select", Select)
    return [str(value) for _label, value in select._options]


async def test_the_list_tags_each_folder_and_shows_shadowed_and_broken_entries() -> None:
    _write_project("court-filings", _PLAIN)
    _write_global("court-filings", _PLAIN)
    _write_global("broken-profile", _BROKEN)
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        library = await _open(app, pilot)
        rows = _rows(library)
        assert rows[0] == "court-filings  project"
        assert rows[1].startswith("broken-profile  global, cannot load: ")
        assert "top_kk" in rows[1]
        assert rows[2] == "court-filings  global, shadowed"
        assert "Default  built-in" in rows
        await _highlight(pilot, library, "court-filings  global, shadowed")
        problem = library.query_one("#library-problem", Static)
        assert _text(problem) == "Hidden by the project profile with the same name."
        assert library.query_one("#library-detail").display


async def test_the_detail_shows_credit_and_what_applying_changes() -> None:
    _write_global("legal-discovery", _LEGAL)
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        library = await _open(app, pilot)
        await _highlight(pilot, library, "legal-discovery  global")
        assert _text(library.query_one("#library-credit", Static)) == (
            "by Jane Doe (@janedoe). Tested on: 4,000 county filings"
        )
        assert _text(library.query_one("#library-description", Static)) == (
            "Scanned legal exhibits."
        )
        table = library.query_one("#library-changes")
        assert await _until(pilot, lambda: table.display)
        assert _table_rows(library, "#library-changes") == [
            ["layout_detection", "off", "on", " reindex "]
        ]
        await _highlight(pilot, library, "Default  built-in")
        title = library.query_one("#library-changes-title", Static)
        assert await _until(pilot, lambda: _text(title) == "Applying it changes nothing.")
        assert not table.display


async def test_credit_renders_tested_on_through_the_shared_formatter() -> None:
    _write_global("legal-discovery", _LEGAL)
    app = _SettingsApp()
    with mock.patch.object(profiles, "tested_on_line", return_value="SENTINEL-TESTED-ON"):
        async with app.run_test(size=(120, 40)) as pilot:
            library = await _open(app, pilot)
            await _highlight(pilot, library, "legal-discovery  global")
            credit = library.query_one("#library-credit", Static)
            assert await _until(pilot, lambda: "SENTINEL-TESTED-ON" in _text(credit))


async def test_a_diff_that_fails_shows_no_changes() -> None:
    _write_global("legal-discovery", _LEGAL)
    app = _SettingsApp()
    with mock.patch.object(profiles, "diff", side_effect=ValueError("gone")) as diff:
        async with app.run_test(size=(120, 40)) as pilot:
            library = await _open(app, pilot)
            await _highlight(pilot, library, "legal-discovery  global")
            assert await _until(pilot, lambda: diff.call_count > 0)
            await app.workers.wait_for_complete()
            assert not library.query_one("#library-changes-title").display
            assert not library.query_one("#library-changes").display


async def test_a_broken_entry_shows_its_reason_and_no_changes() -> None:
    _write_global("broken-profile", _BROKEN)
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        library = await _open(app, pilot)
        broken = next(row for row in _rows(library) if row.startswith("broken-profile"))
        await _highlight(pilot, library, broken)
        assert "top_kk" in _text(library.query_one("#library-problem", Static))
        await app.workers.wait_for_complete()
        assert not library.query_one("#library-changes-title").display


async def test_at_80x24_the_library_shows_the_list_only_and_fits() -> None:
    app = _SettingsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        library = await _open(app, pilot)
        assert library.has_class("-narrow")
        assert not library.query_one("#library-detail").display
        screen_region = library.region
        for widget_id in ("#library-body", "#library-list", "#library-hints"):
            assert _inside(library.query_one(widget_id).region, screen_region), widget_id
        hints = _text(library.query_one("#library-hints", Static))
        for label in ("Apply", "Duplicate", "Rename", "Delete", "Export", "Import", "Close"):
            assert label in hints
        assert "Share" not in hints
        assert "s" not in {binding.key for binding in ProfileLibrary.BINDINGS}


async def test_narrow_layout_shows_the_selected_profiles_credit() -> None:
    _write_global("legal-discovery", _LEGAL)
    app = _SettingsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        library = await _open(app, pilot)
        await _highlight(pilot, library, "legal-discovery  global")
        narrow_credit = library.query_one("#library-narrow-credit", Static)
        assert await _until(
            pilot,
            lambda: (
                _text(narrow_credit) == "by Jane Doe (@janedoe). Tested on: 4,000 county filings"
            ),
        )
        assert narrow_credit.display
        assert _inside(narrow_credit.region, library.region)
        await _highlight(pilot, library, "Default  built-in")
        assert await _until(pilot, lambda: not narrow_credit.display)


async def test_enter_opens_the_apply_dialog_and_the_tab_follows(sources) -> None:
    app = _SettingsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        library = await _open(app, pilot)
        rows = _rows(library)
        for _ in range(rows.index("Research papers  built-in")):
            await pilot.press("down")
        await pilot.press("enter")
        await _dialog(app, pilot, ApplyProfileDialog)
        await _press(pilot, app.screen.query_one("#apply-apply", ConfirmPill))
        assert await _until(pilot, lambda: app.screen is library)
        select = _settings(app).query_one("#profile-select", Select)
        assert await _until(pilot, lambda: select.value == "Research papers")
    assert profiles.active(ProfileStore()).name == "Research papers"


async def test_duplicate_by_keyboard_at_80x24_refuses_a_builtin_name_then_writes() -> None:
    _write_global("legal-discovery", _LEGAL)
    app = _SettingsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        library = await _open(app, pilot)
        await _highlight(pilot, library, "legal-discovery  global")
        await pilot.press("d")
        dialog = await _dialog(app, pilot, SaveProfileDialog)
        name = dialog.query_one("#save-name", Input)
        assert name.value == "legal-discovery copy"
        name.value = ""
        await pilot.press(*"Default")
        error = dialog.query_one("#save-error", Static)
        assert await _until(
            pilot, lambda: _text(error) == "A built-in profile already has this name."
        )
        await pilot.press("enter")
        assert await _until(pilot, lambda: app.screen is dialog)
        for pill in dialog.query("#save-actions ConfirmPill"):
            assert _inside(pill.region, dialog.region), pill.id
        name.value = ""
        await pilot.press(*"my filings", "enter")
        assert await _until(pilot, lambda: "my filings  global" in _rows(library))
        assert app.screen is library
        assert await _until(pilot, lambda: "my filings" in _select_names(app))
    copied = tomllib.loads((_global_dir() / "my-filings.toml").read_text(encoding="utf-8"))
    assert copied["profile"]["tested_on"] == "4,000 county filings"
    assert copied["values"]["chunk_size"] == 512


async def test_duplicate_can_save_to_this_project() -> None:
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        library = await _open(app, pilot)
        await _highlight(pilot, library, "Research papers  built-in")
        await pilot.press("d")
        dialog = await _dialog(app, pilot, SaveProfileDialog)
        dialog.query_one("#save-folder", Select).value = ProfileFolder.PROJECT
        await _press(pilot, dialog.query_one("#save-save", ConfirmPill))
        assert await _until(pilot, lambda: "Research papers copy  project" in _rows(library))
    assert (_project_dir() / "research-papers-copy.toml").is_file()


async def test_rename_moves_the_file_and_the_project_follows() -> None:
    _write_global("legal-discovery", _LEGAL)
    profiles.apply(ProfileStore(), "legal-discovery")
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        library = await _open(app, pilot, "legal-discovery")
        await _highlight(pilot, library, "legal-discovery  global")
        await pilot.press("r")
        dialog = await _dialog(app, pilot, SaveProfileDialog)
        assert not dialog.query("#save-folder")
        name = dialog.query_one("#save-name", Input)
        assert name.value == "legal-discovery"
        assert _text(dialog.query_one("#save-error", Static)) == ""
        name.value = "court"
        await pilot.press("enter")
        assert await _until(pilot, lambda: "court  global" in _rows(library))
        line = _settings(app).query_one("#profile-line-name", Static)
        assert await _until(pilot, lambda: _text(line) == "court")
        await _toasted(app, pilot, "Renamed legal-discovery to court.")
    assert not (_global_dir() / "legal-discovery.toml").exists()
    assert (_global_dir() / "court.toml").is_file()
    assert profiles.active(ProfileStore()).name == "court"


@pytest.mark.parametrize("key", ["r", "x"])
async def test_a_builtin_profile_refuses_rename_and_delete_with_the_core_message(key) -> None:
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        library = await _open(app, pilot)
        await _highlight(pilot, library, "Scanned archive  built-in")
        await pilot.press(key)
        await _toasted(app, pilot, "Scanned archive ships with lilbee and cannot be changed")
        await app.workers.wait_for_complete()
        assert app.screen is library


async def test_delete_asks_first_then_removes_the_file() -> None:
    path = _write_global("old-one", _PLAIN)
    app = _SettingsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        library = await _open(app, pilot)
        await _highlight(pilot, library, "old-one  global")
        await pilot.press("x")
        await _dialog(app, pilot, ConfirmDialog)
        await pilot.press("n")
        assert await _until(pilot, lambda: app.screen is library)
        assert path.is_file()
        await pilot.press("x")
        await _dialog(app, pilot, ConfirmDialog)
        await pilot.press("y")
        assert await _until(pilot, lambda: "old-one  global" not in _rows(library))
        assert await _until(pilot, lambda: "old-one" not in _select_names(app))
    assert not path.exists()


@pytest.mark.parametrize("key", ["enter", "d", "r", "x", "e"])
async def test_an_entry_its_name_does_not_pick_is_refused(key) -> None:
    _write_project("court-filings", _PLAIN)
    hidden = _write_global("court-filings", _PLAIN)
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        library = await _open(app, pilot)
        await _highlight(pilot, library, "court-filings  global, shadowed")
        await pilot.press(key)
        await _toasted(app, pilot, "The name court-filings picks another file")
        await app.workers.wait_for_complete()
        assert app.screen is library
    assert hidden.is_file()
    assert (_project_dir() / "court-filings.toml").is_file()


async def test_export_writes_a_file_at_the_path_given(tmp_path) -> None:
    _write_global("legal-discovery", _LEGAL)
    out = tmp_path / "out"
    out.mkdir()
    app = _SettingsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        library = await _open(app, pilot)
        await _highlight(pilot, library, "legal-discovery  global")
        await pilot.press("e")
        dialog = await _dialog(app, pilot, ProfilePathDialog)
        path = dialog.query_one("#path-input", Input)
        assert path.value == str(Path.cwd())
        assert not dialog.query("#path-folder")
        path.value = str(out)
        await pilot.press("enter")
        await _toasted(app, pilot, "Exported legal-discovery to")
    exported = tomllib.loads((out / "legal-discovery.toml").read_text(encoding="utf-8"))
    assert exported["values"]["chunk_size"] == 512


async def test_export_to_an_existing_file_toasts_and_keeps_it(tmp_path) -> None:
    taken = tmp_path / "taken.toml"
    taken.write_text("keep", encoding="utf-8")
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        library = await _open(app, pilot)
        await _highlight(pilot, library, "Default  built-in")
        await pilot.press("e")
        dialog = await _dialog(app, pilot, ProfilePathDialog)
        dialog.query_one("#path-input", Input).value = str(taken)
        await _press(pilot, dialog.query_one("#path-ok", ConfirmPill))
        await _toasted(app, pilot, "already exists")
    assert taken.read_text(encoding="utf-8") == "keep"


async def test_import_copies_a_file_into_the_folder_chosen(tmp_path) -> None:
    source = tmp_path / "handout.toml"
    source.write_text(_PLAIN, encoding="utf-8")
    app = _SettingsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        library = await _open(app, pilot)
        await pilot.press("i")
        dialog = await _dialog(app, pilot, ProfilePathDialog)
        path = dialog.query_one("#path-input", Input)
        assert path.value == ""
        ok = dialog.query_one("#path-ok", ConfirmPill)
        assert ok.has_class("-disabled")
        assert _text(dialog.query_one("#path-error", Static)) == "Type a path."
        await pilot.press("enter")
        assert await _until(pilot, lambda: app.screen is dialog)
        dialog.query_one("#path-folder", Select).value = ProfileFolder.PROJECT
        path.value = str(source)
        assert await _until(pilot, lambda: not ok.has_class("-disabled"))
        await _press(pilot, ok)
        assert await _until(pilot, lambda: "handout  project" in _rows(library))
        assert await _until(pilot, lambda: "handout" in _select_names(app))
    assert (_project_dir() / "handout.toml").is_file()


async def test_import_of_a_missing_file_toasts_and_writes_nothing(tmp_path) -> None:
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await _open(app, pilot)
        await pilot.press("i")
        dialog = await _dialog(app, pilot, ProfilePathDialog)
        dialog.query_one("#path-input", Input).value = str(tmp_path / "nope.toml")
        await pilot.press("enter")
        await _toasted(app, pilot, "Cannot read the file")
    assert not _global_dir().exists() or not list(_global_dir().iterdir())


async def test_escape_closes_the_library_and_the_path_dialog_cancels() -> None:
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        library = await _open(app, pilot)
        await pilot.press("i")
        dialog = await _dialog(app, pilot, ProfilePathDialog)
        await pilot.press("escape")
        assert await _until(pilot, lambda: app.screen is library)
        await pilot.press("i")
        dialog = await _dialog(app, pilot, ProfilePathDialog)
        await _press(pilot, dialog.query_one("#path-cancel", ConfirmPill))
        assert await _until(pilot, lambda: app.screen is library)
        await pilot.press("escape")
        assert await _until(pilot, lambda: app.screen is _settings(app))
    assert not _global_dir().exists() or not list(_global_dir().iterdir())


async def test_typing_action_letters_into_the_rename_input_does_not_leak() -> None:
    _write_global("legal-discovery", _LEGAL)
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        library = await _open(app, pilot)
        await _highlight(pilot, library, "legal-discovery  global")
        await pilot.press("r")
        dialog = await _dialog(app, pilot, SaveProfileDialog)
        name = dialog.query_one("#save-name", Input)
        name.value = ""
        await pilot.press(*"dxei")
        assert name.value == "dxei"
        assert app.screen is dialog
    assert (_global_dir() / "legal-discovery.toml").is_file()


async def test_typing_action_letters_into_the_import_path_input_does_not_leak() -> None:
    app = _SettingsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await _open(app, pilot)
        await pilot.press("i")
        dialog = await _dialog(app, pilot, ProfilePathDialog)
        path = dialog.query_one("#path-input", Input)
        await pilot.press(*"drex")
        assert path.value == "drex"
        assert app.screen is dialog
    assert not _global_dir().exists() or not list(_global_dir().iterdir())


def test_entry_label_names_the_folder_and_each_problem() -> None:
    _write_global("broken-profile", _BROKEN)
    entry = next(e for e in ProfileStore().scan().entries if e.name == "broken-profile")
    assert str(entry_label(entry)).startswith("broken-profile  global, cannot load: ")
