"""The Profile tab, its + button, and the shared look of the profile dialogs, in a Textual pilot."""

import re

import pytest
from textual.widgets import Input, OptionList, Select, Static

import tests._suite_data_root  # noqa: F401  # must run before lilbee loads
from lilbee.cli.tui.screens.profile_dialogs import (
    ApplyProfileDialog,
    ProfilePathDialog,
    SaveProfileDialog,
)
from lilbee.cli.tui.screens.profile_library import ProfileLibrary
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog, ConfirmPill
from lilbee.core.profile_files import PROFILES_DIRNAME, ProfileFolder, ProfileStore
from lilbee.core.system import default_data_dir
from tests._profile_fixtures import (  # noqa: F401 -- autouse fixtures, applied by import
    gate_releases_at_once,
    isolated_cfg,
)
from tests.test_tui_profile_tab import (
    _dialog,
    _loaded,
    _press,
    _SettingsApp,
    _text,
    _until,
    _write_global,
)

_PLAIN = '[profile]\nname = "{name}"\n\n[values]\nchunk_size = 400\n'
SIZES = [(120, 36), (80, 24)]
BUILTINS = [
    "Default",
    "Code repository",
    "Notes and markdown",
    "Research papers",
    "Scanned archive",
]


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path / "global-data"))
    assert not str(default_data_dir()).startswith("/Users/tobias/Library")


def _names(app) -> list[str]:
    select = next(s for s in app.screen_stack if s.query("#profile-select")).query_one(
        "#profile-select", Select
    )
    return [str(value) for _label, value in select._options]


async def test_the_dropdown_lists_default_then_builtins_then_user_profiles() -> None:
    _write_global("zeta", _PLAIN.format(name="Zeta"))
    _write_global("aardvark", _PLAIN.format(name="Aardvark"))
    expected = [*BUILTINS, "Aardvark", "Zeta"]
    assert ProfileStore().scan().usable_names() == expected, "the /profile completion list"
    app = _SettingsApp()
    async with app.run_test(size=(120, 36)) as pilot:
        await _loaded(app, pilot, "Default")
        assert _names(app) == expected, "the dropdown"


async def test_the_plus_button_saves_a_new_named_profile() -> None:
    app = _SettingsApp()
    async with app.run_test(size=(120, 36)) as pilot:
        screen = await _loaded(app, pilot, "Default")
        plus = screen.query("#profile-new")
        assert plus, "no + button with id profile-new beside the dropdown"
        assert plus.first().region.y == screen.query_one("#profile-select").region.y
        await _press(pilot, plus.first())
        dialog = await _dialog(app, pilot, SaveProfileDialog)
        name = dialog.query_one("#save-name", Input).value
        assert name, "the dialog offers a free name"
        assert _text(dialog.query_one("#save-error", Static)) == "", "no error on open"
        await _press(pilot, dialog.query_one("#save-save", ConfirmPill))
        await _loaded(app, pilot, name)
    saved = list((default_data_dir() / PROFILES_DIRNAME).glob("*.toml"))
    assert len(saved) == 1, saved
    assert ProfileStore().scan().find(name).folder is ProfileFolder.GLOBAL


@pytest.mark.parametrize("size", SIZES)
async def test_save_as_opens_without_an_error(size) -> None:
    app = _SettingsApp()
    async with app.run_test(size=size) as pilot:
        screen = await _loaded(app, pilot, "Default")
        await _press(pilot, screen.query_one("#profile-save_as", ConfirmPill))
        save = await _dialog(app, pilot, SaveProfileDialog)
        err = save.query_one("#save-error", Static)
        assert _text(err) == "", f"Save as opens with: {_text(err)!r}"


@pytest.mark.parametrize("size", SIZES)
async def test_the_import_dialog_opens_without_an_error(size) -> None:
    app = _SettingsApp()
    async with app.run_test(size=size) as pilot:
        await _loaded(app, pilot, "Default")
        app.push_screen(
            ProfilePathDialog(
                title="Import a profile",
                explain="x",
                action_label="Import",
                folders=(ProfileFolder.GLOBAL, ProfileFolder.PROJECT),
            )
        )
        path = await _dialog(app, pilot, ProfilePathDialog)
        err = path.query_one("#path-error", Static)
        assert _text(err) == "", f"Import opens with: {_text(err)!r}"


@pytest.mark.parametrize("size", SIZES)
async def test_dialog_pills_and_sizes_are_the_same_in_every_dialog(size) -> None:
    app = _SettingsApp()
    async with app.run_test(size=size) as pilot:
        screen = await _loaded(app, pilot, "Default")
        await _press(pilot, screen.query_one("#profile-save_as", ConfirmPill))
        save = await _dialog(app, pilot, SaveProfileDialog)
        for pill in save.query(ConfirmPill):
            assert pill.styles.background.a >= 0.15, f"{pill.id} resting pill is invisible"
        assert save.query_one("#save-title").styles.margin.bottom == 1
        assert save.query_one("#save-explain").styles.margin.bottom == 1
        assert save.query_one("#save-body").region.height <= size[1]
        save.dismiss(None)
        await pilot.pause()
        screen.query_one("#profile-select", Select).value = "Code repository"
        apply = await _dialog(app, pilot, ApplyProfileDialog)
        scroll = apply.query_one("#apply-scroll")
        assert scroll.size.height <= scroll.virtual_size.height, "empty rows inside the Apply body"
        assert apply.query_one("#apply-body").region.height <= size[1]
        assert apply.query_one("#apply-title").styles.margin.bottom == 1


@pytest.mark.parametrize("size", SIZES)
async def test_the_library_is_capped_and_its_hints_take_one_row(size) -> None:
    app = _SettingsApp()
    async with app.run_test(size=size) as pilot:
        screen = await _loaded(app, pilot, "Default")
        await _press(pilot, screen.query_one("#profile-manage", ConfirmPill))
        library = await _dialog(app, pilot, ProfileLibrary)
        body = library.query_one("#library-body")
        if size[0] >= 100:
            assert body.region.width <= 100 and body.region.height <= 28, body.region
        assert body.region.height <= size[1]
        assert library.query_one("#library-hints").region.height == 1, "hints wrap"


@pytest.mark.parametrize("size", SIZES)
async def test_the_profile_tab_matches_the_other_tabs(size) -> None:
    app = _SettingsApp()
    async with app.run_test(size=size) as pilot:
        screen = await _loaded(app, pilot, "Default")
        title = screen.query("#profile-title")
        assert title, "no #profile-title"
        assert "select" in _text(title.first()), "the title has no select pill"
        assert not [w for w in screen.query(".setting-title") if "set by you" in _text(w)]
        assert screen.query_one("#profile-select").parent.has_class("setting-editor-row")
        first = screen.query_one("#profile-save_as", ConfirmPill)
        # _loaded returns when the select shows the name, which can come before the action
        # pills get their labels. Until then the empty Update pill is shown and pushes this
        # one right.
        assert await _until(pilot, lambda: bool(_text(first))), "the action pills got no labels"
        await pilot.pause()
        assert first.styles.background.a >= 0.15, "resting pill is invisible"
        assert first.region.x == title.first().region.x + 1, "pill edge is off the heading column"


async def test_the_plus_button_offers_the_next_free_copy_name() -> None:
    _write_global("default copy", _PLAIN.format(name="Default copy"))
    app = _SettingsApp()
    async with app.run_test(size=(120, 36)) as pilot:
        screen = await _loaded(app, pilot, "Default")
        await _press(pilot, screen.query_one("#profile-new", ConfirmPill))
        dialog = await _dialog(app, pilot, SaveProfileDialog)
        assert dialog.query_one("#save-name", Input).value == "Default copy 2"


async def test_save_as_offers_no_name() -> None:
    app = _SettingsApp()
    async with app.run_test(size=(120, 36)) as pilot:
        screen = await _loaded(app, pilot, "Default")
        await _press(pilot, screen.query_one("#profile-save_as", ConfirmPill))
        dialog = await _dialog(app, pilot, SaveProfileDialog)
        assert dialog.query_one("#save-name", Input).value == ""


async def test_the_error_shows_once_the_name_is_edited() -> None:
    app = _SettingsApp()
    async with app.run_test(size=(120, 36)) as pilot:
        screen = await _loaded(app, pilot, "Default")
        await _press(pilot, screen.query_one("#profile-save_as", ConfirmPill))
        dialog = await _dialog(app, pilot, SaveProfileDialog)
        error = dialog.query_one("#save-error", Static)
        assert not error.display
        await pilot.press("a", "backspace")
        assert await _until(pilot, lambda: error.display and _text(error) != "")
        assert dialog.query_one("#save-save", ConfirmPill).has_class("-disabled")


async def test_the_user_pill_shows_only_while_there_are_changes() -> None:
    app = _SettingsApp()
    async with app.run_test(size=(120, 36)) as pilot:
        screen = await _loaded(app, pilot, "Default")
        title = screen.query_one("#profile-changes-title", Static)
        assert "you" not in _text(title).split("Your changes", 1)[1]


async def test_library_rows_pad_the_name_so_the_folder_tags_align() -> None:
    _write_global("a", _PLAIN.format(name="A"))
    app = _SettingsApp()
    async with app.run_test(size=(120, 36)) as pilot:
        screen = await _loaded(app, pilot, "Default")
        await _press(pilot, screen.query_one("#profile-manage", ConfirmPill))
        library = await _dialog(app, pilot, ProfileLibrary)
        listing = library.query_one("#library-list", OptionList)
        assert await _until(pilot, lambda: listing.option_count > 0)
        rows = [str(option.prompt) for option in listing.options]
        columns = {re.search(r"(built-in|global)", row).start() for row in rows}
        assert len(rows) > 1 and len(columns) == 1, rows


@pytest.mark.parametrize("size", SIZES)
async def test_a_long_profile_name_is_cut_and_every_library_row_stays_on_one_line(size) -> None:
    long_name = "L" * 40
    _write_global("long", _PLAIN.format(name=long_name))
    app = _SettingsApp()
    async with app.run_test(size=size) as pilot:
        screen = await _loaded(app, pilot, "Default")
        await _press(pilot, screen.query_one("#profile-manage", ConfirmPill))
        library = await _dialog(app, pilot, ProfileLibrary)
        listing = library.query_one("#library-list", OptionList)
        assert await _until(pilot, lambda: listing.option_count > 1)
        await pilot.pause()
        rows = [option.prompt for option in listing.options]
        row_width = listing.scrollable_content_region.width - 2
        # Only the 120-column list is narrower than the longest row; at 80 columns it fits.
        assert any("…" in str(row) for row in rows) == (len(long_name) + 2 + 8 > row_width)
        assert all(row.cell_length <= row_width for row in rows), (row_width, rows)
        assert listing.virtual_size.height == listing.option_count, "a row wraps"
        columns = {re.search(r"(built-in|global)", str(row)).start() for row in rows}
        assert len(columns) == 1, rows


@pytest.mark.parametrize("size", SIZES)
async def test_the_confirm_dialog_pills_keep_the_shared_look(size) -> None:
    app = _SettingsApp()
    async with app.run_test(size=size) as pilot:
        await _loaded(app, pilot, "Default")
        app.push_screen(ConfirmDialog("Title", "Message"))
        dialog = await _dialog(app, pilot, ConfirmDialog)
        yes = dialog.query_one("#confirm-yes", ConfirmPill)
        no = dialog.query_one("#confirm-no", ConfirmPill)
        assert yes.has_class("-primary") and not no.has_class("-primary")
        assert no.styles.background.a >= 0.15 and yes.styles.background.a >= 0.15
        assert yes.styles.background != no.styles.background


def _catalog_named(*names: str):
    from pathlib import Path

    from lilbee.core.profile_files import ProfileCatalog, ProfileEntry

    return ProfileCatalog(
        entries=tuple(
            ProfileEntry(
                name=name,
                folder=ProfileFolder.GLOBAL,
                path=Path(f"/nowhere/{index}.toml"),
                file=None,
                error=None,
                shadowed_by=None,
            )
            for index, name in enumerate(names)
        )
    )


@pytest.mark.parametrize("length", [10, 34, 35, 36, 40])
def test_the_offered_copy_name_fits_the_name_limit(length: int) -> None:
    from lilbee.cli.tui.screens.profile_dialogs import free_copy_name
    from lilbee.core.profile_files import MAX_NAME_LENGTH, name_problem

    name = "a" * length
    offered = free_copy_name(name, ProfileFolder.GLOBAL, _catalog_named(name))
    assert offered.endswith(" copy")
    assert len(offered) <= MAX_NAME_LENGTH
    assert name_problem(offered, ProfileFolder.GLOBAL, _catalog_named(name)) is None


def test_the_offered_copy_name_drops_a_trailing_space_from_the_cut_base() -> None:
    from lilbee.cli.tui.screens.profile_dialogs import free_copy_name

    name = "a" * 34 + " b" * 3
    offered = free_copy_name(name, ProfileFolder.GLOBAL, _catalog_named(name))
    assert "  " not in offered
    assert offered == "a" * 34 + " copy"


def test_the_offered_copy_name_numbers_a_long_name_that_is_taken() -> None:
    from lilbee.cli.tui.screens.profile_dialogs import free_copy_name

    name = "a" * 40
    taken = _catalog_named(name, "a" * 35 + " copy")
    assert free_copy_name(name, ProfileFolder.GLOBAL, taken) == "a" * 33 + " copy 2"


def test_the_offered_copy_name_ends_when_every_short_name_is_taken() -> None:
    from lilbee.cli.tui.screens.profile_dialogs import COPY_NAME_TRIES, free_copy_name
    from lilbee.core.profile_files import NameProblem, name_problem

    name = "a" * 40
    names = ["a" * 35 + " copy"] + [
        f"{'a' * (35 - len(f' {n}'))} copy {n}" for n in range(2, COPY_NAME_TRIES + 1)
    ]
    taken = _catalog_named(name, *names)
    offered = free_copy_name(name, ProfileFolder.GLOBAL, taken)
    assert name_problem(offered, ProfileFolder.GLOBAL, taken) is NameProblem.TAKEN


async def test_the_plus_button_with_a_long_active_name_offers_a_name_the_dialog_accepts() -> None:
    from lilbee.app import profiles as app_profiles

    long_name = "b" * 38
    _write_global("long", _PLAIN.format(name=long_name))
    app_profiles.apply(ProfileStore(), long_name)
    app = _SettingsApp()
    async with app.run_test(size=(120, 36)) as pilot:
        screen = await _loaded(app, pilot, long_name)
        await _press(pilot, screen.query_one("#profile-new", ConfirmPill))
        dialog = await _dialog(app, pilot, SaveProfileDialog)
        assert dialog.query_one("#save-name", Input).value == "b" * 35 + " copy"
        assert not dialog.query_one("#save-save", ConfirmPill).has_class("-disabled")


@pytest.mark.parametrize("action", ["Import", "Export"])
@pytest.mark.parametrize("size", SIZES)
async def test_a_path_dialog_shows_its_error_when_the_user_clears_the_path(action, size) -> None:
    app = _SettingsApp()
    async with app.run_test(size=size) as pilot:
        await _loaded(app, pilot, "Default")
        app.push_screen(
            ProfilePathDialog(
                title=f"{action} a profile",
                explain="x",
                action_label=action,
                folders=(ProfileFolder.GLOBAL,),
            )
        )
        dialog = await _dialog(app, pilot, ProfilePathDialog)
        error = dialog.query_one("#path-error", Static)
        ok = dialog.query_one("#path-ok", ConfirmPill)
        assert _text(error) == "" and ok.has_class("-disabled")
        await pilot.press("a")
        assert await _until(pilot, lambda: not ok.has_class("-disabled"))
        assert _text(error) == ""
        await pilot.press("backspace")
        assert await _until(pilot, lambda: error.display and _text(error) != "")
        assert ok.has_class("-disabled")


@pytest.mark.parametrize("size", SIZES)
async def test_the_dialog_pills_keep_one_row_and_show_focus_and_disabled(size) -> None:
    from lilbee.core.profile_files import ProfileCatalog

    app = _SettingsApp()
    async with app.run_test(size=size) as pilot:
        await _loaded(app, pilot, "Default")
        app.push_screen(ConfirmDialog("Title", "Message"))
        confirm = await _dialog(app, pilot, ConfirmDialog)
        pills = list(confirm.query(ConfirmPill))
        assert pills
        for pill in pills:
            assert pill.size.height == 1
            assert pill.styles.background.a >= 0.15, f"{pill.id} resting pill is invisible"
        focused = app.focused
        assert isinstance(focused, ConfirmPill)
        assert (
            focused.styles.background.hex
            != next(p for p in pills if p is not focused).styles.background.hex
        )
        confirm.dismiss(False)
        await pilot.pause()
        app.push_screen(
            SaveProfileDialog(
                title="t",
                explain="e",
                catalog=ProfileCatalog(entries=()),
                folders=(ProfileFolder.GLOBAL,),
                initial="",
            )
        )
        save = await _dialog(app, pilot, SaveProfileDialog)
        disabled = save.query_one("#save-save", ConfirmPill)
        assert disabled.has_class("-disabled")
        assert disabled.styles.color.a < save.query_one("#save-cancel", ConfirmPill).styles.color.a


@pytest.mark.parametrize("size", SIZES)
async def test_the_description_has_a_gap_below_the_dropdown_row(size) -> None:
    app = _SettingsApp()
    async with app.run_test(size=size) as pilot:
        screen = await _loaded(app, pilot, "Default")
        row = screen.query_one("#profile-select").parent
        description = screen.query_one("#profile-description", Static)
        assert description.display
        assert description.region.y >= row.region.y + row.region.height + 1


async def test_a_failed_profile_scan_tells_the_user_why(monkeypatch) -> None:
    def _fail() -> None:
        raise PermissionError("denied: /locked/profiles")

    monkeypatch.setattr("lilbee.cli.tui.screens.settings.load_snapshot", _fail)
    app = _SettingsApp()
    async with app.run_test(size=(120, 36)) as pilot:
        assert await _until(pilot, lambda: bool(app.screen.query("#profile-description")))
        line = app.screen.query_one("#profile-description", Static)
        assert await _until(
            pilot, lambda: line.display and "denied: /locked/profiles" in _text(line)
        )
        assert _text(line).startswith("Could not read the profiles")
