"""The click census: every surface opens, takes a click, and a click acts like Enter."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.geometry import Size
from textual.screen import Screen
from textual.widgets import Static

from lilbee.cli.tui.screens.chat import ChatScreen
from lilbee.cli.tui.screens.model_picker import ModelPickerModal
from lilbee.cli.tui.screens.profile_library import ProfileLibrary
from lilbee.cli.tui.screens.startup_gate import StartupGate
from lilbee.cli.tui.screens.status import StatusScreen
from lilbee.cli.tui.widgets.dialog import Dialog
from lilbee.cli.tui.widgets.model_list import ModelList
from lilbee.cli.tui.widgets.status_bar import ViewTab
from lilbee.core.config import cfg
from lilbee.core.config.enums import ChatMode
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import LilbeeAppHost
from tests._tui_census import (
    SIZE_IDS,
    SIZES,
    NoBuilder,
    SurfaceApp,
    backdrop_click_effects,
    build,
    click_effects,
    discover_surfaces,
    focus_labels,
    is_reached,
    key_effects,
    mouse_close_controls,
    open_surface,
    probe_widget,
    settle,
)
from tests.test_tui_small_fixes import _picker_rows

# A surface test opens the surface afresh for each widget, twice; the default 60 s is too short.
pytestmark = pytest.mark.timeout(600)

SURFACES = discover_surfaces()
SURFACE_IDS = [c.__name__ for c in SURFACES]
MIN_SURFACES = 20
# Widgets whose click is an open question for the owner: the census skips them, and the
# expected failure below names the hold and turns red when a click starts to match Enter.
HELD = {(StatusScreen, "DataTable#docs-table")}
# The gate is a loading screen with nothing to focus; every other surface has a control.
NO_FOCUS = {StartupGate}
# The library closes from the Close link in its hint row, a Static that takes no focus.
CLOSES_BY_LINK = {ProfileLibrary}
DIALOGS = [c for c in SURFACES if issubclass(c, Dialog) and c not in CLOSES_BY_LINK]


def test_the_walk_finds_the_surfaces() -> None:
    assert len(SURFACES) >= MIN_SURFACES


def test_a_new_surface_without_a_builder_fails_the_census() -> None:
    class NeedsArguments(Screen[None]):
        def __init__(self, title: str) -> None:
            super().__init__()
            self.title_text = title

    with pytest.raises(NoBuilder, match="NeedsArguments needs arguments"):
        build(NeedsArguments)


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", SURFACES, ids=SURFACE_IDS)
async def test_every_surface_opens_with_a_focusable_widget(cls: type) -> None:
    app = SurfaceApp(cls)
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await open_surface(pilot, app)
        assert type(screen) is cls
        assert bool(screen.focus_chain) == (cls not in NO_FOCUS)


@pytest.mark.asyncio
async def test_the_profile_library_closes_from_its_close_link() -> None:
    app = SurfaceApp(ProfileLibrary)
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await open_surface(pilot, app)
        hints = screen.query_one("#library-hints")
        await pilot.click(hints, offset=(str(hints.content).rindex("Close"), 0))
        await settle(pilot)
        assert app.screen is not screen


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", DIALOGS, ids=[c.__name__ for c in DIALOGS])
async def test_every_dialog_can_be_closed_with_the_mouse(cls: type) -> None:
    assert await mouse_close_controls(cls, (120, 40)), f"{cls.__name__} closes by key only"


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
@pytest.mark.parametrize("cls", SURFACES, ids=SURFACE_IDS)
async def test_every_focusable_widget_is_reached_and_a_click_does_what_enter_does(
    cls: type, size: tuple[int, int]
) -> None:
    app = SurfaceApp(cls)
    missed: list[str] = []
    async with app.run_test(size=size) as pilot:
        screen = await open_surface(pilot, app)
        count = len(screen.focus_chain)
        for widget in screen.focus_chain:
            widget.scroll_visible(animate=False)
            await pilot.pause()
            if not is_reached(screen, widget):
                missed.append(f"{widget!r} region={widget.region}")
    assert not missed, f"a click does not reach: {missed}"

    key_only: list[str] = []
    for index in range(count):
        by_key = await probe_widget(cls, size, index, "enter")
        if not by_key.owns_enter or by_key.text_entry or (cls, by_key.name) in HELD:
            continue
        by_click = await probe_widget(cls, size, index, "click")
        lost = sorted(by_key.effects - by_click.effects)
        if lost:
            key_only.append(f"{by_key.name}: Enter does {lost}, a click does not")
    assert not key_only, key_only


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="HELD for the owner: Enter selects a docs row, a click gives a header event; "
    "the issue names no row action, so a click on a row stays as it is until the owner rules.",
)
async def test_a_click_on_a_status_docs_row_selects_it_like_enter() -> None:
    names = await focus_labels(StatusScreen, (120, 40))
    index = names.index("DataTable#docs-table")
    by_key = await probe_widget(StatusScreen, (120, 40), index, "enter")
    by_click = await probe_widget(StatusScreen, (120, 40), index, "click")
    assert by_key.effects
    assert by_key.effects <= by_click.effects


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", SURFACES, ids=SURFACE_IDS)
async def test_every_shown_binding_has_a_click_path(cls: type) -> None:
    from textual.widgets import Footer

    size = (120, 40)
    app = SurfaceApp(cls)
    async with app.run_test(size=size) as pilot:
        screen = await open_surface(pilot, app)
        footer = bool(screen.query(Footer))
        shown = sorted(
            key
            for key, active in screen.active_bindings.items()
            if active.binding.show and active.enabled
        )
    if footer or not shown:
        return
    clicks = await backdrop_click_effects(cls, size)
    for index in range(len(await focus_labels(cls, size))):
        clicks |= await click_effects(cls, size, index)
    key_only: list[str] = []
    for key in shown:
        effects = await key_effects(cls, size, key)
        if effects and not effects <= clicks:
            key_only.append(f"{key} does {sorted(effects - clicks)} that no click does")
    assert not key_only, key_only


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SIZES, ids=SIZE_IDS)
async def test_a_click_on_the_chat_mode_pill_sets_the_chat_mode(size: tuple[int, int]) -> None:
    cfg.chat_mode = ChatMode.SEARCH.value
    app = SurfaceApp(ChatScreen)
    async with app.run_test(size=size) as pilot:
        screen = await open_surface(pilot, app)
        pill = screen.query_one("#chat-mode-chat")
        pill.scroll_visible(animate=False)
        await pilot.pause()
        await pilot.click(pill)
        assert await wait_until(pilot, lambda: cfg.chat_mode == ChatMode.CHAT.value)


class _Plain(App[None]):
    def compose(self) -> ComposeResult:
        yield Static("x", id="subject")


@pytest.mark.asyncio
async def test_settle_waits_for_a_worker() -> None:
    done: list[bool] = []

    async def work() -> None:
        await asyncio.sleep(0.2)
        done.append(True)

    app = _Plain()
    async with app.run_test() as pilot:
        app.run_worker(work())
        await settle(pilot)
        assert done == [True]


@pytest.mark.asyncio
async def test_settle_waits_for_an_animation() -> None:
    app = _Plain()
    async with app.run_test() as pilot:
        subject = app.query_one("#subject", Static)
        subject.styles.animate("opacity", 0.0, duration=0.3)
        await settle(pilot)
        assert subject.styles.opacity == 0.0


def test_a_view_tab_is_as_wide_as_its_label_before_it_mounts() -> None:
    tab = ViewTab("Catalog")
    width = tab.get_content_width(Size(80, 1), Size(80, 24))
    assert width == len("  Catalog  ")


@pytest.mark.asyncio
async def test_the_view_tabs_fit_in_80_columns() -> None:
    app = SurfaceApp(ChatScreen)
    async with app.run_test(size=(80, 24)) as pilot:
        screen: Any = await open_surface(pilot, app)
        tabs = [t for t in screen.query(ViewTab) if t.display]
        assert tabs
        assert max(t.region.right for t in tabs) <= 80


def _shown(screen: Any, selector: str) -> list[str | None]:
    return [w.id for w in screen.query(selector) if w.display]


@pytest.mark.asyncio
@pytest.mark.parametrize(("width", "dots"), [(80, False), (99, False), (100, True), (120, True)])
async def test_the_tab_dots_show_from_100_columns_and_hide_below(width: int, dots: bool) -> None:
    cfg.wiki = True
    app = SurfaceApp(ChatScreen)
    async with app.run_test(size=(width, 24)) as pilot:
        screen: Any = await open_surface(pilot, app)
        assert bool(_shown(screen, ".view-tab-sep")) is dots
        if dots:
            assert len(_shown(screen, ".view-tab-sep")) == len(screen.query(ViewTab)) - 1


@pytest.mark.asyncio
async def test_the_tab_dots_follow_a_resize() -> None:
    cfg.wiki = True
    app = SurfaceApp(ChatScreen)
    async with app.run_test(size=(120, 24)) as pilot:
        screen: Any = await open_surface(pilot, app)
        assert _shown(screen, ".view-tab-sep")
        await pilot.resize_terminal(80, 24)
        assert await wait_until(pilot, lambda: not _shown(screen, ".view-tab-sep"))
        await pilot.resize_terminal(120, 24)
        assert await wait_until(pilot, lambda: bool(_shown(screen, ".view-tab-sep")))


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [80, 120])
@pytest.mark.parametrize("wiki", [True, False])
async def test_the_wiki_tab_and_its_dot_follow_the_wiki_setting(width: int, wiki: bool) -> None:
    cfg.wiki = wiki
    app = SurfaceApp(ChatScreen)
    async with app.run_test(size=(width, 24)) as pilot:
        screen: Any = await open_surface(pilot, app)
        assert ("view-tab-wiki" in _shown(screen, ".view-tab")) is wiki
        if width >= 100:
            assert ("view-tab-sep-wiki" in _shown(screen, ".view-tab-sep")) is wiki
            assert _shown(screen, ".view-tab-sep")


class _PickerHost(LilbeeAppHost):
    """The app host with a model picker, holding more rows than the screen can show."""

    def compose(self) -> ComposeResult:
        yield Static("")

    def on_mount(self) -> None:
        self.push_screen(
            ModelPickerModal(scope="chat", options=_picker_rows(40), pick_later=lambda ref: None)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (80, 14), (120, 40)], ids=["80x24", "80x14", "120x40"])
async def test_the_picker_footer_and_cancel_stay_in_view_over_a_long_list(
    size: tuple[int, int],
) -> None:
    app = _PickerHost()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        modal = app.screen
        assert isinstance(modal, ModelPickerModal)
        assert await wait_until(pilot, lambda: modal.query_one("#picker-list", ModelList).row_count)
        await settle(pilot)
        root = modal.query_one("#picker-root")
        footer = modal.query_one("#picker-footer")
        assert is_reached(modal, modal.query_one("#dialog-cancel"))
        assert root.region.contains_region(footer.region)
        assert modal.region.contains_region(root.region)
