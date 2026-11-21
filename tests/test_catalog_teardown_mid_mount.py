"""Tearing the app down while the catalog is still mounting must not raise.

Textual's ``Widget.mount``/``mount_all`` no-op once the app is exiting or the
widget is being pruned, so a ``TabbedContent`` caught in that window mounts its
tab strip with no ``Tab`` children. A strip built with ``initial=`` then hits
``Tabs._on_mount``, which assigns the initial tab unconditionally and raises
``ValueError: No Tab with id ...``. Under parallel pytest the teardown lands in
that window often enough to redden CI at random.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import pytest
from textual.app import ComposeResult
from textual.pilot import Pilot
from textual.widget import Widget
from textual.widgets import Footer, TabbedContent, TabPane

from lilbee.cli.tui.screens.catalog import CatalogScreen, GridSection
from lilbee.cli.tui.screens.catalog_utils import TAB_CHAT, TAB_DISCOVER, LocalCatalogRow
from lilbee.cli.tui.widgets.discover_rails import DiscoverRails
from lilbee.cli.tui.widgets.model_grid import ModelGrid
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import LilbeeAppHost, pump_until


class _CatalogApp(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        yield Footer()


@pytest.mark.parametrize("hops", range(9))
async def test_teardown_during_catalog_mount_is_clean(hops: int) -> None:
    """Shutting down `hops` event-loop hops into the mount must not raise."""
    app = _CatalogApp()
    with patch.object(CatalogScreen, "_fetch_remote_models"):
        async with app.run_test():
            app.push_screen(CatalogScreen())
            for _ in range(hops):
                await asyncio.sleep(0)


async def test_catalog_still_lands_on_chat_tab() -> None:
    """Dropping ``initial=`` must not cost the Chat landing tab.

    Without ``initial=`` the strip starts on the first pane and Chat is
    activated a refresh later, so this waits the activation out rather than
    assuming one pause covers it; a bare pause passes locally and fails on a
    loaded runner.
    """
    app = _CatalogApp()
    with patch.object(CatalogScreen, "_fetch_remote_models"):
        async with app.run_test(size=(120, 40)) as pilot:
            await app.push_screen(CatalogScreen())
            tabs = app.screen.query_one("#catalog-tabs", TabbedContent)
            await pump_until(pilot, lambda: tabs.active == TAB_CHAT)
            assert tabs.active == TAB_CHAT


def _row(name: str) -> LocalCatalogRow:
    return LocalCatalogRow(
        name=name,
        task="chat",
        params="--",
        size="--",
        quant="--",
        downloads="--",
        featured=False,
        installed=False,
        sort_downloads=0,
        sort_size=0.0,
        ref=f"{name}-ref",
        backend="native",
    )


_CATALOG_FETCHES = (
    "_fetch_remote_models",
    "_fetch_frontier_models",
    "_fetch_installed_names",
    "_fetch_families",
    "_ensure_task_initial_fetch",
)


def _sections(*headings: str) -> list[GridSection]:
    return [GridSection(heading=h, rows=[_row(f"{h}-row")]) for h in headings]


def _shown_grid_names(screen: CatalogScreen) -> list[str | None]:
    return [g.name for g in screen._grid_container.displayed_children if isinstance(g, ModelGrid)]


@contextlib.asynccontextmanager
async def _chat_grid() -> AsyncIterator[tuple[CatalogScreen, Pilot[None]]]:
    """The catalog on its Chat tab with every fetch stubbed, so only the test remounts."""
    app = _CatalogApp()
    with contextlib.ExitStack() as stack:
        for name in _CATALOG_FETCHES:
            stack.enter_context(patch.object(CatalogScreen, name))
        async with app.run_test(size=(120, 40)) as pilot:
            screen = CatalogScreen()
            await app.push_screen(screen)
            tabs = app.screen.query_one("#catalog-tabs", TabbedContent)
            await pump_until(pilot, lambda: tabs.active == TAB_CHAT)
            # The screen's own first paint of the tab must land before a test remounts.
            painted = await wait_until(pilot, lambda: TAB_CHAT in screen._grid_cache_keys)
            assert painted, "the Chat tab never painted"
            yield screen, pilot


async def test_a_remount_mounts_every_section_before_it_returns() -> None:
    """Every section is in the pane when the remount returns; nothing is left to run later."""
    async with _chat_grid() as (screen, _pilot):
        screen._remount_grid_sections(_sections("First", "Second"), hf_count=0)
        assert _shown_grid_names(screen) == ["First", "Second"]
        hints = [w for w in screen._grid_container.displayed_children if w.has_class("scroll-hint")]
        assert len(hints) == 1, f"the remount left {len(hints)} scroll hints below the sections"


async def test_back_to_back_remounts_leave_focus_on_a_shown_grid() -> None:
    """Two remounts in one tick end with one copy of each section and a live grid focused."""
    async with _chat_grid() as (screen, pilot):
        screen._remount_grid_sections(_sections("Picks", "Chat"), hf_count=0)
        screen._remount_grid_sections(_sections("Picks", "Installed", "Chat"), hf_count=0)
        settled = await wait_until(
            pilot, lambda: all(w.is_attached for w in screen.query(ModelGrid))
        )
        assert settled
        focused = screen.focused
        assert isinstance(focused, ModelGrid), focused
        assert focused.is_attached, "focus is on a grid that was removed"
        assert focused.name == "Picks"
        assert _shown_grid_names(screen) == ["Picks", "Installed", "Chat"]


async def test_a_refresh_during_a_removal_counts_only_the_shown_grids() -> None:
    """Grids still being removed do not count as sections an in-place update can reuse."""
    async with _chat_grid() as (screen, _pilot):
        screen._remount_grid_sections(_sections("Picks"), hf_count=0)
        screen._remount_grid_sections(_sections("Picks", "Chat"), hf_count=0)
        sections = _sections("Picks", "Installed", "Chat")
        assert not screen._extend_grid_sections_in_place(sections, hf_count=0)


async def test_a_remount_returns_the_cursor_to_the_focused_section() -> None:
    """The section that held focus keeps it across a remount, at the same card."""
    async with _chat_grid() as (screen, _pilot):
        two_rows = GridSection(heading="Chat", rows=[_row("a"), _row("b")])
        screen._remount_grid_sections([*_sections("Picks"), two_rows], hf_count=0)
        chat = next(g for g in screen.query(ModelGrid) if g.name == "Chat")
        chat.highlighted = 1
        screen.set_focus(chat)
        screen._remount_grid_sections([*_sections("Picks", "Installed"), two_rows], hf_count=0)
        focused = screen.focused
        assert isinstance(focused, ModelGrid)
        assert focused is not chat
        assert (focused.name, focused.highlighted) == ("Chat", 1)


_ORIG_ACTIVATE_INITIAL_TAB = CatalogScreen._activate_initial_tab
_ORIG_PANE_FOCUSED = TabbedContent._on_tab_pane_focused


def _pane_id_of(widget: Widget | None) -> str | None:
    node = widget
    while node is not None:
        if isinstance(node, TabPane):
            return node.id
        node = node.parent  # type: ignore[assignment]
    return None


@contextlib.asynccontextmanager
async def _one_pressed_in_the_first_paint() -> AsyncIterator[
    tuple[CatalogScreen, Pilot[None], list[tuple[str | None, str]]]
]:
    """Chat is painted and 1 is pressed before the initial tab settles.

    The initial-tab step is held until after the key. Textual's pane-focus handler
    (``TabbedContent._on_tab_pane_focused``) is held too and delivered after the
    step, the order a loaded host showed. Yields the screen, the pilot and every
    ``(pane of the widget, pane shown)`` pair of a focus the screen set.
    """
    held: list[CatalogScreen] = []
    held_focus: list[tuple[TabbedContent, TabPane.Focused]] = []
    app = _CatalogApp()
    with contextlib.ExitStack() as stack:
        for name in _CATALOG_FETCHES:
            stack.enter_context(patch.object(CatalogScreen, name))
        stack.enter_context(
            patch.object(
                CatalogScreen,
                "_build_family_rows",
                lambda self, search: [_row("Pick"), _row("Plain")],
            )
        )
        stack.enter_context(
            patch.object(CatalogScreen, "_activate_initial_tab", lambda self: held.append(self))
        )
        stack.enter_context(
            patch.object(
                TabbedContent,
                "_on_tab_pane_focused",
                lambda self, event: held_focus.append((self, event)),
            )
        )
        async with app.run_test(size=(120, 40)) as pilot:
            screen = CatalogScreen()
            await app.push_screen(screen)
            assert await wait_until(pilot, lambda: bool(held)), "the initial-tab step never ran"
            tabs = screen.query_one("#catalog-tabs", TabbedContent)
            assert tabs.active == TAB_DISCOVER
            focused_panes: list[tuple[str | None, str]] = []
            orig_set_focus = screen.set_focus

            def record_focus(widget: Widget | None, *args: Any, **kwargs: Any) -> None:
                if widget is not None:
                    focused_panes.append((_pane_id_of(widget), tabs.active))
                orig_set_focus(widget, *args, **kwargs)

            screen.set_focus = record_focus  # type: ignore[method-assign]
            screen._refresh_view()  # a worker landing paints the Chat pane
            chat = screen.query_one(f"#{TAB_CHAT}", TabPane)
            assert list(chat.query(ModelGrid)), "the paint mounted no Chat grid"
            await pilot.press("1")
            await pilot.pause()
            _ORIG_ACTIVATE_INITIAL_TAB(screen)
            await pilot.pause()
            for owner, event in held_focus:
                _ORIG_PANE_FOCUSED(owner, event)
            held_focus.clear()
            await wait_until(pilot, lambda: tabs.active != TAB_DISCOVER, max_pauses=30)
            yield screen, pilot, focused_panes


async def test_a_paint_of_a_hidden_pane_moves_no_focus_into_it() -> None:
    """A remount of the Chat pane while Discover is shown sets no focus and keeps Discover."""
    async with _one_pressed_in_the_first_paint() as (screen, _pilot, focused_panes):
        tabs = screen.query_one("#catalog-tabs", TabbedContent)
        assert tabs.active == TAB_DISCOVER, f"the catalog switched to {tabs.active} after 1"
        hidden = [pair for pair in focused_panes if pair[0] != pair[1]]
        assert hidden == [], "focus was set inside a pane that was not shown"


async def test_discover_arrows_stay_on_the_rails_after_a_hidden_paint() -> None:
    """Arrow keys on Discover move only among the rail grids after a hidden Chat paint."""
    async with _one_pressed_in_the_first_paint() as (screen, pilot, _focused_panes):
        rails = screen.query_one("#discover-rails", DiscoverRails)
        rails.set_rails(
            for_you=[_row("Alpha"), _row("Beta")],
            collection=[_row("Gamma")],
            fresh=[_row("Delta"), _row("Epsilon")],
        )
        await pilot.pause()
        rail_grids = list(rails.query(ModelGrid))
        assert rail_grids, "the rails mounted no grid"
        for _ in range(8):
            await pilot.press("down")
            await pilot.pause()
            assert screen.focused in rail_grids, f"arrows left the rails: {screen.focused!r}"
