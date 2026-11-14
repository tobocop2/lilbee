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
from unittest.mock import patch

import pytest
from textual.app import ComposeResult
from textual.pilot import Pilot
from textual.widgets import Footer, TabbedContent

from lilbee.cli.tui.screens.catalog import CatalogScreen, GridSection
from lilbee.cli.tui.screens.catalog_utils import TAB_CHAT, LocalCatalogRow
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
