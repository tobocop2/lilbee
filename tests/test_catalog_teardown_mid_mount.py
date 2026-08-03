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
from unittest.mock import patch

import pytest
from textual.app import ComposeResult
from textual.widgets import Footer, TabbedContent

from lilbee.cli.tui.screens.catalog import CatalogScreen, GridSection
from lilbee.cli.tui.screens.catalog_utils import TAB_CHAT, LocalCatalogRow
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


async def test_deferred_section_mount_after_grid_teardown_is_clean() -> None:
    """The staged tail mount must bail once the tab's grid container is gone.

    ``_remount_grid_sections`` mounts the first section, then defers the rest
    via ``call_after_refresh``. When the pane is unmounted before that callback
    runs (screen teardown, tab remount), the callback used to re-query
    '#grid-chat' and crash with NoMatches. Capture the deferred callable and
    run it after removing the container so the ordering is forced, not timed.
    """
    app = _CatalogApp()
    with contextlib.ExitStack() as stack:
        for name in _CATALOG_FETCHES:
            stack.enter_context(patch.object(CatalogScreen, name))
        async with app.run_test(size=(120, 40)) as pilot:
            screen = CatalogScreen()
            await app.push_screen(screen)
            tabs = app.screen.query_one("#catalog-tabs", TabbedContent)
            await pump_until(pilot, lambda: tabs.active == TAB_CHAT)
            sections = [
                GridSection(heading="First", rows=[_row("row-one")]),
                GridSection(heading="Second", rows=[_row("row-two")]),
            ]
            container = screen._grid_container
            deferred: list = []
            with patch.object(
                screen,
                "call_after_refresh",
                side_effect=lambda cb, *a, **kw: deferred.append((cb, a, kw)),
            ):
                screen._remount_grid_sections(sections, hf_count=0)
            assert deferred, "expected the tail mount to be deferred"
            # Unmount the pane before the deferred tail mount runs.
            await container.remove()
            await pilot.pause()
            assert not container.is_running
            callback, args, kwargs = deferred[-1]
            callback(*args, **kwargs)
            assert not screen.query(".section-heading")
