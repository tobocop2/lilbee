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
from unittest.mock import patch

import pytest
from textual.app import ComposeResult
from textual.widgets import Footer, TabbedContent

from lilbee.cli.tui.screens.catalog import CatalogScreen
from lilbee.cli.tui.screens.catalog_utils import TAB_CHAT
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
