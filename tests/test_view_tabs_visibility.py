"""Per-screen ViewTabs visibility. Guards against the page indicator
being mounted in compose() but invisible at render time."""

from __future__ import annotations

from unittest import mock

import pytest

from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.widgets.status_bar import ViewTabs
from lilbee.config import cfg


@pytest.fixture(autouse=True)
def _isolated_cfg(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_dir = tmp_path / "data"
    cfg.data_root = tmp_path
    cfg.documents_dir = tmp_path / "documents"
    cfg.models_dir = tmp_path / "models"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.chat_model = "test-chat-model.gguf"
    cfg.embedding_model = "test-embed-model"
    cfg.subprocess_embed = False
    cfg.wiki = True
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.documents_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.lancedb_dir.mkdir(parents=True, exist_ok=True)
    yield
    for field_name in type(snapshot).model_fields:
        setattr(cfg, field_name, getattr(snapshot, field_name))


@pytest.fixture(autouse=True)
def _mock_services():
    from lilbee.services import set_services

    mock_svc = mock.MagicMock()
    mock_svc.provider.list_models.return_value = []
    mock_svc.searcher._embedder.embedding_available.return_value = True
    set_services(mock_svc)
    try:
        yield mock_svc
    finally:
        set_services(None)


@pytest.fixture(autouse=True)
def _patch_chat_setup():
    with (
        mock.patch(
            "lilbee.cli.tui.screens.chat.ChatScreen._needs_setup",
            return_value=False,
        ),
        mock.patch(
            "lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready",
            return_value=True,
        ),
    ):
        yield


@pytest.mark.parametrize("view_name", ["Catalog", "Status", "Settings", "Tasks", "Wiki"])
async def test_view_tabs_visible_on_screen(view_name: str) -> None:
    """ViewTabs must be mounted AND occupy a non-zero region on every main screen.

    A passing test on every screen confirms the page indicator renders
    correctly app-wide; a failure on any one would expose a real layout
    regression where compose() yields the widget but CSS or sibling
    layout hides it.
    """
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        # Chat starts in insert mode; escape to normal so [/] navigation works.
        await pilot.press("escape")
        await pilot.pause()
        app.switch_view(view_name)
        await pilot.pause()
        await pilot.pause()  # second tick: switch_screen completes via call_later

        tabs = app.screen.query_one(ViewTabs)
        assert tabs.is_mounted, f"ViewTabs not mounted on {view_name}"
        assert tabs.region.height > 0, f"ViewTabs has zero height on {view_name}"
        assert str(tabs.styles.display) != "none", f"ViewTabs display:none on {view_name}"


async def test_view_tabs_visible_on_chat() -> None:
    """ViewTabs is also visible on the chat screen (the screen the friend
    cited as the only one where it works)."""
    from lilbee.cli.tui.screens.chat import ChatScreen

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, ChatScreen)
        tabs = app.screen.query_one(ViewTabs)
        assert tabs.is_mounted
        assert tabs.region.height > 0
        assert str(tabs.styles.display) != "none"


async def test_view_tabs_docks_at_top_not_bottom() -> None:
    """Airline-style header: ViewTabs sits at the top of every screen
    (above the screen content), and the Footer keybindings stay at the
    bottom. Asserting via region.y instead of compose-tree position so
    the test catches CSS regressions too."""
    from textual.widgets import Footer

    from lilbee.cli.tui.screens.chat import ChatScreen

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, ChatScreen)
        tabs = app.screen.query_one(ViewTabs)
        footer = app.screen.query_one(Footer)
        assert tabs.region.y < footer.region.y, (
            f"ViewTabs (y={tabs.region.y}) must sit above Footer "
            f"(y={footer.region.y}); airline-style header."
        )
        # ViewTabs is in the top half of a 40-row terminal.
        assert tabs.region.y < 20


async def test_view_tabs_active_view_tracks_screen_changes() -> None:
    """Regression: ViewTabs.on_mount captured app.active_view once, but
    switch_view's _finish callback updates app.active_view AFTER the new
    screen mounts. The pill highlight lagged the actual screen by one
    navigation step. Fix pushes the new active_view into the live ViewTabs
    after _finish runs."""
    from lilbee.cli.tui.screens.chat import ChatScreen

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, ChatScreen)
        assert app.screen.query_one(ViewTabs).active_view == "Chat"

        for view in ("Catalog", "Status", "Settings", "Tasks"):
            app.switch_view(view)
            await pilot.pause()
            await pilot.pause()  # one extra tick for call_later(_finish)
            tabs = app.screen.query_one(ViewTabs)
            assert tabs.active_view == view, (
                f"After switch to {view}, ViewTabs.active_view was "
                f"{tabs.active_view!r}, expected {view!r}"
            )
