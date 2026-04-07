"""Navigation flow tests — verify keyboard-driven TUI interactions.

Every test uses pilot.press() to simulate actual keystrokes, never
action_* methods directly. This catches key resolution, focus routing,
and event bubbling bugs that unit tests miss.
"""

from __future__ import annotations

from unittest import mock

import pytest
from textual.widgets import Footer, Input

from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.screens.catalog import CatalogScreen
from lilbee.cli.tui.screens.chat import ChatScreen
from lilbee.cli.tui.screens.settings import SettingsScreen
from lilbee.cli.tui.screens.status import StatusScreen
from lilbee.cli.tui.screens.task_center import TaskCenter
from lilbee.cli.tui.widgets.grid_select import GridSelect
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
    cfg.vision_model = ""
    cfg.subprocess_embed = False
    cfg.wiki = False
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.documents_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    yield
    for field_name in type(snapshot).model_fields:
        setattr(cfg, field_name, getattr(snapshot, field_name))


@pytest.fixture(autouse=True)
def _mock_services():
    mock_svc = mock.MagicMock()
    mock_svc.provider.list_models.return_value = []
    mock_svc.searcher._embedder.embedding_available.return_value = True
    with mock.patch("lilbee.services.get_services", return_value=mock_svc):
        yield mock_svc


@pytest.fixture(autouse=True)
def _patch_chat_setup():
    with mock.patch(
        "lilbee.cli.tui.screens.chat.ChatScreen._needs_setup",
        return_value=False,
    ), mock.patch(
        "lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready",
        return_value=True,
    ):
        yield


SCREEN_TYPES = [ChatScreen, CatalogScreen, StatusScreen, SettingsScreen, TaskCenter]


async def test_bracket_keys_cycle_all_screens():
    """Press ] through all 5 views from normal mode (Escape first on Chat)."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, ChatScreen)

        # Chat starts in insert mode — Escape to normal mode first
        await pilot.press("escape")
        await pilot.pause()

        expected = [CatalogScreen, StatusScreen, SettingsScreen, TaskCenter, ChatScreen]
        for screen_type in expected:
            await pilot.press("right_square_bracket")
            await pilot.pause()
            assert isinstance(app.screen, screen_type), (
                f"Expected {screen_type.__name__}, got {type(app.screen).__name__}"
            )


async def test_bracket_keys_cycle_backward():
    """Press [ to go backward through views."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        # Escape to normal mode so ] works
        await pilot.press("escape")
        await pilot.pause()

        await pilot.press("left_square_bracket")
        await pilot.pause()
        assert isinstance(app.screen, TaskCenter)

        await pilot.press("left_square_bracket")
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen)


async def test_bracket_keys_work_from_settings():
    """Navigate to Settings, press ], verify screen changes to Tasks."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.switch_view("Settings")
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen)

        await pilot.press("right_square_bracket")
        await pilot.pause()
        assert isinstance(app.screen, TaskCenter)


async def test_settings_filter_not_focused_on_mount():
    """Settings should focus scroll container, not the search Input."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.switch_view("Settings")
        await pilot.pause()

        search = app.screen.query_one("#settings-search", Input)
        assert app.screen.focused is not search


async def test_settings_slash_focuses_filter():
    """/ focuses the filter, Enter blurs it."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.switch_view("Settings")
        await pilot.pause()

        await pilot.press("slash")
        await pilot.pause()
        search = app.screen.query_one("#settings-search", Input)
        assert app.screen.focused is search

        await pilot.press("enter")
        await pilot.pause()
        assert app.screen.focused is not search


async def test_settings_escape_from_filter_stays():
    """Escape from filter blurs it but stays on Settings."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.switch_view("Settings")
        await pilot.pause()

        await pilot.press("slash")
        await pilot.pause()
        assert isinstance(app.screen.focused, Input)

        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen)
        assert not isinstance(app.screen.focused, Input)


async def test_settings_jk_scrolls_not_types():
    """j should scroll, not type into the filter input."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.switch_view("Settings")
        await pilot.pause()

        await pilot.press("j")
        await pilot.pause()
        search = app.screen.query_one("#settings-search", Input)
        assert search.value == ""


async def test_grid_arrows_stay_on_catalog():
    """Right arrow in catalog grid mode should move grid cursor, not switch screens."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.switch_view("Catalog")
        await pilot.pause()

        await pilot.press("right")
        await pilot.pause()
        assert isinstance(app.screen, CatalogScreen)


async def test_footer_present_on_screens():
    """Every screen should have a Footer widget."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        views = ["Chat", "Catalog", "Status", "Settings", "Tasks"]
        for view in views:
            app.switch_view(view)
            await pilot.pause()
            footers = app.screen.query(Footer)
            assert len(footers) > 0, f"{view} screen has no Footer"


async def test_help_panel_toggle():
    """? opens HelpPanel, ? again closes it."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        # Escape to normal mode so ? isn't typed into Input
        await pilot.press("escape")
        await pilot.pause()

        await pilot.press("question_mark")
        await pilot.pause()
        # HelpPanel may be on the screen or app level
        has_panel = bool(app.screen.query("HelpPanel") or app.query("HelpPanel"))
        assert has_panel, "HelpPanel should be visible"

        await pilot.press("question_mark")
        await pilot.pause()
        has_panel = bool(app.screen.query("HelpPanel") or app.query("HelpPanel"))
        assert not has_panel, "HelpPanel should be hidden"
