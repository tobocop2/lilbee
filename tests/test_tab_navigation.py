"""Tab walks every focusable widget on every TUI screen.

Universal Tab navigation contract: pressing Tab on any screen advances
focus through every selectable widget, then wraps. This test drives a
real LilbeeApp through each top-level screen, captures the focus chain
by repeatedly pressing Tab, and asserts the chain visits the expected
widgets. The expected widgets per screen are the minimum set we want
keyboard-only users to reach; new focusable widgets should grow this
fixture.
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest

from conftest import TEST_EMBED_REF, TEST_LOCAL_REF
from lilbee.cli.tui.app import LilbeeApp
from lilbee.core.config import cfg


@pytest.fixture(autouse=True)
def _isolated_cfg(tmp_path) -> Any:
    snapshot = cfg.model_copy()
    cfg.data_dir = tmp_path / "data"
    cfg.data_root = tmp_path
    cfg.documents_dir = tmp_path / "documents"
    cfg.models_dir = tmp_path / "models"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.chat_model = TEST_LOCAL_REF
    cfg.embedding_model = TEST_EMBED_REF
    cfg.subprocess_embed = False
    cfg.wiki = False
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.documents_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.lancedb_dir.mkdir(parents=True, exist_ok=True)
    yield
    for field_name in type(snapshot).model_fields:
        setattr(cfg, field_name, getattr(snapshot, field_name))


@pytest.fixture(autouse=True)
def _mock_services() -> Any:
    from lilbee.core.services import set_services

    mock_svc = mock.MagicMock()
    mock_svc.provider.list_models.return_value = []
    mock_svc.searcher._embedder.embedding_available.return_value = True
    set_services(mock_svc)
    try:
        yield mock_svc
    finally:
        set_services(None)


@pytest.fixture(autouse=True)
def _patch_chat_setup() -> Any:
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


async def _walk_tab_chain(app: LilbeeApp, pilot: Any, max_presses: int = 60) -> list[str]:
    """Press Tab up to ``max_presses`` times; return the focused widget id chain.

    Stops early once the chain repeats (i.e. a full cycle). Returns the
    sequence of focused widget ids (or the widget class name when id is
    ``None``)."""
    chain: list[str] = []
    seen_at_index: dict[str, int] = {}
    for _ in range(max_presses):
        focused = app.focused
        marker = focused.id if focused is not None and focused.id else type(focused).__name__
        if marker in seen_at_index:
            break
        seen_at_index[marker] = len(chain)
        chain.append(marker)
        await pilot.press("tab")
        await pilot.pause()
    return chain


@pytest.mark.xfail(
    reason="ChatScreen's priority=True Tab→complete binding intercepts focus_next "
    "even when no completion overlay is open. Tracked in beads.",
    strict=False,
)
async def test_chat_tab_chain_includes_view_tabs() -> None:
    """Tab from ChatScreen should visit the ViewTabs and chat input."""
    app = LilbeeApp()
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        chain = await _walk_tab_chain(app, pilot, max_presses=40)
        for view in ("view-tab-chat", "view-tab-catalog", "view-tab-settings"):
            assert view in chain, f"Tab walk missed {view}: {chain}"
        assert "chat-input" in chain, f"chat-input missing from {chain}"


async def test_settings_tab_chain_visits_tabs_widget_and_search() -> None:
    """Tab from SettingsScreen visits the search input and the group Tabs."""
    app = LilbeeApp()
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        app.switch_view("Settings")
        await pilot.pause(0.2)
        chain = await _walk_tab_chain(app, pilot, max_presses=80)
        # Search input is reachable from the Settings screen.
        assert "settings-search" in chain, f"settings-search missing from {chain}"
        # ContentTabs (TabbedContent's tab strip) is reachable. Textual
        # gives it a stable id like 'tabs' inside ContentTabs; checking
        # by class name keeps the test resilient to internal renames.
        assert any(name in chain for name in ("ContentTabs", "Tabs")), (
            f"Tabs widget missing from {chain}"
        )


@pytest.mark.xfail(
    reason="GridSelect's Tab→LeaveDown handler routes through self.focus_next() "
    "which loops back to catalog-grid instead of advancing to catalog-search. "
    "Tracked in beads.",
    strict=False,
)
async def test_catalog_tab_chain_visits_search() -> None:
    """Tab from CatalogScreen should visit the search input."""
    app = LilbeeApp()
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        app.switch_view("Catalog")
        await pilot.pause(0.2)
        chain = await _walk_tab_chain(app, pilot, max_presses=40)
        assert "catalog-search" in chain, f"catalog-search missing from {chain}"
