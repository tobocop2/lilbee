"""Tab walks every focusable widget on every TUI screen.

Universal Tab navigation contract: pressing Tab on any non-input widget
advances focus through every selectable widget, then wraps. Text inputs
are a documented carve-out: in chat input, search inputs, and editor
fields, Tab inserts a literal tab character so users can type tabs in
messages or settings. To leave a focused input, press Escape (chat) or
click out (search/editor).

These tests drive a real ``LilbeeApp`` through each top-level screen,
walk the chain by pressing Tab from a non-input start widget, and
assert the chain visits the expected widgets. They also assert the
text-input carve-out: Tab pressed inside chat input inserts ``\\t``
without advancing focus.
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

    Stops early when the same widget instance is focused twice (a full
    cycle). Each entry in the returned list is the focused widget's id
    or class name; multiple unnamed widgets of the same class produce
    duplicate entries, which is intentional.
    """
    chain: list[str] = []
    seen_ids: set[int] = set()
    for _ in range(max_presses):
        focused = app.focused
        if focused is None:
            break
        if id(focused) in seen_ids:
            break
        seen_ids.add(id(focused))
        marker = focused.id if focused.id else type(focused).__name__
        chain.append(marker)
        await pilot.press("tab")
        await pilot.pause()
    return chain


async def test_chat_tab_chain_includes_view_tabs() -> None:
    """Tab walk from a view tab on ChatScreen visits every focusable widget.

    Starts the walk from the chat view tab so chat input's literal-tab
    carve-out doesn't terminate the chain early.
    """
    from lilbee.cli.tui.widgets.status_bar import ViewTabs

    app = LilbeeApp()
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause(0.2)
        view_tabs = app.screen.query_one(ViewTabs)
        view_tabs.query_one("#view-tab-chat").focus()
        await pilot.pause()
        chain = await _walk_tab_chain(app, pilot, max_presses=40)
        for view in ("view-tab-chat", "view-tab-catalog", "view-tab-settings"):
            assert view in chain, f"Tab walk missed {view}: {chain}"


async def test_chat_input_tab_inserts_literal_tab() -> None:
    """Tab pressed while chat input has focus inserts ``\\t``, not focus_next.

    Mirrors a typical text-editor contract: Tab is a literal character
    inside the prompt so users can paste indented code or type tabs.
    """
    from lilbee.cli.tui.widgets.chat_input import ChatInput

    app = LilbeeApp()
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause(0.2)
        inp = app.screen.query_one("#chat-input", ChatInput)
        inp.focus()
        inp.value = "hello"
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()
        assert inp.value == "hello\t", f"Tab did not insert: {inp.value!r}"
        assert app.focused is inp, "Focus moved away from chat input on Tab"


async def test_settings_tab_chain_visits_group_tabs_widget() -> None:
    """Tab from SettingsScreen visits the group Tabs strip."""
    app = LilbeeApp()
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        app.switch_view("Settings")
        await pilot.pause(0.2)
        chain = await _walk_tab_chain(app, pilot, max_presses=80)
        assert any(name in chain for name in ("ContentTabs", "Tabs")), (
            f"Tabs widget missing from {chain}"
        )


async def test_settings_tab_rolls_over_to_next_pane() -> None:
    """Tab past the last editor in a pane activates the next group tab."""
    from textual.widgets import TabbedContent

    from lilbee.cli.tui.screens.settings import SettingsScreen, _LazyGroupBody

    app = LilbeeApp()
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        app.switch_view("Settings")
        await pilot.pause(0.2)
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        screen.populate_all_panes()
        await pilot.pause()
        tabs = screen.query_one(TabbedContent)
        starting_pane = tabs.active
        body = screen.query_one(f"#{starting_pane}-body", _LazyGroupBody)
        focusables = [w for w in body.query("*") if w.focusable]
        assert focusables, "active pane should expose at least one focusable"
        focusables[-1].focus()
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause(0.3)
        assert tabs.active != starting_pane, (
            f"Tab from last field should activate the next pane (still {tabs.active})"
        )


async def test_catalog_tab_chain_visits_search() -> None:
    """Pressing / on CatalogScreen reveals + focuses the search input.

    The search input is hidden by default; the catalog footer surfaces
    the / binding so the user can opt in. Once revealed, the Tab walk
    must visit it so screen-reader and keyboard-only users can route
    out of the search box without touching the mouse.
    """
    from lilbee.cli.tui.widgets.status_bar import ViewTabs

    app = LilbeeApp()
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause()
        app.switch_view("Catalog")
        await pilot.pause(0.3)
        view_tabs = app.screen.query_one(ViewTabs)
        view_tabs.query_one("#view-tab-catalog").focus()
        await pilot.pause()
        await pilot.press("slash")
        await pilot.pause()
        focused_id = app.focused.id if app.focused else None
        assert focused_id == "catalog-search", (
            f"slash should focus catalog-search, focused={focused_id!r}"
        )
        chain = await _walk_tab_chain(app, pilot, max_presses=40)
        assert "catalog-search" in chain, f"catalog-search missing from {chain}"
