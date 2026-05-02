"""Tests for the GridListToggle widget on the catalog screen."""

from __future__ import annotations

from unittest import mock

from textual import events
from textual.app import App, ComposeResult

from lilbee.cli.tui.widgets.grid_list_toggle import GridListToggle


class _ToggleApp(App[None]):
    def compose(self) -> ComposeResult:
        yield GridListToggle()


async def test_toggle_renders_grid_active_by_default() -> None:
    app = _ToggleApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        toggle = app.query_one(GridListToggle)
        rendered = str(toggle.render())
        assert "Grid" in rendered
        assert "List" in rendered
        assert "·" in rendered


async def test_set_grid_repaints_active_half() -> None:
    app = _ToggleApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        toggle = app.query_one(GridListToggle)
        toggle.set_grid(False)
        await pilot.pause()
        toggle.set_grid(True)
        await pilot.pause()
        # No exception means the repaint cycled both states cleanly.


async def test_action_select_grid_calls_screen_toggle_when_on_list() -> None:
    """Calling action_select_grid only fires when currently on list."""
    app = _ToggleApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        toggle = app.query_one(GridListToggle)
        toggle.set_grid(False)  # currently list
        with mock.patch.object(toggle, "_call_screen_toggle") as mocked:
            toggle.action_select_grid()
        mocked.assert_called_once()


async def test_action_select_grid_noop_when_already_grid() -> None:
    app = _ToggleApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        toggle = app.query_one(GridListToggle)
        with mock.patch.object(toggle, "_call_screen_toggle") as mocked:
            toggle.action_select_grid()
        mocked.assert_not_called()


async def test_action_select_list_noop_when_already_list() -> None:
    app = _ToggleApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        toggle = app.query_one(GridListToggle)
        toggle.set_grid(False)
        with mock.patch.object(toggle, "_call_screen_toggle") as mocked:
            toggle.action_select_list()
        mocked.assert_not_called()


async def test_action_flip_calls_screen_toggle() -> None:
    app = _ToggleApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        toggle = app.query_one(GridListToggle)
        with mock.patch.object(toggle, "_call_screen_toggle") as mocked:
            toggle.action_flip()
        mocked.assert_called_once()


async def test_on_click_calls_screen_toggle() -> None:
    app = _ToggleApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        toggle = app.query_one(GridListToggle)
        click = mock.MagicMock(spec=events.Click)
        with mock.patch.object(toggle, "_call_screen_toggle") as mocked:
            toggle.on_click(click)
        mocked.assert_called_once()
        click.stop.assert_called_once()


async def test_call_screen_toggle_no_op_when_not_catalog_screen() -> None:
    """The standalone _ToggleApp's screen is not a CatalogScreen, so the toggle
    silently no-ops rather than crashing."""
    app = _ToggleApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        toggle = app.query_one(GridListToggle)
        # Direct call should not raise even though screen is not CatalogScreen.
        toggle._call_screen_toggle()
