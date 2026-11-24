"""Quitting while one of our screens or widgets mounts exits cleanly.

Each real-exit case quits from inside the node's compose, so the exit lands while it
mounts. Textual then mounts none of its children but still sends Mount.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import pytest
from textual import events
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import TabbedContent

from lilbee.cli.tui import app as app_module
from lilbee.cli.tui.mount_guard import MountWhenAttached
from lilbee.cli.tui.screens.memories import MemoriesScreen
from lilbee.cli.tui.screens.model_picker import ModelPickerModal
from lilbee.cli.tui.screens.settings import PROFILE_PANE_ID, SettingsScreen
from lilbee.cli.tui.screens.settings_engines import EnginesPane
from lilbee.cli.tui.screens.settings_presets import PresetDialog, PresetsPane
from lilbee.cli.tui.screens.status import StatusScreen
from lilbee.cli.tui.screens.task_center import TaskCenter
from lilbee.cli.tui.screens.wiki_drafts import WikiDraftsScreen
from lilbee.cli.tui.widgets.fleet_body import FleetBody
from lilbee.cli.tui.widgets.model_bar import ModelOption
from lilbee.cli.tui.widgets.session_list import SessionListPanel
from lilbee.cli.tui.widgets.status_bar import ViewTabs
from tests._lilbee_app_test_host import LilbeeAppHost

_W = TypeVar("_W", bound=Widget)


def _exiting(cls: type[_W]) -> type[_W]:
    """Return a subclass of ``cls`` that quits the app from inside its compose."""

    def compose(self: Widget) -> ComposeResult:
        self.app.exit()
        yield from cls.compose(self)

    namespace = {"compose": compose, "__module__": cls.__module__}
    return type(f"Exiting{cls.__name__}", (cls,), namespace)


class _Host(LilbeeAppHost):
    def __init__(self, start: Callable[[App], None]) -> None:
        super().__init__()
        self._start = start

    def on_ready(self) -> None:
        self._start(self)


def _push(make: Callable[[], Screen]) -> Callable[[App], None]:
    return lambda app: app.push_screen(make())


def _hold(make: Callable[[], Widget]) -> Callable[[App], None]:
    class _Holder(Screen[None]):
        def compose(self) -> ComposeResult:
            yield make()

    return _push(_Holder)


def _options() -> list[ModelOption]:
    return [ModelOption("Qwen3 0.6B", "qwen3-0.6b"), ModelOption("Llama 8B", "llama-8b")]


def _picker() -> Screen:
    return _exiting(ModelPickerModal)(scope="chat", options=_options(), pick_later=lambda _r: None)


def _profile(app: App) -> None:
    # switch_view replaces the screen on top, so one must be pushed first, as in the real app
    app.push_screen(Screen())
    app.open_profile_tab()


SITES: dict[str, Callable[[App], None]] = {
    "model-picker": _push(_picker),
    "settings-profile": _profile,
    "status": _push(_exiting(StatusScreen)),
    "task-center": _push(_exiting(TaskCenter)),
    "memories": _push(_exiting(MemoriesScreen)),
    "wiki-drafts": _push(_exiting(WikiDraftsScreen)),
    "view-tabs": _hold(_exiting(ViewTabs)),
    "fleet-body": _hold(_exiting(FleetBody)),
    "session-list": _hold(_exiting(SessionListPanel)),
    "preset-dialog": _push(_exiting(PresetDialog)),
}


@pytest.mark.parametrize("site", sorted(SITES))
async def test_quit_while_mounting_exits_cleanly(site, monkeypatch):
    monkeypatch.setitem(app_module._VIEW_FACTORIES, "Settings", _exiting(SettingsScreen))
    app = _Host(SITES[site])
    await app.run_async(headless=True, size=(120, 40))
    assert app._exception is None, f"{site} rc={app.return_code} {app._exception!r}"
    assert app.return_code == 0


GUARDED: dict[str, Callable[[], Any]] = {
    "model-picker": lambda: ModelPickerModal(
        scope="chat", options=_options(), pick_later=lambda _r: None
    ),
    "preset-dialog": PresetDialog,
    "settings": SettingsScreen,
    "status": StatusScreen,
    "task-center": TaskCenter,
    "memories": MemoriesScreen,
    "wiki-drafts": WikiDraftsScreen,
    "view-tabs": ViewTabs,
    "fleet-body": FleetBody,
    "session-list": SessionListPanel,
    "engines-pane": EnginesPane,
    "presets-pane": PresetsPane,
}


@pytest.mark.parametrize("site", sorted(GUARDED))
async def test_unattached_node_drops_mount(site, monkeypatch):
    """A node outside a running app gets no call of its mount handler."""
    node = GUARDED[site]()
    calls: list[object] = []
    monkeypatch.setattr(type(node), "on_mount", lambda _self: calls.append(1))
    assert isinstance(node, MountWhenAttached)
    await node.on_event(events.Mount())
    assert calls == []


async def test_profile_tab_requested_before_mount_shows_on_mount():
    """The Profile tab asked for before the screen mounts is active once it has."""
    app = LilbeeAppHost()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(Screen())
        app.open_profile_tab()
        await pilot.pause()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        assert screen.query_one("#settings-tabs", TabbedContent).active == PROFILE_PANE_ID


async def test_profile_tab_requested_on_a_mounted_screen_shows_after_refresh():
    app = LilbeeAppHost()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(Screen())
        app.switch_view("Settings")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        assert screen.is_mounted
        tabs = screen.query_one("#settings-tabs", TabbedContent)
        tabs.active = next(pane for pane in screen._pane_ids if pane != PROFILE_PANE_ID)
        await pilot.pause()
        screen.request_profile_tab()
        await pilot.pause()
        await pilot.pause()
        assert tabs.active == PROFILE_PANE_ID
