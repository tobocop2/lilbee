"""Quitting while one of our screens or widgets mounts exits cleanly.

Each case quits from inside the node's compose, so the exit lands while it
mounts. Textual then mounts none of its children but still sends Mount.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import pytest
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widget import Widget

from lilbee.cli.tui import app as app_module
from lilbee.cli.tui.screens.model_picker import ModelPickerModal
from lilbee.cli.tui.screens.settings import SettingsScreen
from lilbee.cli.tui.screens.settings_engines import EnginesPane
from lilbee.cli.tui.screens.settings_presets import PresetsPane
from lilbee.cli.tui.screens.status import StatusScreen
from lilbee.cli.tui.screens.task_center import TaskCenter
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

    return type(f"Exiting{cls.__name__}", (cls,), {"compose": compose})


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


def _picker() -> Screen:
    opts = [ModelOption("Qwen3 0.6B", "qwen3-0.6b"), ModelOption("Llama 8B", "llama-8b")]
    return _exiting(ModelPickerModal)(scope="chat", options=opts, pick_later=lambda _r: None)


def _profile(app: App) -> None:
    app.open_profile_tab()


SITES: dict[str, Callable[[App], None]] = {
    "model-picker": _push(_picker),
    "settings-profile": _profile,
    "status": _push(_exiting(StatusScreen)),
    "task-center": _push(_exiting(TaskCenter)),
    "view-tabs": _hold(_exiting(ViewTabs)),
    "fleet-body": _hold(_exiting(FleetBody)),
    "session-list": _hold(_exiting(SessionListPanel)),
    "engines-pane": _hold(_exiting(EnginesPane)),
    "presets-pane": _hold(_exiting(PresetsPane)),
}


@pytest.mark.parametrize("site", sorted(SITES))
async def test_quit_while_mounting_exits_cleanly(site, monkeypatch):
    monkeypatch.setitem(app_module._VIEW_FACTORIES, "Settings", _exiting(SettingsScreen))
    app = _Host(SITES[site])
    await app.run_async(headless=True, size=(120, 40))
    assert app._exception is None, f"{site} rc={app.return_code} {app._exception!r}"
    assert app.return_code == 0
