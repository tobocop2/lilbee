"""The crawl and preset dialogs keep their buttons on screen, whatever the form height."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from textual.app import ComposeResult
from textual.errors import NoWidget
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Static

from lilbee.app.presets import ServerPreset
from lilbee.cli.tui.screens.settings_presets import PresetDialog
from lilbee.cli.tui.widgets.crawl_dialog import CrawlDialog
from lilbee.providers.roles import WorkerRole
from lilbee.providers.server_registry import RerankWire
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import LilbeeAppHost

SMALL = (80, 24)
LARGE = (120, 60)

_PRESET = ServerPreset(
    name="gpu-box",
    base_url="http://gpu.local:8080",
    roles=(WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.RERANK),
    has_key=True,
    valid=True,
    rerank_wire=RerankWire.V1,
    memory_reserve_gb=None,
    reserve_note=None,
)

# (dialog factory, button ids, last form field id)
DIALOGS: list[tuple[Callable[[], Screen], tuple[str, ...], str]] = [
    (CrawlDialog, ("crawl-submit", "crawl-cancel"), "crawl-cookie-input"),
    (PresetDialog, ("preset-save", "preset-cancel"), "preset-key-input"),
    (
        lambda: PresetDialog(_PRESET),
        ("preset-save", "preset-clear-key", "preset-cancel"),
        "preset-key-input",
    ),
]
DIALOG_IDS = ["crawl", "preset-add", "preset-edit"]


class _App(LilbeeAppHost):
    def __init__(self, dialog: Screen) -> None:
        super().__init__()
        self._dialog = dialog

    def compose(self) -> ComposeResult:
        yield Static("")

    def on_mount(self) -> None:
        self.push_screen(self._dialog)


def _shown(screen: Screen, widget: Widget) -> bool:
    """Whether both corners of *widget* are drawn on the screen, with nothing over them."""
    region = widget.region
    corners = ((region.x, region.y), (region.right - 1, region.bottom - 1))
    try:
        return all(screen.get_widget_at(x, y)[0] is widget for x, y in corners)
    except NoWidget:
        return False


@pytest.mark.parametrize("size", [SMALL, LARGE], ids=["80x24", "120x60"])
@pytest.mark.parametrize(("make", "buttons", "last_field"), DIALOGS, ids=DIALOG_IDS)
async def test_buttons_are_on_screen_without_focus(make, buttons, last_field, size) -> None:
    app = _App(make())
    async with app.run_test(size=size) as pilot:
        await wait_until(pilot, lambda: app.screen.query(f"#{buttons[0]}").first() is not None)
        await pilot.pause()
        screen = app.screen
        assert not any(screen.query_one(f"#{b}").has_focus for b in buttons)
        for button_id in buttons:
            assert _shown(screen, screen.query_one(f"#{button_id}")), button_id
        # The form scrolls above the buttons; it does not run under them.
        form = screen.query_one("#dialog-form")
        assert form.region.bottom <= screen.query_one(f"#{buttons[0]}").region.y


@pytest.mark.parametrize(("make", "buttons", "last_field"), DIALOGS, ids=DIALOG_IDS)
async def test_buttons_stay_on_screen_when_the_form_scrolls(make, buttons, last_field) -> None:
    app = _App(make())
    async with app.run_test(size=SMALL) as pilot:
        await wait_until(pilot, lambda: app.screen.query(f"#{buttons[0]}").first() is not None)
        await pilot.pause()
        screen = app.screen
        form = screen.query_one("#dialog-form")
        assert form.max_scroll_y > 0, "the form must be taller than its area at 80x24"
        screen.query_one(f"#{last_field}").focus()
        await pilot.pause()
        assert form.scroll_y > 0
        for button_id in buttons:
            assert _shown(screen, screen.query_one(f"#{button_id}")), button_id
