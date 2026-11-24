"""The crawl and preset forms scroll their fields; the button row stays on screen."""

from __future__ import annotations

import pytest
from textual.app import ComposeResult
from textual.css.query import NoMatches
from textual.errors import NoWidget
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Static

from lilbee.app.presets import ServerPreset
from lilbee.cli.tui.screens.settings_presets import PresetDialog
from lilbee.cli.tui.widgets.crawl_dialog import CrawlDialog
from lilbee.providers.roles import WorkerRole
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import LilbeeAppHost


class _DialogApp(LilbeeAppHost):
    """Opens one dialog over the host."""

    def __init__(self, dialog: Screen) -> None:
        super().__init__()
        self._dialog = dialog

    def compose(self) -> ComposeResult:
        yield Static("")

    def on_mount(self) -> None:
        self.push_screen(self._dialog)


def _shown(screen: Screen, widget: Widget) -> bool:
    """True when the screen draws *widget* at both ends of its row, so no parent clips it."""
    region = widget.region
    ends = ((region.x, region.y), (region.right - 1, region.bottom - 1))
    try:
        return all(screen.get_widget_at(x, y)[0] is widget for x, y in ends)
    except NoWidget:
        return False


def _buttons(screen: Screen, ids: tuple[str, ...]) -> list[Widget]:
    try:
        return [screen.query_one(f"#{button_id}") for button_id in ids]
    except NoMatches:
        return []


def _preset() -> ServerPreset:
    return ServerPreset(
        name="local",
        base_url="http://localhost:1234/v1",
        roles=(WorkerRole.CHAT,),
        has_key=True,
        valid=True,
        rerank_wire=None,
        memory_reserve_gb=None,
        reserve_note=None,
    )


@pytest.mark.parametrize("size", [(80, 24), (60, 20)])
@pytest.mark.parametrize(
    ("make", "ids"),
    [
        (lambda: CrawlDialog(), ("crawl-submit", "crawl-cancel")),
        (lambda: PresetDialog(), ("preset-save", "preset-cancel")),
        (lambda: PresetDialog(_preset()), ("preset-save", "preset-cancel")),
    ],
    ids=["crawl", "preset-add", "preset-edit"],
)
async def test_the_buttons_stay_on_screen_without_a_focus_change(
    make, ids: tuple[str, ...], size: tuple[int, int]
) -> None:
    app = _DialogApp(make())
    async with app.run_test(size=size) as pilot:

        def visible() -> bool:
            buttons = _buttons(app.screen, ids)
            return len(buttons) == len(ids) and all(_shown(app.screen, b) for b in buttons)

        assert await wait_until(pilot, visible), [
            (b.id, b.region) for b in _buttons(app.screen, ids)
        ]


@pytest.mark.parametrize(
    ("make", "scroll_id"),
    [
        (lambda: CrawlDialog(), "crawl-scroll"),
        (lambda: PresetDialog(), "preset-scroll"),
        (lambda: PresetDialog(_preset()), "preset-scroll"),
    ],
    ids=["crawl", "preset-add", "preset-edit"],
)
async def test_a_form_taller_than_the_terminal_scrolls_its_fields(make, scroll_id: str) -> None:
    app = _DialogApp(make())
    async with app.run_test(size=(80, 24)) as pilot:

        def scrolls() -> bool:
            found = app.screen.query(f"#{scroll_id}")
            return bool(found) and found.first().max_scroll_y > 0

        assert await wait_until(pilot, scrolls)
        scroll = app.screen.query_one(f"#{scroll_id}")
        assert scroll.region.bottom <= app.screen.size.height
