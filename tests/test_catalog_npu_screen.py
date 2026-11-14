"""The catalog screen shows an NPU row, or the note that this host cannot run it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from textual.app import ComposeResult
from textual.pilot import Pilot
from textual.widgets import Static

from _async_wait import wait_until
from conftest import stub_hf_listing
from lilbee.app.services import get_services
from lilbee.catalog import query as _query
from lilbee.cli.tui.screens.catalog import CatalogScreen
from lilbee.cli.tui.widgets.model_grid import ModelGrid
from lilbee.providers.fleet import host_caps
from lilbee.providers.fleet.engines.lemonade import LEMONADE
from lilbee.providers.fleet.engines.lemonade_models import parse_models
from tests._lilbee_app_test_host import LilbeeAppHost

PINNED = json.loads(
    (Path(__file__).parent / "fixtures" / "engine_model_lists" / "lemonade-pinned.json").read_text(
        encoding="utf-8"
    )
)
XDNA2 = host_caps.collect(
    system="Linux", machine="x86_64", cpu_info={}, nvidia=None, accel_devices=[("0x1022", "0x17f0")]
)
NO_NPU = host_caps.collect(
    system="Linux", machine="x86_64", cpu_info={}, nvidia=None, accel_devices=[]
)
SIZES = pytest.mark.parametrize("size", [(80, 24), (120, 40)], ids=["80x24", "120x40"])
CARD_SIZE_LINE = "4B · ~3.7 GB"
"""The size line of the card of the ``qwen3:4b`` row."""


class _App(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        yield Static("host")


def _text(screen: CatalogScreen) -> str:
    return "\n".join(str(w.render()) for w in screen.query(Static))


def _cards(screen: CatalogScreen) -> str:
    """The text a user reads on the model cards: each grid draws its own lines."""
    lines = (
        grid.render_line(y).text
        for grid in screen.query(ModelGrid)
        for y in range(grid.size.height)
    )
    return "\n".join(lines)


def _browse(monkeypatch: pytest.MonkeyPatch, host: host_caps.HostCaps) -> None:
    monkeypatch.setattr(_query, "get_picks", lambda: [])
    monkeypatch.setattr(_query, "session_host", lambda: host)
    stub_hf_listing(monkeypatch, {})
    rows = parse_models(PINNED)
    monkeypatch.setattr(get_services().hf_client, "fetch_engine_models", lambda engine: rows)


async def _search_chat_tab(pilot: Pilot[None], screen: CatalogScreen, text: str) -> None:
    await pilot.press("2")
    await pilot.press("slash")
    await wait_until(pilot, lambda: screen._filter_open)
    for char in text:
        await pilot.press("colon" if char == ":" else char)


@SIZES
async def test_an_npu_host_shows_the_row_and_no_hidden_note(
    monkeypatch: pytest.MonkeyPatch, size: tuple[int, int]
) -> None:
    _browse(monkeypatch, XDNA2)
    async with _App().run_test(size=size) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await _search_chat_tab(pilot, screen, "qwen3:4b")
        shown = await wait_until(pilot, lambda: CARD_SIZE_LINE in _cards(screen), timeout=3.0)
        cards, text = _cards(screen), _text(screen)
        assert shown, cards
        assert "qwen3:4b" in cards
        assert "hidden" not in text
        assert not {"None", "ModelFormat", "XDNA", "xdna"} & set(cards.split())
        assert "None" not in text


@SIZES
async def test_a_host_without_the_npu_shows_the_hidden_note_and_no_row(
    monkeypatch: pytest.MonkeyPatch, size: tuple[int, int]
) -> None:
    _browse(monkeypatch, NO_NPU)
    reason = LEMONADE.host_gate(NO_NPU).reason
    note = f"1 hidden (This host cannot run it. {reason})"
    async with _App().run_test(size=size) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await _search_chat_tab(pilot, screen, "qwen3:4b")
        shown = await wait_until(pilot, lambda: note in _text(screen), timeout=3.0)
        assert shown, _text(screen)
        assert CARD_SIZE_LINE not in _cards(screen)
        assert "None" not in _text(screen)
