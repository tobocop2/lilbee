"""Tests for the bounded test waits."""

from __future__ import annotations

import asyncio
import time

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import Static

from tests._async_wait import subtree_mounted, wait_until


class _Pilot:
    """A pilot whose pause yields for a short, real interval."""

    async def pause(self) -> None:
        await asyncio.sleep(0.01)


def _true_after(seconds: float):
    deadline = time.monotonic() + seconds
    return lambda: time.monotonic() >= deadline


@pytest.mark.asyncio
async def test_wait_until_keeps_pumping_for_its_time_budget() -> None:
    assert await wait_until(_Pilot(), _true_after(0.2), max_pauses=1, timeout=5.0)


@pytest.mark.asyncio
async def test_wait_until_with_no_time_budget_stops_after_its_pauses() -> None:
    assert not await wait_until(_Pilot(), _true_after(5.0), max_pauses=1)


class _GatedRow(Static):
    """A row whose inner widget joins the tree only once the gate opens."""

    gate: asyncio.Event

    def compose(self) -> ComposeResult:
        yield Static("inner", id="inner")

    async def mount_composed_widgets(self, widgets: list[Widget]) -> None:
        await self.gate.wait()
        await super().mount_composed_widgets(widgets)


@pytest.mark.asyncio
async def test_subtree_mounted_is_false_until_the_inner_widgets_mount() -> None:
    app = App[None]()
    async with app.run_test() as pilot:
        root = Vertical(id="root")
        await app.mount(root)
        assert root.is_mounted
        row = _GatedRow(id="row")
        row.gate = asyncio.Event()
        root.mount(row)
        # No pilot pause here: a pause waits on every widget, and the gated row cannot answer.
        await asyncio.sleep(0.05)
        # The root is mounted already; only its later child is not, so the walk must see it.
        assert root.is_mounted
        assert not subtree_mounted(root)
        assert not row.query("#inner")
        row.gate.set()
        assert await wait_until(pilot, lambda: subtree_mounted(root))
        assert row.query("#inner")
