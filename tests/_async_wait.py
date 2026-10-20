"""Bounded waits for tests: a pilot.pause() loop for Textual state, a poll for thread state.

Several TUI tests assert on state that updates via Textual's message
cascade (``TabbedContent.active`` settling after assignment, worker
results landing via ``call_from_thread``, etc.). A single ``await
pilot.pause()`` is usually enough on the macOS / Linux runners but is
unreliably enough on the slower Windows runner -- the cascade may need
a handful of message-loop ticks to settle. The helper here pauses for
a budget of pauses and seconds, returning as soon as *predicate* is true.
The caller still runs its real assertion after; the helper only buys time.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from textual.widget import Widget

if TYPE_CHECKING:
    from textual.app import App
    from textual.pilot import Pilot

WidgetT = TypeVar("WidgetT", bound=Widget)


async def wait_until(
    pilot: Pilot,
    predicate: Callable[[], bool],
    *,
    max_pauses: int = 50,
    timeout: float = 0.0,
) -> bool:
    """Pump the Textual message loop until *predicate* is true or the budget is spent.

    The budget is at least *max_pauses* pauses and at least *timeout* seconds.
    Returns the final predicate value so callers can choose to assert on
    it or fall through to their existing assertion. ``max_pauses=50``
    covers Windows-CI worker timing without sleeping forever on a hung
    test.
    """
    deadline = time.monotonic() + timeout
    pauses = 0
    while not predicate():
        if pauses >= max_pauses and time.monotonic() >= deadline:
            return False
        await pilot.pause()
        pauses += 1
    return True


def poll_until(predicate: Callable[[], bool], timeout: float = 10.0) -> bool:
    """Poll *predicate* until it holds or *timeout* seconds pass; returns whether it held."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.02)
    return True


async def wait_for_widget(
    pilot: Pilot,
    app: App,
    selector: str,
    widget_type: type[WidgetT] = Widget,  # type: ignore[assignment]
    *,
    timeout: float = 5.0,
) -> WidgetT:
    """Wait until the current screen holds *selector*, then return that widget.

    Some screens mount sections after they become current, so a query made
    right after the screen switch can miss a widget that is still coming.
    """
    assert await wait_until(pilot, lambda: bool(app.screen.query(selector)), timeout=timeout), (
        f"{selector} never mounted"
    )
    return app.screen.query_one(selector, widget_type)
