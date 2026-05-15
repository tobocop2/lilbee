"""Bounded pilot.pause() loop for waiting on Textual async state in tests.

Several TUI tests assert on state that updates via Textual's message
cascade (``TabbedContent.active`` settling after assignment, worker
results landing via ``call_from_thread``, etc.). A single ``await
pilot.pause()`` is usually enough on the macOS / Linux runners but is
unreliably enough on the slower Windows runner -- the cascade may need
a handful of message-loop ticks to settle. The helper here pauses up
to *max_pauses* times, returning as soon as *predicate* is true. The
caller still runs its real assertion after; the helper only buys time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from textual.pilot import Pilot


async def wait_until(
    pilot: Pilot,
    predicate: Callable[[], bool],
    *,
    max_pauses: int = 50,
) -> bool:
    """Pump the Textual message loop until *predicate* is true or the budget is spent.

    Returns the final predicate value so callers can choose to assert on
    it or fall through to their existing assertion. ``max_pauses=50``
    covers Windows-CI worker timing without sleeping forever on a hung
    test.
    """
    if predicate():
        return True
    for _ in range(max_pauses):
        await pilot.pause()
        if predicate():
            return True
    return False
