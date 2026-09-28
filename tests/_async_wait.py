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
    from textual.widget import Widget


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


async def press_widget(
    pilot: Pilot,
    widget: Widget,
    key: str = "enter",
    *,
    max_pauses: int = 50,
) -> None:
    """Focus *widget* and press *key* on it.

    ``Widget.focus()`` only queues a ``call_later`` that sets focus on a
    later message-loop tick. ``Screen.set_focus()`` is the synchronous
    primitive it defers to, and calling it directly assigns focus with no
    such delay, but only once ``widget.focusable`` is true, which is not
    guaranteed on the first poll right after a dialog mounts (its layout
    may not have settled yet). The call is retried on every failed poll
    for that reason, not to survive something else repeatedly taking
    focus back.
    """

    def _focused() -> bool:
        if not widget.has_focus and widget.focusable:
            widget.screen.set_focus(widget)
        return widget.has_focus

    assert await wait_until(pilot, _focused, max_pauses=max_pauses), widget
    await pilot.press(key)
