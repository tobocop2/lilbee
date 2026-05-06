"""Animated Knight-Rider scanner-bar header for the assistant bubble."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from textual.content import Content
from textual.timer import Timer
from textual.widgets import Static

_CSS_FILE = Path(__file__).parent / "thinking_header.tcss"

_BLOCK_FILLED = "▰"
_BLOCK_EMPTY = "▱"
_TRACK_CELLS = 9
"""Visible width of the bouncing track. Wider than the old 5-cell snake
so the back-and-forth motion reads as deliberate scanning."""

_FRAME_INTERVAL = 0.1
"""100 ms per frame: motion without renderer thrash on large chat logs."""

_DIM_STYLE = "$text-muted"
_FILL_STYLE = "bold $success"


def _bounce_position(frame: int) -> int:
    """Return the lit cell index for *frame* in a Knight-Rider bounce.

    Cycle length is ``2 * (cells - 1)``: forward sweep 0..cells-1, then
    backward sweep cells-2..1, repeating.
    """
    cycle = 2 * (_TRACK_CELLS - 1)
    step = frame % cycle
    if step < _TRACK_CELLS:
        return step
    return cycle - step


def _frame_content(frame: int) -> Content:
    """Render the bouncing-block track for *frame* as styled content."""
    pos = _bounce_position(frame)
    parts: list[Content] = []
    for i in range(_TRACK_CELLS):
        if i == pos:
            parts.append(Content.styled(_BLOCK_FILLED, _FILL_STYLE))
        else:
            parts.append(Content.styled(_BLOCK_EMPTY, _DIM_STYLE))
    return Content.assemble(*parts)


class ThinkingHeader(Static):
    """Single Static that animates a Knight-Rider bouncing block."""

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

    def __init__(self) -> None:
        super().__init__(_frame_content(0), classes="thinking-header")
        self._frame: int = 0
        self._timer: Timer | None = None
        self._target: Callable[[Content], None] | None = None

    def on_mount(self) -> None:
        self._timer = self.set_interval(_FRAME_INTERVAL, self._tick)

    def on_unmount(self) -> None:
        self.stop()

    def stop(self) -> None:
        """Stop the animator timer; safe to call repeatedly."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def redirect_to(self, target: Callable[[Content], None] | None) -> None:
        """Send each frame's content to *target* instead of painting self."""
        self._target = target

    def _tick(self) -> None:
        self._frame += 1
        content = _frame_content(self._frame)
        if self._target is not None:
            self._target(content)
        else:
            self.update(content)
