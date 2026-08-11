"""Progress bar renderables for the Task Center row.

Returns plain strings so the enclosing Static's CSS colors the bar
per task state (active / done / failed / cancelled). Using Rich Text
here would hard-code a palette and fight the theme.
"""

from __future__ import annotations

from lilbee.cli.tui.messages import progress_bar_glyphs

_BAR_WIDTH = 60


def progress_cell(percent: float, width: int = _BAR_WIDTH) -> str:
    """Render a 0-100 percent value as a bar + trailing %."""
    fill, track = progress_bar_glyphs()
    pct = max(0.0, min(percent, 100.0))
    filled = int(width * pct / 100)
    bar = fill * filled + track * (width - filled)
    return f"{bar}  {pct:5.1f}%"


def indeterminate_cell(tick: int, width: int = _BAR_WIDTH) -> str:
    """Render an indeterminate pulse bar by sliding a 3-char window."""
    fill, track = progress_bar_glyphs()
    pos = tick % (width + 3)
    cells = [track] * width
    for offset in (-1, 0, 1):
        i = pos + offset
        if 0 <= i < width:
            cells[i] = fill
    return "".join(cells) + "   ···"


def frozen_indeterminate_cell(width: int = _BAR_WIDTH) -> str:
    """Render a non-animating indeterminate bar for terminal-state rows.

    The bar is fully filled and trailing dots are dropped so the row no
    longer reads as 'work in progress' once the task has finished. The
    trailing percentage is omitted because an indeterminate task never
    had a real percentage to report.
    """
    return progress_bar_glyphs()[0] * width
