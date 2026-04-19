"""Block-character progress bar renderables for the Task Center row.

Returns plain strings so the enclosing Static's CSS colors the bar
per task state (active / done / failed / cancelled). Using Rich Text
here would hard-code a palette and fight the theme.
"""

from __future__ import annotations

_BAR_WIDTH = 60
_FULL = "█"
_EMPTY = "░"


def progress_cell(percent: float, width: int = _BAR_WIDTH) -> str:
    """Render a 0-100 percent value as a block-character bar + trailing %."""
    pct = max(0.0, min(percent, 100.0))
    filled = int(width * pct / 100)
    bar = _FULL * filled + _EMPTY * (width - filled)
    return f"{bar}  {pct:5.1f}%"


def indeterminate_cell(tick: int, width: int = _BAR_WIDTH) -> str:
    """Render an indeterminate pulse bar by sliding a 3-char window."""
    pos = tick % (width + 3)
    cells = [_EMPTY] * width
    for offset in (-1, 0, 1):
        i = pos + offset
        if 0 <= i < width:
            cells[i] = _FULL
    return "".join(cells) + "   ···"
