"""Block-character progress bar renderables for use in DataTable cells.

Returns Rich ``Text`` renderables so callers can place them directly into
``DataTable.update_cell(...)`` without widget wrappers.
"""

from __future__ import annotations

from rich.text import Text

_BAR_WIDTH = 20
_FULL = "█"
_EMPTY = "░"
_INDETERMINATE_TRAIL = "   …  "
_DETERMINATE_STYLE = "bold cyan"
_INDETERMINATE_STYLE = "dim cyan"


def progress_cell(percent: float, width: int = _BAR_WIDTH) -> Text:
    """Render a 0-100 percent value as a block-character bar + trailing %."""
    pct = max(0.0, min(percent, 100.0))
    filled = int(width * pct / 100)
    bar = _FULL * filled + _EMPTY * (width - filled)
    return Text(f"{bar} {pct:5.1f}%", style=_DETERMINATE_STYLE)


def indeterminate_cell(tick: int, width: int = _BAR_WIDTH) -> Text:
    """Render an indeterminate pulse bar by sliding a 3-char window."""
    pos = tick % (width + 3)
    cells = [_EMPTY] * width
    for offset in (-1, 0, 1):
        i = pos + offset
        if 0 <= i < width:
            cells[i] = _FULL
    return Text("".join(cells) + _INDETERMINATE_TRAIL, style=_INDETERMINATE_STYLE)
