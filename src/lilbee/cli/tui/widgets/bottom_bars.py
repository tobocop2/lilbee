"""Single dock-bottom container for the per-screen bottom stack.

Textual's ``dock: bottom`` does not stack siblings: every dock-bottom
widget lands at the same edge row and overlaps, so only the
last-composed widget paints. Each screen composes exactly one
``BottomBars`` holding the per-screen bottom widgets
(``TaskBar``, ``ViewTabs``, ``Footer``, and on chat the prompt area)
so they stack vertically instead of colliding.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual.containers import Vertical

_CSS_FILE = Path(__file__).parent / "bottom_bars.tcss"


class BottomBars(Vertical):
    """Vertical container docked to the screen's bottom edge.

    Children stack top-to-bottom inside the container so TaskBar,
    ViewTabs, and Footer each get their own row instead of colliding
    at the screen's bottom edge.
    """

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")
