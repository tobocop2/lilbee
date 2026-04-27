"""Vertical container docked to the screen's top edge.

Mirrors BottomBars: Textual's ``dock: top`` does not stack siblings,
so multiple top-docked widgets need a single Vertical wrapper.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual.containers import Vertical

_CSS_FILE = Path(__file__).parent / "top_bars.tcss"


class TopBars(Vertical):
    """Top-edge dock wrapper that stacks children vertically."""

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")
