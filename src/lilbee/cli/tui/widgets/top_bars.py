"""Vertical container docked to the screen's top edge."""

from __future__ import annotations

from textual.containers import Vertical


class TopBars(Vertical):
    """Top-edge dock wrapper that stacks children vertically."""

    DEFAULT_CSS = """
    TopBars {
        dock: top;
        height: auto;
        width: 100%;
    }
    """
