"""Common base for the non-modal side drawers."""

from __future__ import annotations

from textual.containers import Vertical


class Drawer(Vertical):
    """A non-modal side drawer that owns the keyboard while focus is inside it.

    The chat screen's vim mode treats enter / i / a / o as conversation keys and
    swallows them. A drawer's own controls need those keys, so the chat screen
    asks whether focus sits under a Drawer rather than naming each drawer class:
    a new drawer inherits the exemption instead of silently eating its own enter.
    """
