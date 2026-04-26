"""Single dock-top container for the per-screen top stack.

Mirror of ``BottomBars`` for the top edge: Textual's ``dock: top`` does
not stack siblings, so each screen composes one ``TopBars`` holding the
top-mounted widgets (``ViewTabs``, and on chat also ``ModelBar``) and
they stack vertically inside it instead of overlapping.
"""

from __future__ import annotations

from textual.containers import Vertical


class TopBars(Vertical):
    """Vertical container docked to the screen's top edge.

    Children stack top-to-bottom inside the container so ModelBar (on
    chat) and ViewTabs each get their own row instead of colliding at
    the screen's top edge.
    """

    DEFAULT_CSS = """
    TopBars {
        dock: top;
        height: auto;
        width: 100%;
    }
    """
