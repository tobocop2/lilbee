"""BottomBars -- single dock-bottom container for the per-screen bottom stack.

Textual's ``dock: bottom`` places a widget at the bottom edge of its
container, but it does NOT stack multiple dock-bottom siblings on top
of each other -- every dock-bottom widget lands at the SAME edge row
and overlaps. (See ``textual._arrange._arrange_dock_widgets``: each
docked widget's region is computed as ``Region(0, height -
widget_height, ...)`` independently, with no offset for earlier dock
siblings.) The last-composed widget paints on top, hiding the others.

Every lilbee screen has three dock-bottom widgets -- ``TaskBar``,
``ViewTabs``, and ``Footer`` -- which collided into the same bottom
row, leaving only ``Footer`` visible. The chat screen additionally
has ``#chat-prompt-area`` docked bottom, making it even worse.

The fix is to dock a single container (this widget) to the bottom
edge and let its children flow vertically inside it. Each screen
composes exactly one ``BottomBars`` holding ``TaskBar``, ``ViewTabs``,
and ``Footer`` (and, on the chat screen, the prompt area as the
first child).
"""

from __future__ import annotations

from textual.containers import Vertical


class BottomBars(Vertical):
    """Vertical container docked to the screen's bottom edge.

    Children stack top-to-bottom inside the container so TaskBar,
    ViewTabs, and Footer each get their own row instead of colliding
    at the screen's bottom edge.
    """

    DEFAULT_CSS = """
    BottomBars {
        dock: bottom;
        height: auto;
        width: 100%;
    }
    """
