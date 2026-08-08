"""Textual's command palette with the search icon replaced.

Textual seeds ``SearchIcon`` with a magnifying-glass emoji, which the system
emoji font renders in its own colors at double width, making it the one element
on screen that ignores the theme.

``SearchIcon`` is built inside ``CommandPalette.compose`` and ``icon`` is a
reactive, so the icon is set after mount rather than by overriding ``compose``,
which would mean copying the library's layout. The query is deliberately not
``query_one``: if a future Textual restructures the palette, the icon reverts
rather than raising out of the Ctrl+P action.
"""

from __future__ import annotations

from textual.command import CommandPalette, SearchIcon

from lilbee.cli.tui import messages as msg


class LilbeeCommandPalette(CommandPalette):
    """Command palette whose search icon is a plain single-width glyph."""

    def on_mount(self) -> None:
        for icon in self.query(SearchIcon):
            icon.icon = msg.COMMAND_PALETTE_ICON
