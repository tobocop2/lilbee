"""Shared key vocabulary for the browse screens.

One fragment so sibling screens cannot drift: q goes back, Escape
dismisses-or-goes-back, j/k walk the list, g/G jump to the ends. Screens
spread these into ``BINDINGS`` and implement the actions against their own
list widget. A focused Input or TextArea consumes printable keys before any
binding fires, so the letters are typing-safe without focus guards.
"""

from __future__ import annotations

from textual.binding import Binding, BindingType

BROWSE_LIST_BINDINGS: list[BindingType] = [
    Binding("j", "cursor_down", "Nav", show=False),
    Binding("k", "cursor_up", "Nav", show=False),
    Binding("g", "jump_top", "Top", show=False),
    Binding("G", "jump_bottom", "End", show=False),
]


def browse_back_bindings(*, escape_action: str = "go_back") -> list[BindingType]:
    """The back pair. Screens with a search overlay pass ``dismiss_or_back``
    so Escape clears the overlay first and leaves on the second press."""
    return [
        Binding("q", "go_back", "Back", show=True),
        Binding("escape", escape_action, "Back", show=False),
    ]
