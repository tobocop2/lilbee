"""Input subclass that lets screen-nav keys ([ and ]) bubble up.

Textual's default `Input.check_consume_key` returns True for every printable
character, which tells the binding dispatcher to skip any ancestor binding for
those keys even if that binding is marked `priority=True`. That means the
app-level screen navigation keys declared as `[` / `]` silently get typed into
the focused input instead of switching screens.

This subclass exempts `left_square_bracket` and `right_square_bracket` from the
consume check so those keys bubble up to the screen/app bindings unchanged.
All other printable characters are still captured normally, preserving the
full Input behavior for regular text entry.
"""

from __future__ import annotations

from textual.widgets import Input

_BUBBLE_KEYS: frozenset[str] = frozenset({"left_square_bracket", "right_square_bracket"})


class NavAwareInput(Input):
    """Input that does not consume the `[` / `]` screen-nav keys."""

    def check_consume_key(self, key: str, character: str | None) -> bool:
        if key in _BUBBLE_KEYS:
            return False
        return super().check_consume_key(key, character)
