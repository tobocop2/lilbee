"""Common base for the non-modal side drawers."""

from __future__ import annotations

from textual.await_remove import AwaitRemove
from textual.containers import Vertical
from textual.widget import Widget


class Drawer(Vertical):
    """A non-modal side drawer that owns the keyboard while focus is inside it.

    The chat screen's vim mode treats esc / enter / i / a / o as conversation keys
    and swallows them. A drawer's own controls need those keys, so the chat screen
    asks whether focus sits under a Drawer rather than naming each drawer class:
    a new drawer inherits the exemption instead of silently eating its own enter.
    """

    def __init__(self, *, id: str) -> None:
        super().__init__(id=id)
        self._return_focus: Widget | None = None

    def on_compose(self) -> None:
        """Remember the focus from before the drawer opened; children mount after this.

        Focus inside another drawer is looked past to that drawer's own return target,
        since that drawer can close first.
        """
        focused = self.screen.focused
        holder = drawer_holding(focused)
        self._return_focus = focused if holder is None else holder._return_focus

    def remove(self) -> AwaitRemove:
        """Close the drawer, handing focus it holds back to where it was before it opened."""
        previous = self._return_focus
        if self.has_focus_within and previous is not None and previous.is_attached:
            self.screen.set_focus(previous)
        return super().remove()


def drawer_holding(widget: Widget | None) -> Drawer | None:
    """The drawer that contains *widget*, or None when it sits outside every drawer."""
    if widget is None:
        return None
    return next((node for node in widget.ancestors_with_self if isinstance(node, Drawer)), None)
