"""Base of every lilbee modal dialog."""

from __future__ import annotations

from typing import Any

from textual.await_complete import AwaitComplete
from textual.screen import ModalScreen, Screen, ScreenResultType


def pop_screens_above(screen: Screen[Any]) -> None:
    """Close each screen above *screen*, top first, so that *screen* is on top.

    The pops are not awaited, because one of the screens above can be the caller,
    and Textual refuses that await.
    """
    while not screen.is_active:
        screen.app.screen.dismiss()


class Dialog(ModalScreen[ScreenResultType]):
    """A modal that closes once, and closes no screen but itself and those above it."""

    def dismiss(self, result: ScreenResultType | None = None) -> AwaitComplete:
        """Cancel the screens above the dialog, then close it with *result*.

        A dialog that is already closed does nothing: a second Enter pressed before the
        dialog closes still sends it an answer, and Textual's dismiss pops whichever
        screen is on top.
        """
        if self not in self.app.screen_stack:
            return AwaitComplete()
        pop_screens_above(self)
        return super().dismiss(result)
