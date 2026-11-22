"""Base of every lilbee modal dialog."""

from __future__ import annotations

from textual.await_complete import AwaitComplete
from textual.screen import ModalScreen, ScreenResultType


class Dialog(ModalScreen[ScreenResultType]):
    """A modal that closes once, and closes no screen but itself and those above it."""

    def dismiss(self, result: ScreenResultType | None = None) -> AwaitComplete:
        """Cancel the screens above the dialog, then close it with *result*.

        A dialog that is already closed does nothing: a second Enter pressed before the
        dialog closes still sends it an answer, and Textual's dismiss pops whichever
        screen is on top. The screens above are popped with no await, because one of them
        can be the caller, and Textual refuses that await.
        """
        if self not in self.app.screen_stack:
            return AwaitComplete()
        while not self.is_active:
            self.app.screen.dismiss()
        return super().dismiss(result)
