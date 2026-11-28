"""Base of every lilbee modal dialog."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from textual.await_complete import AwaitComplete
from textual.containers import Horizontal
from textual.screen import ModalScreen, Screen, ScreenResultType
from textual.widgets import Button

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.mount_guard import MountWhenAttached


def pop_screens_above(screen: Screen[Any], retry: Callable[[], object]) -> bool:
    """Close each screen above *screen*, top first; return whether *screen* is now on top.

    A screen above that has not mounted yet stays open, and *retry* runs once it has
    mounted: Textual still runs the mount handler of a screen closed before its mount,
    and that handler then reads a stack that no longer holds it. A *screen* that is no
    longer on the stack returns False and schedules nothing. The pops are not awaited,
    because one of the screens above can be the caller, and Textual refuses that await.
    """
    app = screen.app
    while screen in app.screen_stack and not screen.is_active:
        top = app.screen
        if not top.is_mounted:
            top.call_later(retry)
            return False
        top.dismiss()
    return screen in app.screen_stack


class CancelButton(Button):
    """A compact button that closes its dialog with no answer."""

    DEFAULT_CSS = """
    CancelButton {
        width: auto;
        min-width: 8;
        height: 1;
        padding: 0 1;
        margin-left: 1;
    }
    """

    def __init__(self, label: str = msg.PROFILE_CANCEL_LABEL) -> None:
        super().__init__(label, id="dialog-cancel", compact=True)
        # The button theme draws a border at every state; inline style beats all of them.
        self.styles.border = ("none", "white")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.screen.dismiss(None)


class DialogFooter(Horizontal):
    """The bottom row of a dialog: a hint that takes the width, then a close button."""

    DEFAULT_CSS = """
    DialogFooter {
        height: auto;
        width: 100%;
    }
    DialogFooter > Static {
        width: 1fr;
    }
    """


class Dialog(MountWhenAttached, ModalScreen[ScreenResultType]):
    """A modal that closes once, and closes no screen but itself and those above it."""

    def dismiss(self, result: ScreenResultType | None = None) -> AwaitComplete:
        """Cancel the screens above the dialog, then close it with *result*.

        A dialog that is already closed does nothing: a second Enter pressed before the
        dialog closes still sends it an answer, and Textual's dismiss pops whichever
        screen is on top.
        """

        def _again() -> None:
            # Not awaited: the retry runs on the screen above, which this dismiss closes.
            self.dismiss(result)

        if not pop_screens_above(self, _again):
            return AwaitComplete()
        return super().dismiss(result)
