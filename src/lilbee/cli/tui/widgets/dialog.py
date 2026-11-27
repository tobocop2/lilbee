"""Base of every lilbee modal dialog."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from textual.await_complete import AwaitComplete
from textual.screen import ModalScreen, Screen, ScreenResultType

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
