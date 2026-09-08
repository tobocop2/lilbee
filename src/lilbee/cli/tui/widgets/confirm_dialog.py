"""Reusable confirmation modal dialog."""

from __future__ import annotations

from typing import ClassVar

from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Center, Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Label, Static

from lilbee.cli.tui import messages as msg


class ConfirmPill(Static, can_focus=True):
    """Focusable yes/no pill; Enter, Space or a click picks it."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "select", "Pick", show=False),
        Binding("space", "select", "Pick", show=False),
    ]

    class Picked(Message):
        """A pill was picked, carrying the answer it stands for."""

        def __init__(self, answer: bool) -> None:
            super().__init__()
            self.answer = answer

    def __init__(self, label: str, *, pill_id: str, answer: bool) -> None:
        super().__init__(label, id=pill_id)
        self._answer = answer

    def action_select(self) -> None:
        self.post_message(self.Picked(self._answer))

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.action_select()


class ConfirmDialog(ModalScreen[bool]):
    """Modal yes/no dialog that returns True (confirmed) or False (cancelled)."""

    CSS_PATH = "confirm_dialog.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("y", "confirm", "Yes", show=True),
        # Fallback for when no pill holds focus; a focused pill takes enter itself.
        Binding("enter", "confirm", "Confirm", show=False),
        Binding("n", "cancel", "No", show=True),
        Binding("escape", "cancel", "Cancel", show=False),
        # `app.` because the focus actions live on the App, as Screen's own
        # tab binding has them.
        Binding("left", "app.focus_previous", "Previous", show=False),
        Binding("right", "app.focus_next", "Next", show=False),
    ]

    def __init__(self, title: str, message: str) -> None:
        super().__init__()
        self._title = title
        self._message = message

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self._title, id="confirm-title")
            yield Label(self._message, id="confirm-message")
            with Center(), Horizontal(id="confirm-buttons"):
                yield ConfirmPill(msg.CONFIRM_YES_LABEL, pill_id="confirm-yes", answer=True)
                yield ConfirmPill(msg.CONFIRM_NO_LABEL, pill_id="confirm-no", answer=False)

    @on(ConfirmPill.Picked)
    def _on_pill_picked(self, event: ConfirmPill.Picked) -> None:
        event.stop()
        self.dismiss(event.answer)

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)
