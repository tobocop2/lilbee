"""Reusable confirmation modal dialog."""

from __future__ import annotations

from typing import ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Center, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static


class ConfirmDialog(ModalScreen[bool]):
    """Modal yes/no dialog that returns True (confirmed) or False (cancelled)."""

    CSS_PATH = "confirm_dialog.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("y", "confirm", "Yes", show=True),
        # Fallback when no button holds focus; a focused button consumes
        # enter itself, so tabbing to No and pressing enter cancels.
        Binding("enter", "confirm", "Confirm", show=False),
        Binding("n", "cancel", "No", show=True),
        Binding("escape", "cancel", "Cancel", show=False),
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
                yield Button("Yes (y)", id="confirm-yes")
                yield Button("No (n)", id="confirm-no")

    @on(Button.Pressed)
    def _on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(event.button.id == "confirm-yes")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)
