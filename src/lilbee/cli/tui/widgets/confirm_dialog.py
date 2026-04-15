"""Reusable confirmation modal dialog."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Center, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static


class ConfirmDialog(ModalScreen[bool]):
    """Modal yes/no dialog that returns True (confirmed) or False (cancelled)."""

    DEFAULT_CSS = """
    ConfirmDialog {
        align: center middle;
    }
    ConfirmDialog > Vertical {
        width: 50;
        height: auto;
        max-height: 12;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }
    ConfirmDialog #confirm-title {
        text-style: bold;
        width: 100%;
        content-align: center middle;
        margin-bottom: 1;
    }
    ConfirmDialog #confirm-message {
        width: 100%;
        content-align: center middle;
        margin-bottom: 1;
    }
    ConfirmDialog Center {
        width: 100%;
        height: auto;
    }
    ConfirmDialog Button {
        margin: 0 1;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("y", "confirm", "Yes", show=True),
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
            with Center():
                yield Button("Yes (y)", variant="error", id="confirm-yes")
                yield Button("No (n)", variant="default", id="confirm-no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm-yes")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)
