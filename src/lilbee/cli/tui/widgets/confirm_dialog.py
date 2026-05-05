"""Reusable confirmation modal dialog."""

from __future__ import annotations

from typing import ClassVar

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Center, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, Static


class _ConfirmPill(Static, can_focus=True):
    """Pill-styled clickable label used as Yes/No in :class:`ConfirmDialog`."""

    def __init__(self, label: str, *, dialog_id: str) -> None:
        super().__init__(label, id=dialog_id, classes="confirm-pill")
        self._dialog_id = dialog_id

    def on_click(self, event: events.Click) -> None:
        event.stop()
        screen = self.screen
        if isinstance(screen, ConfirmDialog):
            screen.dismiss(self._dialog_id == "confirm-yes")


class ConfirmDialog(ModalScreen[bool]):
    """Modal yes/no dialog that returns True (confirmed) or False (cancelled)."""

    CSS_PATH = "confirm_dialog.tcss"

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
            with Center(), Horizontal(id="confirm-buttons"):
                yield _ConfirmPill("Yes (y)", dialog_id="confirm-yes")
                yield _ConfirmPill("No (n)", dialog_id="confirm-no")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)
