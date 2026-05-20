"""Modal that confirms pulling an unsupported-architecture model."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from lilbee.cli.tui import messages as msg


class ConfirmUnsupportedModal(ModalScreen[bool]):
    """Asks the user whether to proceed with an unsupported-arch pull."""

    CSS_PATH: ClassVar[str] = "confirm_unsupported.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "dismiss(False)", "Cancel", show=True),
        Binding("q", "dismiss(False)", "Cancel", show=False),
        Binding("y", "dismiss(True)", "Pull anyway", show=False),
        Binding("n", "dismiss(False)", "Cancel", show=False),
    ]

    def __init__(self, *, architecture: str) -> None:
        super().__init__()
        self._architecture = architecture or "unknown"

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-unsupported-root"):
            yield Static(msg.COMPAT_MODAL_TITLE, id="confirm-unsupported-title")
            yield Static(
                msg.COMPAT_MODAL_BODY.format(arch=self._architecture),
                id="confirm-unsupported-body",
            )
            with Horizontal(id="confirm-unsupported-actions"):
                yield Button(
                    msg.COMPAT_MODAL_CANCEL, id="confirm-unsupported-cancel", variant="default"
                )
                yield Button(
                    msg.COMPAT_MODAL_CONFIRM, id="confirm-unsupported-confirm", variant="warning"
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm-unsupported-confirm")
