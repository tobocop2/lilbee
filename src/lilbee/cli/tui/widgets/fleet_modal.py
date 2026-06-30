"""Fleet modal overlay: FleetBody hosted in a dismissible ModalScreen."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.screen import ModalScreen

from lilbee.cli.tui.widgets.fleet_body import FleetBody


class FleetModal(ModalScreen[None]):
    """GPU fleet panel as a modal overlay, dismissed with Escape."""

    CSS_PATH = "fleet_modal.tcss"
    AUTO_FOCUS = "#placement-gpus"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Close", show=False),
        # priority so the editor controls don't swallow ctrl+r/s/x
        Binding("ctrl+r", "preview", "Preview", show=True, priority=True),
        Binding("ctrl+s", "apply", "Apply", show=True, priority=True),
        Binding("ctrl+x", "clear", "Auto", show=True, priority=True),
    ]

    def compose(self) -> ComposeResult:
        yield FleetBody()

    def action_cancel(self) -> None:
        """Dismiss the overlay, returning to the screen underneath."""
        self.dismiss()

    def action_preview(self) -> None:
        """Delegate preview to FleetBody."""
        self.query_one(FleetBody).action_preview()

    def action_apply(self) -> None:
        """Delegate apply to FleetBody."""
        self.query_one(FleetBody).action_apply()

    def action_clear(self) -> None:
        """Delegate clear (auto) to FleetBody."""
        self.query_one(FleetBody).action_clear()
