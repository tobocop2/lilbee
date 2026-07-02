"""Fleet side drawer: FleetBody docked right so chat stays live on the left."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical

from lilbee.cli.tui.widgets.fleet_body import FleetBody

_CSS_FILE = Path(__file__).parent / "fleet_drawer.tcss"
_DRAWER_CSS = _CSS_FILE.read_text(encoding="utf-8")


class FleetDrawer(Vertical):
    """GPU fleet panel as a non-modal right-side drawer; closed with esc or ctrl+g.

    Docked to the right so the screen underneath reflows to the left and stays
    interactive -- the chat prompt keeps working while the live GPU bars move in
    the drawer. The preview/apply/clear keys are scoped here, so they fire only
    while focus is inside the drawer, never from the chat prompt.
    """

    DEFAULT_CSS: ClassVar[str] = _DRAWER_CSS

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Close", show=False),
        # priority so a focused toggle child doesn't swallow ctrl+r/s/x
        Binding("ctrl+r", "preview", "Preview", show=True, priority=True),
        Binding("ctrl+s", "apply", "Apply", show=True, priority=True),
        Binding("ctrl+x", "clear", "Auto", show=True, priority=True),
    ]

    def __init__(self) -> None:
        super().__init__(id="fleet-drawer")

    def compose(self) -> ComposeResult:
        yield FleetBody()

    # No on_mount focus grab: opening the drawer must NOT steal focus from the
    # chat prompt, so you can keep typing while it stays open. Click a toggle (or
    # tab) to focus the drawer; ctrl+g closes it from anywhere, esc when focused.

    def action_close(self) -> None:
        """Remove the drawer, returning full width to the screen underneath."""
        self.remove()

    def action_preview(self) -> None:
        """Delegate preview (ctrl+r) to the hosted FleetBody."""
        self.query_one(FleetBody).action_preview()

    def action_apply(self) -> None:
        """Delegate apply (ctrl+s) to the hosted FleetBody."""
        self.query_one(FleetBody).action_apply()

    def action_clear(self) -> None:
        """Delegate clear/auto (ctrl+x) to the hosted FleetBody."""
        self.query_one(FleetBody).action_clear()
