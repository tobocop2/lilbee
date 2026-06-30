"""Fleet screen: thin host for FleetBody (GPU table + placement editor)."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.screen import Screen

from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.widgets.fleet_body import FleetBody


class FleetScreen(Screen[None]):
    """GPU fleet viewer and interactive placement editor."""

    # Lilbee always hosts screens on a LilbeeApp, so narrowing the type lets
    # the screen call switch_view without per-call type: ignore comments.
    app: LilbeeApp  # type: ignore[assignment]

    CSS_PATH = "fleet.tcss"
    AUTO_FOCUS = "#placement-gpus"
    HELP = "Configure GPU placement. ctrl+r preview, ctrl+s apply, ctrl+x auto, q back."

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "go_back", "Back", show=True),
        Binding("escape", "go_back", "Back", show=False),
        # priority so they fire even when a button/editor child has focus.
        Binding("ctrl+r", "preview", "Preview", show=True, priority=True),
        Binding("ctrl+s", "apply", "Apply", show=True, priority=True),
        Binding("ctrl+x", "clear", "Auto", show=True, priority=True),
    ]

    def compose(self) -> ComposeResult:
        from textual.widgets import Footer

        from lilbee.cli.tui.widgets.bottom_bars import BottomBars
        from lilbee.cli.tui.widgets.status_bar import ViewTabs
        from lilbee.cli.tui.widgets.task_bar import TaskBar
        from lilbee.cli.tui.widgets.top_bars import TopBars

        with TopBars():
            yield ViewTabs()
        yield FleetBody()
        with BottomBars():
            yield TaskBar()
            yield Footer()

    def action_preview(self) -> None:
        """Delegate preview to FleetBody."""
        self.query_one(FleetBody).action_preview()

    def action_apply(self) -> None:
        """Delegate apply to FleetBody."""
        self.query_one(FleetBody).action_apply()

    def action_clear(self) -> None:
        """Delegate clear (auto) to FleetBody."""
        self.query_one(FleetBody).action_clear()

    def action_go_back(self) -> None:
        """Return to Chat via the guarded switch_view (inverse of Fleet entry)."""
        self.app.switch_view("Chat")
