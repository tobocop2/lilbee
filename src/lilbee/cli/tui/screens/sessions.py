"""Full-screen Sessions view: browse and manage saved conversations."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.screen import Screen

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.widgets.session_list import SessionListPanel

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp


class SessionsScreen(Screen[None]):
    """Manage saved conversations: filter, resume, rename, delete, new."""

    app: LilbeeApp  # type: ignore[assignment]

    CSS_PATH = "sessions.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "go_back", "Back", show=True),
    ]

    def compose(self) -> ComposeResult:
        from textual.widgets import Footer

        from lilbee.cli.tui.widgets.bottom_bars import BottomBars
        from lilbee.cli.tui.widgets.status_bar import ViewTabs
        from lilbee.cli.tui.widgets.task_bar import TaskBar
        from lilbee.cli.tui.widgets.top_bars import TopBars

        with TopBars():
            yield ViewTabs()
        yield SessionListPanel(focus_filter=False)
        with BottomBars():
            yield TaskBar()
            yield Footer()

    def action_go_back(self) -> None:
        self.app.switch_view(msg.DEFAULT_VIEW)

    @on(SessionListPanel.Resumed)
    def _on_resumed(self, event: SessionListPanel.Resumed) -> None:
        self.app.resume_session(event.session_id)

    @on(SessionListPanel.NewChat)
    def _on_new_chat(self, _event: SessionListPanel.NewChat) -> None:
        self.app.new_chat()

    @on(SessionListPanel.CloseRequested)
    def _on_close(self, _event: SessionListPanel.CloseRequested) -> None:
        self.app.switch_view(msg.DEFAULT_VIEW)
