"""Full-screen Sessions view: browse and manage saved conversations."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import BindingType
from textual.screen import Screen

from lilbee.cli.tui.browse_bindings import BROWSE_LIST_BINDINGS, browse_back_bindings
from lilbee.cli.tui.widgets.session_list import SessionListPanel

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp


class SessionsScreen(Screen[None]):
    """Manage saved conversations: filter, resume, rename, delete, new."""

    app: LilbeeApp  # type: ignore[assignment]

    CSS_PATH = "sessions.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        *browse_back_bindings(),
        *BROWSE_LIST_BINDINGS,
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
        self.app.go_back()

    def action_cursor_down(self) -> None:
        self.query_one(SessionListPanel).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one(SessionListPanel).action_cursor_up()

    def action_jump_top(self) -> None:
        self.query_one(SessionListPanel).jump_to(0)

    def action_jump_bottom(self) -> None:
        self.query_one(SessionListPanel).jump_to(-1)

    @on(SessionListPanel.Resumed)
    def _on_resumed(self, event: SessionListPanel.Resumed) -> None:
        self.app.resume_session(event.session_id)

    @on(SessionListPanel.NewChat)
    def _on_new_chat(self, _event: SessionListPanel.NewChat) -> None:
        self.app.new_chat()

    @on(SessionListPanel.CloseRequested)
    def _on_close(self, _event: SessionListPanel.CloseRequested) -> None:
        # Same semantics as q: return to where the user came from.
        self.app.go_back()
