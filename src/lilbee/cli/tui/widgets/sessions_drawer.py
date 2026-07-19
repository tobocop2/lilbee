"""Sessions side drawer: the session list docked left so chat stays live.

Non-modal, mirroring the Fleet drawer. Docked left so the screen underneath
reflows to the right and the chat prompt keeps working while the drawer is open.
Resume, new-chat, and close come from the embedded panel as messages.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from textual import on
from textual.app import ComposeResult
from textual.screen import Screen

from lilbee.cli.tui.widgets.drawer import Drawer
from lilbee.cli.tui.widgets.session_list import SessionListPanel

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp

_DRAWER_CSS = (Path(__file__).parent / "sessions_drawer.tcss").read_text(encoding="utf-8")

# CSS class set on the host screen so app.tcss insets the docked bars to the
# space right of the drawer.
_HOST_OPEN_CLASS = "sessions-open"


class SessionsDrawer(Drawer):
    """Session switcher as a non-modal left-side drawer; closed with esc or ctrl+o."""

    DEFAULT_CSS: ClassVar[str] = _DRAWER_CSS

    app: LilbeeApp  # type: ignore[assignment]

    def __init__(self) -> None:
        super().__init__(id="sessions-drawer")
        self._host: Screen[object] | None = None

    def compose(self) -> ComposeResult:
        yield SessionListPanel()

    def on_mount(self) -> None:
        """Inset the host screen's docked bars so the drawer sits beside them."""
        self._host = self.screen
        self._host.add_class(_HOST_OPEN_CLASS)

    def on_unmount(self) -> None:
        """Restore the bars to full width when the drawer closes by any path."""
        if self._host is not None:
            self._host.remove_class(_HOST_OPEN_CLASS)

    @on(SessionListPanel.Resumed)
    def _on_resumed(self, event: SessionListPanel.Resumed) -> None:
        self.app.resume_session(event.session_id)
        self.remove()

    @on(SessionListPanel.NewChat)
    def _on_new_chat(self, _event: SessionListPanel.NewChat) -> None:
        self.app.new_chat()
        self.remove()

    @on(SessionListPanel.CloseRequested)
    def _on_close(self, _event: SessionListPanel.CloseRequested) -> None:
        self.remove()
