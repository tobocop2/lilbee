"""The line above the Settings tabs: the active profile and how many values you set."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Static

from lilbee.cli.tui import messages as msg

if TYPE_CHECKING:
    from lilbee.cli.tui.screens.profile_tab import ProfileSnapshot


class ProfileLinePill(Static, can_focus=True):
    """The profile's name; Enter, Space or a click jumps to the Profile tab."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "jump", "Profile tab", show=False),
        Binding("space", "jump", "Profile tab", show=False),
    ]

    class Jump(Message):
        """Asks the settings screen to show the Profile tab."""

    def action_jump(self) -> None:
        self.post_message(self.Jump())

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.action_jump()


class ProfileLine(Horizontal):
    """Shows the active profile, the count of your values, and a warning when its file is off."""

    def compose(self) -> ComposeResult:
        yield Static(msg.PROFILE_LINE_LABEL, id="profile-line-label")
        yield ProfileLinePill("", id="profile-line-name", markup=False)
        yield Static("", id="profile-line-count")
        yield Static("", id="profile-line-status")

    def show(self, snapshot: ProfileSnapshot) -> None:
        """Fill the line from *snapshot*."""
        self.query_one("#profile-line-name", ProfileLinePill).update(snapshot.active.name)
        count = self.query_one("#profile-line-count", Static)
        count.update(msg.profile_count_text(len(snapshot.active.changes)))
        status = self.query_one("#profile-line-status", Static)
        status_text = msg.PROFILE_STATUS_PILL[snapshot.active.status]
        status.update(status_text)
        status.display = bool(status_text)
