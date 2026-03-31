"""Navigation bar — cmus-style numbered view tabs."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

_VIEWS = ["Chat", "Models", "Status", "Settings"]


class NavBar(Widget):
    """Persistent tab bar showing numbered views. Active view is highlighted."""

    DEFAULT_CSS = """
    NavBar {
        dock: bottom;
        height: 1;
        background: $surface;
    }
    NavBar > Static {
        width: auto;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("h", "view_left", "Left", show=False),
        Binding("left", "view_left", "Left", show=False),
        Binding("l", "view_right", "Right", show=False),
        Binding("right", "view_right", "Right", show=False),
        Binding("1", "go_to_view_0", "1", show=False),
        Binding("2", "go_to_view_1", "2", show=False),
        Binding("3", "go_to_view_2", "3", show=False),
        Binding("4", "go_to_view_3", "4", show=False),
    ]

    active_view: reactive[str] = reactive("Chat")

    def compose(self) -> ComposeResult:
        yield Static(id="nav-bar-content")

    def watch_active_view(self, value: str) -> None:
        self._refresh_display()

    def on_mount(self) -> None:
        self._refresh_display()

    def _refresh_display(self) -> None:
        parts: list[str] = []
        for i, name in enumerate(_VIEWS, 1):
            if name == self.active_view:
                parts.append(f"[bold reverse] {i}:{name} [/]")
            else:
                parts.append(f" {i}:{name} ")
        parts.append("  [dim]?[/] Help  [dim]^c[/] Quit")
        content = self.query_one("#nav-bar-content", Static)
        content.update("".join(parts))

    def _change_view(self, view_name: str) -> None:
        """Change to a different view and navigate to its screen."""
        if view_name not in _VIEWS:
            return
        self.active_view = view_name
        if view_name == "Chat":
            self.app.pop_screen()
            return
        from lilbee.cli.tui.screens.catalog import CatalogScreen
        from lilbee.cli.tui.screens.settings import SettingsScreen
        from lilbee.cli.tui.screens.status import StatusScreen

        if view_name == "Models":
            self.app.push_screen(CatalogScreen())
        elif view_name == "Status":
            self.app.push_screen(StatusScreen())
        elif view_name == "Settings":
            self.app.push_screen(SettingsScreen())

    def action_view_left(self) -> None:
        """Navigate to the previous view (h or left arrow)."""
        current_idx = _VIEWS.index(self.active_view)
        prev_idx = (current_idx - 1) % len(_VIEWS)
        self._change_view(_VIEWS[prev_idx])

    def action_view_right(self) -> None:
        """Navigate to the next view (l or right arrow)."""
        current_idx = _VIEWS.index(self.active_view)
        next_idx = (current_idx + 1) % len(_VIEWS)
        self._change_view(_VIEWS[next_idx])

    def action_go_to_view_0(self) -> None:
        self._change_view(_VIEWS[0])

    def action_go_to_view_1(self) -> None:
        self._change_view(_VIEWS[1])

    def action_go_to_view_2(self) -> None:
        self._change_view(_VIEWS[2])

    def action_go_to_view_3(self) -> None:
        self._change_view(_VIEWS[3])
