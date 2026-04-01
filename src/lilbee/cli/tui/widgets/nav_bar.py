"""Navigation bar — view tabs with optional task status indicator."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from lilbee.cli.tui import messages as msg

_VIEWS = msg.NAV_VIEWS


class NavBar(Widget):
    """Persistent tab bar showing views. Active view is highlighted."""

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

    active_view: reactive[str] = reactive("Chat")
    active_task_text: reactive[str] = reactive("")

    def compose(self) -> ComposeResult:
        yield Static(id="nav-bar-content")

    def watch_active_view(self, value: str) -> None:
        self._refresh_display()

    def watch_active_task_text(self, value: str) -> None:
        self._refresh_display()

    def on_mount(self) -> None:
        if hasattr(self.app, "_active_view"):
            self.active_view = self.app._active_view  # type: ignore[union-attr]
        self._refresh_display()

    def _refresh_display(self) -> None:
        parts: list[str] = []
        for name in _VIEWS:
            if name == self.active_view:
                parts.append(f"[bold reverse] {name} [/]")
            else:
                parts.append(f" {name} ")
        if self.active_task_text:
            parts.append(f"  [bold]{self.active_task_text}[/]")
        parts.append(msg.NAV_HELP_QUIT)
        content = self.query_one("#nav-bar-content", Static)
        content.update("".join(parts))
