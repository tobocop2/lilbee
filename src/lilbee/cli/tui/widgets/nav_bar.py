"""Navigation bar — cmus-style numbered view tabs."""

from __future__ import annotations

from textual.app import ComposeResult
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
