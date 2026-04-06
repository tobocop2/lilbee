"""StatusBar — unified bottom bar with mode indicator, view tabs, and hints."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from lilbee.cli.tui import messages as msg

_VIEWS = msg.NAV_VIEWS

_MODE_STYLES: dict[str, str] = {
    msg.MODE_NORMAL: "bold white on dark_blue",
    msg.MODE_INSERT: "bold white on dark_green",
}


class StatusBar(Widget):
    """Persistent bottom bar: mode indicator + view tabs + binding hints."""

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 1;
        background: $surface;
    }
    StatusBar > Static {
        width: auto;
    }
    """

    active_view: reactive[str] = reactive("Chat")
    mode_text: reactive[str] = reactive("")

    def compose(self) -> ComposeResult:
        yield Static(id="status-bar-content")

    def on_mount(self) -> None:
        self.active_view = getattr(self.app, "active_view", "Chat")
        self._refresh()

    def watch_active_view(self, value: str) -> None:
        self._refresh()

    def watch_mode_text(self, value: str) -> None:
        self._refresh()

    def _refresh(self) -> None:
        parts: list[str] = []
        if self.mode_text:
            style = _MODE_STYLES.get(self.mode_text, "bold white on dark_green")
            parts.append(f"[{style}] {self.mode_text} [/] ")
        for name in _VIEWS:
            if name == self.active_view:
                parts.append(f"[bold reverse] {name} [/]")
            else:
                parts.append(f" [dim]{name}[/] ")
        parts.append(msg.STATUS_BAR_HINTS)
        self.query_one("#status-bar-content", Static).update("".join(parts))
