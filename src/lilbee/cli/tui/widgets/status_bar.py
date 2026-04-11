"""ViewTabs — view tab strip with mode indicator."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from lilbee.cli.tui import messages as msg

_MODE_STYLES: dict[str, str] = {
    msg.MODE_NORMAL: "bold white on dark_blue",
    msg.MODE_INSERT: "bold white on dark_green",
}

_DEFAULT_MODE_STYLE = "bold white on dark_red"


class ViewTabs(Widget):
    """View tab strip with mode indicator."""

    DEFAULT_CSS = """
    ViewTabs {
        dock: bottom;
        height: 1;
        width: 100%;
        background: $surface;
    }
    ViewTabs > Static {
        width: auto;
    }
    """
    active_view: reactive[str] = reactive(msg.DEFAULT_VIEW)
    mode_text: reactive[str] = reactive("")

    def compose(self) -> ComposeResult:
        yield Static(id="view-tabs-content")

    def on_mount(self) -> None:
        self.active_view = getattr(self.app, "active_view", msg.DEFAULT_VIEW)
        self._refresh()

    def watch_active_view(self, value: str) -> None:
        self._refresh()

    def watch_mode_text(self, value: str) -> None:
        self._refresh()

    def _refresh(self) -> None:
        parts: list[str] = []
        if self.mode_text:
            style = _MODE_STYLES.get(self.mode_text, _DEFAULT_MODE_STYLE)
            parts.append(f"[{style}] {self.mode_text} [/] ")
        for name in msg.get_nav_views():
            if name == self.active_view:
                parts.append(f"[bold reverse] {name} [/]")
            else:
                parts.append(f" [dim]{name}[/] ")
        self.query_one("#view-tabs-content", Static).update("".join(parts))
