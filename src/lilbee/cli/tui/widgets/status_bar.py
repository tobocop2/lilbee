"""ViewTabs — view tab strip with mode indicator."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.content import Content
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.pill import DOT_SEP, pill

_MODE_COLORS: dict[str, str] = {
    msg.MODE_NORMAL: "$primary",
    msg.MODE_INSERT: "$success",
}

_DEFAULT_MODE_COLOR = "$error"


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
        if not self.is_mounted:
            return
        parts: list[Content | str | tuple[str, str]] = []

        # Build tab strip
        tab_parts: list[Content | str | tuple[str, str]] = []
        for name in msg.get_nav_views():
            if name == self.active_view:
                tab_parts.append(pill(f" {name} ", "$primary", "$text"))
            else:
                tab_parts.append((f" {name} ", "dim"))
        # Join with dot separators
        joined: list[Content | str | tuple[str, str]] = []
        for i, part in enumerate(tab_parts):
            if i > 0:
                joined.append((DOT_SEP, "$text-muted"))
            joined.append(part)
        parts.extend(joined)

        # Mode pill (right-aligned)
        if self.mode_text:
            color = _MODE_COLORS.get(self.mode_text, _DEFAULT_MODE_COLOR)
            parts.append("  ")
            parts.append(pill(f" {self.mode_text} ", color, "$text"))

        self.query_one("#view-tabs-content", Static).update(Content.assemble(*parts))
