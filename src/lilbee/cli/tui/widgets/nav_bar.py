"""Navigation bar — view tabs with optional task status indicator."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.events import Click
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from lilbee.cli.tui import messages as msg

_VIEWS = msg.NAV_VIEWS


class NavBar(Widget):
    """Persistent tab bar showing views. Active view is highlighted."""

    DEFAULT_CSS = """
    NavBar {
        dock: top;
        height: 1;
        background: $surface;
    }
    NavBar > Static {
        width: auto;
    }
    """

    active_view: reactive[str] = reactive("Chat")
    active_task_text: reactive[str] = reactive("")
    mode_text: reactive[str] = reactive("")

    def compose(self) -> ComposeResult:
        yield Static(id="nav-bar-content")

    def watch_active_view(self, value: str) -> None:
        self._refresh_display()

    def watch_active_task_text(self, value: str) -> None:
        self._refresh_display()

    def watch_mode_text(self, value: str) -> None:
        self._refresh_display()

    def on_mount(self) -> None:
        self.active_view = getattr(self.app, "active_view", "Chat")
        self._refresh_display()

    def _refresh_display(self) -> None:
        parts: list[str] = []
        if self.mode_text:
            if "NORMAL" in self.mode_text:
                parts.append(f"[bold white on dark_blue] {self.mode_text} [/] ")
            else:
                parts.append(f"[bold white on dark_green] {self.mode_text} [/] ")
        for name in _VIEWS:
            if name == self.active_view:
                parts.append(f"[bold reverse] {name} [/]")
            elif name == "Tasks" and self.active_task_text:
                parts.append(f" [bold yellow]{name}[/] ")
            else:
                parts.append(f" {name} ")
        if self.active_task_text:
            parts.append(f"  [bold]{self.active_task_text}[/]")
        parts.append(msg.NAV_HELP_QUIT)
        content = self.query_one("#nav-bar-content", Static)
        content.update("".join(parts))

    def on_click(self, event: Click) -> None:
        """Switch view when a view name in the bar is clicked."""
        from lilbee.cli.tui.app import LilbeeApp

        view = _view_at_x(event.x)
        if view is not None and isinstance(self.app, LilbeeApp):
            self.app.switch_view(view)


def _view_regions() -> list[tuple[int, int, str]]:
    """Return (start_x, end_x, view_name) for each view label.

    Each label is rendered as `` {name} `` (space-padded), so the
    width of each segment is ``len(name) + 2``.
    """
    regions: list[tuple[int, int, str]] = []
    offset = 0
    for name in _VIEWS:
        width = len(name) + 2  # leading + trailing space
        regions.append((offset, offset + width, name))
        offset += width
    return regions


def _view_at_x(x: int) -> str | None:
    """Return the view name at column *x*, or None if outside view labels."""
    for start, end, name in _view_regions():
        if start <= x < end:
            return name
    return None
