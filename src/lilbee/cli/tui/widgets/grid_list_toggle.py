"""Visible Grid ↔ List toggle for the catalog screen.

Click or focus+Enter on either half flips the view; keeps the existing
``v`` keybinding. Renders as plain bold-vs-muted text with a dot
divider, matching the Search/Chat toggle aesthetic.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual import events
from textual.binding import Binding, BindingType
from textual.content import Content
from textual.widgets import Static

from lilbee.cli.tui import messages as msg

_CSS_FILE = Path(__file__).parent / "grid_list_toggle.tcss"


class GridListToggle(Static, can_focus=True):
    """Flip the catalog between grid and list views; mirrors the ``v`` binding."""

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "flip", "Toggle view", show=False),
        Binding("space", "flip", "Toggle view", show=False),
        Binding("left", "select_grid", "Grid view", show=False),
        Binding("right", "select_list", "List view", show=False),
    ]

    def __init__(self) -> None:
        super().__init__(id="grid-list-toggle")
        self._is_grid: bool = True

    def set_grid(self, is_grid: bool) -> None:
        """Sync the active half to *is_grid* and repaint."""
        self._is_grid = is_grid
        if self.is_mounted:
            self._refresh()

    def on_mount(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        grid_label = self._render_label(msg.CATALOG_VIEW_GRID, active=self._is_grid)
        list_label = self._render_label(msg.CATALOG_VIEW_LIST, active=not self._is_grid)
        divider = Content.styled(" · ", "$text-muted")
        self.update(Content.assemble(grid_label, divider, list_label))

    @staticmethod
    def _render_label(label: str, *, active: bool) -> Content:
        if active:
            return Content.styled(label, "bold $primary")
        return Content.styled(label, "$text-muted")

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self._call_screen_toggle()

    def action_flip(self) -> None:
        self._call_screen_toggle()

    def action_select_grid(self) -> None:
        if not self._is_grid:
            self._call_screen_toggle()

    def action_select_list(self) -> None:
        if self._is_grid:
            self._call_screen_toggle()

    def _call_screen_toggle(self) -> None:
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        screen = self.screen
        if isinstance(screen, CatalogScreen):
            screen.action_toggle_view()
