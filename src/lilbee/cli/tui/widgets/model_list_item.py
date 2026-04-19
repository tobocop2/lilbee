"""ModelListItem: single-row widget for the catalog list view.

The list view mirrors the grid view's visual language (pills, dim specs)
but at one row per model instead of cards in a grid. Each item renders
two lines: bold name plus task/backend/featured pills on line 1, dim
`params | quant | size` plus `downloads` or `installed` pill on line 2.
Focusable so the screen can drive keyboard navigation with j/k/g/G.
Clicking or pressing enter posts a Selected message the screen catches.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from textual import containers, widgets
from textual.app import ComposeResult
from textual.binding import Binding
from textual.content import Content
from textual.events import Click
from textual.message import Message

from lilbee.cli.tui.pill import pill
from lilbee.cli.tui.widgets.catalog_theme import MIDDLE_DOT, TASK_COLORS
from lilbee.models import FEATURED_STAR

if TYPE_CHECKING:
    from lilbee.cli.tui.screens.catalog import TableRow

_CSS_FILE = Path(__file__).parent / "model_list_item.tcss"


class ModelListItem(containers.VerticalGroup, can_focus=True):
    """A single model row, rendered as two stacked lines."""

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

    BINDINGS: ClassVar = [Binding("enter", "select", "Select", show=False)]

    @dataclass
    class Selected(Message):
        """Posted when the user activates a list item via click or Enter."""

        item: ModelListItem

        @property
        def control(self) -> ModelListItem:
            return self.item

    def __init__(self, row: TableRow) -> None:
        self._row = row
        super().__init__()

    @property
    def row(self) -> TableRow:
        return self._row

    def action_select(self) -> None:
        self.post_message(self.Selected(self))

    def on_click(self, event: Click) -> None:
        event.stop()
        self.focus()
        self.post_message(self.Selected(self))

    def compose(self) -> ComposeResult:
        row = self._row
        yield widgets.Static(_build_head(row), id="list-head")
        with containers.HorizontalGroup(id="list-meta"):
            yield widgets.Label(_build_specs(row.params, row.quant, row.size), id="list-specs")
            status = _build_status(row)
            if status is not None:
                yield widgets.Label(status, id="list-status")


def _build_head(row: TableRow) -> Content:
    """Bold name plus task/backend pills, featured-star prefix if applicable."""
    bg = TASK_COLORS.get(row.task, "$primary")
    parts: list[Content] = []
    if row.featured:
        parts.append(Content.styled(f"{FEATURED_STAR} ", "$warning"))
    parts.append(Content.styled(row.name, "bold"))
    parts.append(Content("  "))
    parts.append(pill(row.task, bg, "$text"))
    if row.backend:
        parts.append(Content(" "))
        parts.append(pill(row.backend, "$accent", "$text"))
    return Content.assemble(*parts)


def _build_specs(params: str, quant: str, size: str) -> Content:
    """Build the specs line: params middle-dot quant middle-dot size."""
    parts = [p for p in (params, quant, size) if p and p != "--"]
    if not parts:
        return Content.styled("--", "$text-muted")
    return Content.styled(f" {MIDDLE_DOT} ".join(parts), "$text-muted")


def _build_status(row: TableRow) -> Content | None:
    """Build the status pill for installed or download count."""
    if row.installed:
        return pill("installed", "$success", "$text")
    if row.sort_downloads > 0:
        return Content.styled(f"↓ {row.downloads}", "$text-muted")
    return None
