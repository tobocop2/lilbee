"""ModelCard — compact card widget for the catalog grid view."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual import containers, widgets
from textual.app import ComposeResult
from textual.content import Content
from textual.reactive import reactive

from lilbee.cli.tui.pill import pill
from lilbee.models import ModelTask

if TYPE_CHECKING:
    from lilbee.cli.tui.screens.catalog import TableRow

MIDDLE_DOT = "·"

_TASK_COLORS: dict[str, str] = {
    ModelTask.CHAT: "$primary",
    ModelTask.EMBEDDING: "$secondary",
    ModelTask.VISION: "$warning",
}


class ModelCard(containers.VerticalGroup):
    """A single model card displaying name, task pill, specs, and status."""

    DEFAULT_CSS = """
    ModelCard {
        height: auto;
        border: tall $surface-lighten-2;
        padding: 0 1;
        pointer: pointer;
    }
    ModelCard:hover {
        background: $panel;
    }
    ModelCard #card-header {
        grid-size: 3 1;
        grid-columns: 1fr auto auto;
        height: auto;
    }
    ModelCard #card-name {
        text-style: bold;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }
    ModelCard #card-pick {
        width: auto;
        margin: 0 1 0 0;
    }
    ModelCard #card-task {
        text-align: right;
    }
    ModelCard #card-info {
        text-style: dim;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }
    ModelCard #card-status {
        text-style: dim;
    }
    """

    selected: reactive[bool] = reactive(False)

    def __init__(self, row: TableRow) -> None:
        self._row = row
        super().__init__()

    @property
    def row(self) -> TableRow:
        return self._row

    def watch_selected(self, selected: bool) -> None:
        self.set_class(selected, "-selected")

    def compose(self) -> ComposeResult:
        row = self._row
        bg = _TASK_COLORS.get(row.task, "$primary")
        with containers.Grid(id="card-header"):
            yield widgets.Label(row.name, id="card-name")
            if row.featured:
                yield widgets.Label(pill("pick", "$warning", "$text"), id="card-pick")
            yield widgets.Label(pill(row.task, bg, "$text"), id="card-task")
        specs = _build_specs(row.params, row.quant, row.size)
        yield widgets.Label(specs, id="card-info")
        status = _build_status(row)
        if status is not None:
            yield widgets.Label(status, id="card-status")


def _build_specs(params: str, quant: str, size: str) -> Content:
    """Build the specs line: params · quant · size."""
    parts = [p for p in (params, quant, size) if p and p != "--"]
    if not parts:
        return Content("--")
    return Content(f" {MIDDLE_DOT} ".join(parts))


def _build_status(row: TableRow) -> Content | None:
    """Build the status pill for installed or download count."""
    if row.installed:
        return pill("installed", "$success", "$text")
    if row.sort_downloads > 0:
        return Content.styled(f"↓ {row.downloads}", "$text-muted")
    return None
