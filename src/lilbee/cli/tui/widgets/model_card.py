"""ModelCard — compact card widget for the catalog grid view."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from textual import containers, widgets
from textual.app import ComposeResult
from textual.content import Content
from textual.reactive import reactive

from lilbee.cli.tui.pill import pill
from lilbee.models import ModelTask

if TYPE_CHECKING:
    from lilbee.cli.tui.screens.catalog import TableRow

_CSS_FILE = Path(__file__).parent / "model_card.tcss"

MIDDLE_DOT = "·"

_TASK_COLORS: dict[str, str] = {
    ModelTask.CHAT: "$primary",
    ModelTask.EMBEDDING: "$secondary",
    ModelTask.VISION: "$warning",
}


class ModelCard(containers.VerticalGroup):
    """A single model card displaying name, task pill, specs, and status."""

    # Widget CSS lives in model_card.tcss so it gets syntax highlighting and
    # matches the convention used for screens. Textual's Widget class only
    # supports DEFAULT_CSS (there is no widget-level CSS_PATH), so we load the
    # file once at import time.
    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

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
        from lilbee.cli.tui import messages as msg

        row = self._row
        bg = _TASK_COLORS.get(row.task, "$primary")
        yield widgets.Label(row.name, id="card-name")
        with containers.HorizontalGroup(id="card-pills"):
            if row.featured:
                yield widgets.Label(pill("pick", "$warning", "$text"), id="card-pick")
            yield widgets.Label(pill(row.task, bg, "$text"), id="card-task")
            if row.backend:
                yield widgets.Label(pill(row.backend, "$accent", "$text"), id="card-backend")
        specs = _build_specs(row.params, row.quant, row.size)
        yield widgets.Label(specs, id="card-info")
        status = _build_status(row)
        if status is not None:
            yield widgets.Label(status, id="card-status")
        # Subtle "Enter to install" hint; CSS shows it only when the card
        # is highlighted (GridSelect cursor), hides for installed cards.
        if not row.installed:
            yield widgets.Label(msg.SETUP_CARD_HINT, id="card-hint")


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
