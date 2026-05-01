"""ModelCard: compact card widget for the catalog grid view.

Renders both ``LocalCatalogRow`` (installable / installed GGUFs) and
``FrontierCatalogRow`` (cloud chat models) via type dispatch so the
catalog can show a Frontier section above local sections without a
second widget class.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from textual import containers, widgets
from textual.app import ComposeResult
from textual.content import Content
from textual.reactive import reactive

from lilbee.cli.tui.pill import pill
from lilbee.cli.tui.screens.catalog_utils import (
    CatalogRow,
    FrontierCatalogRow,
    KeyStatus,
    LocalCatalogRow,
)
from lilbee.cli.tui.widgets.catalog_theme import MIDDLE_DOT, TASK_COLORS

if TYPE_CHECKING:
    pass

_CSS_FILE = Path(__file__).parent / "model_card.tcss"


class ModelCard(containers.VerticalGroup):
    """A single model card displaying name, task pill, specs, and status."""

    # Widget CSS lives in model_card.tcss so it gets syntax highlighting and
    # matches the convention used for screens. Textual's Widget class only
    # supports DEFAULT_CSS (there is no widget-level CSS_PATH), so we load the
    # file once at import time.
    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

    selected: reactive[bool] = reactive(False)

    def __init__(self, row: CatalogRow) -> None:
        self._row = row
        super().__init__()
        # The card uses a class to style frontier rows distinctly from
        # local rows in the grid (e.g. accent border, key-status badge).
        if isinstance(row, FrontierCatalogRow):  # frontier branch
            self.add_class("-frontier")

    @property
    def row(self) -> CatalogRow:
        return self._row

    def watch_selected(self, selected: bool) -> None:
        self.set_class(selected, "-selected")

    def compose(self) -> ComposeResult:
        # Dispatch on row type. isinstance is annotated per AGENTS.md
        # so future readers see why a runtime check is the right tool
        # here (sealed union dispatch, not a bandaid).
        row = self._row
        if isinstance(row, FrontierCatalogRow):  # sealed-union dispatch
            yield from _compose_frontier(row)
        else:
            yield from _compose_local(row)


def _compose_local(row: LocalCatalogRow) -> ComposeResult:
    from lilbee.cli.tui import messages as msg

    bg = TASK_COLORS.get(row.task, "$primary")
    yield widgets.Label(row.name, id="card-name")
    with containers.HorizontalGroup(id="card-pills"):
        if row.featured:
            yield widgets.Label(pill("pick", "$warning", "$text"), id="card-pick")
        yield widgets.Label(pill(row.task, bg, "$text"), id="card-task")
        if row.backend:
            yield widgets.Label(pill(row.backend, "$accent", "$text"), id="card-backend")
    specs = _build_specs(row.params, row.quant, row.size)
    yield widgets.Label(specs, id="card-info")
    status = _build_local_status(row)
    if status is not None:
        yield widgets.Label(status, id="card-status")
    # Subtle "Enter to install" hint; CSS shows it only when the card
    # is highlighted (GridSelect cursor), hides for installed cards.
    if not row.installed:
        yield widgets.Label(msg.SETUP_CARD_HINT, id="card-hint")


def _compose_frontier(row: FrontierCatalogRow) -> ComposeResult:
    yield widgets.Label(row.name, id="card-name")
    with containers.HorizontalGroup(id="card-pills"):
        yield widgets.Label(pill(row.provider, "$accent", "$text"), id="card-backend")
        yield widgets.Label(_key_status_pill(row.key_status), id="card-status")
    info = Content.styled(f"Cloud via {row.provider} API", "$text-muted")
    yield widgets.Label(info, id="card-info")


def _key_status_pill(status: KeyStatus) -> Content:
    if status == KeyStatus.READY:
        return pill("ready", "$success", "$text")
    return pill("needs key", "$warning", "$text")


def _build_specs(params: str, quant: str, size: str) -> Content:
    """Build the specs line: params · quant · size."""
    parts = [p for p in (params, quant, size) if p and p != "--"]
    if not parts:
        return Content("--")
    return Content(f" {MIDDLE_DOT} ".join(parts))


def _build_local_status(row: LocalCatalogRow) -> Content | None:
    """Build the status pill for installed or download count."""
    if row.installed:
        return pill("installed", "$success", "$text")
    if row.sort_downloads > 0:
        return Content.styled(f"↓ {row.downloads}", "$text-muted")
    return None
