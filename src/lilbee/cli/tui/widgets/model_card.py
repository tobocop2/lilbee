"""ModelCard: compact card widget for the catalog grid view.

Renders both ``LocalCatalogRow`` (installable / installed GGUFs) and
``FrontierCatalogRow`` (cloud chat models) via type dispatch so the
catalog can show a Frontier section above local sections without a
second widget class.

A card composes its name + pills + specs + status + (highlight-only)
hint into a single ``Static`` rendering one ``Content`` object. This
keeps the per-card widget count to two (the container + one Static)
so the catalog's grid view can scale to thousands of rows without
saturating the compositor with mount/reflow work.
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
    CatalogRowKind,
    FrontierCatalogRow,
    LocalCatalogRow,
)
from lilbee.cli.tui.widgets.catalog_card_shared import (
    _key_status_pill,
    _local_card_lines,
    _truncate_name,
)

if TYPE_CHECKING:
    pass

_CSS_FILE = Path(__file__).parent / "model_card.tcss"


class ModelCard(containers.VerticalGroup):
    """A single model card displaying name, task pill, specs, and status."""

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

    selected: reactive[bool] = reactive(False)

    def __init__(self, row: CatalogRow) -> None:
        self._row = row
        super().__init__()
        if row.kind == CatalogRowKind.FRONTIER:
            self.add_class("-frontier")

    @property
    def row(self) -> CatalogRow:
        return self._row

    def watch_selected(self, selected: bool) -> None:
        self.set_class(selected, "-selected")
        # Re-render so the highlight-only hint appears / disappears.
        # Cheap: one Static.update() per highlight move beats the
        # CSS-driven display: none toggle on a separate Label widget.
        try:
            body = self.query_one(".card-body", widgets.Static)
        except Exception:
            return
        body.update(_render(self._row, selected=selected))

    def compose(self) -> ComposeResult:
        yield widgets.Static(
            _render(self._row, selected=self.selected),
            classes="card-body",
            markup=False,
        )


def _render(row: CatalogRow, *, selected: bool) -> Content:
    """Compose the full card content (name + pills + specs + status + hint)."""
    if row.kind == CatalogRowKind.FRONTIER:
        return _render_frontier(row)
    return _render_local(row, selected=selected)


def _render_local(row: LocalCatalogRow, *, selected: bool) -> Content:
    """Auto-height presentation of the shared card slots: absent slots are omitted."""
    slots = _local_card_lines(row, selected=selected)
    return Content("\n").join([line for line in slots if line is not None])


def _render_frontier(row: FrontierCatalogRow) -> Content:
    name = Content.styled(_truncate_name(row.name), "bold")
    backend_pill = pill(row.provider, "$accent", "$text")
    status_pill = _key_status_pill(row.key_status)
    pill_line = Content(" ").join([backend_pill, status_pill])
    info = Content.styled(f"Cloud via {row.provider} API", "$text-muted")
    return Content("\n").join([name, pill_line, info])


# _local_card_lines / _key_status_pill / _truncate_name and the compat chip
# live in catalog_card_shared.
