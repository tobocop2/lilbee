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
    KeyStatus,
    LocalCatalogRow,
)
from lilbee.cli.tui.widgets.catalog_theme import MIDDLE_DOT, TASK_COLORS

if TYPE_CHECKING:
    pass

_CSS_FILE = Path(__file__).parent / "model_card.tcss"

_NAME_MAX_CHARS = 28
"""Maximum displayed model-name length; longer names are ellipsis-truncated."""

_ELLIPSIS = "…"


def _truncate_name(name: str) -> str:
    """Return *name* shortened to ``_NAME_MAX_CHARS`` with an ellipsis tail."""
    if len(name) <= _NAME_MAX_CHARS:
        return name
    return name[: _NAME_MAX_CHARS - 1].rstrip() + _ELLIPSIS


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
    from lilbee.cli.tui import messages as msg

    bg = TASK_COLORS.get(row.task, "$primary")
    name = Content.styled(_truncate_name(row.name), "bold")
    pills: list[Content] = []
    if row.featured:
        pills.append(pill("pick", "$warning", "$text"))
    pills.append(pill(row.task, bg, "$text"))
    if row.backend:
        pills.append(pill(row.backend, "$accent", "$text"))
    pill_line = Content(" ").join(pills)
    specs = _build_specs(row.params, row.quant, row.size)
    status = _build_local_status(row)

    parts: list[Content] = [name, pill_line, specs]
    if status is not None:
        parts.append(status)
    if selected:
        hint = msg.INSTALLED_CARD_HINT if row.installed else msg.SETUP_CARD_HINT
        parts.append(Content.styled(hint, "$text-muted 40% italic"))
    return Content("\n").join(parts)


def _render_frontier(row: FrontierCatalogRow) -> Content:
    name = Content.styled(_truncate_name(row.name), "bold")
    backend_pill = pill(row.provider, "$accent", "$text")
    status_pill = _key_status_pill(row.key_status)
    pill_line = Content(" ").join([backend_pill, status_pill])
    info = Content.styled(f"Cloud via {row.provider} API", "$text-muted")
    return Content("\n").join([name, pill_line, info])


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
