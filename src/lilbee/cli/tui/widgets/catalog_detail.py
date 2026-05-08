"""Right-pane detail drawer for the catalog screen.

Focus-following: the catalog screen wires ``ModelGrid.Highlighted`` to
``CatalogDetailDrawer.update_for_row``. The drawer renders the focused
row's name, fit chip, every size variant with its per-variant fit, the
license, and a description preview. Visibility toggles via the
``-collapsed`` CSS class so width changes are a single layout pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.content import Content
from textual.widgets import Static

from lilbee.cli.tui.pill import pill
from lilbee.cli.tui.screens.catalog_utils import (
    CatalogRow,
    FrontierCatalogRow,
    LocalCatalogRow,
    SizeVariant,
)
from lilbee.runtime.hardware import FitChip, FitLevel

_CSS_FILE = Path(__file__).parent / "catalog_detail.tcss"

_FIT_LEVEL_BACKGROUND: dict[FitLevel, str] = {
    FitLevel.FITS: "$success",
    FitLevel.TIGHT: "$warning",
    FitLevel.WONT_RUN: "$error",
}

_EMPTY_HINT = "Highlight a model to see details."


class CatalogDetailDrawer(Vertical):
    """Right-side panel that mirrors the highlighted catalog row.

    Designed as a passive renderer: the screen calls update_for_row on
    every ``ModelGrid.Highlighted`` event. There is no event subscription
    inside the drawer so it stays test-friendly and decoupled from the
    grid widget's message routing.
    """

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8") if _CSS_FILE.exists() else ""

    def compose(self) -> ComposeResult:
        yield Static(_EMPTY_HINT, id="catalog-detail-name", classes="catalog-detail-name")
        yield Static("", id="catalog-detail-fit", classes="catalog-detail-fit")
        yield Static("", id="catalog-detail-sizes", classes="catalog-detail-sizes")
        yield Static("", id="catalog-detail-license", classes="catalog-detail-license")
        yield Static("", id="catalog-detail-description", classes="catalog-detail-description")

    def update_for_row(self, row: CatalogRow | None) -> None:
        """Render the drawer for *row*; clearing back to the empty hint when None."""
        if row is None:
            self._clear()
            return
        if isinstance(row, FrontierCatalogRow):
            self._render_frontier(row)
            return
        self._render_local(row)

    def _clear(self) -> None:
        self.query_one("#catalog-detail-name", Static).update(_EMPTY_HINT)
        for selector in (
            "#catalog-detail-fit",
            "#catalog-detail-sizes",
            "#catalog-detail-license",
            "#catalog-detail-description",
        ):
            self.query_one(selector, Static).update("")

    def _render_local(self, row: LocalCatalogRow) -> None:
        self.query_one("#catalog-detail-name", Static).update(row.name)
        fit_widget = self.query_one("#catalog-detail-fit", Static)
        if row.fit is not None:
            fit_widget.update(_render_fit_pill(row.fit))
        else:
            fit_widget.update("")
        sizes = self.query_one("#catalog-detail-sizes", Static)
        sizes.update(_render_sizes_block(row.size_variants))
        license_widget = self.query_one("#catalog-detail-license", Static)
        license_widget.update(_license_text(row))
        description = self.query_one("#catalog-detail-description", Static)
        description.update(_description_text(row))

    def _render_frontier(self, row: FrontierCatalogRow) -> None:
        self.query_one("#catalog-detail-name", Static).update(row.name)
        self.query_one("#catalog-detail-fit", Static).update("")
        self.query_one("#catalog-detail-sizes", Static).update("")
        self.query_one("#catalog-detail-license", Static).update(f"Provider  {row.provider}")
        self.query_one("#catalog-detail-description", Static).update(
            f"Cloud model accessed via the {row.provider} API."
        )


def _render_fit_pill(fit: FitChip) -> Content:
    if fit.level is FitLevel.FITS:
        text = f"fits +{fit.headroom_gb:.1f} GB"
    elif fit.level is FitLevel.TIGHT:
        text = f"tight +{max(0.0, fit.headroom_gb):.1f} GB"
    else:
        text = f"won't {fit.headroom_gb:.1f} GB"
    return pill(text, _FIT_LEVEL_BACKGROUND[fit.level], "$text")


def _render_sizes_block(variants: list[SizeVariant]) -> str:
    """Multi-line plain-text listing of every variant the row carries."""
    if not variants:
        return ""
    lines = ["Sizes"]
    for v in variants:
        suffix = ""
        if v.fit is not None:
            if v.fit.level is FitLevel.FITS:
                suffix = "  ✓"
            elif v.fit.level is FitLevel.TIGHT:
                suffix = "  ⚠"
            else:
                suffix = "  ✗"
        lines.append(f"  {v.label}  {v.size_gb:.1f} GB{suffix}")
    return "\n".join(lines)


def _license_text(_row: LocalCatalogRow) -> str:
    """License placeholder; CatalogModel/ModelFamily don't carry one yet.

    Kept as a stub so callers have a stable seam: future plumbing for
    per-row license strings (HF metadata fetch, family-level config) can
    fill this in without touching the drawer's render path.
    """
    return ""


def _description_text(row: LocalCatalogRow) -> str:
    if row.catalog_model is not None and row.catalog_model.description:
        return row.catalog_model.description
    if row.family is not None and row.family.description:
        return row.family.description
    return ""
