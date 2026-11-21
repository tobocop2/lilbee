"""Right-pane detail drawer for the catalog screen.

Focus-following: the catalog screen wires ``ModelGrid.Highlighted`` to
``CatalogDetailDrawer.update_for_row``. The drawer renders the focused
row's name, fit chip, every size variant with its per-variant fit, the
license, and a description preview. Visibility toggles via the
``-collapsed`` CSS class so width changes are a single layout pass.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import ClassVar

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.content import Content
from textual.widgets import Static

from lilbee.app import engine_service
from lilbee.app.engine_choice import engine_for_ref
from lilbee.catalog.formats import ModelFormat
from lilbee.catalog.models import CatalogModel
from lilbee.catalog.types import ModelCompat
from lilbee.cli.tui.engine_mark import engine_mark, format_size
from lilbee.cli.tui.screens.catalog_utils import (
    CatalogRow,
    CatalogRowKind,
    FrontierCatalogRow,
    LocalCatalogRow,
    SizeVariant,
)
from lilbee.cli.tui.widgets.catalog_card_shared import _render_fit_pill
from lilbee.core.config import cfg
from lilbee.providers.fleet.engines.choice import AUTO, NoEngineError
from lilbee.runtime.hardware import FitChip, FitLevel

_CSS_FILE = Path(__file__).parent / "catalog_detail.tcss"

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
        yield Static(
            _EMPTY_HINT, id="catalog-detail-name", classes="catalog-detail-name", markup=False
        )
        yield Static("", id="catalog-detail-fit", classes="catalog-detail-fit")
        yield Static("", id="catalog-detail-sizes", classes="catalog-detail-sizes")
        yield Static(
            "", id="catalog-detail-license", classes="catalog-detail-license", markup=False
        )
        yield Static("", id="catalog-detail-compat", classes="catalog-detail-compat")
        yield Static("", id="catalog-detail-engine", classes="catalog-detail-engine")
        yield Static("", id="catalog-detail-formats", classes="catalog-detail-formats")
        yield Static(
            "",
            id="catalog-detail-description",
            classes="catalog-detail-description",
            markup=False,
        )

    def update_for_row(self, row: CatalogRow | None) -> None:
        """Render the drawer for *row*; clearing back to the empty hint when None."""
        if row is None:
            self._clear()
            return
        if row.kind == CatalogRowKind.FRONTIER:
            self._render_frontier(row)
            return
        self._render_local(row)

    def _clear(self) -> None:
        self.query_one("#catalog-detail-name", Static).update(_EMPTY_HINT)
        for selector in (
            "#catalog-detail-fit",
            "#catalog-detail-sizes",
            "#catalog-detail-license",
            "#catalog-detail-compat",
            "#catalog-detail-engine",
            "#catalog-detail-formats",
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
        compat_widget = self.query_one("#catalog-detail-compat", Static)
        compat_widget.update(_compat_sentence(row))
        self.query_one("#catalog-detail-engine", Static).update(_engine_line(row))
        self.query_one("#catalog-detail-formats", Static).update(_formats_block(row))
        description = self.query_one("#catalog-detail-description", Static)
        description.update(_description_text(row))

    def _render_frontier(self, row: FrontierCatalogRow) -> None:
        self.query_one("#catalog-detail-name", Static).update(row.name)
        self.query_one("#catalog-detail-fit", Static).update("")
        self.query_one("#catalog-detail-sizes", Static).update("")
        self.query_one("#catalog-detail-license", Static).update(f"Provider  {row.provider}")
        self.query_one("#catalog-detail-compat", Static).update("")
        self.query_one("#catalog-detail-engine", Static).update("")
        self.query_one("#catalog-detail-formats", Static).update("")
        self.query_one("#catalog-detail-description", Static).update(
            f"Cloud model accessed via the {row.provider} API."
        )


def _render_sizes_block(variants: list[SizeVariant]) -> str:
    """Multi-line plain-text listing of every variant the row carries."""
    if not variants:
        return ""
    lines = ["Sizes"]
    for v in variants:
        suffix = f"  {_fit_glyph(v.fit)}" if v.fit is not None else ""
        lines.append(f"  {v.label}  {v.size_gb:.1f} GB{suffix}")
    return "\n".join(lines)


def _fit_glyph(fit: FitChip) -> str:
    if fit.level is FitLevel.FITS:
        return "✓"
    if fit.level is FitLevel.TIGHT:
        return "⚠"
    return "✗"


def _license_text(_row: LocalCatalogRow) -> str:
    """License placeholder; CatalogModel/ModelFamily don't carry one yet.

    Kept as a stub so callers have a stable seam: future plumbing for
    per-row license strings (HF metadata fetch, family-level config) can
    fill this in without touching the drawer's render path.
    """
    return ""


def _engine_name(ref: str) -> str | None:
    """The engine a run of the model *ref* uses now, or None when no installed engine loads it.

    This is the lookup a run makes, so a stored per-model engine choice shows here.
    """
    try:
        return engine_for_ref(ref, cfg).name
    except NoEngineError:
        return None


def _row_ref(row: LocalCatalogRow) -> str:
    """The reference a run of the row's model uses."""
    return row.catalog_model.pull_ref if row.catalog_model is not None else row.ref


def _engine_line(row: LocalCatalogRow) -> Content:
    """The labelled engine line: the engine that loads the file, or that none is installed."""
    from lilbee.cli.tui import messages as msg

    engine = _engine_name(_row_ref(row))
    if engine is None:
        return Content(msg.CATALOG_DETAIL_ENGINE_LABEL + msg.CATALOG_DETAIL_ENGINE_NOT_INSTALLED)
    return Content.assemble(
        msg.CATALOG_DETAIL_ENGINE_LABEL, engine_mark(engine), msg.CATALOG_DETAIL_ENGINE_CHOSEN
    )


def _compat_sentence(row: LocalCatalogRow) -> Content:
    """Build the architecture-compatibility sentence for the detail drawer."""
    from lilbee.cli.tui import messages as msg

    if row.compat is ModelCompat.SUPPORTED:
        return Content(msg.COMPAT_DETAIL_SENTENCE_SUPPORTED)
    arch = row.catalog_model.architecture if row.catalog_model is not None else ""
    template = {
        ModelCompat.UNSUPPORTED: msg.COMPAT_DETAIL_SENTENCE_UNSUPPORTED,
        ModelCompat.UNKNOWN: msg.COMPAT_DETAIL_SENTENCE_UNKNOWN,
    }[row.compat]
    return Content(template.format(arch=arch or "unknown") if "{arch}" in template else template)


def _format_engine(model: CatalogModel, fmt: ModelFormat) -> Content:
    """The engine of one format: the installed engine, or the one to download with its size."""
    from lilbee.cli.tui import messages as msg

    name = _engine_name(replace(model, format=fmt).pull_ref)
    if name is not None:
        return engine_mark(name)
    options = engine_service.engine_options(replace(model, format=fmt).pull_ref)
    missing = [o for o in options if o.name != AUTO and not o.installed]
    if not missing:
        return Content(msg.CATALOG_FORMAT_NO_ENGINE)
    option = missing[0]
    if option.download_bytes is None:
        return engine_mark(option.name)
    size = msg.CATALOG_FORMAT_DOWNLOAD.format(size=format_size(option.download_bytes))
    return Content.assemble(engine_mark(option.name), " ", (f"({size})", "$text-muted"))


def _formats_block(row: LocalCatalogRow) -> Content:
    """One line for each format of the model: size and fit where known, and the engine.

    Only the format the row would pull has a size and a fit: the catalog fetch reads one size
    for each repo. The block is empty when the model ships one format.
    """
    from lilbee.cli.tui import messages as msg

    model = row.catalog_model
    if model is None or len(model.formats) <= 1:
        return Content("")
    lines = [Content(msg.CATALOG_FORMATS_HEADING)]
    for fmt in sorted(model.formats):
        parts: list[str | Content] = [f"  {fmt.value}  "]
        if fmt == model.format:
            parts.append(f"{model.size_gb:.1f} GB  ")
            if row.fit is not None:
                parts.append(f"{_fit_glyph(row.fit)}  ")
        parts.append(_format_engine(model, fmt))
        lines.append(Content.assemble(*parts))
    return Content("\n").join(lines)


def _description_text(row: LocalCatalogRow) -> str:
    if row.catalog_model is not None and row.catalog_model.description:
        return row.catalog_model.description
    if row.family is not None and row.family.description:
        return row.family.description
    return ""
