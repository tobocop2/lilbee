"""Rendering helpers shared by the catalog card, grid, and list widgets.

These were duplicated across ``model_card`` and ``model_grid`` (name
truncation, the spec/status pills) and the fit-level color map was also
copied into ``catalog_detail``. Keeping them here means a change to the card
surface is made once. ``model_card`` / ``model_grid`` re-export the names they
use, so callers (and tests) can keep importing them from those modules.
"""

from __future__ import annotations

from textual.content import Content

from lilbee.cli.tui.pill import pill
from lilbee.cli.tui.screens.catalog_utils import KeyStatus, LocalCatalogRow
from lilbee.cli.tui.widgets.catalog_theme import MIDDLE_DOT
from lilbee.runtime.hardware import FitChip, FitLevel

_NAME_MAX_CHARS = 28
_ELLIPSIS = "…"

# Pill background per fit level, shared by the grid card and the detail drawer.
_FIT_LEVEL_BACKGROUND: dict[FitLevel, str] = {
    FitLevel.FITS: "$success",
    FitLevel.TIGHT: "$warning",
    FitLevel.WONT_RUN: "$error",
}


def _render_fit_pill(fit: FitChip) -> Content:
    """Verbose fit chip with signed headroom GB, used by the detail drawer.

    Negative headroom means the model overflows available memory; the won't-run
    label reports the shortfall as a positive amount.
    """
    if fit.level is FitLevel.FITS:
        text = f"fits +{fit.headroom_gb:.1f} GB"
    elif fit.level is FitLevel.TIGHT:
        text = f"tight +{max(0.0, fit.headroom_gb):.1f} GB"
    else:
        text = f"won't run, short by {abs(fit.headroom_gb):.1f} GB"
    return pill(text, _FIT_LEVEL_BACKGROUND[fit.level], "$text")


def _truncate_name(name: str) -> str:
    """Return *name* shortened to ``_NAME_MAX_CHARS`` with an ellipsis tail."""
    if len(name) <= _NAME_MAX_CHARS:
        return name
    return name[: _NAME_MAX_CHARS - 1].rstrip() + _ELLIPSIS


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
