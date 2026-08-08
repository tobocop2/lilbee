"""Rendering helpers shared by the catalog card, grid, and list widgets.

The local-card body (name, pill lines, specs, status, hint) is built here
once; ``model_grid`` renders the slots as a fixed-height card and
``model_card`` as an auto-height card. ``model_list`` keeps its own two-line
text presentation but draws its labels, colors, and spec strip from the same
maps so the three surfaces cannot drift apart.
"""

from __future__ import annotations

from textual.content import Content

from lilbee.catalog.types import ModelCompat
from lilbee.cli.tui.pill import pill
from lilbee.cli.tui.screens.catalog_utils import (
    NATIVE_BACKEND,
    KeyStatus,
    LocalCatalogRow,
    SizeVariant,
)
from lilbee.cli.tui.widgets.catalog_theme import MIDDLE_DOT, TASK_COLORS
from lilbee.runtime.hardware import FitChip, FitLevel

_NAME_MAX_CHARS = 28
_ELLIPSIS = "…"

# Pill background per fit level, shared by the grid card and the detail drawer.
_FIT_LEVEL_BACKGROUND: dict[FitLevel, str] = {
    FitLevel.FITS: "$success",
    FitLevel.TIGHT: "$warning",
    FitLevel.WONT_RUN: "$error",
}

_FIT_LEVEL_LABEL_COMPACT: dict[FitLevel, str] = {
    FitLevel.FITS: "fits",
    FitLevel.TIGHT: "tight",
    FitLevel.WONT_RUN: "won't run",
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


def _fit_pill_compact(fit: FitChip) -> Content:
    """Card-side compact fit chip: just ``fits`` / ``tight`` / ``won't run``."""
    return pill(_FIT_LEVEL_LABEL_COMPACT[fit.level], _FIT_LEVEL_BACKGROUND[fit.level], "$text")


def _compat_label(compat: ModelCompat) -> str | None:
    """Compat text for non-SUPPORTED rows, or None for SUPPORTED."""
    from lilbee.cli.tui import messages as msg

    if compat is ModelCompat.SUPPORTED:
        return None
    if compat is ModelCompat.UNSUPPORTED:
        return msg.COMPAT_PILL_UNSUPPORTED
    return msg.COMPAT_PILL_UNKNOWN


def _compat_pill(compat: ModelCompat) -> Content | None:
    """Return the compat chip Content for non-SUPPORTED rows, or None for SUPPORTED."""
    label = _compat_label(compat)
    if label is None:
        return None
    if compat is ModelCompat.UNSUPPORTED:
        return pill(label, "$warning", "$text")
    return pill(label, "$panel", "$text-muted")


def _truncate_name(name: str) -> str:
    """Return *name* shortened to ``_NAME_MAX_CHARS`` with an ellipsis tail."""
    if len(name) <= _NAME_MAX_CHARS:
        return name
    return name[: _NAME_MAX_CHARS - 1].rstrip() + _ELLIPSIS


def _key_status_pill(status: KeyStatus) -> Content:
    if status == KeyStatus.READY:
        return pill("ready", "$success", "$text")
    return pill("needs key", "$warning", "$text")


def _spec_strip(params: str, quant: str, size: str) -> str:
    """Dot-joined spec fragments, skipping empty / placeholder values."""
    parts = [p for p in (params, quant, size) if p and p != "--"]
    return f" {MIDDLE_DOT} ".join(parts)


def _build_specs(params: str, quant: str, size: str) -> Content:
    """Build the specs line: params · quant · size."""
    text = _spec_strip(params, quant, size)
    return Content(text) if text else Content("--")


def _build_local_status(row: LocalCatalogRow) -> Content | None:
    """Build the status pill for installed or download count."""
    if row.installed:
        return pill("installed", "$success", "$text")
    if row.sort_downloads > 0:
        return Content.styled(f"↓ {row.downloads}", "$text-muted")
    return None


def _local_card_lines(
    row: LocalCatalogRow, *, selected: bool, body_width: int | None = None
) -> list[Content | None]:
    """Semantic card slots: name, primary pills, secondary pills, specs, status, hint.

    A slot is None when the row has nothing for it; the grid paints None as a
    blank line (fixed-height card) while the wizard card omits the slot
    (auto-height). *body_width* selects the grid presentation: the compact fit
    pill and the inline size-variant strip, both sized to the card column.
    """
    from lilbee.cli.tui import messages as msg

    bg = TASK_COLORS.get(row.task, "$primary")
    name = Content.styled(_truncate_name(row.name), "bold")
    # Two pill rows so wide secondary chips (fit + 'unsupported') don't push
    # the card border out of alignment on narrow grid columns.
    primary_pills: list[Content] = []
    if row.featured:
        primary_pills.append(pill("pick", "$warning", "$text"))
    primary_pills.append(pill(row.task, bg, "$text"))
    # Drop the 'native' backend pill on cards to free horizontal space; the
    # backend is implied for local models. Remote backends (ollama, etc.)
    # still surface their pill since that's a meaningful distinction.
    if row.backend and row.backend != NATIVE_BACKEND:
        primary_pills.append(pill(row.backend, "$accent", "$text"))
    primary_line = Content(" ").join(primary_pills)

    secondary_pills: list[Content] = []
    if body_width is not None and row.fit is not None:
        # Grid card uses the compact 'fits' / 'tight' / "won't run" label only;
        # the headroom GB lives in the detail drawer where the wider pane
        # can render it without competing for card width.
        secondary_pills.append(_fit_pill_compact(row.fit))
    compat_chip = _compat_pill(row.compat)
    if compat_chip is not None:
        secondary_pills.append(compat_chip)
    secondary_line = Content(" ").join(secondary_pills) if secondary_pills else None

    # Family card with multiple quants: replace the simple specs line
    # with an inline chip strip so the user sees every available size
    # at a glance without expanding into the drawer.
    if body_width is not None and len(row.size_variants) > 1:
        specs = _build_size_variant_strip(row.size_variants, body_width)
    else:
        specs = _build_specs(row.params, row.quant, row.size)
    status = _build_local_status(row)

    hint: Content | None = None
    if selected:
        hint_text = msg.INSTALLED_CARD_HINT if row.installed else msg.SETUP_CARD_HINT
        hint = Content.styled(hint_text, "$text-muted 40% italic")
    return [name, primary_line, secondary_line, specs, status, hint]


def _build_size_variant_strip(variants: list[SizeVariant], width: int) -> Content:
    """Inline chip strip showing the quants of a family-aggregated card.

    Renders compact 'Q4 · Q5 · F16' style chips so the eye reads the
    available sizes at a glance. A family that varies by parameter count
    rather than quant would render identical chips, so colliding quants
    fall back to the full per-variant label. Per-variant fit colors aren't
    applied here; the drawer (right pane) carries the full fit-per-size
    detail when a card is highlighted.

    Duplicate labels collapse, then chips that do not fit *width* are dropped
    and counted as ``+N``: a family can hold more variants than a card column
    has room for, and the long fallback labels reach that limit quickly.
    """
    quants = [v.quant if v.quant != "--" else v.label for v in variants]
    labels = quants if len(set(quants)) == len(quants) else [v.label for v in variants]
    # Two repos in one family can share a parameter count and quant, which
    # renders the same chip twice and reads as a bug.
    labels = list(dict.fromkeys(labels))
    sep = f" {MIDDLE_DOT} "
    shown = len(labels)
    while shown > 1:
        text = sep.join(labels[:shown])
        hidden = len(labels) - shown
        if hidden:
            text = f"{text}  +{hidden}"
        if len(text) <= width:
            return Content.styled(text, "$text-muted")
        shown -= 1
    return Content.styled(labels[0][:width] if labels else "", "$text-muted")
