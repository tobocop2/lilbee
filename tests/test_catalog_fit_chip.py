"""Tests for the per-card hardware-fit chip rendering."""

from __future__ import annotations

from lilbee.cli.tui.widgets.model_grid import _FIT_LEVEL_BACKGROUND, _fit_pill
from lilbee.runtime.hardware import FitChip, FitLevel


def test_fits_chip_uses_success_background() -> None:
    assert _FIT_LEVEL_BACKGROUND[FitLevel.FITS] == "$success"


def test_tight_chip_uses_warning_background() -> None:
    assert _FIT_LEVEL_BACKGROUND[FitLevel.TIGHT] == "$warning"


def test_wont_run_chip_uses_error_background() -> None:
    assert _FIT_LEVEL_BACKGROUND[FitLevel.WONT_RUN] == "$error"


def test_fits_pill_renders_positive_headroom() -> None:
    chip = FitChip(level=FitLevel.FITS, headroom_gb=8.3)
    pill_content = _fit_pill(chip)
    assert "fits +8.3 GB" in pill_content.plain


def test_tight_pill_clamps_headroom_at_zero_for_display() -> None:
    """Tight at exact 0 GB headroom shows '+0.0 GB', not negative; the
    headroom math may produce -0.0 from float arithmetic but the chip
    still reads as nonnegative."""
    chip = FitChip(level=FitLevel.TIGHT, headroom_gb=0.0)
    pill_content = _fit_pill(chip)
    assert "tight +0.0 GB" in pill_content.plain


def test_wont_run_pill_shows_negative_headroom() -> None:
    chip = FitChip(level=FitLevel.WONT_RUN, headroom_gb=-2.0)
    pill_content = _fit_pill(chip)
    assert "won't run, short by 2.0 GB" in pill_content.plain
