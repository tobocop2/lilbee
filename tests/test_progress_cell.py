"""Tests for the block-character progress cell renderables."""

from __future__ import annotations

from lilbee.cli.tui.widgets.progress_cell import (
    _BAR_WIDTH,
    _EMPTY,
    _FULL,
    indeterminate_cell,
    progress_cell,
)


def test_progress_cell_renders_bar_and_percent() -> None:
    text = progress_cell(42.5)
    rendered = text.plain
    assert _FULL in rendered
    assert _EMPTY in rendered
    assert "42.5%" in rendered


def test_progress_cell_at_zero_is_all_empty() -> None:
    rendered = progress_cell(0).plain
    assert _FULL not in rendered
    assert rendered.count(_EMPTY) == _BAR_WIDTH
    assert "0.0%" in rendered


def test_progress_cell_at_hundred_is_all_full() -> None:
    rendered = progress_cell(100).plain
    assert rendered.count(_FULL) == _BAR_WIDTH
    assert "100.0%" in rendered


def test_progress_cell_clamps_negative_to_zero() -> None:
    rendered = progress_cell(-5).plain
    assert rendered.count(_EMPTY) == _BAR_WIDTH
    assert "0.0%" in rendered


def test_progress_cell_clamps_over_hundred_to_hundred() -> None:
    rendered = progress_cell(130).plain
    assert rendered.count(_FULL) == _BAR_WIDTH
    assert "100.0%" in rendered


def test_progress_cell_respects_custom_width() -> None:
    rendered = progress_cell(50, width=10).plain
    assert rendered.count(_FULL) == 5
    assert rendered.count(_EMPTY) == 5


def test_indeterminate_cell_contains_pulse_window() -> None:
    rendered = indeterminate_cell(5).plain
    assert _FULL in rendered
    assert "%" not in rendered


def test_indeterminate_cell_slides_with_tick() -> None:
    first = indeterminate_cell(0).plain
    later = indeterminate_cell(5).plain
    assert first != later


def test_indeterminate_cell_wraps_on_long_tick() -> None:
    rendered = indeterminate_cell(1000).plain
    assert _FULL in rendered
