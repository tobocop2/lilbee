"""Tests for the block-character progress cell renderables."""

from __future__ import annotations

from unittest import mock

from lilbee.cli.tui.messages import (
    PROGRESS_BAR_FILL,
    PROGRESS_BAR_FILL_BLOCK,
    PROGRESS_BAR_TRACK,
    PROGRESS_BAR_TRACK_BLOCK,
    progress_bar_glyphs,
)
from lilbee.cli.tui.widgets.progress_cell import (
    _BAR_WIDTH,
    frozen_indeterminate_cell,
    indeterminate_cell,
    progress_cell,
)

# Whatever pair the running terminal selects; the bar must be built from it.
_FULL, _EMPTY = progress_bar_glyphs()


def test_progress_cell_renders_bar_and_percent() -> None:
    rendered = progress_cell(42.5)
    assert _FULL in rendered
    assert _EMPTY in rendered
    assert "42.5%" in rendered


def test_progress_cell_at_zero_is_all_empty() -> None:
    rendered = progress_cell(0)
    assert _FULL not in rendered
    assert rendered.count(_EMPTY) == _BAR_WIDTH
    assert "0.0%" in rendered


def test_progress_cell_at_hundred_is_all_full() -> None:
    rendered = progress_cell(100)
    assert rendered.count(_FULL) == _BAR_WIDTH
    assert "100.0%" in rendered


def test_progress_cell_clamps_negative_to_zero() -> None:
    rendered = progress_cell(-5)
    assert rendered.count(_EMPTY) == _BAR_WIDTH
    assert "0.0%" in rendered


def test_progress_cell_clamps_over_hundred_to_hundred() -> None:
    rendered = progress_cell(130)
    assert rendered.count(_FULL) == _BAR_WIDTH
    assert "100.0%" in rendered


def test_progress_cell_respects_custom_width() -> None:
    rendered = progress_cell(50, width=10)
    assert rendered.count(_FULL) == 5
    assert rendered.count(_EMPTY) == 5


def test_indeterminate_cell_contains_pulse_window() -> None:
    rendered = indeterminate_cell(5)
    assert _FULL in rendered
    assert "%" not in rendered


def test_indeterminate_cell_slides_with_tick() -> None:
    first = indeterminate_cell(0)
    later = indeterminate_cell(5)
    assert first != later


def test_indeterminate_cell_wraps_on_long_tick() -> None:
    rendered = indeterminate_cell(1000)
    assert _FULL in rendered


def test_frozen_indeterminate_cell_is_fully_filled() -> None:
    """terminal indeterminate rows render a full static bar."""
    rendered = frozen_indeterminate_cell()
    assert rendered == _FULL * _BAR_WIDTH


def test_frozen_indeterminate_cell_drops_trailing_dots() -> None:
    """no '···' tail on frozen bars (that trail reads as 'still
    working')."""
    assert "·" not in frozen_indeterminate_cell()


def test_frozen_indeterminate_cell_respects_custom_width() -> None:
    rendered = frozen_indeterminate_cell(width=8)
    assert rendered == _FULL * 8


def test_glyphs_are_blocks_only_where_the_terminal_tiles_them() -> None:
    with mock.patch("lilbee.cli.tui.color_compat.draws_block_bars", return_value=True):
        assert progress_bar_glyphs() == (PROGRESS_BAR_FILL_BLOCK, PROGRESS_BAR_TRACK_BLOCK)
    with mock.patch("lilbee.cli.tui.color_compat.draws_block_bars", return_value=False):
        assert progress_bar_glyphs() == (PROGRESS_BAR_FILL, PROGRESS_BAR_TRACK)


def test_progress_cell_uses_the_selected_pair_in_both_modes() -> None:
    for block, fill, track in (
        (True, PROGRESS_BAR_FILL_BLOCK, PROGRESS_BAR_TRACK_BLOCK),
        (False, PROGRESS_BAR_FILL, PROGRESS_BAR_TRACK),
    ):
        with mock.patch("lilbee.cli.tui.color_compat.draws_block_bars", return_value=block):
            rendered = progress_cell(50, width=10)
            assert rendered.count(fill) == 5
            assert rendered.count(track) == 5
            assert frozen_indeterminate_cell(width=4) == fill * 4
            assert track in indeterminate_cell(0, width=8)
