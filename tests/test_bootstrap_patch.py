"""The onefile bootstrap patch must stay in step with the shared wordmark."""

from __future__ import annotations

import pathlib

import pytest

from lilbee.runtime.bee_logo import (
    AMBER_BRIGHT_XTERM,
    AMBER_DIM_XTERM,
    AMBER_MID_XTERM,
    BEE_LINES,
)

_PATCH = (
    pathlib.Path(__file__).resolve().parents[1] / "tools/wheel-build/onefile-bootstrap-lilbee.patch"
)

_ADDED_PREFIX = "+"


@pytest.fixture(scope="module")
def added_lines() -> list[str]:
    """The lines the patch adds to the Nuitka bootstrap."""
    text = _PATCH.read_text()
    return [line[1:] for line in text.splitlines() if line.startswith(_ADDED_PREFIX)]


def test_patch_exists():
    """The build script applies this patch by path; a rename must fail loudly here."""
    assert _PATCH.is_file()


def test_c_logo_matches_the_python_wordmark(added_lines):
    """Every non-blank wordmark row must appear verbatim as a C string literal."""
    body = "\n".join(added_lines)
    for row in BEE_LINES:
        if not row.strip():
            continue
        assert f'"{row}",' in body, f"wordmark row missing from the C bootstrap: {row!r}"


def test_c_logo_has_no_extra_rows(added_lines):
    """The C array must hold exactly the non-blank wordmark rows, in order."""
    row_starts = ('"@', '"!', '" :', '": ')
    literals = [
        line.strip().removeprefix('"').removesuffix('",')
        for line in added_lines
        if line.strip().startswith(row_starts)
    ]
    expected = [row for row in BEE_LINES if row.strip()]
    assert literals == expected


def test_bootstrap_uses_the_shared_amber_ramp(added_lines):
    """The bootstrap paints the wordmark and bar from the shared palette."""
    body = "\n".join(added_lines)
    for index in (AMBER_DIM_XTERM, AMBER_MID_XTERM, AMBER_BRIGHT_XTERM):
        assert f"38;5;{index}m" in body, f"xterm index {index} missing from the C bootstrap"


def test_bootstrap_bar_uses_the_splash_block_glyphs(added_lines):
    """The bar reads as the same surface as the splash, not an ASCII fallback."""
    body = "\n".join(added_lines)
    for glyph in ("\\u2593", "\\u2592", "\\u2591"):
        assert glyph in body, f"block glyph {glyph} missing from the bootstrap bar"
    assert "'='" not in body, "the bar should not fall back to ASCII"


def test_progress_is_suppressed_off_a_terminal(added_lines):
    """A piped or redirected launch must not emit escape codes."""
    body = "\n".join(added_lines)
    assert "isatty(STDERR_FILENO)" in body


def test_stamp_fast_path_is_guarded_by_payload_size(added_lines):
    """A stamp from a different payload must not be trusted."""
    body = "\n".join(added_lines)
    assert "_NUITKA_ONEFILE_PAYLOAD_SIZE_INT" in body
    assert "stamped_size !=" in body


def test_stamp_fast_path_requires_an_executable_entry_point(added_lines):
    """A stamp naming a missing binary must fall back to a full extraction."""
    body = "\n".join(added_lines)
    assert "access(stamped_main, X_OK)" in body


def test_patch_is_posix_guarded(added_lines):
    """Windows keeps Nuitka's stock bootstrap, which has its own splash support."""
    body = "\n".join(added_lines)
    assert "#if !defined(_WIN32) && !defined(__MSYS__)" in body


def test_bootstrap_frame_matches_the_splash_frame_geometry(added_lines):
    """The splash repaints the bootstrap's frame in place, so the row counts must agree.

    The splash writes len(BEE_LINES) wordmark rows, a separator, and the bar. The
    bootstrap parks the cursor that many rows above the bar. A mismatch would draw
    the splash over the wrong lines.
    """
    body = "\n".join(added_lines)
    expected = len(BEE_LINES) + 1
    assert f"#define LILBEE_FRAME_ROWS_ABOVE_BAR {expected}" in body


def test_interactive_launch_keeps_the_wordmark_on_screen(added_lines):
    """Erasing the frame before exec left the terminal blank until the splash started."""
    body = "\n".join(added_lines)
    assert "lilbee_progress_end(argc == 1)" in body
    assert "keep_logo ? " in body
