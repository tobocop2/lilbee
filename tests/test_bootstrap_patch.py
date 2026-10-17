"""The onefile bootstrap patch must stay in step with the shared wordmark."""

from __future__ import annotations

import os
import pathlib
import re
import select
import shutil
import subprocess
import sys
import time

import pytest

from lilbee.runtime.bee_logo import (
    BEE_LINES,
    ROSE_BRIGHT_XTERM,
    ROSE_DIM_XTERM,
    ROSE_MID_XTERM,
)
from lilbee.runtime.onefile_cache import BOOTSTRAP_MANIFEST_NAME

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


def test_bootstrap_uses_the_shared_rose_ramp(added_lines):
    """The bootstrap paints the wordmark and bar from the shared palette."""
    body = "\n".join(added_lines)
    for index in (ROSE_DIM_XTERM, ROSE_MID_XTERM, ROSE_BRIGHT_XTERM):
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


def test_manifest_name_matches_the_startup_sweep(added_lines):
    """The sweep only deletes a directory that holds the manifest this patch writes."""
    body = "\n".join(added_lines)
    assert f'#define LILBEE_MANIFEST_NAME "{BOOTSTRAP_MANIFEST_NAME}"' in body


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


def test_bootstrap_centres_on_the_python_wordmark_width(added_lines):
    """Both stages centre with (columns - width) / 2; a drifting width constant
    would draw the bootstrap and the splash at different offsets."""
    from lilbee.runtime.bee_logo import LOGO_WIDTH

    assert f"#define LILBEE_LOGO_WIDTH {LOGO_WIDTH}" in added_lines
    assert any("TIOCGWINSZ" in line for line in added_lines)


_CURSOR_CODE = re.compile(r"\x1b\[\?25([lh])")
_WORDMARK_MARK = "@@@"
_PTY_WAIT_SECONDS = 30.0
_PTY_POLL_SECONDS = 0.05


def _run_progress_on_a_pty(tmp_path: pathlib.Path, added_lines: list[str], steps: str) -> str:
    """Compile the patch's progress functions with a driver and return what they write to a pty."""
    text = "\n".join(added_lines)
    head = text[text.index("#if !defined(_WIN32)") : text.index("static void lilbee_stamp_path")]
    end_of_progress = text.index("#endif\n", text.index("static void lilbee_progress_end"))
    body = text[text.index("#define LILBEE_FRAME_ROWS_ABOVE_BAR") : end_of_progress]
    source = tmp_path / "driver.c"
    source.write_text(
        "#include <stdio.h>\n#include <stdbool.h>\n#include <stdlib.h>\n#include <string.h>\n"
        f"{head}\n{body}\n#endif\n"
        f"int main(void) {{ lilbee_progress_begin(); {steps} return 0; }}\n",
        encoding="utf-8",
    )
    binary = tmp_path / "driver"
    subprocess.run(["cc", "-o", str(binary), str(source)], check=True)
    import pty  # POSIX only; the only caller is skipped on Windows

    master, slave = pty.openpty()
    proc = subprocess.Popen([str(binary)], stderr=slave, stdout=slave, close_fds=True)
    chunks: list[bytes] = []
    try:
        deadline = time.monotonic() + _PTY_WAIT_SECONDS
        while True:
            exited = proc.poll() is not None
            timeout = 0 if exited else min(max(deadline - time.monotonic(), 0), _PTY_POLL_SECONDS)
            if not select.select([master], [], [], timeout)[0]:
                if exited or time.monotonic() >= deadline:
                    break
                continue
            chunks.append(os.read(master, 65536))
    finally:
        os.close(slave)
        os.close(master)
        proc.kill()
        proc.wait()
    return b"".join(chunks).decode("utf-8", "replace")


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("cc") is None,
    reason="needs a POSIX pty and a C compiler",
)
class TestBootstrapCursor:
    """What the bootstrap writes to a terminal decides where the cursor sits."""

    def test_the_cursor_is_hidden_before_the_wordmark_is_drawn(self, added_lines, tmp_path):
        out = _run_progress_on_a_pty(
            tmp_path, added_lines, "lilbee_progress_update(50, 100); lilbee_progress_end(true);"
        )
        first_hide = out.index("\x1b[?25l")
        assert first_hide < out.index(_WORDMARK_MARK)

    def test_an_interactive_launch_hands_the_hidden_cursor_on(self, added_lines, tmp_path):
        out = _run_progress_on_a_pty(
            tmp_path, added_lines, "lilbee_progress_update(50, 100); lilbee_progress_end(true);"
        )
        assert _CURSOR_CODE.findall(out)[-1] == "l"

    def test_a_launch_with_arguments_shows_the_cursor_again(self, added_lines, tmp_path):
        out = _run_progress_on_a_pty(
            tmp_path, added_lines, "lilbee_progress_update(50, 100); lilbee_progress_end(false);"
        )
        assert _CURSOR_CODE.findall(out)[-1] == "h"

    def test_a_failed_extraction_does_not_leave_the_cursor_hidden(self, added_lines, tmp_path):
        out = _run_progress_on_a_pty(
            tmp_path, added_lines, "lilbee_progress_update(50, 100); exit(2);"
        )
        assert _CURSOR_CODE.findall(out)[-1] == "h"
