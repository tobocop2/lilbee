"""T4 TUI: rotating-file log routing under <data_root>/logs/tui.log."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from drivers.tui import TuiSession

from conftest import TUI_BOOT_TIMEOUT

_LOG_FLUSH_GRACE = 2.0


@pytest.mark.tui
def test_tui_session_writes_rotating_log_file(tui: TuiSession, lilbee_data: Path) -> None:
    """A booted TUI populates <data_root>/logs/tui.log so debugging is possible.

    bb-pmyi replaced the silent strip-and-discard handler with a
    RotatingFileHandler. Without this test, a regression that drops the file
    handler would only surface when a user reported missing logs.
    """
    tui.wait_for("lilbee", timeout=TUI_BOOT_TIMEOUT)
    tui.send("\x11")  # Ctrl+Q exits the app; cleanup runs inside run_tui's finally.
    deadline = time.monotonic() + _LOG_FLUSH_GRACE
    log_path = lilbee_data / "logs" / "tui.log"
    while time.monotonic() < deadline:
        if log_path.exists():
            break
        time.sleep(0.1)
    assert log_path.exists(), f"expected TUI log at {log_path}; logs dir absent"
