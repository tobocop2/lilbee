"""Phase-0 GO/NO-GO harness smoke.

Validates that pywinpty/ptyprocess + pyte can drive a real lilbee binary
(or wheel-installed `lilbee` script) and read back screen state. If this
fails on any OS, half the matrix design changes; see plan.

Run:
    LILBEE_QA_BIN=$(which lilbee) python -m pytest tools/qa/_phase0_harness_smoke.py -v

Deleted after Phase 0 lands green on all three OSes.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from drivers.tui import TuiSession, lilbee_env

_BOOT_TIMEOUT_SECONDS = 30.0


def _resolve_lilbee_bin() -> str:
    explicit = os.environ.get("LILBEE_QA_BIN")
    if explicit:
        return explicit
    discovered = shutil.which("lilbee")
    if discovered:
        return discovered
    pytest.skip("lilbee binary not found; set LILBEE_QA_BIN or install lilbee")


@pytest.fixture
def lilbee_pty(tmp_path: Path) -> Iterator[TuiSession]:
    bin_path = _resolve_lilbee_bin()
    data_dir = tmp_path / "lilbee-data"
    data_dir.mkdir()
    session = TuiSession([bin_path, "--version"], env=lilbee_env(data_dir))
    try:
        yield session
    finally:
        session.close()


def test_phase0_lilbee_version_runs_under_pty(lilbee_pty: TuiSession) -> None:
    """The GO/NO-GO: lilbee --version prints something readable via PTY+pyte."""
    lilbee_pty.wait_for("lilbee", timeout=_BOOT_TIMEOUT_SECONDS)
