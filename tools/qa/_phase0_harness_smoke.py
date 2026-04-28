"""Phase-0 GO/NO-GO harness smoke.

Validates that pywinpty/ptyprocess + pyte can drive a real lilbee binary
(or wheel-installed `lilbee` script) and read back screen state. If this
fails on any OS, half the matrix design changes; see plan.

Run:
    LILBEE_QA_BIN=$(which lilbee) python -m pytest tools/qa/_phase0_harness_smoke.py -v

Deleted after Phase 0 lands green on all three OSes.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pyte
import pytest

if sys.platform == "win32":
    from winpty import PtyProcess  # type: ignore[import-not-found]
else:
    from ptyprocess import PtyProcess  # type: ignore[import-not-found]


_DEFAULT_COLS = 120
_DEFAULT_ROWS = 40
_BOOT_TIMEOUT_SECONDS = 30.0
_POLL_INTERVAL_SECONDS = 0.2
_READ_CHUNK_BYTES = 65536


def _resolve_lilbee_bin() -> str:
    explicit = os.environ.get("LILBEE_QA_BIN")
    if explicit:
        return explicit
    discovered = shutil.which("lilbee")
    if discovered:
        return discovered
    pytest.skip("lilbee binary not found; set LILBEE_QA_BIN or install lilbee")


def _read_into(stream: pyte.ByteStream, proc: PtyProcess) -> None:
    try:
        chunk = proc.read(_READ_CHUNK_BYTES)
    except (BlockingIOError, TimeoutError, EOFError, OSError):
        return
    if not chunk:
        return
    payload = chunk if isinstance(chunk, bytes) else chunk.encode("utf-8", "replace")
    stream.feed(payload)


def _wait_for(
    screen: pyte.Screen,
    stream: pyte.ByteStream,
    proc: PtyProcess,
    needle: str,
    timeout: float,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _read_into(stream, proc)
        if needle.lower() in "\n".join(screen.display).lower():
            return True
        time.sleep(_POLL_INTERVAL_SECONDS)
    return False


@pytest.fixture
def lilbee_pty(tmp_path: Path) -> Iterator[tuple[pyte.Screen, pyte.ByteStream, PtyProcess]]:
    bin_path = _resolve_lilbee_bin()
    data_dir = tmp_path / "lilbee-data"
    data_dir.mkdir()

    env = os.environ.copy()
    env["LILBEE_DATA"] = str(data_dir)
    env["LILBEE_NO_SPLASH"] = "1"
    env["LILBEE_LOG_LEVEL"] = "WARNING"

    screen = pyte.Screen(_DEFAULT_COLS, _DEFAULT_ROWS)
    stream = pyte.ByteStream()
    stream.attach(screen)

    proc = PtyProcess.spawn(
        [bin_path, "--version"],
        dimensions=(_DEFAULT_ROWS, _DEFAULT_COLS),
        env=env,
    )
    if sys.platform != "win32":
        with contextlib.suppress(AttributeError, OSError):
            os.set_blocking(proc.fd, False)
    try:
        yield screen, stream, proc
    finally:
        with contextlib.suppress(Exception):
            proc.terminate(force=True)


def test_phase0_lilbee_version_runs_under_pty(
    lilbee_pty: tuple[pyte.Screen, pyte.ByteStream, PtyProcess],
) -> None:
    """The GO/NO-GO: lilbee --version prints something readable via PTY+pyte."""
    screen, stream, proc = lilbee_pty
    found = _wait_for(screen, stream, proc, "lilbee", timeout=_BOOT_TIMEOUT_SECONDS)
    assert found, f"did not see 'lilbee' in screen output. screen:\n{chr(10).join(screen.display)}"
