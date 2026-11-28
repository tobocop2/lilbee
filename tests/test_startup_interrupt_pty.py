"""A Ctrl+C at each stage of the interactive start ends quietly with status 130."""

from __future__ import annotations

import os
import pty
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs a POSIX pty")

STAGES = ["lilbee.app.services", "lilbee.runtime.splash", "lilbee.cli"]
_DEADLINE_SECONDS = 90.0

_RUNNER = """
import importlib.abc
import os
import signal
import sys

signal.signal(signal.SIGINT, signal.default_int_handler)
stage = sys.argv[1]


class _Injector(importlib.abc.MetaPathFinder):
    fired = False

    def find_spec(self, name, path=None, target=None):
        if name == stage and not _Injector.fired:
            _Injector.fired = True
            sys.stderr.write("INJECTED " + stage + "\\n")
            sys.stderr.flush()
            os.kill(os.getpid(), signal.SIGINT)
        return None


sys.meta_path.insert(0, _Injector())
sys.argv = ["lilbee"]
from lilbee.runtime.launcher import main

main()
"""


def _run_in_pty(runner: Path, stage: str, data_dir: Path, *, splash: bool) -> tuple[int, str]:
    """Run the launcher in a pty with a Ctrl+C injected at *stage*; return its status and output."""
    master, slave = pty.openpty()
    env = dict(os.environ, LILBEE_DATA=str(data_dir))
    if splash:
        env.pop("LILBEE_NO_SPLASH", None)
    else:
        env["LILBEE_NO_SPLASH"] = "1"
    proc = subprocess.Popen(
        [sys.executable, str(runner), stage],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
    )
    os.close(slave)
    chunks: list[bytes] = []
    deadline = time.monotonic() + _DEADLINE_SECONDS
    try:
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.5)
            if ready:
                try:
                    data = os.read(master, 4096)
                except OSError:
                    break
                if not data:
                    break
                chunks.append(data)
            elif proc.poll() is not None:
                break
        status = proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        os.close(master)
    return status, b"".join(chunks).decode("utf-8", errors="replace")


@pytest.mark.parametrize("splash", [False, True], ids=["no-splash", "splash"])
def test_ctrl_c_at_each_startup_stage_exits_130_without_traceback(
    tmp_path: Path, splash: bool
) -> None:
    assert STAGES
    runner = tmp_path / "runner.py"
    runner.write_text(_RUNNER, encoding="utf-8")
    for stage in STAGES:
        status, output = _run_in_pty(runner, stage, tmp_path / "data", splash=splash)
        assert f"INJECTED {stage}" in output, f"no interrupt was sent at {stage}"
        assert "Traceback" not in output, f"traceback at {stage}: {output}"
        assert "KeyboardInterrupt" not in output, f"interrupt text at {stage}: {output}"
        assert status == 130, f"status {status} at {stage}: {output}"
