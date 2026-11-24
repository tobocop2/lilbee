"""The exit of sync and analyze with a leftover worker of the loop's pool, through the entry."""

from __future__ import annotations

import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from lilbee.app.services import _STRAGGLER_BUDGET_S, _STRAGGLER_NOTE
from tests._signals import child_preexec
from tests.conftest import clean_env

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal delivery")

_TERMINATED = 128 + signal.SIGTERM
# The exit waits the straggler budget for the worker, then the engine stop; the rest is
# the interpreter's start. A kill at the ceiling ends a child whose exit hangs (-9).
_EXIT_BOUND_S = _STRAGGLER_BUDGET_S + 15.0
_FAILURE_CEILING_S = 40.0

# The real entry point, with a worker the command leaves on its loop's pool.
_ENTRY = Path(__file__).with_name("_stuck_worker_entry.py")

_COMMANDS = ["sync", "analyze"]


class _Command:
    """One run of a lilbee command in a fresh interpreter, its output read line by line."""

    def __init__(self, tmp_path: Path, command: str, *, work: str, hold: bool) -> None:
        data = tmp_path / "data"
        data.mkdir()
        self.lines: list[str] = []
        self.exit_status = -1
        self.elapsed = 0.0
        self.started = time.monotonic()
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(_ENTRY), command, "--data-dir", str(data)],
            env={
                **clean_env(data),
                "HF_HUB_OFFLINE": "1",
                "X_WORK_S": work,
                "X_HOLD": "1" if hold else "0",
            },
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            preexec_fn=child_preexec(),
        )
        # The kill turns a blocked read of the output into an EOF.
        self.ceiling = threading.Timer(_FAILURE_CEILING_S, self.process.kill)
        self.ceiling.daemon = True
        self.ceiling.start()

    @property
    def output(self) -> str:
        return "".join(self.lines)

    def read_until(self, marker: str) -> None:
        assert self.process.stdout is not None
        for line in iter(self.process.stdout.readline, ""):
            self.lines.append(line)
            if line.strip() == marker:
                return
        raise AssertionError(f"the command ended before {marker}:\n{self.output}")

    def status(self) -> tuple[int, float]:
        """Read the rest of the output; return the exit status and the seconds since the start."""
        assert self.process.stdout is not None
        self.lines.extend(iter(self.process.stdout.readline, ""))
        status = self.process.wait()
        self.ceiling.cancel()
        return status, time.monotonic() - self.started

    def end(self) -> None:
        self.ceiling.cancel()
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        if self.process.stdout is not None:
            self.process.stdout.close()


def _run(tmp_path: Path, command: str, *, work: str, signal_after: str | None = None) -> _Command:
    """Run *command* to its end; with *signal_after*, send SIGTERM once that line shows."""
    run = _Command(tmp_path, command, work=work, hold=signal_after == "HOLDING")
    try:
        if signal_after is not None:
            run.read_until(signal_after)
            run.process.send_signal(signal.SIGTERM)
        run.exit_status, run.elapsed = run.status()
    finally:
        run.end()
    return run


@pytest.mark.parametrize("command", _COMMANDS)
def test_a_stuck_worker_leaves_the_command_status_within_the_bound(
    tmp_path: Path, command: str
) -> None:
    run = _run(tmp_path, command, work="stuck")
    assert run.exit_status == 0, run.output
    assert run.elapsed < _EXIT_BOUND_S, run.output
    assert _STRAGGLER_NOTE.strip() in run.output, run.output


@pytest.mark.parametrize("command", _COMMANDS)
def test_a_worker_that_ends_within_the_budget_finishes_before_the_exit(
    tmp_path: Path, command: str
) -> None:
    run = _run(tmp_path, command, work="1.0")
    assert run.exit_status == 0, run.output
    assert any(line.strip() == "WORKER DONE" for line in run.lines), run.output
    assert _STRAGGLER_NOTE.strip() not in run.output, run.output


@posix_only
@pytest.mark.parametrize("command", _COMMANDS)
def test_a_terminate_during_the_command_with_a_stuck_worker_ends_with_its_status(
    tmp_path: Path, command: str
) -> None:
    run = _run(tmp_path, command, work="stuck", signal_after="HOLDING")
    assert run.exit_status == _TERMINATED, run.output
    assert run.elapsed < _EXIT_BOUND_S, run.output


@posix_only
@pytest.mark.parametrize("command", _COMMANDS)
def test_a_terminate_while_the_exit_waits_on_a_stuck_worker_ends_with_its_status(
    tmp_path: Path, command: str
) -> None:
    run = _run(tmp_path, command, work="stuck", signal_after="COMMAND DONE")
    assert run.exit_status == _TERMINATED, run.output
    assert run.elapsed < _EXIT_BOUND_S, run.output
