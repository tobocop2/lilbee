"""Evidence for a child run that stalls: its output tail, its process tree and its stacks.

A test that starts a child process under a time limit learns only "timed out" when the limit
passes, and the limit kills the child with the evidence. ``StallWatch`` takes a snapshot at a
share of the limit, while the child still lives. ``StalledRun`` is a ``subprocess.TimeoutExpired``
whose text holds the snapshot and the tail of the child's output, so the failure explains itself.

The snapshot lists every descendant of this process: its state, its command, and on Linux its
kernel wait channel and signal masks from ``/proc``. When ``py-spy`` is on ``PATH`` and the
kernel lets this user attach, it adds a stack dump of each descendant. Other systems skip the
``/proc`` fields and still get the tree.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from types import TracebackType

import psutil

# The snapshot is taken at this share of the limit, before the limit kills the child.
SNAPSHOT_SHARE = 0.75
OUTPUT_TAIL_CHARS = 3000
_PY_SPY = "py-spy"
_PY_SPY_LIMIT_S = 5.0
_PY_SPY_TAIL_CHARS = 3500
_COMMAND_CHARS = 160
_JOIN_WAIT_S = 10.0
_STATUS_FIELDS = ("State", "PPid", "SigBlk", "SigIgn", "SigCgt")
_PROC = Path("/proc")


def _proc_text(pid: int, name: str) -> str:
    """The text of ``/proc/<pid>/<name>``, or a short reason it cannot be read."""
    try:
        return (_PROC / str(pid) / name).read_text(encoding="utf-8").strip()
    except OSError as err:
        return f"unreadable ({err.strerror or err})"


def _status_fields(pid: int) -> str:
    lines = _proc_text(pid, "status").splitlines()
    return " ".join(
        line.replace("\t", " ") for line in lines if line.startswith(tuple(_STATUS_FIELDS))
    )


def _py_spy_dump(pid: int, py_spy: str | None) -> str:
    """The stacks of one process, or the reason there are none."""
    if py_spy is None:
        return f"{_PY_SPY} is not on PATH"
    try:
        done = subprocess.run(            [py_spy, "dump", "--pid", str(pid)],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=_PY_SPY_LIMIT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as err:
        return f"{_PY_SPY} failed: {err!r}"
    text = done.stdout[-_PY_SPY_TAIL_CHARS:] if done.returncode == 0 else done.stderr
    return text.strip() or f"{_PY_SPY} exited {done.returncode} with no output"


def _describe(proc: psutil.Process, *, on_linux: bool, with_stacks: bool) -> list[str]:
    try:
        command = " ".join(proc.cmdline())[:_COMMAND_CHARS]
        lines = [f"pid={proc.pid} ppid={proc.ppid()} status={proc.status()} {command}"]
    except psutil.Error as err:
        return [f"pid={proc.pid} gone: {err}"]
    if on_linux:
        lines.append(f"  {_status_fields(proc.pid)}")
        lines.append(f"  wchan={_proc_text(proc.pid, 'wchan')}")
    if with_stacks:
        lines.append("  " + _py_spy_dump(proc.pid, shutil.which(_PY_SPY)).replace("\n", "\n  "))
    return lines


def process_tree_report(*, with_stacks: bool = True) -> str:
    """Every descendant of this process, one block each, with its stacks when asked."""
    on_linux = _PROC.joinpath("self", "status").exists()
    lines = [f"process tree of pid {os.getpid()} at {time.strftime('%H:%M:%S')}"]
    for proc in psutil.Process().children(recursive=True):
        lines.extend(_describe(proc, on_linux=on_linux, with_stacks=with_stacks))
    if len(lines) == 1:
        lines.append("  no descendants")
    return "\n".join(lines)


def _tail(output: str | bytes | None) -> str:
    """The end of a child's output; a timeout hands it back as bytes on POSIX."""
    if output is None:
        return ""
    text = output.decode("utf-8", errors="replace") if isinstance(output, bytes) else output
    return text[-OUTPUT_TAIL_CHARS:]


class StalledRun(subprocess.TimeoutExpired):
    """A timed-out child run whose text carries the output tail and the snapshot."""

    def __init__(self, expired: subprocess.TimeoutExpired, snapshot: str) -> None:
        super().__init__(expired.cmd, expired.timeout, output=expired.output)
        self.snapshot = snapshot

    def __str__(self) -> str:
        return (
            f"{super().__str__()}\n"
            f"--- last {OUTPUT_TAIL_CHARS} characters of the child's output ---\n"
            f"{_tail(self.output)}\n"
            f"--- {self.snapshot}"
        )


class StallWatch:
    """Take one snapshot of the process tree at a share of the limit, unless the run ends first."""

    def __init__(self, timeout_s: float, *, share: float = SNAPSHOT_SHARE) -> None:
        self._delay_s = timeout_s * share
        self._done = threading.Event()
        self._snapshot = ""
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> StallWatch:
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._done.set()
        self._thread.join(_JOIN_WAIT_S)

    def _run(self) -> None:
        if not self._done.wait(self._delay_s):
            self._snapshot = process_tree_report()

    def stalled(self, expired: subprocess.TimeoutExpired) -> StalledRun:
        """The expired run with the snapshot attached; waits for a snapshot still in progress."""
        self._thread.join(_JOIN_WAIT_S)
        return StalledRun(expired, self._snapshot or "no snapshot was taken")
