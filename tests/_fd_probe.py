"""Branch-only probe for a closed lock descriptor.

It records each flock owner and dumps a closer's stack. It is active only when
LILBEE_TEST_FD_PROBE_DIR is set; .github/diag/fd_closer.bt sends SIGUSR2 on a
close of a held fd.
"""

import contextlib
import faulthandler
import fcntl
import os
import signal
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import IO, TextIO

import pytest

_ENV_DIR = "LILBEE_TEST_FD_PROBE_DIR"
# fd_closer.bt arms a process that closes this number; the close itself fails with EBADF.
_ARM_FD = 1070596096
_LOCK_OPS = fcntl.LOCK_SH | fcntl.LOCK_EX
_STACK_LIMIT = 26


class FdProbe:
    """Writes one owner record per successful flock and keeps the faulthandler dump file open."""

    def __init__(self, out_dir: Path) -> None:
        pid = os.getpid()
        self._owners: TextIO = (out_dir / f"fdp-owner-{pid}.txt").open("a", encoding="utf-8")
        self._dumps: TextIO = (out_dir / f"fdp-dump-{pid}.txt").open("a", encoding="utf-8")
        self._write_lock = threading.Lock()
        self._real_flock: Callable[[int, int], None] = fcntl.flock

    def install(self) -> None:
        faulthandler.register(signal.SIGUSR2, file=self._dumps, all_threads=True)
        fcntl.flock = self._flock  # type: ignore[assignment]
        with contextlib.suppress(OSError):
            os.close(_ARM_FD)

    def _flock(self, fd: int | IO[bytes], operation: int) -> None:
        self._real_flock(fd, operation)  # type: ignore[arg-type]
        if operation & _LOCK_OPS:
            self._record(fd if isinstance(fd, int) else fd.fileno(), operation)

    def _record(self, fd: int, operation: int) -> None:
        try:
            path = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            path = "?"
        stack = traceback.format_stack(limit=_STACK_LIMIT)[:-2]
        lines = [
            f"FDP-OWNER pid={os.getpid()} tid={threading.get_native_id()} fd={fd} op={operation} "
            f"ns={time.monotonic_ns()} path={path}\n"
        ]
        lines += [f"FDP-OWNER-STACK {frame.strip().splitlines()[0]}\n" for frame in stack]
        with self._write_lock:
            self._owners.writelines(lines)
            self._owners.flush()


def pytest_configure(config: pytest.Config) -> None:
    raw = os.environ.get(_ENV_DIR, "").strip()
    if not raw:
        return
    out_dir = Path(raw)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe = FdProbe(out_dir)
    probe.install()
    config.pluginmanager.register(probe, "lilbee-fd-probe")
