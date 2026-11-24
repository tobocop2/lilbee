"""Watch ``LILBEE_PARENT_PID`` and shut down when the parent process exits."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import psutil

log = logging.getLogger(__name__)

POLL_INTERVAL_SECS = 2.0
PARENT_PID_ENV = "LILBEE_PARENT_PID"


def parse_parent_pid(env: dict[str, str] | None = None) -> int | None:
    """Return a valid parent PID from the env, or None if unset/garbage."""
    src = env if env is not None else os.environ
    raw = src.get(PARENT_PID_ENV)
    if not raw:
        return None
    try:
        pid = int(raw)
    except ValueError:
        log.warning("%s=%r is not an integer; skipping parent-death monitor", PARENT_PID_ENV, raw)
        return None
    if pid <= 0:
        log.warning("%s=%d is non-positive; skipping parent-death monitor", PARENT_PID_ENV, pid)
        return None
    return pid


_PROC_ROOT = Path("/proc")
_STAT_START_TICKS_INDEX = 19  # field 22 of /proc/<pid>/stat, counted after the "(comm)" field


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _linux_start_ticks(pid: int) -> float:
    """Start time of *pid* in ticks since boot, which a wall-clock step does not change.

    psutil's Linux create_time adds the ``btime`` line of /proc/stat to these ticks and reads
    it again on each call, so a clock step moves it. Raises the psutil errors for a missing
    or unreadable process, as ``psutil.Process`` does.
    """
    try:
        raw = (_PROC_ROOT / str(pid) / "stat").read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, ProcessLookupError):
        raise psutil.NoSuchProcess(pid) from None
    except PermissionError:
        raise psutil.AccessDenied(pid) from None
    try:
        # The comm field may hold spaces and parentheses; the last ")" ends it.
        return float(raw.rpartition(")")[2].split()[_STAT_START_TICKS_INDEX])
    except (ValueError, IndexError):
        raise psutil.AccessDenied(pid) from None


def _start_identity(pid: int) -> float:
    """What tells *pid* apart from a process that later reuses the pid.

    Linux: start ticks since boot. Elsewhere: psutil's create time, which is fixed there.
    Raises ``psutil.NoSuchProcess`` or ``psutil.AccessDenied``.
    """
    if _is_linux():
        return _linux_start_ticks(pid)
    return float(psutil.Process(pid).create_time())


def _parent_start_time(pid: int) -> float | None:
    """Start identity of *pid*, or None if it is gone or unreadable."""
    try:
        return _start_identity(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def _same_process(pid: int, start_time: float | None) -> bool:
    """False when *pid* now belongs to a different process than *start_time*."""
    if start_time is None:
        return True
    try:
        return _start_identity(pid) == start_time
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def _parent_alive(pid: int, start_time: float | None) -> bool:
    """True while *pid* still refers to the original parent process."""
    return psutil.pid_exists(pid) and _same_process(pid, start_time)


def _started_after_us(start_time: float | None) -> bool:
    """Whether the process at the parent pid started after this one, so it is not the parent.

    The parent pid can name a grandparent (an editor, or a console-script wrapper),
    so the test is the create-time order, not ``os.getppid()``.
    """
    return start_time is not None and start_time > _start_identity(os.getpid())


def watch_parent_thread(
    parent_pid: int,
    on_death: Callable[[], None],
    *,
    poll_interval_secs: float = POLL_INTERVAL_SECS,
) -> threading.Thread:
    """Daemon thread that fires *on_death* once *parent_pid* exits or its PID is recycled."""

    def _loop() -> None:
        start_time = _parent_start_time(parent_pid)
        if not _started_after_us(start_time):
            while _parent_alive(parent_pid, start_time):
                time.sleep(poll_interval_secs)
        log.info("%s=%d is no longer alive; triggering shutdown", PARENT_PID_ENV, parent_pid)
        on_death()

    thread = threading.Thread(target=_loop, daemon=True, name="lilbee-parent-monitor")
    thread.start()
    return thread
