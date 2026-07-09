"""Watch ``LILBEE_PARENT_PID`` and shut down when the parent process exits."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections.abc import Callable

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


def _parent_start_time(pid: int) -> float | None:
    """Process create-time for *pid*, or None if it is gone or unreadable."""
    try:
        return float(psutil.Process(pid).create_time())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def _same_process(pid: int, start_time: float | None) -> bool:
    """False when *pid* now belongs to a different process than *start_time*."""
    if start_time is None:
        return True
    try:
        return bool(psutil.Process(pid).create_time() == start_time)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def _parent_alive(pid: int, start_time: float | None) -> bool:
    """True while *pid* still refers to the original parent process."""
    return psutil.pid_exists(pid) and _same_process(pid, start_time)


async def watch_parent_async(
    parent_pid: int,
    on_death: Callable[[], None],
    *,
    poll_interval_secs: float = POLL_INTERVAL_SECS,
) -> None:
    """Poll *parent_pid* until it exits or its PID is recycled, then call *on_death* once."""
    start_time = _parent_start_time(parent_pid)
    while _parent_alive(parent_pid, start_time):
        await asyncio.sleep(poll_interval_secs)
    log.info("%s=%d is no longer alive; triggering shutdown", PARENT_PID_ENV, parent_pid)
    on_death()


def watch_parent_thread(
    parent_pid: int,
    on_death: Callable[[], None],
    *,
    poll_interval_secs: float = POLL_INTERVAL_SECS,
) -> threading.Thread:
    """Daemon thread that fires *on_death* once *parent_pid* exits or its PID is recycled."""

    def _loop() -> None:
        start_time = _parent_start_time(parent_pid)
        while _parent_alive(parent_pid, start_time):
            time.sleep(poll_interval_secs)
        log.info("%s=%d is no longer alive; triggering shutdown", PARENT_PID_ENV, parent_pid)
        on_death()

    thread = threading.Thread(target=_loop, daemon=True, name="lilbee-parent-monitor")
    thread.start()
    return thread
