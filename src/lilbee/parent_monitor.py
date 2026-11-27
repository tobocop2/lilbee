"""Watch ``LILBEE_PARENT_PID`` and shut down when the parent process exits."""

from __future__ import annotations

import ctypes
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


def linux_boot_id() -> str | None:
    """The id of this boot, or None where /proc does not give one.

    Start ticks count from boot, so a record that outlives a reboot must hold this id beside them.
    """
    boot_id_file = _PROC_ROOT / "sys/kernel/random/boot_id"
    try:
        return boot_id_file.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


# KUSER_SHARED_DATA is mapped read-only at this address in every Windows process. BootId is
# the field after AlternativeArchitecture (offset 0x2c0, winsiderss/phnt ntexapi.h:8916), so
# it sits at 0x2c4 (ntexapi.h:8397); the loader increments it on each boot.
_WINDOWS_SHARED_USER_DATA = 0x7FFE0000
_WINDOWS_BOOT_ID_OFFSET = 0x2C4
_MACOS_BOOT_SESSION_KEY = b"kern.bootsessionuuid"
_MACOS_BOOT_SESSION_BUFFER = 64


def _read_u32(address: int) -> int:
    """The 32-bit value at *address* of this process; only the Windows boot read calls it."""
    return ctypes.c_uint32.from_address(address).value


def _macos_boot_session() -> str | None:
    """The kernel's per-boot UUID (``sysctl kern.bootsessionuuid``), or None when the read fails."""
    try:
        libc = ctypes.CDLL(None)
        buffer = ctypes.create_string_buffer(_MACOS_BOOT_SESSION_BUFFER)
        size = ctypes.c_size_t(_MACOS_BOOT_SESSION_BUFFER)
        if libc.sysctlbyname(_MACOS_BOOT_SESSION_KEY, buffer, ctypes.byref(size), None, 0) != 0:
            return None
        return buffer.value.decode("ascii") or None
    except (OSError, AttributeError, UnicodeDecodeError):
        return None


def _windows_boot_id() -> str | None:
    """The loader's boot counter from the shared user data page, or None when it cannot be read."""
    try:
        return str(_read_u32(_WINDOWS_SHARED_USER_DATA + _WINDOWS_BOOT_ID_OFFSET))
    except (OSError, ValueError):
        return None


def boot_identity() -> str | None:
    """A value that is equal for every read in one boot and differs between boots.

    A wall-clock step does not change it. None when the platform gives none or the read fails;
    a caller then has no identity to compare and must not trust a monotonic clock reading.
    """
    if _is_linux():
        return linux_boot_id()
    if sys.platform == "darwin":
        return _macos_boot_session()
    if sys.platform == "win32":
        return _windows_boot_id()
    return None


# proc_pid_rusage flavor RUSAGE_INFO_V0 fills struct rusage_info_v0 (sys/resource.h): a
# 16-byte uuid, eight 64-bit counters, then ri_proc_start_abstime and ri_proc_exit_abstime.
_MACOS_RUSAGE_INFO_V0 = 0
_MACOS_RUSAGE_WORDS = 12
_MACOS_START_ABSTIME_INDEX = 10


def _macos_start_abstime(pid: int) -> int | None:
    """Start of *pid* on the mach absolute clock, which no wall-clock step moves.

    None when *pid* is gone or the read fails. psutil's macOS create time follows the wall
    clock, and after a backward step it is off by twice the step.
    """
    try:
        libc = ctypes.CDLL(None)
        buffer = (ctypes.c_uint64 * _MACOS_RUSAGE_WORDS)()
        if libc.proc_pid_rusage(ctypes.c_int(pid), _MACOS_RUSAGE_INFO_V0, buffer) != 0:
            return None
        return int(buffer[_MACOS_START_ABSTIME_INDEX])
    except (OSError, AttributeError):
        return None


def macos_start_stamp(pid: int) -> tuple[str, int] | None:
    """(boot session, mach start) of *pid*, or None off macOS or when either cannot be read.

    The mach clock restarts at boot, so a record that outlives a reboot holds the boot
    session beside the start. The pair names one process across clock steps and reboots.
    """
    if sys.platform != "darwin":
        return None
    boot = _macos_boot_session()
    start = _macos_start_abstime(pid)
    if boot is None or start is None:
        return None
    return boot, start


def linux_start_stamp(pid: int) -> tuple[str, float] | None:
    """(boot id, start ticks) of *pid*, or None off Linux or when either cannot be read.

    The pair names one process across clock steps and reboots.
    """
    if not _is_linux():
        return None
    boot_id = linux_boot_id()
    if boot_id is None:
        return None
    try:
        return boot_id, _linux_start_ticks(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def linux_age_s(boot_id: str, start_ticks: float) -> float | None:
    """Seconds since a process started, or None when the stamp is not from this boot.

    *boot_id* and *start_ticks* come from :func:`linux_start_stamp`. The age is the boot clock
    minus the start ticks, so a wall-clock step does not move it. None off Linux, on another
    boot, and when the boot id cannot be read.
    """
    if not _is_linux() or linux_boot_id() != boot_id:
        return None
    return time.clock_gettime(time.CLOCK_BOOTTIME) - start_ticks / os.sysconf("SC_CLK_TCK")


def _start_identity(pid: int) -> float:
    """What tells *pid* apart from a process that later reuses the pid.

    Each platform's value holds across a wall-clock step and orders processes by start.
    Linux: start ticks since boot. macOS: the start on the mach absolute clock. Windows:
    psutil's create time, the process creation FILETIME, which a step does not move.
    Raises ``psutil.NoSuchProcess`` or ``psutil.AccessDenied``.
    """
    if _is_linux():
        return _linux_start_ticks(pid)
    if sys.platform == "darwin":
        start = _macos_start_abstime(pid)
        if start is None:
            raise psutil.NoSuchProcess(pid)
        return start
    return float(psutil.Process(pid).create_time())


def _parent_start_time(pid: int) -> float | None:
    """Start identity of *pid*, or None if it is gone or unreadable."""
    try:
        return _start_identity(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def _same_process(pid: int, start_time: float | None) -> bool:
    """False when *pid* now belongs to a different process than *start_time*, or is gone."""
    return start_time is None or _parent_start_time(pid) == start_time


def _parent_alive(pid: int, start_time: float | None) -> bool:
    """True while *pid* still refers to the original parent process."""
    return psutil.pid_exists(pid) and _same_process(pid, start_time)


def _started_after_us(start_time: float | None) -> bool:
    """Whether the process at the parent pid started after this one, so it is not the parent.

    The parent pid can name a grandparent (an editor, or a console-script wrapper),
    so the test is the start order, not ``os.getppid()``.
    """
    own = _parent_start_time(os.getpid())
    return start_time is not None and own is not None and start_time > own


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
