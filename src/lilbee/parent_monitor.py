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


# macOS create time has one-second resolution and a clock step moves it by the step minus
# up to two seconds. Measured on a macos-latest runner across a real +2 h step and the
# restore: the derived start sat 0.9 s and 1.9 s from the spawn reading. A pid reused
# later starts seconds or more after the record.
MONO_START_TOLERANCE_S = 3.0


def started_near_mono(create_time: float, started_mono: float, started_boot: str | None) -> bool:
    """True when a process created at *create_time* started where *started_mono* says.

    psutil's create time and the wall clock move together in a step, so the wall-clock
    age taken from the create time gives a start on the monotonic clock that does not.
    A reading from another boot, or with no boot identity, says nothing about the process.
    """
    if started_boot is None or started_boot != boot_identity():
        return False
    derived = time.monotonic() - (time.time() - create_time)
    return abs(derived - started_mono) <= MONO_START_TOLERANCE_S


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

    Linux: start ticks since boot. Elsewhere: psutil's create time, which a wall-clock step
    moves on macOS; :func:`_same_process` also accepts a start that holds on the monotonic clock.
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


def _mono_stamp(start_time: float | None) -> tuple[float, str] | None:
    """(start on the monotonic clock, boot identity) for a create-time identity, or None.

    None on Linux, where the identity is start ticks, and when the boot cannot be named.
    """
    boot = None if _is_linux() or start_time is None else boot_identity()
    if start_time is None or boot is None:
        return None
    return time.monotonic() - (time.time() - start_time), boot


def _same_process(
    pid: int, start_time: float | None, mono_stamp: tuple[float, str] | None = None
) -> bool:
    """False when *pid* now belongs to a different process than *start_time*.

    A create time that differs still passes when the process's start on the monotonic
    clock, from the same boot, matches *mono_stamp* within the tolerance.
    """
    if start_time is None:
        return True
    try:
        identity = _start_identity(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    if identity == start_time:
        return True
    return mono_stamp is not None and started_near_mono(identity, *mono_stamp)


def _parent_alive(
    pid: int, start_time: float | None, mono_stamp: tuple[float, str] | None = None
) -> bool:
    """True while *pid* still refers to the original parent process."""
    return psutil.pid_exists(pid) and _same_process(pid, start_time, mono_stamp)


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
        mono_stamp = _mono_stamp(start_time)
        if not _started_after_us(start_time):
            while _parent_alive(parent_pid, start_time, mono_stamp):
                time.sleep(poll_interval_secs)
        log.info("%s=%d is no longer alive; triggering shutdown", PARENT_PID_ENV, parent_pid)
        on_death()

    thread = threading.Thread(target=_loop, daemon=True, name="lilbee-parent-monitor")
    thread.start()
    return thread
