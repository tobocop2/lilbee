"""Bind a spawned llama-swap's lifetime to this process, so a crash cannot orphan it.

A graceful exit already tears the fleet down, and a stale engine is reaped on the
next launch. This closes the window in between: when lilbee dies without running
cleanup (SIGKILL, a segfault, a closed terminal), the engine and its VRAM would
otherwise survive until something launches again.

Each platform binds differently, and each falls back to the existing reap when its
mechanism is unavailable, so a failure here never fails a spawn:

* Linux: ``PR_SET_PDEATHSIG`` asks the kernel to signal the child when its parent
  dies. The catch is that "parent" is the *thread* that forked, not the process,
  so the spawn is routed through one long-lived thread whose lifetime is the
  process's. Binding to a transient worker thread would kill the engine the moment
  that thread returned.
* Windows: the child joins a job object marked kill-on-close. The job handle is
  held for the life of the process, so every child in it dies when the handle does.
* macOS: no equivalent primitive, so the spawn is plain and the reap covers it.

Not a third-party library: ``processfamily`` wraps the same primitives but needs
``pywin32`` (the standalone build cannot bundle it) and its ``prctl`` path skips
macOS. The syscalls are a few ctypes lines and the process-lifetime routing is
lilbee-specific, so they stay custom; the single-worker executor is the stdlib's.
"""

from __future__ import annotations

import ctypes
import logging
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)

# Linux prctl(2): PR_SET_PDEATHSIG is option 1. SIGTERM (15) lets llama-swap run
# its own shutdown; the kernel escalates nothing, so a wedged child is still the
# reap's job.
_PR_SET_PDEATHSIG = 1
_PDEATHSIG = 15

# Resolve libc in the parent, at import, so the post-fork child touches only an
# already-loaded handle. A dlopen (or a Python import) after fork can deadlock on
# the linker/import lock a sibling thread may hold, and lilbee is heavily threaded.
_libc: ctypes.CDLL | None = None
if sys.platform.startswith("linux"):  # pragma: no cover - Linux only
    try:
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
    except OSError:
        _libc = None

# Windows job-object constants (winnt.h). Extended-limit information carries the
# kill-on-close flag; the class ordinal is JobObjectExtendedLimitInformation. The
# access rights are the minimum OpenProcess needs to place a pid in a job.
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_OBJECT_EXTENDED_LIMIT_CLASS = 9
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001


def _set_pdeathsig() -> None:  # pragma: no cover - Linux child, runs post-fork
    """Ask the kernel to send SIGTERM to this child when its parent thread dies.

    Runs in the forked child before exec. Touches only pre-resolved handles (libc
    and os are loaded in the parent), since a fork in a threaded process must
    neither allocate nor take a lock a sibling thread may hold. A parent that
    already died between fork and here leaves getppid at 1 (reparented to init),
    so the child exits rather than linger.
    """
    if _libc is not None:
        _libc.prctl(_PR_SET_PDEATHSIG, _PDEATHSIG, 0, 0, 0)
    if os.getppid() == 1:
        os._exit(1)


class _LifetimeSpawner:
    """Runs subprocess spawns on one process-lifetime thread so PR_SET_PDEATHSIG holds.

    The death signal binds to the thread that forks, not the process, so the fork
    must happen on a thread that lives as long as the process. A one-worker
    ``ThreadPoolExecutor`` is exactly that: created on first spawn, its single
    thread is reused for every spawn and only stops at interpreter exit (or the
    test-only ``close``). Spawns are infrequent, so serializing them costs nothing.
    """

    def __init__(self) -> None:
        self._executor: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()

    def spawn(self, *args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="fleet-spawner"
                )
            executor = self._executor
        return executor.submit(subprocess.Popen, *args, **kwargs).result()

    def close(self) -> None:
        """Stop the spawner thread. Unused in production, where it lives forever."""
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True)


_spawner = _LifetimeSpawner()


def _assign_to_kill_on_close_job(pid: int) -> None:  # pragma: no cover - Windows only
    """Put *pid* in a job object that kills its members when this process exits.

    One job is created per child and its handle deliberately leaked: the process
    holding an open kill-on-close handle is what makes the child die with it, and
    the handle is reclaimed by the OS when the process ends.
    """
    # windll is a Windows-only stdlib loader, absent from the type stubs on other
    # platforms; reaching it through an Any-typed alias keeps the checker quiet
    # without an attr-defined ignore, and this runs only on win32.
    ct: Any = ctypes
    kernel32 = ct.windll.kernel32

    class JobObjectBasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in ("r", "w", "o", "rb", "wb", "ob")]

    class JobObjectExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JobObjectBasicLimitInformation),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise OSError("CreateJobObjectW failed")
    info = JobObjectExtendedLimitInformation()
    info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(
        job, _JOB_OBJECT_EXTENDED_LIMIT_CLASS, ctypes.byref(info), ctypes.sizeof(info)
    ):
        raise OSError("SetInformationJobObject failed")
    handle = kernel32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)
    if not handle:
        raise OSError("OpenProcess failed")
    try:
        if not kernel32.AssignProcessToJobObject(job, handle):
            raise OSError("AssignProcessToJobObject failed")
    finally:
        kernel32.CloseHandle(handle)


def _pdeathsig_preexec() -> Callable[[], None] | None:
    """The Linux death-signal preexec, or None where it does not apply.

    None when libc did not resolve, so the spawn is plain and the reap covers it.
    """
    return _set_pdeathsig if _libc is not None else None


def spawn_llama_swap(argv: list[str], **popen_kwargs: Any) -> subprocess.Popen[Any]:
    """Spawn llama-swap bound to this process's lifetime, falling back to a plain spawn.

    The binding is best-effort: if the platform primitive is unavailable, the
    child is spawned anyway and the stale-engine reap on the next launch remains
    the backstop.
    """
    preexec = _pdeathsig_preexec()
    if preexec is not None:
        try:
            return _spawner.spawn(argv, preexec_fn=preexec, **popen_kwargs)
        except OSError:
            log.info("Could not bind the engine to this process; the next launch reaps it.")
            return _spawner.spawn(argv, **popen_kwargs)

    proc = _spawner.spawn(argv, **popen_kwargs)
    if sys.platform == "win32":
        try:
            _assign_to_kill_on_close_job(proc.pid)
        except OSError:
            log.info("Could not bind the engine to this process; the next launch reaps it.")
    return proc
