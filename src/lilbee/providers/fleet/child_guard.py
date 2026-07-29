"""Bind a spawned llama-swap's lifetime to this process, so a crash cannot orphan it.

A graceful exit tears the fleet down and a stale engine is reaped on the next
launch; this closes the window between, when lilbee dies without cleanup (SIGKILL,
segfault, closed terminal). Each platform binds differently and falls back to the
next-launch reap when its primitive is unavailable, so a failure here never fails
a spawn:

* Linux: ``PR_SET_PDEATHSIG``, set in the preexec of one process-lifetime thread
  (the signal binds to the forking thread, not the process).
* Windows: a kill-on-close job object whose handle is held for the process life.
* Elsewhere (macOS, any POSIX host without prctl): a death pipe (see
  :func:`_watch_via_death_pipe`).

``processfamily`` covers only prctl+job (needs pywin32, skips macOS), so the
syscalls stay custom; the executor is the stdlib's.
"""

from __future__ import annotations

import contextlib
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

# prctl(2) PR_SET_PDEATHSIG=1; SIGTERM so llama-swap runs its own shutdown.
_PR_SET_PDEATHSIG = 1
_PDEATHSIG = 15

# Resolved at import so the post-fork child touches an already-loaded handle: a
# dlopen after fork can deadlock on a lock a sibling thread holds.
_libc: ctypes.CDLL | None = None
if sys.platform.startswith("linux"):  # pragma: no cover - Linux only
    try:
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
    except OSError:
        _libc = None

# Windows job-object constants (winnt.h).
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_OBJECT_EXTENDED_LIMIT_CLASS = 9
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001

# Live death-pipe write ends, keyed by guarded pid. os.pipe() returns bare fds the
# GC never closes, so this is the only handle to close one deliberately (on child
# stop) or leave for the kernel to close on our death (which fires the binding).
_death_pipe_write_fds: dict[int, int] = {}


def _make_pdeathsig_preexec(parent_pid: int) -> Callable[[], None] | None:
    """A preexec binding the child's death to *parent_pid*, or None if libc is absent.

    Compares against the real pid, not 1, so a lilbee running as pid 1 (a container
    without an init shim) is not mistaken for a dead parent.
    """
    if _libc is None:
        return None
    libc = _libc

    def _set_pdeathsig() -> None:
        # In the forked child pre-exec: touch only pre-resolved handles, no alloc.
        libc.prctl(_PR_SET_PDEATHSIG, _PDEATHSIG, 0, 0, 0)
        if os.getppid() != parent_pid:
            os._exit(1)

    return _set_pdeathsig


class _LifetimeSpawner:
    """Runs spawns on one process-lifetime thread so PR_SET_PDEATHSIG binds to it.

    The death signal binds to the forking thread, so a one-worker
    ``ThreadPoolExecutor`` (reused for every spawn, stopped only at exit or the
    test-only ``close``) keeps that thread alive for the process.
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
        """Stop the spawner thread. Test-only; in production it lives forever."""
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True)


_spawner = _LifetimeSpawner()


def _assign_to_kill_on_close_job(pid: int) -> None:  # pragma: no cover - Windows only
    """Put *pid* in a kill-on-close job object.

    The job handle is deliberately leaked: holding it open is what kills the child
    when this process ends, and the OS reclaims it then.
    """
    # windll is a Windows-only loader absent from other platforms' stubs; the
    # Any alias keeps the checker quiet without an attr-defined ignore.
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


def _watch_via_death_pipe(pid: int) -> None:
    """Signal *pid* from a detached ``sh`` watcher once a pipe reaches EOF.

    Portable stand-in for prctl/job objects: we hold the pipe's write end until the
    child stops or we die, and the watcher (reading the read end as its stdin)
    signals *pid* at EOF. The write end is O_CLOEXEC, so exec'd children never
    inherit it; a fork-without-exec child would keep the pipe open past our death,
    so a bound child's owner must not fork workers (serve is single-process).

    The read end is passed as the watcher's stdin rather than redirected with
    ``<&N``: dash mis-dups a multi-digit fd there, silently binding the wrong one.
    Best-effort; ``kill -0`` skips an already-dead pid, and :func:`release_death_pipe`
    keeps the pid-recycle window to the child's own stop.
    """
    try:
        read_fd, write_fd = os.pipe()
    except OSError:
        log.info("Could not bind the engine to this process; the next launch reaps it.")
        return
    script = f"read -r _; kill -0 {pid} 2>/dev/null && kill {pid}"
    try:
        _spawner.spawn(
            ["/bin/sh", "-c", script],
            stdin=read_fd,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        os.close(write_fd)
        log.info("Could not bind the engine to this process; the next launch reaps it.")
    else:
        # A crashed-without-release child can leave a stale entry the OS recycles
        # this pid into; close it before the overwrite so it never leaks.
        release_death_pipe(pid)
        _death_pipe_write_fds[pid] = write_fd
    finally:
        os.close(read_fd)


def release_death_pipe(pid: int) -> None:
    """Close the death pipe guarding *pid* so its watcher wakes and exits.

    Called when the guarded child is stopped while this process lives on (fleet
    reload / model switch), instead of leaving the watcher parked until our death.
    A no-op for a pid with no death pipe (kernel-bound platforms, or already released).
    """
    write_fd = _death_pipe_write_fds.pop(pid, None)
    if write_fd is not None:
        with contextlib.suppress(OSError):
            os.close(write_fd)


def spawn_bound_child(
    argv: list[str],
    *,
    bind_lifetime: bool = True,
    death_pipe: bool = True,
    **popen_kwargs: Any,
) -> subprocess.Popen[Any]:
    """Spawn *argv*, by default bound to this process's lifetime.

    ``bind_lifetime`` is False when the child is meant to outlive this process
    (``keep_engine_warm``); the next-launch reap is then the only cleanup.

    ``death_pipe`` is False for a short-lived child: the pipe's watcher lives until
    we die, so binding a child that outlives neither costs a process and an fd for
    nothing. Kernel bindings have no such cost and still apply.

    Pass ``start_new_session=True`` to also put the child in its own group; the
    binding does not depend on it.
    """
    if not bind_lifetime:
        return _spawner.spawn(argv, **popen_kwargs)

    preexec = _make_pdeathsig_preexec(os.getpid())
    if preexec is not None:
        try:
            return _spawner.spawn(argv, preexec_fn=preexec, **popen_kwargs)
        except OSError:
            log.info("Could not set the death signal on the engine; trying the death pipe.")

    proc = _spawner.spawn(argv, **popen_kwargs)
    if sys.platform == "win32":
        try:
            _assign_to_kill_on_close_job(proc.pid)
        except OSError:
            log.info("Could not bind the engine to this process; the next launch reaps it.")
    elif death_pipe:
        _watch_via_death_pipe(proc.pid)
    return proc
