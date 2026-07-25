"""Cross-process lifecycle primitives for the shared engine.

The engine (llama-swap + llama-server processes) is machine-level infrastructure:
any lilbee process may build it, every compatible process binds to it, and the
kernel arbitrates liveness through file locks so no pid bookkeeping can go stale.
The mechanics are agnostic to what they front.
"""

from __future__ import annotations

import hashlib
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from filelock import FileLock, SoftFileLock
from filelock import Timeout as FileLockTimeout

if TYPE_CHECKING:
    from collections.abc import Iterator

log = logging.getLogger(__name__)

ENGINE_DIR_ENV = "LILBEE_ENGINE_DIR"
_BUILD_LOCK_NAME = "engine.lock"
# A legitimate build holds the lock only for the llama-swap spawn (its 30 s boot
# budget); the model itself loads lazily on the first request, outside the lock.
# So a wait longer than this is a wedged holder, not honest contention -- bound
# it rather than block forever, which would deadlock every startup and exit.
_BUILD_LOCK_TIMEOUT_S = 90.0
_USERS_DIRNAME = "engine-users"
_USER_LOCK_SUFFIX = ".lock"
# One opt-in file per installation, keyed by config root (not pid, which changes
# every run). Plain files, not locks: an opt-in outlives its process. Per-install
# so a restart reclaims its own prior mark while a peer's stays distinct.
_KEEP_WARM_SUFFIX = ".keep-warm"
# Throwaway per-process file used to ask whether flock really works here.
_FLOCK_PROBE_PREFIX = ".flock-probe."
# Non-blocking probe: a live peer refuses instantly.
_PROBE_TIMEOUT_S = 0.0
# Finite: infinite acquires trip filelock's thread-local deadlock detection
# after a cross-thread release. The pid-named file has no live contender, so
# this never waits in practice.
_HOLD_TIMEOUT_S = 10.0


def machine_engine_dir() -> Path:
    """The per-OS-user engine slot every lilbee process scans first."""
    from lilbee.core.system import default_state_dir

    override = os.environ.get(ENGINE_DIR_ENV, "").strip()
    if override:
        return Path(override)
    return default_state_dir() / "engine"


def private_engine_dir(config_root: Path) -> Path:
    """The overflow engine dir for one config root, used when the slot is incompatible."""
    return config_root / "data" / "engine"


@contextmanager
def build_lock(engine_dir: Path, *, best_effort: bool = False) -> Iterator[None]:
    """Serialize scan-or-build (and config-change restarts) for *engine_dir*.

    Acquired with a finite timeout so a wedged holder cannot deadlock the machine:
    a blocking acquire would hang every startup behind it and, on the shutdown and
    config-change paths, leave processes unable to exit. A wait is logged so a
    stall is visible. On timeout a build caller raises (it could not acquire the
    engine, better than an unbounded hang); a *best_effort* caller -- teardown and
    config-change, which must not wedge a dying or reconfiguring process -- logs
    and proceeds without the lock.
    """
    engine_dir.mkdir(parents=True, exist_ok=True)
    lock = FileLock(engine_dir / _BUILD_LOCK_NAME)
    try:
        lock.acquire(timeout=_PROBE_TIMEOUT_S)
    except FileLockTimeout:
        log.info(
            "Waiting up to %.0fs for another process to build the engine at %s",
            _BUILD_LOCK_TIMEOUT_S,
            engine_dir,
        )
        try:
            lock.acquire(timeout=_BUILD_LOCK_TIMEOUT_S)
        except FileLockTimeout:
            if not best_effort:
                raise
            log.warning(
                "Engine build lock at %s held past %.0fs; proceeding without it.",
                engine_dir,
                _BUILD_LOCK_TIMEOUT_S,
            )
            yield
            return
    try:
        yield
    finally:
        lock.release()


@lru_cache(maxsize=8)
def kernel_arbitrates_locks(engine_dir: Path) -> bool:
    """Whether *engine_dir*'s filesystem gives real kernel-arbitrated file locks.

    The whole membership scheme rests on the kernel releasing a lock on any
    death, so no pid bookkeeping can go stale. On a filesystem where flock
    returns ENOSYS (FUSE, some NFS mounts) filelock silently rewrites itself to
    SoftFileLock with only a Python warning, and the guarantee is gone: that
    fallback path opens the lock file with O_TRUNC and unlinks it before
    re-acquiring, so a process merely *probing* a live member's lock destroys
    it. live_users_exist would then report an empty slot while members are
    serving, and the last-out stop would kill an engine in use.

    Probed by acquiring a throwaway lock in the dir and checking what filelock
    turned it into; cached, since the answer is a property of the mount.
    """
    engine_dir.mkdir(parents=True, exist_ok=True)
    # Named per process: the probe asks what the filesystem supports, which needs
    # no mutual exclusion. A shared probe file would make every lilbee queue on
    # one lock, and this runs while the in-process build lock is held.
    probe_path = engine_dir / f"{_FLOCK_PROBE_PREFIX}{os.getpid()}"
    lock = FileLock(probe_path, thread_local=False)
    try:
        lock.acquire(timeout=_HOLD_TIMEOUT_S)
    except (FileLockTimeout, OSError):
        # An unusable probe file says nothing about flock support; assume the
        # filesystem is fine rather than refuse the shared slot over it.
        return True
    try:
        return not isinstance(lock, SoftFileLock)
    finally:
        lock.release()
        probe_path.unlink(missing_ok=True)


def _users_dir(engine_dir: Path) -> Path:
    return engine_dir / _USERS_DIRNAME


def _keep_warm_path(engine_dir: Path, config_root: Path) -> Path:
    token = hashlib.blake2b(str(config_root).encode(), digest_size=8).hexdigest()
    return _users_dir(engine_dir) / f"{token}{_KEEP_WARM_SUFFIX}"


def request_keep_warm(engine_dir: Path, config_root: Path) -> None:
    """Record that the installation at *config_root* wants *engine_dir*'s engine warm.

    The slot is shared across installations whose configs differ, so any one's
    opt-in keeps the engine warm even when a default-config sibling is last out.
    Keyed by config root, not pid: a restart of the same installation reclaims
    its own mark. ``stop_engine`` clears the set, so a rebuilt engine starts
    unmarked.
    """
    marker = _keep_warm_path(engine_dir, config_root)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()


def withdraw_keep_warm(engine_dir: Path, config_root: Path) -> None:
    """Drop this installation's opt-in for *engine_dir*, leaving every peer's intact.

    Reclaims the mark this installation left, so flipping the setting off (even
    across a restart) lets the engine stop instead of staying warm forever.
    """
    _keep_warm_path(engine_dir, config_root).unlink(missing_ok=True)


def keep_warm_requested(engine_dir: Path) -> bool:
    """Whether any user of *engine_dir* asked for the engine to stay resident.

    Not gated on liveness: an opt-in means "outlive me", so an exited user's
    marker is the case it exists for.
    """
    return any(_users_dir(engine_dir).glob(f"*{_KEEP_WARM_SUFFIX}"))


def clear_keep_warm(engine_dir: Path) -> None:
    """Forget every persistence opt-in for *engine_dir*; the engine they applied to is gone."""
    for marker in _users_dir(engine_dir).glob(f"*{_KEEP_WARM_SUFFIX}"):
        marker.unlink(missing_ok=True)


def _user_lock_path(engine_dir: Path, pid: int) -> Path:
    return _users_dir(engine_dir) / f"{pid}{_USER_LOCK_SUFFIX}"


@dataclass
class UserLockHold:
    """One process's held membership in an engine's user set."""

    engine_dir: Path
    path: Path
    _lock: FileLock = field(repr=False)

    def release_and_check_last(self) -> bool:
        """Release this hold and report whether no live peers remain.

        Idempotent. The lock file is removed when the last in-process hold
        releases; acquirable peer files belong to dead processes and are
        deleted in passing.
        """
        if self._lock.is_locked:
            self._lock.release()
            if not self._lock.is_locked:
                self.path.unlink(missing_ok=True)
        return not live_users_exist(self.engine_dir)


def hold_user_lock(engine_dir: Path, pid: int | None = None) -> UserLockHold:
    """Hold this process's user lock for *engine_dir* until released or death.

    *pid* names the lock file (defaults to this process); tests pass explicit
    pids to simulate peers from one process.
    """
    path = _user_lock_path(engine_dir, os.getpid() if pid is None else pid)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = _user_file_lock(path)
    lock.acquire(timeout=_HOLD_TIMEOUT_S)
    return UserLockHold(engine_dir=engine_dir, path=path, _lock=lock)


def _user_file_lock(path: Path) -> FileLock:
    """One process-wide reentrant lock instance per user-lock path.

    Not thread-local: acquire and release run on different threads.
    Singleton: two providers in one process hold the same pid-named file,
    and separate instances over fcntl falsely succeed or trip filelock's
    deadlock detection.
    """
    return FileLock(path, thread_local=False, is_singleton=True)


def live_users_exist(engine_dir: Path) -> bool:
    """Probe every user lock file; clean the dead, report whether any refused."""
    users = _users_dir(engine_dir)
    for path in sorted(users.glob(f"*{_USER_LOCK_SUFFIX}")):
        probe = _user_file_lock(path)
        if probe.is_locked:
            # Held by this process; acquiring would reentrantly succeed.
            return True
        try:
            probe.acquire(timeout=_PROBE_TIMEOUT_S)
        except FileLockTimeout:
            return True
        probe.release()
        path.unlink(missing_ok=True)
    return False
