"""Cross-process lifecycle primitives for the shared engine.

The engine (llama-swap + llama-server processes) is machine-level infrastructure:
any lilbee process may build it, every compatible process binds to it, and the
kernel arbitrates liveness through file locks so no pid bookkeeping can go stale.
The mechanics are agnostic to what they front.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from filelock import FileLock
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
# Marks an engine whose users asked it to outlive them. A plain file, not a
# lock: it outlives every process by design.
_KEEP_WARM_NAME = "keep-warm"
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


def _users_dir(engine_dir: Path) -> Path:
    return engine_dir / _USERS_DIRNAME


def request_keep_warm(engine_dir: Path) -> None:
    """Record that a user of *engine_dir* wants the engine to outlive it.

    The opt-in belongs to the engine, not to a process: the machine slot is
    shared across installations whose configs differ, so reading the setting
    from whichever process happens to exit last hands the decision to an
    arbitrary sibling. Any user that opts in marks the engine, and the mark
    lasts exactly as long as the engine instance it describes -- ``stop_engine``
    clears it, so a rebuilt engine starts from whoever opts in next.
    """
    marker = engine_dir / _KEEP_WARM_NAME
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()


def keep_warm_requested(engine_dir: Path) -> bool:
    """Whether any user of *engine_dir* asked for the engine to stay resident."""
    return (engine_dir / _KEEP_WARM_NAME).exists()


def clear_keep_warm(engine_dir: Path) -> None:
    """Forget *engine_dir*'s persistence opt-in; the engine it applied to is gone."""
    (engine_dir / _KEEP_WARM_NAME).unlink(missing_ok=True)


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
