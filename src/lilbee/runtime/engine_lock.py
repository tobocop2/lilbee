"""Cross-process lifecycle primitives for the shared engine.

The engine (llama-swap + llama-server processes) is machine-level infrastructure:
any lilbee process may build it, every compatible process binds to it, and the
kernel arbitrates liveness through file locks so no pid bookkeeping can go stale.
The mechanics are agnostic to what they front.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from filelock import FileLock
from filelock import Timeout as FileLockTimeout

if TYPE_CHECKING:
    from collections.abc import Iterator

ENGINE_DIR_ENV = "LILBEE_ENGINE_DIR"
_BUILD_LOCK_NAME = "engine.lock"
_USERS_DIRNAME = "engine-users"
_USER_LOCK_SUFFIX = ".lock"
# A live peer refuses its lock instantly; this poll exists only to make the
# probe non-blocking, not to wait.
_PROBE_TIMEOUT_S = 0.0


def machine_engine_dir() -> Path:
    """The per-OS-user engine slot every lilbee process scans first."""
    override = os.environ.get(ENGINE_DIR_ENV, "").strip()
    if override:
        return Path(override)
    if sys.platform == "win32":  # pragma: no cover - platform split
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return base / "lilbee" / "engine"


def private_engine_dir(config_root: Path) -> Path:
    """The overflow engine dir for one config root, used when the slot is incompatible."""
    return config_root / "data" / "engine"


@contextmanager
def build_lock(engine_dir: Path) -> Iterator[None]:
    """Serialize scan-or-build (and config-change restarts) for *engine_dir*."""
    engine_dir.mkdir(parents=True, exist_ok=True)
    lock = FileLock(engine_dir / _BUILD_LOCK_NAME)
    with lock:
        yield


def _users_dir(engine_dir: Path) -> Path:
    return engine_dir / _USERS_DIRNAME


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

        Peer lock files that can be acquired belong to dead processes (the
        kernel released their locks) and are deleted in passing; any refusal
        means a live peer. Idempotent: a second call re-runs the peer probe.
        """
        if self._lock.is_locked:
            self._lock.release()
            self.path.unlink(missing_ok=True)
        return not _live_peer_exists(self.engine_dir)


def hold_user_lock(engine_dir: Path, pid: int | None = None) -> UserLockHold:
    """Hold this process's user lock for *engine_dir* until released or death.

    *pid* names the lock file (defaults to this process); tests pass explicit
    pids to simulate peers from one process.
    """
    path = _user_lock_path(engine_dir, os.getpid() if pid is None else pid)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(path)
    lock.acquire()
    return UserLockHold(engine_dir=engine_dir, path=path, _lock=lock)


def _live_peer_exists(engine_dir: Path) -> bool:
    """Probe every user lock file; clean the dead, report whether any refused."""
    users = _users_dir(engine_dir)
    for path in sorted(users.glob(f"*{_USER_LOCK_SUFFIX}")):
        probe = FileLock(path)
        try:
            probe.acquire(timeout=_PROBE_TIMEOUT_S)
        except FileLockTimeout:
            return True
        probe.release()
        path.unlink(missing_ok=True)
    return False
