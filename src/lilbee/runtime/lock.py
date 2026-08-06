"""Cross-process locks: LanceDB write locking and the server singleton.

Write locking combines an in-process mutex with a cross-process file lock
(filelock) so separate processes also coordinate writes. Read consistency is
handled by LanceDB's built-in MVCC via ``read_consistency_interval`` in
``lilbee.data.store``. The server lock makes ``lilbee serve`` a singleton per
data dir.
"""

import json
import logging
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock
from filelock import Timeout as FileLockTimeout

from lilbee.core.config import cfg

log = logging.getLogger(__name__)

# Default timeout (seconds) for acquiring the write lock
LOCK_TIMEOUT = 30.0
# Grace (seconds) for a dying predecessor to release the server lock during a
# restart handoff before a new `lilbee serve` gives up. This budgets the PROMPT
# path only: a predecessor whose llama-swaps honor SIGTERM releases well inside
# it. It deliberately does NOT cover SIGKILL escalation -- the teardown that
# actually holds the lock (_release_engines -> stop_engine -> _stop_stale_swap in
# lilbee.providers.fleet.swap_manager) can spend _ORPHAN_STOP_TIMEOUT_S plus the
# kill and reap waits per group, i.e. tens of seconds across four groups on a
# wedged machine. Sizing this to that worst case would make every ordinary
# restart wait on a pathological one; instead the successor exits with
# LOCK_REFUSAL_EXIT_CODE and the operator retries.
SERVER_LOCK_TIMEOUT = 15.0
_SERVER_LOCK_NAME = "server.lock"
_SCOPE_LOCK_NAME = "server.scope.lock"
_SCOPE_OWNER_NAME = "server.scope.owner.json"
# Minimum blocking wait granted to the in-process mutex even when the file lock
# consumed the whole budget, so a deadline-edge acquire still gets a real attempt.
_MUTEX_MIN_WAIT = 0.1


class LockTimeoutError(TimeoutError):
    """Raised when a lock cannot be acquired within the timeout."""


# In-process write mutex: serializes writers within the same process
_write_mutex = threading.Lock()


def _lock_path(lancedb_dir: Path | None) -> Path:
    return (lancedb_dir if lancedb_dir is not None else cfg.lancedb_dir) / ".lock"


def server_lock_path(data_dir: Path) -> Path:
    """Path of the one-server-per-data-dir lock file."""
    return data_dir / _SERVER_LOCK_NAME


def acquire_server_lock(data_dir: Path, timeout: float = SERVER_LOCK_TIMEOUT) -> FileLock | None:
    """Hold the one-server-per-data-dir lock, or None when a live server owns it.

    The lock is an OS file lock, so the kernel releases it the moment its holder
    exits, however it died; a crashed or killed server leaves no stale state.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    lock = FileLock(server_lock_path(data_dir))
    try:
        lock.acquire(timeout=timeout)
    except FileLockTimeout:
        return None
    return lock


@dataclass(frozen=True)
class ScopeOwner:
    """The data dir the server holding a scope lock is serving, for the refusal message."""

    data_dir: str


@dataclass(frozen=True)
class ScopeHold:
    """A held scope lock plus its owner sidecar; release removes both."""

    lock: FileLock
    owner_path: Path

    def release(self) -> None:
        """Remove the owner sidecar, then free the scope for the next server."""
        self.owner_path.unlink(missing_ok=True)
        self.lock.release()


def acquire_scope_lock(
    scope_dir: Path, data_dir: Path, timeout: float = SERVER_LOCK_TIMEOUT
) -> ScopeHold | None:
    """Hold the one-server-per-scope lock, or None when a live server owns the scope.

    The scope is a directory shared by several would-be servers (the Obsidian
    plugin's shared root). Like the data-dir lock, the OS releases it the moment
    the holder exits. The owner sidecar records which data dir the holder is
    serving so a refused starter can name it in its message.
    """
    scope_dir.mkdir(parents=True, exist_ok=True)
    lock = FileLock(scope_dir / _SCOPE_LOCK_NAME)
    try:
        lock.acquire(timeout=timeout)
    except FileLockTimeout:
        return None
    owner_path = scope_dir / _SCOPE_OWNER_NAME
    owner_path.write_text(json.dumps({"data_dir": str(data_dir)}), encoding="utf-8")
    return ScopeHold(lock, owner_path)


def read_scope_owner(scope_dir: Path) -> ScopeOwner | None:
    """The scope's recorded owner, or None when absent or unreadable."""
    try:
        payload = json.loads((scope_dir / _SCOPE_OWNER_NAME).read_text(encoding="utf-8"))
        return ScopeOwner(data_dir=str(payload["data_dir"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None


@contextmanager
def write_lock(
    lancedb_dir: Path | None = None, timeout: float = LOCK_TIMEOUT
) -> Generator[None, None, None]:
    """Acquire the cross-process file lock then the in-process mutex.

    The file lock lives next to the store's data, so cross-process writers
    coordinate only when they lock the *same* directory: callers pass their
    store's ``lancedb_dir`` (a per-instance ``Lilbee`` uses its own dir).
    ``None`` falls back to the global ``cfg.lancedb_dir``.

    The two stages share one budget: the time spent waiting on the file lock is
    deducted before waiting on the mutex (plus a small ``_MUTEX_MIN_WAIT`` floor),
    so a 30s request cannot stall for roughly twice that.
    """
    deadline = time.monotonic() + timeout
    lock_path = _lock_path(lancedb_dir)
    # The first write to a per-instance store can run before its data dir exists;
    # the file lock cannot be created in a missing directory.
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flock = FileLock(lock_path)
    try:
        flock.acquire(timeout=timeout)
    except FileLockTimeout:
        raise LockTimeoutError("Timed out waiting for exclusive file lock") from None
    try:
        # Floor the mutex budget so a file lock that wins right at the deadline
        # still gets a brief blocking attempt instead of a zero-timeout poll that
        # spuriously fails when another thread holds the mutex for an instant.
        remaining = max(_MUTEX_MIN_WAIT, deadline - time.monotonic())
        acquired = _write_mutex.acquire(timeout=remaining)
        if not acquired:
            raise LockTimeoutError("Timed out waiting for write lock")
        try:
            yield
        finally:
            _write_mutex.release()
    finally:
        flock.release()
