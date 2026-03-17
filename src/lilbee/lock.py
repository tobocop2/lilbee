"""Write locking for LanceDB access.

Combines an in-process mutex with a cross-process file lock (filelock)
so separate processes also coordinate writes. Read consistency is handled
by LanceDB's built-in MVCC via read_consistency_interval in store.py.
"""

import logging
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock
from filelock import Timeout as FileLockTimeout

from lilbee.config import cfg

log = logging.getLogger(__name__)

# Default timeout (seconds) for acquiring the write lock
LOCK_TIMEOUT = 30.0


class LockTimeoutError(TimeoutError):
    """Raised when a lock cannot be acquired within the timeout."""


# In-process write mutex — serializes writers within the same process
_write_mutex = threading.Lock()


def _lock_path() -> Path:
    return cfg.lancedb_dir / ".lock"


@contextmanager
def write_lock(timeout: float = LOCK_TIMEOUT) -> Generator[None, None, None]:
    """Context manager: acquire exclusive file lock then in-process mutex."""
    flock = FileLock(_lock_path())
    try:
        flock.acquire(timeout=timeout)
    except FileLockTimeout:
        raise LockTimeoutError("Timed out waiting for exclusive file lock") from None
    try:
        acquired = _write_mutex.acquire(timeout=timeout)
        if not acquired:
            raise LockTimeoutError("Timed out waiting for write lock")
        try:
            yield
        finally:
            _write_mutex.release()
    finally:
        flock.release()
