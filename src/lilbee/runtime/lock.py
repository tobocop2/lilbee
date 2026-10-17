"""Cross-process locks: LanceDB write locking, the sync mark, and the server singleton.

Write locking combines an in-process mutex with a cross-process file lock
(filelock) so separate processes also coordinate writes. Read consistency is
handled by LanceDB's built-in MVCC via ``read_consistency_interval`` in
``lilbee.data.store``. The server lock makes ``lilbee serve`` a singleton per
data dir.
"""

import asyncio
import errno
import json
import logging
import sqlite3
import sys
import threading
import time
import uuid
from collections.abc import AsyncGenerator, Generator, Iterator
from contextlib import (
    AbstractContextManager,
    ExitStack,
    asynccontextmanager,
    closing,
    contextmanager,
)
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, NoReturn, Self, overload

from filelock import AsyncReadWriteLock, FileLock, ReadWriteLock
from filelock import Timeout as FileLockTimeout

from lilbee.core.config import Config, cfg
from lilbee.runtime.cancellation import CancelSignal, TaskCancelledError

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
_SYNC_LOCK_NAME = "sync.lock"
_SYNC_PROBE_PREFIX = "sync.probe."
# How long the probe waits out a momentary busy before it concludes the filesystem cannot lock.
_PROBE_WAIT_S = 0.3
# Longest attempt of a wait that takes the sync lock in slices; a cancelled wait returns
# within it.
_SYNC_TAKE_SLICE_S = 0.1
# What the filesystem raises when the sync lock cannot be created or taken: OSError
# from making the data root, sqlite3.Error from SQLite (a file that is not a
# database, or a mount that refuses its locks). A busy lock arrives as filelock's
# Timeout instead, which is itself an OSError, so it is caught first.
_SYNC_LOCK_ERRORS = (OSError, sqlite3.Error)
_SYNC_LOCK_REFUSED = "Cannot lock %s (%s); a reset refuses until it can."
_SYNC_RUNNING = "A sync, an import, an add or a wiki build is running on this library."
_RESET_WHEN_DONE = "Reset again when it finishes."
_SYNC_LOCK_UNKNOWN = (
    "Cannot tell whether a sync or import is running: {path} cannot be locked ({error}). "
    "Stop every lilbee process, delete {path}, and {again}."
)
_SYNC_LOCK_NOT_CLOSED = "Could not close %s (%s)."
_DATA_ROOT_UNLOCKABLE = (
    "Cannot tell whether a sync or import is running: {path} cannot be locked ({error})."
)
_SYNC_REFUSED = "Cannot change the library: {path} cannot be locked ({error}). {advice}"
_DELETE_SYNC_LOCK = "Stop every lilbee process, delete {path}, and try again."
_FREE_OPEN_FILES = "Close some programs or raise the limit of open files, then try again."
_MAKE_WRITABLE = "Make {path} writable, then try again."
_SYNC_ADVICE_BY_ERRNO: dict[int | None, str] = {
    errno.EMFILE: _FREE_OPEN_FILES,
    errno.ENFILE: _FREE_OPEN_FILES,
    errno.ENOSPC: "Free some disk space, then try again.",
    errno.EACCES: _MAKE_WRITABLE,
    errno.EPERM: _MAKE_WRITABLE,
    errno.EROFS: _MAKE_WRITABLE,
}
_FIX_THE_CAUSE = "Fix that problem, then try again."
_NO_LOCKING = (
    "{path} is on a filesystem that does not support file locking. Move it to a local disk."
)
_NO_HARD_LINKS = "its filesystem does not support hard links; move it to {disk}"
_LOCAL_DISK = "a local disk"
# The disk to move to, by sys.platform; every other platform gets _LOCAL_DISK.
_HARD_LINK_DISKS = {"darwin": "a disk formatted as APFS"}
_PROBE_LEFT = "Could not delete the lock probe %s (%s); it is safe to delete by hand."
# Minimum blocking wait granted to the in-process mutex even when the file lock
# consumed the whole budget, so a deadline-edge acquire still gets a real attempt.
_MUTEX_MIN_WAIT = 0.1
# The timeout threading reads as "wait until the lock is free".
_NO_DEADLINE = -1


class LockTimeoutError(TimeoutError):
    """Raised when a lock cannot be acquired within the timeout."""


class LockingUnsupportedError(RuntimeError):
    """Raised when the filesystem holding a lock file cannot lock files at all."""


class ResetRefusedError(RuntimeError):
    """Raised when a reset cannot safely run on the data root; nothing was deleted."""


class SyncRunningError(RuntimeError):
    """Raised when syncs must stay off the data root and one is running; nothing was changed."""


class LibraryUnlockableError(RuntimeError):
    """Raised when a sync or a change that keeps syncs off cannot lock the data root."""


class _NoHardLinksError(OSError):
    """Raised by the lock probe on a filesystem that has no hard links."""


class SyncMarkWaitCancelledError(Exception):
    """Raised when a cancel ends a wait for the sync lock; the block did not run."""


class LibraryWaitCancelledError(TaskCancelledError):
    """Raised when a cancel ends the wait of a library hold; nothing was changed."""


# How each error that refuses a writer tells the user to run the command again.
_AGAIN = {ResetRefusedError: "reset again", LibraryUnlockableError: "try again"}


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
        with _enolck_as_unsupported(data_dir):
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
        with _enolck_as_unsupported(scope_dir):
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


@overload
def _open_sync_lock(data_root: Path, *, write: Literal[True]) -> ReadWriteLock: ...
@overload
def _open_sync_lock(
    data_root: Path, *, write: Literal[False], refused: type[RuntimeError]
) -> AsyncReadWriteLock | None: ...
@overload
def _open_sync_lock(
    data_root: Path,
    *,
    write: Literal[False],
    on_caller_thread: Literal[True],
    refused: type[RuntimeError],
) -> ReadWriteLock | None: ...
@overload
def _open_sync_lock(
    data_root: Path, *, write: Literal[True], refused: type[RuntimeError]
) -> ReadWriteLock: ...
def _open_sync_lock(
    data_root: Path,
    *,
    write: bool,
    on_caller_thread: bool = False,
    refused: type[RuntimeError] = ResetRefusedError,
) -> ReadWriteLock | AsyncReadWriteLock | None:
    """Open the data root's sync lock, or raise *refused* when it cannot be opened.

    None is for a reader on a filesystem with no hard links, which runs unmarked.
    The read side is taken on an event loop unless *on_caller_thread* is set.
    """
    if not _data_root_locks(data_root, refused, unmarked_reader=not write):
        return None
    path = data_root / _SYNC_LOCK_NAME
    try:
        kind = ReadWriteLock if write or on_caller_thread else AsyncReadWriteLock
        return kind(path, is_singleton=False)
    except _SYNC_LOCK_ERRORS as exc:
        _sync_lock_unavailable(path, exc, refused, reader=not write)


def _data_root_locks(
    data_root: Path, refused: type[RuntimeError], *, unmarked_reader: bool
) -> bool:
    """True when *data_root* can lock files; False where an unmarked reader runs; else raise."""
    try:
        data_root.mkdir(parents=True, exist_ok=True)
        _require_locking(data_root)
    except _NoHardLinksError as exc:
        if not unmarked_reader:
            _sync_lock_unavailable(data_root, exc, refused, refusal=_DATA_ROOT_UNLOCKABLE)
        # A reset cannot open the lock on this filesystem either, so it refuses.
        log.warning(_SYNC_LOCK_REFUSED, data_root, exc)
        return False
    except _SYNC_LOCK_ERRORS as exc:
        _sync_lock_unavailable(
            data_root, exc, refused, reader=unmarked_reader, refusal=_DATA_ROOT_UNLOCKABLE
        )
    return True


def refuse_where_a_sync_cannot_lock(data_root: Path) -> None:
    """Raise the error a sync raises when it cannot open or take the sync lock of *data_root*.

    The lock is not held on return, and a busy lock is not waited for.
    """
    lock = _open_sync_lock(
        data_root, write=False, on_caller_thread=True, refused=LibraryUnlockableError
    )
    if lock is None:
        return
    try:
        # SQLite reads the lock file at the take, so a file that is not a lock fails here.
        lock.acquire_read(blocking=False)
    except FileLockTimeout:
        # A reset holds the lock, so the lock works; the sync waits for the reset.
        pass
    except _SYNC_LOCK_ERRORS as exc:
        _sync_lock_unavailable(Path(lock.lock_file), exc, LibraryUnlockableError, reader=True)
    finally:
        _close_sync_lock(lock)


def _close_sync_lock(lock: ReadWriteLock) -> None:
    """Close *lock*; a close that fails only warns, so the caller keeps its own result."""
    try:
        lock.close()
    except _SYNC_LOCK_ERRORS as exc:
        log.warning(_SYNC_LOCK_NOT_CLOSED, lock.lock_file, exc)


def _require_locking(data_root: Path) -> None:
    """Raise ``LockingUnsupportedError`` when a lock nobody else holds still reports busy."""
    # Unique per call: a probe shared with another thread or host would read its hold as busy.
    probe_path = data_root / f"{_SYNC_PROBE_PREFIX}{uuid.uuid4().hex}"
    try:
        with closing(ReadWriteLock(probe_path, is_singleton=False)) as probe:
            probe.acquire_write(timeout=_PROBE_WAIT_S)
    except FileLockTimeout:
        raise LockingUnsupportedError(_NO_LOCKING.format(path=data_root)) from None
    except OSError as exc:
        if not _hard_link_refused(exc):
            raise
        disk = _HARD_LINK_DISKS.get(sys.platform, _LOCAL_DISK)
        raise _NoHardLinksError(_NO_HARD_LINKS.format(disk=disk)) from exc
    finally:
        _remove_probe(probe_path)


def _hard_link_refused(error: OSError) -> bool:
    """True when *error* is the lock's hard link, refused by a filesystem that has none."""
    # Only a call on two paths sets filename2, and the link is the lock's only such call.
    return error.errno == errno.ENOTSUP and error.filename2 is not None


@contextmanager
def _enolck_as_unsupported(directory: Path) -> Generator[None, None, None]:
    """Raise ``LockingUnsupportedError`` for a file lock the filesystem refuses with ENOLCK."""
    try:
        yield
    except OSError as exc:
        if exc.errno != errno.ENOLCK:
            raise
        raise LockingUnsupportedError(_NO_LOCKING.format(path=directory)) from exc


def _remove_probe(probe_path: Path) -> None:
    """Delete a lock probe; a leftover probe is harmless, so a failure only warns."""
    try:
        probe_path.unlink(missing_ok=True)
    except OSError as exc:
        log.warning(_PROBE_LEFT, probe_path, exc)


def _sync_lock_unavailable(
    path: Path,
    error: Exception,
    refused: type[RuntimeError],
    *,
    reader: bool = False,
    refusal: str = _SYNC_LOCK_UNKNOWN,
) -> NoReturn:
    """Raise *refused* with the text for a *reader* or a writer that cannot lock *path*."""
    text = _SYNC_REFUSED if reader else refusal
    advice = _sync_advice(path, error)
    raise refused(
        text.format(path=path, error=_cause_text(error), again=_AGAIN[refused], advice=advice)
    ) from error


def _cause_text(error: Exception) -> str:
    """The operating system's words for *error*, without the file that it names."""
    # The file can be the lock probe, whose name means nothing to the user.
    if isinstance(error, OSError) and error.strerror is not None:
        return f"[Errno {error.errno}] {error.strerror}"
    return str(error)


def _sync_advice(path: Path, error: Exception) -> str:
    """What to do about *error*, which stopped a sync from locking *path*."""
    # SQLite raises the bare DatabaseError for a file that is not one of its databases.
    if path.name == _SYNC_LOCK_NAME and type(error) is sqlite3.DatabaseError:
        return _DELETE_SYNC_LOCK.format(path=path)
    code = error.errno if isinstance(error, OSError) else None
    return _SYNC_ADVICE_BY_ERRNO.get(code, _FIX_THE_CAUSE).format(path=path)


def _raise_cancel_behind(error: Exception) -> None:
    """Raise the cancel that *error* replaced, when it replaced one."""
    # filelock raises a failed take or close in place of the cancel that interrupted it.
    if isinstance(error.__context__, asyncio.CancelledError):
        raise error.__context__ from None


class _SyncLockAttempt:
    """One try at a side of the sync lock, *seconds* long at most; ``busy`` is how it failed."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.busy: FileLockTimeout | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _kind: object, error: BaseException | None, _traceback: object) -> bool:
        if isinstance(error, _SYNC_LOCK_ERRORS):
            _raise_cancel_behind(error)
        # Only a busy lock is waited out.
        if isinstance(error, FileLockTimeout):
            self.busy = error
        return self.busy is not None


def _sync_lock_wait(
    cancel: CancelSignal | None, wait: float | None = None
) -> Iterator[_SyncLockAttempt]:
    """Every wait for a side of the sync lock: yield one attempt after another until one takes it.

    The caller takes the lock inside the ``with`` of each attempt, for its ``seconds``.
    *cancel* is read after every attempt, the one that took the lock included: a set
    *cancel* raises ``SyncMarkWaitCancelledError``, and the caller closes the lock.
    A lock still busy *wait* seconds after the first attempt raises filelock's
    ``Timeout``; None sets no limit. A wait with a limit and no *cancel* is one
    attempt *wait* seconds long: a writer has its place ahead of later readers only
    during an attempt, so with a *cancel* a later reader can pass it.
    """
    if cancel is None and wait is not None:
        attempt = _SyncLockAttempt(wait)
        yield attempt
        if attempt.busy is not None:
            raise attempt.busy
        return
    deadline = None if wait is None else time.monotonic() + wait
    while True:
        left = _SYNC_TAKE_SLICE_S if deadline is None else deadline - time.monotonic()
        attempt = _SyncLockAttempt(max(0.0, min(_SYNC_TAKE_SLICE_S, left)))
        yield attempt
        if _is_set(cancel):
            raise SyncMarkWaitCancelledError
        if attempt.busy is None:
            return
        if deadline is not None and time.monotonic() >= deadline:
            raise attempt.busy


def _is_set(cancel: CancelSignal | None) -> bool:
    return cancel is not None and cancel.is_set()


@asynccontextmanager
async def sync_running(
    data_root: Path, cancel: CancelSignal | None = None
) -> AsyncGenerator[None, None]:
    """Mark a sync or import running on *data_root*, across processes; they share the mark.

    The wait for a reset to finish runs off the event loop. A *cancel* that is set
    during that wait, or when the mark is taken, raises ``asyncio.CancelledError``:
    the block does not run and the mark is free.
    """
    # Opening the lock reads and writes files, so it runs off the event loop too.
    lock = await asyncio.to_thread(
        _open_sync_lock, data_root, write=False, refused=LibraryUnlockableError
    )
    if lock is None:
        yield
        return
    try:
        try:
            for attempt in _sync_lock_wait(cancel):
                with attempt:
                    await lock.acquire_read(timeout=attempt.seconds)
        except SyncMarkWaitCancelledError:
            raise asyncio.CancelledError from None
        except _SYNC_LOCK_ERRORS as exc:
            _sync_lock_unavailable(Path(lock.lock_file), exc, LibraryUnlockableError, reader=True)
        yield
    finally:
        try:
            await lock.close()
        except _SYNC_LOCK_ERRORS as exc:
            # Logged only: the block keeps its own result, error or cancel.
            log.warning(_SYNC_LOCK_NOT_CLOSED, lock.lock_file, exc)
            # filelock keeps the worker thread of a lock whose close raised; wait until it ends.
            await asyncio.to_thread(lock.executor.shutdown)
            _raise_cancel_behind(exc)


@contextmanager
def source_keys_in_use(
    data_root: Path, cancel: CancelSignal | None = None
) -> Generator[None, None, None]:
    """Hold the sync mark of *data_root* for a block that reads and writes source keys.

    Waits while a reset holds syncs off. A *cancel* that is
    set during that wait, or when the mark is taken, raises
    ``SyncMarkWaitCancelledError``: the block does not run and the mark is free.
    """
    lock = _open_sync_lock(
        data_root, write=False, on_caller_thread=True, refused=LibraryUnlockableError
    )
    if lock is None:
        yield
        return
    try:
        try:
            for attempt in _sync_lock_wait(cancel):
                with attempt:
                    lock.acquire_read(timeout=attempt.seconds)
        except _SYNC_LOCK_ERRORS as exc:
            if _is_set(cancel):
                raise SyncMarkWaitCancelledError from None
            _sync_lock_unavailable(Path(lock.lock_file), exc, LibraryUnlockableError, reader=True)
        yield
    finally:
        _close_sync_lock(lock)


@contextmanager
def _no_sync_running(data_root: Path) -> Generator[None, None, None]:
    """Keep syncs off *data_root* for the block; raise ``ResetRefusedError`` unless it can."""
    try:
        lock = _open_sync_lock(data_root, write=True)
    except LockingUnsupportedError as exc:
        raise ResetRefusedError(str(exc)) from exc
    with closing(lock):
        try:
            lock.acquire_write(blocking=False)
        except FileLockTimeout:
            raise ResetRefusedError(f"{_SYNC_RUNNING} {_RESET_WHEN_DONE}") from None
        except _SYNC_LOCK_ERRORS as exc:
            _sync_lock_unavailable(Path(lock.lock_file), exc, ResetRefusedError)
        yield


@contextmanager
def _syncs_held_off(
    data_root: Path, running: str, wait: float = 0.0, cancel: CancelSignal | None = None
) -> Generator[None, None, None]:
    """Keep syncs off *data_root* for the block; a sync that outlasts *wait* raises *running*.

    Raises ``SyncRunningError`` with the text *running*. A data root that cannot
    be locked raises ``LibraryUnlockableError`` or ``LockingUnsupportedError``.
    A *cancel* that is set during the wait, or when the lock is taken, raises
    ``SyncMarkWaitCancelledError``: the block does not run and the lock is free.
    """
    lock = _open_sync_lock(data_root, write=True, refused=LibraryUnlockableError)
    with closing(lock):
        try:
            for attempt in _sync_lock_wait(cancel, wait):
                with attempt:
                    lock.acquire_write(timeout=attempt.seconds, blocking=attempt.seconds > 0)
        except FileLockTimeout:
            raise SyncRunningError(running) from None
        except _SYNC_LOCK_ERRORS as exc:
            _sync_lock_unavailable(Path(lock.lock_file), exc, LibraryUnlockableError)
        yield


class _HoldKind(StrEnum):
    """What a hold of a library is for; the kind selects the take of the sync lock."""

    CHANGE = "change"
    RESET = "reset"


@dataclass(frozen=True)
class LibraryHold:
    """What a hold of a library yields: its block has syncs off on the data root of *config*.

    Only ``_library_hold`` makes one. Every path a function uses under the hold
    comes from its *config*.
    """

    config: Config


@contextmanager
def _library_hold(
    config: Config,
    kind: _HoldKind,
    *,
    running: str = _SYNC_RUNNING,
    wait: float = 0.0,
    cancel: CancelSignal | None = None,
) -> Generator[LibraryHold, None, None]:
    """Keep syncs off *config*'s data root as *kind* does, for the block.

    A reset takes the sync lock with no wait and reads no *running*, *wait* or
    *cancel*. A *cancel* that ends the wait for the sync lock raises
    ``LibraryWaitCancelledError``: the block does not run and the lock is free.
    """
    if kind is _HoldKind.RESET:
        syncs_off = _no_sync_running(config.data_root)
    else:
        syncs_off = _syncs_held_off(config.data_root, running, wait, cancel)
    with ExitStack() as held:
        try:
            held.enter_context(syncs_off)
        except SyncMarkWaitCancelledError:
            # No cause: a cancel that carries an error is reported as that error.
            raise LibraryWaitCancelledError from None
        yield LibraryHold(config)


def library_held(
    config: Config, running: str, wait: float = 0.0, cancel: CancelSignal | None = None
) -> AbstractContextManager[LibraryHold]:
    """Keep syncs off the data root of *config* for the block.

    A sync that outlasts *wait* raises ``SyncRunningError`` with the text *running*.
    A *cancel* that is set during the wait, or when the lock is taken, raises
    ``LibraryWaitCancelledError``: the block does not run and the lock is free.
    """
    return _library_hold(config, _HoldKind.CHANGE, running=running, wait=wait, cancel=cancel)


def library_held_for_reset(config: Config) -> AbstractContextManager[LibraryHold]:
    """Keep syncs off *config*'s data root for the block of a reset, with no wait.

    A sync that runs, or a sync lock that cannot be taken, raises ``ResetRefusedError``.
    """
    return _library_hold(config, _HoldKind.RESET)


@contextmanager
def write_lock(
    lancedb_dir: Path | None = None, timeout: float | None = LOCK_TIMEOUT
) -> Generator[None, None, None]:
    """Acquire the cross-process file lock then the in-process mutex.

    The file lock lives next to the store's data, so cross-process writers
    coordinate only when they lock the *same* directory: callers pass their
    store's ``lancedb_dir`` (a per-instance ``Lilbee`` uses its own dir).
    ``None`` falls back to the global ``cfg.lancedb_dir``.

    The two stages share one budget: the time spent waiting on the file lock is
    deducted before waiting on the mutex (plus a small ``_MUTEX_MIN_WAIT`` floor),
    so a 30s request cannot stall for roughly twice that. A *timeout* of None
    waits for both without a deadline.
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    lock_path = _lock_path(lancedb_dir)
    # The first write to a per-instance store can run before its data dir exists;
    # the file lock cannot be created in a missing directory.
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flock = FileLock(lock_path)
    try:
        with _enolck_as_unsupported(lock_path.parent):
            # None takes the lock's own timeout, which waits without a deadline.
            flock.acquire(timeout=timeout)
    except FileLockTimeout:
        raise LockTimeoutError("Timed out waiting for exclusive file lock") from None
    try:
        # Floor the mutex budget so a file lock that wins right at the deadline
        # still gets a brief blocking attempt instead of a zero-timeout poll that
        # spuriously fails when another thread holds the mutex for an instant.
        remaining = (
            _NO_DEADLINE if deadline is None else max(_MUTEX_MIN_WAIT, deadline - time.monotonic())
        )
        acquired = _write_mutex.acquire(timeout=remaining)
        if not acquired:
            raise LockTimeoutError("Timed out waiting for write lock")
        try:
            yield
        finally:
            _write_mutex.release()
    finally:
        flock.release()
