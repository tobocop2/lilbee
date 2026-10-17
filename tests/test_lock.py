"""Tests for write locking and file locking."""

import asyncio
import errno
import logging
import os
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Generator
from contextlib import closing, contextmanager, nullcontext
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from filelock import FileLock, ReadWriteLock
from filelock import Timeout as FileLockTimeout

from lilbee.core.config import cfg
from lilbee.runtime.lock import (
    _SYNC_TAKE_SLICE_S,
    LibraryUnlockableError,
    LockingUnsupportedError,
    LockTimeoutError,
    ResetRefusedError,
    SyncMarkWaitCancelledError,
    _lock_path,
    _no_sync_running,
    _sync_lock_wait,
    acquire_scope_lock,
    acquire_server_lock,
    read_scope_owner,
    server_lock_path,
    source_keys_in_use,
    sync_running,
    write_lock,
)
from tests.conftest import cancel_set_when_the_sync_lock_is_granted, reset_held_on_a_thread


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path):
    """Point cfg.lancedb_dir at tmp_path for file lock isolation."""
    snapshot = cfg.model_copy()
    cfg.lancedb_dir = tmp_path / "lancedb_test"
    cfg.lancedb_dir.mkdir(parents=True)
    yield
    for name in type(snapshot).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


class TestWriteLock:
    def test_basic(self):
        with write_lock(timeout=2):
            pass

    def test_releases_on_error(self):
        with pytest.raises(RuntimeError, match="boom"), write_lock(timeout=2):
            raise RuntimeError("boom")
        # Lock should be released: a subsequent write lock should succeed
        with write_lock(timeout=1):
            pass

    def test_serializes_writers(self):
        """Two write_lock() calls cannot overlap."""
        events: list[str] = []
        writer1_entered = threading.Event()
        writer1_release = threading.Event()

        def writer1() -> None:
            with write_lock(timeout=2):
                writer1_entered.set()
                events.append("w1_start")
                writer1_release.wait(timeout=5)
                events.append("w1_end")

        def writer2() -> None:
            writer1_entered.wait(timeout=5)
            with write_lock(timeout=5):
                events.append("w2")

        t1 = threading.Thread(target=writer1)
        t2 = threading.Thread(target=writer2)
        t1.start()
        t2.start()
        time.sleep(0.05)
        writer1_release.set()
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert events.index("w1_end") < events.index("w2")

    def test_timeout(self):
        """write_lock times out when another writer holds it."""
        entered = threading.Event()
        release = threading.Event()
        timed_out = threading.Event()

        def holder() -> None:
            with write_lock(timeout=2):
                entered.set()
                release.wait(timeout=5)

        def waiter() -> None:
            entered.wait(timeout=5)
            with pytest.raises(LockTimeoutError), write_lock(timeout=0.05):
                pass
            timed_out.set()

        t1 = threading.Thread(target=holder)
        t2 = threading.Thread(target=waiter)
        t1.start()
        t2.start()
        t2.join(timeout=5)
        assert timed_out.is_set()
        release.set()
        t1.join(timeout=5)

    def test_mutex_timeout(self):
        """write_lock raises when the in-process mutex times out."""
        from lilbee.runtime.lock import _write_mutex

        _write_mutex.acquire()
        try:
            with pytest.raises(LockTimeoutError, match="write lock"), write_lock(timeout=0.05):
                pass
        finally:
            _write_mutex.release()

    def test_lock_file_created(self):
        """Lock file is created at the expected path."""
        expected = _lock_path(None)
        with write_lock(timeout=2):
            assert expected.exists()

    def test_lock_path_uses_passed_dir(self, tmp_path):
        """A passed lancedb_dir keys the lock file; None falls back to cfg."""
        other = tmp_path / "other_db"
        assert _lock_path(other) == other / ".lock"
        assert _lock_path(None) == cfg.lancedb_dir / ".lock"

    def test_write_lock_targets_passed_dir(self, tmp_path):
        """write_lock(dir) creates the lock file under that dir, not cfg's.

        A per-instance store writes to its own lancedb_dir; the file lock must
        live there or cross-process writers never coordinate.
        """
        other = tmp_path / "other_db"
        other.mkdir()
        with write_lock(other, timeout=2):
            assert (other / ".lock").exists()
        assert not (cfg.lancedb_dir / ".lock").exists()

    def test_timeout_budget_is_split_across_stages(self, monkeypatch, tmp_path):
        """The file-lock wait is deducted from the mutex wait (single budget).

        Previously each stage got the full timeout, so a 30s request could stall
        ~60s. The mutex must receive only the budget the file lock left.
        """
        from filelock import FileLock

        import lilbee.runtime.lock as lockmod

        clock = {"t": 1000.0}
        monkeypatch.setattr(lockmod.time, "monotonic", lambda: clock["t"])

        class FakeMutex:
            def __init__(self) -> None:
                self.captured: float | None = None

            def acquire(self, timeout: float = -1) -> bool:
                self.captured = timeout
                return True

            def release(self) -> None: ...

        fake = FakeMutex()
        monkeypatch.setattr(lockmod, "_write_mutex", fake)

        real_flock_acquire = FileLock.acquire

        def slow_flock_acquire(self, timeout=None, **kw):
            clock["t"] += 22.0  # the file lock consumed 22s of the budget
            return real_flock_acquire(self, timeout=timeout, **kw)

        monkeypatch.setattr(FileLock, "acquire", slow_flock_acquire)

        with write_lock(tmp_path, timeout=30.0):
            pass
        assert fake.captured == pytest.approx(8.0, abs=0.5)  # 30 - 22 remaining


class TestServerLock:
    def test_acquire_holds_and_creates_lock_file(self, tmp_path: Path):
        lock = acquire_server_lock(tmp_path, timeout=0.1)
        assert lock is not None
        assert lock.is_locked
        assert server_lock_path(tmp_path).exists()
        lock.release()

    def test_second_acquire_refused_while_held(self, tmp_path: Path):
        holder = acquire_server_lock(tmp_path, timeout=0.1)
        assert holder is not None
        assert acquire_server_lock(tmp_path, timeout=0.05) is None
        holder.release()

    def test_reacquire_after_release(self, tmp_path: Path):
        first = acquire_server_lock(tmp_path, timeout=0.1)
        assert first is not None
        first.release()
        second = acquire_server_lock(tmp_path, timeout=0.05)
        assert second is not None
        second.release()

    def test_acquire_creates_missing_data_dir(self, tmp_path: Path):
        missing = tmp_path / "nested" / "data"
        lock = acquire_server_lock(missing, timeout=0.1)
        assert lock is not None
        lock.release()


class TestScopeLock:
    def test_acquire_writes_owner_sidecar(self, tmp_path: Path):
        hold = acquire_scope_lock(tmp_path, tmp_path / "vaults" / "a", timeout=0.1)
        assert hold is not None
        owner = read_scope_owner(tmp_path)
        assert owner is not None
        assert owner.data_dir == str(tmp_path / "vaults" / "a")
        hold.release()

    def test_second_acquire_refused_and_owner_readable(self, tmp_path: Path):
        holder = acquire_scope_lock(tmp_path, tmp_path / "vaults" / "a", timeout=0.1)
        assert holder is not None
        assert acquire_scope_lock(tmp_path, tmp_path / "vaults" / "b", timeout=0.05) is None
        owner = read_scope_owner(tmp_path)
        assert owner is not None and owner.data_dir.endswith("a")
        holder.release()

    def test_release_removes_owner_sidecar_and_frees_the_scope(self, tmp_path: Path):
        first = acquire_scope_lock(tmp_path, tmp_path / "vaults" / "a", timeout=0.1)
        assert first is not None
        first.release()
        assert read_scope_owner(tmp_path) is None
        second = acquire_scope_lock(tmp_path, tmp_path / "vaults" / "b", timeout=0.05)
        assert second is not None
        second.release()

    def test_corrupt_owner_sidecar_reads_as_none(self, tmp_path: Path):
        (tmp_path / "server.scope.owner.json").write_text("not json{{{")
        assert read_scope_owner(tmp_path) is None

    def test_acquire_creates_missing_scope_dir(self, tmp_path: Path):
        missing = tmp_path / "shared" / "root"
        hold = acquire_scope_lock(missing, tmp_path / "data", timeout=0.1)
        assert hold is not None
        hold.release()


_SQLITE_HEADER = b"SQLite format 3\x00"
_NO_HARD_LINKS = "its filesystem does not support hard links"
_APFS_ADVICE = f"{_NO_HARD_LINKS}; move it to a disk formatted as APFS"
_LOCAL_DISK_ADVICE = f"{_NO_HARD_LINKS}; move it to a local disk"
_LINK_REFUSED = OSError(errno.ENOTSUP, "Operation not supported", "sync.probe", None, "lock.db")
_OPEN_REFUSED = OSError(errno.ENOTSUP, "Operation not supported", "sync.probe")
_READ_ONLY = OSError(errno.EROFS, "Read-only file system")
# A hard link that fails on a filesystem that has hard links.
_LINK_DENIED = OSError(errno.EACCES, "Permission denied", "sync.probe", None, "lock.db")
_LINK_DISK_FULL = OSError(errno.ENOSPC, "No space left on device", "sync.probe", None, "lock.db")
_LINK_TOO_MANY = OSError(errno.EMLINK, "Too many links", "sync.probe", None, "lock.db")
_FIX_ADVICE = "Fix that problem, then try again."
_WRITABLE_ADVICE = "Make {path} writable, then try again."


_RESET_IN_ANOTHER_PROCESS = """
import sys
from pathlib import Path
from lilbee.runtime.lock import ResetRefusedError, _no_sync_running
try:
    with _no_sync_running(Path(sys.argv[1])):
        pass
except ResetRefusedError:
    sys.exit(3)
"""
_RESET_HOLDS = "reset holds"
_HELD_RESET_IN_ANOTHER_PROCESS = f"""
import sys
from pathlib import Path
from lilbee.runtime.lock import _no_sync_running
with _no_sync_running(Path(sys.argv[1])):
    print("{_RESET_HOLDS}", flush=True)
    sys.stdin.read()
"""
# How long a test waits for a step that must happen; reaching it fails the test.
_FAILURE_TIMEOUT_S = 20.0


@pytest.fixture
def no_lock_daemon():
    """Make every SQLite lock report busy until its timeout, as on a mount without a lock daemon."""
    unblock = threading.Event()

    def _busy(lock: ReadWriteLock, timeout: float = -1, *, blocking: bool = True) -> None:
        if blocking:
            unblock.wait(None if timeout == -1 else timeout)
        raise FileLockTimeout(lock.lock_file)

    with (
        mock.patch.object(ReadWriteLock, "acquire_read", _busy),
        mock.patch.object(ReadWriteLock, "acquire_write", _busy),
    ):
        yield unblock
    unblock.set()


@pytest.fixture
def no_lock_daemon_for_file_locks():
    """Make every file lock fail with ENOLCK, as on a mount without a lock daemon."""
    import fcntl  # POSIX only; the tests that use this fixture skip on Windows

    def _flock(_fd: int, _operation: int) -> None:
        raise OSError(errno.ENOLCK, "No locks available")

    with mock.patch.object(fcntl, "flock", _flock):
        yield


@contextmanager
def _reset_held_in_another_process(data_root: Path) -> Generator[None, None, None]:
    """Hold a reset on *data_root* in a child process until the block ends."""
    holder = subprocess.Popen(
        [sys.executable, "-c", _HELD_RESET_IN_ANOTHER_PROCESS, str(data_root)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    try:
        assert holder.stdout is not None
        assert any(_RESET_HOLDS in line for line in holder.stdout)
        yield
    finally:
        holder.communicate(timeout=30)


@contextmanager
def _sync_lock_close_refused() -> Generator[None, None, None]:
    """Close each sync lock, then raise from the close as a disk that fails at that moment does."""
    real_close = ReadWriteLock.close

    def _close(lock: ReadWriteLock) -> None:
        real_close(lock)
        if Path(lock.lock_file).name == "sync.lock":
            raise sqlite3.OperationalError("disk I/O error")

    with mock.patch.object(ReadWriteLock, "close", _close):
        yield


@contextmanager
def _sync_take_that_ends_busy() -> Generator[threading.Event, None, None]:
    """Yield an event that is set when a take of the read side of a sync lock ends busy."""
    ended_busy = threading.Event()
    real_take = ReadWriteLock.acquire_read

    def _take(lock: ReadWriteLock, *args: object, **kwargs: object) -> object:
        try:
            return real_take(lock, *args, **kwargs)
        except FileLockTimeout:
            if Path(lock.lock_file).name == "sync.lock":
                ended_busy.set()
            raise

    with mock.patch.object(ReadWriteLock, "acquire_read", _take):
        yield ended_busy


@contextmanager
def _sync_lock_close_held_then_refused() -> Generator[
    tuple[threading.Event, threading.Event], None, None
]:
    """Yield (closing, finish): a close of a sync lock sets *closing* and raises after *finish*."""
    closing, finish = threading.Event(), threading.Event()
    real_close = ReadWriteLock.close

    def _close(lock: ReadWriteLock) -> None:
        real_close(lock)
        if Path(lock.lock_file).name == "sync.lock":
            closing.set()
            finish.wait(_FAILURE_TIMEOUT_S)
            raise sqlite3.OperationalError("disk I/O error")

    with mock.patch.object(ReadWriteLock, "close", _close):
        try:
            yield closing, finish
        finally:
            finish.set()


@contextmanager
def _recorded_read_takes() -> Generator[tuple[list[float], threading.Event], None, None]:
    """Record the timeout of each take of a read lock while the block runs.

    Yields the timeouts and an event that is set when the first take is recorded.
    """
    waits: list[float] = []
    first_take = threading.Event()
    real_acquire = ReadWriteLock.acquire_read

    def _acquire(lock: ReadWriteLock, timeout: float = -1, *, blocking: bool = True) -> None:
        waits.append(timeout)
        first_take.set()
        real_acquire(lock, timeout, blocking=blocking)

    with mock.patch.object(ReadWriteLock, "acquire_read", _acquire):
        yield waits, first_take


@contextmanager
def _mark_held_on_a_thread(data_root: Path) -> Generator[Callable[[], None], None, None]:
    """Hold the sync mark of *data_root* on a thread, as a running sync does.

    Yields the call that ends the hold; the hold ends with the block at the latest.
    """
    held, release = threading.Event(), threading.Event()

    def _sync() -> None:
        with source_keys_in_use(data_root):
            held.set()
            release.wait(3 * _FAILURE_TIMEOUT_S)

    def _end() -> None:
        release.set()
        holder.join(_FAILURE_TIMEOUT_S)

    holder = threading.Thread(target=_sync)
    holder.start()
    try:
        assert held.wait(_FAILURE_TIMEOUT_S)
        yield _end
    finally:
        _end()


@contextmanager
def _sync_write_takes(
    after: Callable[[str], object] = lambda _outcome: None,
) -> Generator[list[str], None, None]:
    """Record each take of the write side of a sync lock as "busy" or "granted".

    *after* is called with that word when the take ends, before the caller sees it.
    """
    takes: list[str] = []
    real_take = ReadWriteLock.acquire_write

    def _take(lock: ReadWriteLock, *args: object, **kwargs: object) -> object:
        if Path(lock.lock_file).name != "sync.lock":
            return real_take(lock, *args, **kwargs)
        try:
            taken = real_take(lock, *args, **kwargs)
        except FileLockTimeout:
            takes.append("busy")
            after("busy")
            raise
        takes.append("granted")
        after("granted")
        return taken

    with mock.patch.object(ReadWriteLock, "acquire_write", _take):
        yield takes


def _a_sync_can_start(data_root: Path) -> bool:
    """True when the read side of the sync lock of *data_root* is free to take now."""
    with closing(ReadWriteLock(data_root / "sync.lock", is_singleton=False)) as probe:
        try:
            probe.acquire_read(blocking=False)
        except FileLockTimeout:
            return False
    return True


def _lock_warnings(caplog) -> int:
    return sum(r.name == "lilbee.runtime.lock" and r.levelname == "WARNING" for r in caplog.records)


def _threads_left_since(before: set[threading.Thread]) -> list[str]:
    """Names of the threads started since *before*, without the event loop's own pool."""
    return [
        thread.name
        for thread in set(threading.enumerate()) - before
        if not thread.name.startswith("asyncio_")
    ]


class TestSyncMark:
    """Syncs share the data root's mark; a reset needs it free, in any process."""

    async def test_syncs_share_the_mark(self, tmp_path: Path) -> None:
        async with sync_running(tmp_path), sync_running(tmp_path):
            with pytest.raises(ResetRefusedError), _no_sync_running(tmp_path):
                pass
        with _no_sync_running(tmp_path):
            pass

    @pytest.mark.parametrize("reset_held", [reset_held_on_a_thread, _reset_held_in_another_process])
    async def test_a_sync_cancelled_while_it_waits_returns_while_the_reset_still_holds(
        self, tmp_path: Path, reset_held
    ) -> None:
        entered = []

        async def _sync() -> None:
            async with sync_running(tmp_path):
                entered.append("sync")

        before = set(threading.enumerate())
        with reset_held(tmp_path), _recorded_read_takes() as (waits, first_take):
            sync_task = asyncio.create_task(_sync())
            assert await asyncio.to_thread(first_take.wait, _FAILURE_TIMEOUT_S)
            sync_task.cancel()
            await asyncio.wait({sync_task}, timeout=_FAILURE_TIMEOUT_S)
            returned = sync_task.done()
            with (
                pytest.raises(ResetRefusedError, match="is running on this library"),
                _no_sync_running(tmp_path),
            ):
                pass

        assert returned, "one cancel did not end the wait for the reset"
        assert sync_task.cancelled()
        assert entered == []
        assert waits
        assert all(0 < wait <= _SYNC_TAKE_SLICE_S for wait in waits), waits
        assert _threads_left_since(before) == []
        with _no_sync_running(tmp_path):
            pass

    @pytest.mark.parametrize("reset_held", [reset_held_on_a_thread, _reset_held_in_another_process])
    async def test_a_set_cancel_ends_the_wait_of_a_sync_while_the_reset_still_holds(
        self, tmp_path: Path, reset_held
    ) -> None:
        cancel = threading.Event()
        entered = []

        async def _sync() -> None:
            async with sync_running(tmp_path, cancel):
                entered.append("sync")

        before = set(threading.enumerate())
        with reset_held(tmp_path), _recorded_read_takes() as (_waits, first_take):
            sync_task = asyncio.create_task(_sync())
            assert await asyncio.to_thread(first_take.wait, _FAILURE_TIMEOUT_S)
            cancel.set()
            # A guard against a hang; the task is never cancelled, only the signal is set.
            await asyncio.wait({sync_task}, timeout=_FAILURE_TIMEOUT_S)
            returned = sync_task.done()
            with (
                pytest.raises(ResetRefusedError, match="is running on this library"),
                _no_sync_running(tmp_path),
            ):
                pass

        assert returned, "the set cancel did not end the wait for the reset"
        assert sync_task.cancelled()
        assert entered == []
        assert _threads_left_since(before) == []
        with _no_sync_running(tmp_path):
            pass

    async def test_a_sync_with_an_unset_cancel_waits_for_the_reset_and_runs(
        self, tmp_path: Path
    ) -> None:
        cancel = threading.Event()
        order: list[str] = []

        async def _sync() -> None:
            async with sync_running(tmp_path, cancel):
                order.append("sync ran")

        with reset_held_on_a_thread(tmp_path), _sync_take_that_ends_busy() as found_the_mark_held:
            sync_task = asyncio.create_task(_sync())
            assert await asyncio.to_thread(found_the_mark_held.wait, _FAILURE_TIMEOUT_S)
            order.append("reset ended")
        await asyncio.wait_for(sync_task, _FAILURE_TIMEOUT_S)

        assert order == ["reset ended", "sync ran"]

    @pytest.mark.parametrize("cancel_set", ["before a free take", "at the grant", "after a wait"])
    async def test_a_sync_granted_the_lock_with_its_cancel_set_does_not_run(
        self, tmp_path: Path, caplog, cancel_set: str
    ) -> None:
        cancel = threading.Event()
        entered: list[str] = []

        async def _sync() -> None:
            async with sync_running(tmp_path, cancel):
                entered.append("sync")

        if cancel_set == "before a free take":
            cancel.set()
        before = set(threading.enumerate())
        with (
            caplog.at_level(logging.WARNING, logger="lilbee.runtime.lock"),
            cancel_set_when_the_sync_lock_is_granted(cancel) as grants,
        ):
            if cancel_set == "after a wait":
                with (
                    reset_held_on_a_thread(tmp_path),
                    _sync_take_that_ends_busy() as found_the_mark_held,
                ):
                    sync_task = asyncio.create_task(_sync())
                    assert await asyncio.to_thread(found_the_mark_held.wait, _FAILURE_TIMEOUT_S)
                    assert not cancel.is_set()
            else:
                sync_task = asyncio.create_task(_sync())
            # A guard against a hang; the task is never cancelled, only the signal is set.
            await asyncio.wait({sync_task}, timeout=_FAILURE_TIMEOUT_S)

        assert grants == ["granted"], "the sync never held the lock"
        assert sync_task.cancelled()
        assert entered == []
        assert _lock_warnings(caplog) == 0
        assert _threads_left_since(before) == []
        with _no_sync_running(tmp_path):
            pass

    async def test_a_wiki_writer_cancelled_while_it_waits_takes_the_mark_in_short_slices(
        self, tmp_path: Path
    ) -> None:
        cancel = threading.Event()
        outcome: list[BaseException] = []

        def _writer() -> None:
            try:
                with source_keys_in_use(tmp_path, cancel):
                    outcome.append(AssertionError("the block ran while the reset held"))
            except SyncMarkWaitCancelledError as exc:
                outcome.append(exc)

        with reset_held_on_a_thread(tmp_path), _recorded_read_takes() as (waits, first_take):
            writer = threading.Thread(target=_writer)
            writer.start()
            try:
                assert await asyncio.to_thread(first_take.wait, _FAILURE_TIMEOUT_S)
            finally:
                cancel.set()
                # A guard against a hang; the outcome below is what the test asserts.
                await asyncio.to_thread(writer.join, _FAILURE_TIMEOUT_S)

        assert [type(exc) for exc in outcome] == [SyncMarkWaitCancelledError]
        assert waits
        assert all(0 < wait <= _SYNC_TAKE_SLICE_S for wait in waits), waits
        with _no_sync_running(tmp_path):
            pass

    @pytest.mark.parametrize("cancel_set", ["before a free take", "at the grant", "after a wait"])
    def test_a_wiki_writer_granted_the_mark_with_its_cancel_set_does_not_run(
        self, tmp_path: Path, cancel_set: str
    ) -> None:
        cancel = threading.Event()
        outcome: list[object] = []

        def _writer() -> None:
            try:
                with source_keys_in_use(tmp_path, cancel):
                    outcome.append("the block ran")
            except SyncMarkWaitCancelledError as exc:
                outcome.append(type(exc))

        if cancel_set == "before a free take":
            cancel.set()
        with cancel_set_when_the_sync_lock_is_granted(cancel) as grants:
            if cancel_set == "after a wait":
                with (
                    reset_held_on_a_thread(tmp_path),
                    _sync_take_that_ends_busy() as found_the_mark_held,
                ):
                    writer = threading.Thread(target=_writer)
                    writer.start()
                    assert found_the_mark_held.wait(_FAILURE_TIMEOUT_S)
                    assert not cancel.is_set()
                # A guard against a hang; the outcome below is what the test asserts.
                writer.join(_FAILURE_TIMEOUT_S)
            else:
                _writer()

        if cancel_set != "before a free take":
            assert grants == ["granted"], "the writer never held the mark"
        assert outcome == [SyncMarkWaitCancelledError]
        with _no_sync_running(tmp_path):
            pass

    def test_a_wiki_writer_with_an_unset_cancel_waits_for_the_reset_and_runs(
        self, tmp_path: Path
    ) -> None:
        order: list[str] = []

        def _writer() -> None:
            with source_keys_in_use(tmp_path, threading.Event()):
                order.append("writer ran")

        with reset_held_on_a_thread(tmp_path), _sync_take_that_ends_busy() as found_the_mark_held:
            writer = threading.Thread(target=_writer)
            writer.start()
            assert found_the_mark_held.wait(_FAILURE_TIMEOUT_S)
            order.append("reset ended")
        writer.join(_FAILURE_TIMEOUT_S)

        assert order == ["reset ended", "writer ran"]

    @pytest.mark.parametrize("cancelled_in", ["the wait", "the block"])
    async def test_a_close_error_does_not_replace_the_cancel_of_a_sync(
        self, tmp_path: Path, caplog, cancelled_in: str
    ) -> None:
        entered = asyncio.Event()
        left_at_the_return: list[str] = []

        async def _sync() -> None:
            try:
                async with sync_running(tmp_path):
                    entered.set()
                    await asyncio.Event().wait()
            finally:
                # Nothing is awaited between the return of the mark and this line.
                left_at_the_return.extend(_threads_left_since(before))

        waits = cancelled_in == "the wait"
        holder = reset_held_on_a_thread(tmp_path, ends_after_s=3 * _FAILURE_TIMEOUT_S)
        with (
            holder if waits else nullcontext() as resetter,
            _sync_lock_close_refused(),
            _sync_take_that_ends_busy() as found_the_mark_held,
            caplog.at_level(logging.WARNING, logger="lilbee.runtime.lock"),
        ):
            before = set(threading.enumerate())
            sync_task = asyncio.create_task(_sync())
            if waits:
                assert await asyncio.to_thread(found_the_mark_held.wait, _FAILURE_TIMEOUT_S)
            else:
                await asyncio.wait_for(entered.wait(), _FAILURE_TIMEOUT_S)
            sync_task.cancel()
            await asyncio.wait({sync_task}, timeout=_FAILURE_TIMEOUT_S)
            returned = sync_task.done()
            # The reset holds until this block ends, so the sync returned while it held.
            reset_still_holds = resetter is not None and resetter.is_alive()

        assert returned, "one cancel did not end the sync"
        assert reset_still_holds == waits
        assert sync_task.cancelled()
        assert entered.is_set() != waits
        assert f"Could not close {tmp_path / 'sync.lock'} (disk I/O error)." in caplog.text
        assert left_at_the_return == []
        with _no_sync_running(tmp_path):
            pass

    async def test_a_cancel_during_a_close_that_fails_is_still_raised(
        self, tmp_path: Path, caplog
    ) -> None:
        async def _sync() -> str:
            async with sync_running(tmp_path):
                pass
            return "finished"

        with (
            _sync_lock_close_held_then_refused() as (closing, finish),
            caplog.at_level(logging.WARNING, logger="lilbee.runtime.lock"),
        ):
            sync_task = asyncio.create_task(_sync())
            assert await asyncio.to_thread(closing.wait, _FAILURE_TIMEOUT_S)
            sync_task.cancel()
            finish.set()
            await asyncio.wait({sync_task}, timeout=_FAILURE_TIMEOUT_S)

        assert sync_task.cancelled(), "the close error replaced the cancel"
        assert f"Could not close {tmp_path / 'sync.lock'} (disk I/O error)." in caplog.text
        with _no_sync_running(tmp_path):
            pass

    @pytest.mark.parametrize("sync_error", [None, ValueError("the sync failed")])
    async def test_a_close_error_is_logged_and_keeps_the_outcome_of_the_sync(
        self, tmp_path: Path, caplog, sync_error: Exception | None
    ) -> None:
        async def _sync() -> str:
            async with sync_running(tmp_path):
                if sync_error is not None:
                    raise sync_error
                return "finished"

        before = set(threading.enumerate())
        with (
            _sync_lock_close_refused(),
            caplog.at_level(logging.WARNING, logger="lilbee.runtime.lock"),
        ):
            if sync_error is None:
                assert await _sync() == "finished"
            else:
                with pytest.raises(ValueError, match="the sync failed"):
                    await _sync()

        assert f"Could not close {tmp_path / 'sync.lock'} (disk I/O error)." in caplog.text
        assert _threads_left_since(before) == []
        with _no_sync_running(tmp_path):
            pass

    async def test_a_sync_cancelled_as_its_take_succeeds_frees_the_mark(
        self, tmp_path: Path
    ) -> None:
        still_referenced: list[ReadWriteLock] = []
        taken = threading.Event()
        proceed = threading.Event()
        real_acquire = ReadWriteLock.acquire_read

        def _acquire(lock: ReadWriteLock, *args, **kwargs) -> None:
            still_referenced.append(lock)
            real_acquire(lock, *args, **kwargs)
            taken.set()
            proceed.wait(5)

        entered = []

        async def _sync() -> None:
            async with sync_running(tmp_path):
                entered.append("sync")

        with mock.patch.object(ReadWriteLock, "acquire_read", _acquire):
            sync_task = asyncio.create_task(_sync())
            assert await asyncio.to_thread(taken.wait, 5)
            sync_task.cancel()
            asyncio.get_running_loop().call_later(0.05, proceed.set)
            await asyncio.wait({sync_task}, timeout=5)

        assert sync_task.cancelled()
        assert entered == []
        assert len(still_referenced) == 1
        with _no_sync_running(tmp_path):
            pass

    async def test_a_sync_cancelled_as_its_take_fails_does_not_run(
        self, tmp_path: Path, caplog
    ) -> None:
        taking = threading.Event()
        proceed = threading.Event()

        def _refused(lock: ReadWriteLock, *args, **kwargs) -> None:
            taking.set()
            proceed.wait(5)
            raise sqlite3.OperationalError("disk I/O error")

        entered = []

        async def _sync() -> None:
            async with sync_running(tmp_path):
                entered.append("sync")

        with (
            mock.patch.object(ReadWriteLock, "acquire_read", _refused),
            caplog.at_level(logging.WARNING, logger="lilbee.runtime.lock"),
        ):
            sync_task = asyncio.create_task(_sync())
            assert await asyncio.to_thread(taking.wait, 5)
            sync_task.cancel()
            asyncio.get_running_loop().call_later(0.05, proceed.set)
            await asyncio.wait({sync_task}, timeout=5)

        assert sync_task.cancelled()
        assert entered == []
        assert _lock_warnings(caplog) == 0

    async def test_a_sync_leaves_no_thread_behind(self, tmp_path: Path) -> None:
        before = set(threading.enumerate())
        async with sync_running(tmp_path):
            pass

        assert _threads_left_since(before) == []

    @pytest.mark.parametrize("side", ["acquire_read", "acquire_write"])
    async def test_the_mark_is_free_at_the_end_of_the_block_not_at_garbage_collection(
        self, tmp_path: Path, side: str
    ) -> None:
        still_referenced: list[ReadWriteLock] = []
        real_acquire = {
            "acquire_read": ReadWriteLock.acquire_read,
            "acquire_write": ReadWriteLock.acquire_write,
        }[side]

        def _acquire(lock: ReadWriteLock, *args, **kwargs) -> None:
            if Path(lock.lock_file).name == "sync.lock":
                still_referenced.append(lock)
            real_acquire(lock, *args, **kwargs)

        with mock.patch.object(ReadWriteLock, side, _acquire):
            if side == "acquire_read":
                async with sync_running(tmp_path):
                    pass
            else:
                with _no_sync_running(tmp_path):
                    pass

        assert len(still_referenced) == 1
        with _no_sync_running(tmp_path):
            pass

    async def test_a_sync_probes_the_data_root_off_the_event_loop(self, tmp_path: Path) -> None:
        probed_on: list[int] = []
        real_acquire_write = ReadWriteLock.acquire_write

        def _acquire_write(lock: ReadWriteLock, *args, **kwargs) -> None:
            probed_on.append(threading.get_ident())
            real_acquire_write(lock, *args, **kwargs)

        with mock.patch.object(ReadWriteLock, "acquire_write", _acquire_write):
            async with sync_running(tmp_path):
                pass

        assert len(probed_on) == 1
        assert threading.get_ident() not in probed_on

    async def test_a_reset_in_another_process_is_refused_while_a_sync_runs(
        self, tmp_path: Path
    ) -> None:
        import subprocess
        import sys

        def _reset_elsewhere() -> int:
            return subprocess.run(
                [sys.executable, "-c", _RESET_IN_ANOTHER_PROCESS, str(tmp_path)],
                timeout=60,
            ).returncode

        async with sync_running(tmp_path):
            assert await asyncio.to_thread(_reset_elsewhere) == 3
        assert await asyncio.to_thread(_reset_elsewhere) == 0

    async def test_a_sync_starts_only_after_a_reset_ends(self, tmp_path: Path) -> None:
        held = threading.Event()
        release = threading.Event()

        def _reset() -> None:
            with _no_sync_running(tmp_path):
                held.set()
                release.wait(5)

        entered = asyncio.Event()

        async def _sync() -> None:
            async with sync_running(tmp_path):
                entered.set()

        resetter = threading.Thread(target=_reset)
        resetter.start()
        assert held.wait(5)
        sync_task = asyncio.create_task(_sync())
        started = time.monotonic()
        await asyncio.sleep(0.3)
        assert time.monotonic() - started < 2, "the wait for the reset blocked the event loop"
        assert not entered.is_set()
        release.set()
        await asyncio.wait_for(sync_task, 5)
        assert entered.is_set()
        resetter.join(timeout=5)

    async def test_the_mark_lives_in_a_missing_data_root(self, tmp_path: Path) -> None:
        root = tmp_path / "not-yet"
        async with sync_running(root):
            assert root.is_dir()

    async def test_a_lock_file_that_is_not_a_database_refuses_a_sync_and_a_reset(
        self, tmp_path: Path, caplog
    ) -> None:
        junk = b"not a lock database " * 8
        assert not junk.startswith(_SQLITE_HEADER)
        lock_file = tmp_path / "sync.lock"
        lock_file.write_bytes(junk)
        ran = []
        with caplog.at_level(logging.WARNING, logger="lilbee.runtime.lock"):
            with pytest.raises(LibraryUnlockableError) as refused_sync:
                async with sync_running(tmp_path):
                    ran.append("sync")
            with (
                pytest.raises(ResetRefusedError, match="file is not a database") as caught,
                _no_sync_running(tmp_path),
            ):
                ran.append("reset")

        assert ran == []
        assert str(refused_sync.value) == (
            f"Cannot change the library: {lock_file} cannot be locked (file is not a database). "
            f"Stop every lilbee process, delete {lock_file}, and try again."
        )
        assert type(refused_sync.value.__cause__) is sqlite3.DatabaseError
        assert f"delete {lock_file}" in str(caught.value)
        assert _lock_warnings(caplog) == 0
        assert "Traceback" not in caplog.text
        assert lock_file.read_bytes() == junk
        lock_file.unlink()
        async with sync_running(tmp_path):
            ran.append("sync")
        with _no_sync_running(tmp_path):
            ran.append("reset")
        assert ran == ["sync", "reset"]

    async def test_a_lock_file_that_cannot_be_opened_refuses_a_sync_and_a_reset(
        self, tmp_path: Path, caplog
    ) -> None:
        refused = sqlite3.OperationalError("disk I/O error")

        def _probe_only(path: Path, *args, **kwargs) -> ReadWriteLock:
            if Path(path).name.startswith("sync.probe."):
                return ReadWriteLock(path, *args, **kwargs)
            raise refused

        ran = []
        with (
            mock.patch("lilbee.runtime.lock.ReadWriteLock", _probe_only),
            mock.patch("lilbee.runtime.lock.AsyncReadWriteLock", side_effect=refused),
            caplog.at_level(logging.WARNING, logger="lilbee.runtime.lock"),
        ):
            with pytest.raises(LibraryUnlockableError) as refused_sync:
                async with sync_running(tmp_path):
                    ran.append("sync")
            with (
                pytest.raises(LibraryUnlockableError) as refused_wiki,
                source_keys_in_use(tmp_path),
            ):
                ran.append("wiki")
            with (
                pytest.raises(ResetRefusedError, match="disk I/O error") as caught,
                _no_sync_running(tmp_path),
            ):
                ran.append("reset")

        assert ran == []
        lock_file = tmp_path / "sync.lock"
        assert str(refused_sync.value) == (
            f"Cannot change the library: {lock_file} cannot be locked (disk I/O error). "
            f"{_FIX_ADVICE}"
        )
        assert str(refused_wiki.value) == str(refused_sync.value)
        assert refused_sync.value.__cause__ is refused
        assert f"delete {lock_file}" in str(caught.value)
        assert caught.value.__cause__ is refused
        assert _lock_warnings(caplog) == 0

    @pytest.mark.parametrize(
        ("refused", "advice"),
        [
            pytest.param(
                OSError(errno.EMFILE, "Too many open files"),
                "Close some programs or raise the limit of open files, then try again.",
                id="too-many-open-files",
            ),
            pytest.param(
                OSError(errno.ENFILE, "Too many open files in system"),
                "Close some programs or raise the limit of open files, then try again.",
                id="file-table-full",
            ),
            pytest.param(
                OSError(errno.ENOSPC, "No space left on device"),
                "Free some disk space, then try again.",
                id="disk-full",
            ),
            pytest.param(OSError(errno.EACCES, "Permission denied"), _WRITABLE_ADVICE, id="denied"),
            pytest.param(
                OSError(errno.EPERM, "Operation not permitted"),
                _WRITABLE_ADVICE,
                id="not-permitted",
            ),
            pytest.param(
                sqlite3.OperationalError("unable to open database file"),
                _FIX_ADVICE,
                id="sqlite-cannot-open",
            ),
        ],
    )
    async def test_the_advice_of_a_refused_sync_fits_what_stopped_the_open(
        self, tmp_path: Path, refused: Exception, advice: str
    ) -> None:
        ran = []
        with (
            mock.patch("lilbee.runtime.lock.AsyncReadWriteLock", side_effect=refused),
            pytest.raises(LibraryUnlockableError) as refused_sync,
        ):
            async with sync_running(tmp_path):
                ran.append("sync")

        assert ran == []
        lock_file = tmp_path / "sync.lock"
        assert str(refused_sync.value) == (
            f"Cannot change the library: {lock_file} cannot be locked ({refused}). "
            + advice.format(path=lock_file)
        )
        assert "delete" not in str(refused_sync.value)
        assert not lock_file.exists()

    async def test_a_lock_that_cannot_be_taken_refuses_a_sync_and_a_reset(
        self, tmp_path: Path, caplog
    ) -> None:
        real_acquire_write = ReadWriteLock.acquire_write

        def _refused(lock: ReadWriteLock, *args, **kwargs) -> None:
            if Path(lock.lock_file).name.startswith("sync.probe."):
                real_acquire_write(lock, *args, **kwargs)
                return
            raise sqlite3.OperationalError("disk I/O error")

        before = set(threading.enumerate())
        ran = []
        with (
            mock.patch.object(ReadWriteLock, "acquire_read", _refused),
            mock.patch.object(ReadWriteLock, "acquire_write", _refused),
            caplog.at_level(logging.WARNING, logger="lilbee.runtime.lock"),
        ):
            with pytest.raises(LibraryUnlockableError) as refused_sync:
                async with sync_running(tmp_path):
                    ran.append("sync")
            with (
                pytest.raises(LibraryUnlockableError) as refused_wiki,
                source_keys_in_use(tmp_path),
            ):
                ran.append("wiki")
            with (
                pytest.raises(ResetRefusedError, match="disk I/O error") as caught,
                _no_sync_running(tmp_path),
            ):
                ran.append("reset")

        assert ran == []
        lock_file = tmp_path / "sync.lock"
        assert str(refused_sync.value) == (
            f"Cannot change the library: {lock_file} cannot be locked (disk I/O error). "
            f"{_FIX_ADVICE}"
        )
        assert str(refused_wiki.value) == str(refused_sync.value)
        assert f"delete {lock_file}" in str(caught.value)
        assert _lock_warnings(caplog) == 0
        assert _threads_left_since(before) == []
        async with sync_running(tmp_path):
            ran.append("sync")
        with _no_sync_running(tmp_path):
            ran.append("reset")
        assert ran == ["sync", "reset"]

    async def test_an_error_outside_the_filesystem_class_propagates(self, tmp_path: Path) -> None:
        from unittest import mock

        with (
            mock.patch("lilbee.runtime.lock.ReadWriteLock", side_effect=RuntimeError("bug")),
            pytest.raises(RuntimeError, match="bug"),
        ):
            async with sync_running(tmp_path):
                pass

    async def test_a_mount_without_locking_refuses_a_sync(
        self, tmp_path: Path, no_lock_daemon: threading.Event
    ) -> None:
        ran = []

        async def _sync() -> None:
            async with sync_running(tmp_path):
                ran.append("sync")

        try:
            with pytest.raises(LockingUnsupportedError) as caught:
                await asyncio.wait_for(_sync(), 5)
        finally:
            no_lock_daemon.set()

        assert ran == []
        assert str(caught.value) == (
            f"{tmp_path} is on a filesystem that does not support file locking. "
            "Move it to a local disk."
        )
        assert not list(tmp_path.glob("sync.probe.*"))

    @pytest.mark.usefixtures("no_lock_daemon")
    def test_a_mount_without_locking_refuses_a_reset_without_blaming_a_sync(
        self, tmp_path: Path
    ) -> None:
        ran = []
        with pytest.raises(ResetRefusedError) as caught, _no_sync_running(tmp_path):
            ran.append("reset")

        assert ran == []
        assert f"{tmp_path} is on a filesystem that does not support file locking" in str(
            caught.value
        )
        assert "is running on this library" not in str(caught.value)
        assert not list(tmp_path.glob("sync.probe.*"))

    def test_a_probe_busy_for_a_moment_lets_a_reset_run(self, tmp_path: Path) -> None:
        real_acquire_write = ReadWriteLock.acquire_write

        def _free_after_a_moment(
            lock: ReadWriteLock, timeout: float = -1, *, blocking: bool = True
        ) -> None:
            if Path(lock.lock_file).name.startswith("sync.probe.") and (
                not blocking or 0 <= timeout < 0.1
            ):
                raise FileLockTimeout(lock.lock_file)
            real_acquire_write(lock, timeout, blocking=blocking)

        ran = []
        with (
            mock.patch.object(ReadWriteLock, "acquire_write", _free_after_a_moment),
            _no_sync_running(tmp_path),
        ):
            ran.append("reset")

        assert ran == ["reset"]

    @pytest.mark.parametrize(
        "refused",
        [
            pytest.param(sqlite3.OperationalError("disk I/O error"), id="io-error"),
            pytest.param(sqlite3.DatabaseError("file is not a database"), id="not-a-database"),
        ],
    )
    async def test_a_probe_that_errors_names_the_data_directory_not_the_lock_file(
        self, tmp_path: Path, caplog, refused: sqlite3.Error
    ) -> None:
        real_acquire_write = ReadWriteLock.acquire_write

        def _probe_errors(lock: ReadWriteLock, *args, **kwargs) -> None:
            if Path(lock.lock_file).name.startswith("sync.probe."):
                raise refused
            real_acquire_write(lock, *args, **kwargs)

        ran = []
        with (
            mock.patch.object(ReadWriteLock, "acquire_write", _probe_errors),
            caplog.at_level(logging.WARNING, logger="lilbee.runtime.lock"),
        ):
            with pytest.raises(LibraryUnlockableError) as refused_sync:
                async with sync_running(tmp_path):
                    ran.append("sync")
            with pytest.raises(ResetRefusedError) as caught, _no_sync_running(tmp_path):
                ran.append("reset")

        assert ran == []
        assert str(refused_sync.value) == (
            f"Cannot change the library: {tmp_path} cannot be locked ({refused}). {_FIX_ADVICE}"
        )
        assert refused_sync.value.__cause__ is refused
        assert str(caught.value) == (
            f"Cannot tell whether a sync or import is running: {tmp_path} cannot be locked "
            f"({refused})."
        )
        assert _lock_warnings(caplog) == 0
        assert not list(tmp_path.glob("sync.probe.*"))

    @pytest.mark.skipif(sys.platform != "darwin", reason="a failed hard link stops only macOS")
    async def test_a_macos_disk_without_hard_links_runs_a_sync_and_refuses_a_reset(
        self, tmp_path: Path, caplog
    ) -> None:
        def _no_link(src: str, dst: str, **_kwargs: object) -> None:
            raise OSError(errno.ENOTSUP, "Operation not supported", str(src), None, str(dst))

        ran = []
        with (
            mock.patch.object(os, "link", _no_link),
            caplog.at_level(logging.WARNING, logger="lilbee.runtime.lock"),
        ):
            async with sync_running(tmp_path):
                ran.append("sync")
            with pytest.raises(ResetRefusedError) as caught, _no_sync_running(tmp_path):
                ran.append("reset")

        assert ran == ["sync"]
        assert str(caught.value) == (
            f"Cannot tell whether a sync or import is running: {tmp_path} cannot be locked "
            f"({_APFS_ADVICE})."
        )
        assert [r.getMessage() for r in caplog.records if r.name == "lilbee.runtime.lock"] == [
            f"Cannot lock {tmp_path} ({_APFS_ADVICE}); a reset refuses until it can."
        ]
        assert [path.name for path in tmp_path.iterdir()] == [cfg.lancedb_dir.name]

    @pytest.mark.parametrize(
        ("platform", "reported"),
        [
            pytest.param("darwin", _APFS_ADVICE, id="link-on-macos"),
            pytest.param("linux", _LOCAL_DISK_ADVICE, id="link-on-linux"),
        ],
    )
    async def test_a_probe_on_a_disk_without_hard_links_is_deleted_and_runs_a_sync(
        self, tmp_path: Path, caplog, monkeypatch, platform: str, reported: str
    ) -> None:
        monkeypatch.setattr("lilbee.runtime.lock.sys", SimpleNamespace(platform=platform))

        def _created_then_refused(path: Path, *_args, **_kwargs) -> ReadWriteLock:
            Path(path).touch()
            raise _LINK_REFUSED

        ran = []
        with (
            mock.patch("lilbee.runtime.lock.ReadWriteLock", _created_then_refused),
            caplog.at_level(logging.WARNING, logger="lilbee.runtime.lock"),
        ):
            async with sync_running(tmp_path):
                ran.append("sync")
            with source_keys_in_use(tmp_path):
                ran.append("wiki")
            with pytest.raises(ResetRefusedError) as caught, _no_sync_running(tmp_path):
                ran.append("reset")

        assert ran == ["sync", "wiki"]
        assert str(caught.value) == (
            f"Cannot tell whether a sync or import is running: {tmp_path} cannot be locked "
            f"({reported})."
        )
        assert [r.getMessage() for r in caplog.records if r.name == "lilbee.runtime.lock"] == [
            f"Cannot lock {tmp_path} ({reported}); a reset refuses until it can."
        ] * 2
        assert [path.name for path in tmp_path.iterdir()] == [cfg.lancedb_dir.name]

    @pytest.mark.parametrize(
        ("refusal", "platform", "reported", "advice"),
        [
            pytest.param(
                _OPEN_REFUSED,
                "darwin",
                f"[Errno {errno.ENOTSUP}] Operation not supported",
                _FIX_ADVICE,
                id="open-not-supported",
            ),
            pytest.param(
                _READ_ONLY,
                "darwin",
                "[Errno 30] Read-only file system",
                _WRITABLE_ADVICE,
                id="read-only",
            ),
            pytest.param(
                _LINK_DENIED,
                "darwin",
                "[Errno 13] Permission denied",
                _WRITABLE_ADVICE,
                id="link-denied",
            ),
            pytest.param(
                _LINK_DISK_FULL,
                "darwin",
                "[Errno 28] No space left on device",
                "Free some disk space, then try again.",
                id="link-disk-full",
            ),
            pytest.param(
                _LINK_TOO_MANY,
                "linux",
                "[Errno 31] Too many links",
                _FIX_ADVICE,
                id="link-too-many",
            ),
        ],
    )
    async def test_a_probe_that_cannot_be_opened_is_deleted_and_refuses_a_sync_and_a_reset(
        self,
        tmp_path: Path,
        caplog,
        monkeypatch,
        refusal: OSError,
        platform: str,
        reported: str,
        advice: str,
    ) -> None:
        monkeypatch.setattr("lilbee.runtime.lock.sys", SimpleNamespace(platform=platform))

        def _created_then_refused(path: Path, *_args, **_kwargs) -> ReadWriteLock:
            Path(path).touch()
            raise refusal

        ran = []
        with (
            mock.patch("lilbee.runtime.lock.ReadWriteLock", _created_then_refused),
            caplog.at_level(logging.WARNING, logger="lilbee.runtime.lock"),
        ):
            with pytest.raises(LibraryUnlockableError) as refused_sync:
                async with sync_running(tmp_path):
                    ran.append("sync")
            with pytest.raises(ResetRefusedError) as caught, _no_sync_running(tmp_path):
                ran.append("reset")

        assert ran == []
        assert str(refused_sync.value) == (
            f"Cannot change the library: {tmp_path} cannot be locked ({reported}). "
            + advice.format(path=tmp_path)
        )
        assert refused_sync.value.__cause__ is refusal
        assert str(caught.value) == (
            f"Cannot tell whether a sync or import is running: {tmp_path} cannot be locked "
            f"({reported})."
        )
        assert _lock_warnings(caplog) == 0
        assert [path.name for path in tmp_path.iterdir()] == [cfg.lancedb_dir.name]

    def test_a_busy_probe_is_probed_again_on_the_next_reset(self, tmp_path: Path) -> None:
        real_acquire_write = ReadWriteLock.acquire_write
        probes = []

        def _first_probe_busy(lock: ReadWriteLock, *args, **kwargs) -> None:
            if Path(lock.lock_file).name.startswith("sync.probe."):
                probes.append(lock.lock_file)
                if len(probes) == 1:
                    raise FileLockTimeout(lock.lock_file)
            real_acquire_write(lock, *args, **kwargs)

        ran = []
        with mock.patch.object(ReadWriteLock, "acquire_write", _first_probe_busy):
            with (
                pytest.raises(ResetRefusedError, match="does not support file locking"),
                _no_sync_running(tmp_path),
            ):
                ran.append("refused reset")
            with _no_sync_running(tmp_path):
                ran.append("reset")

        assert ran == ["reset"]
        assert len(probes) == 2

    async def test_a_lockable_mount_leaves_no_probe_behind(self, tmp_path: Path) -> None:
        async with sync_running(tmp_path):
            with pytest.raises(ResetRefusedError, match="is running on this library"):
                with _no_sync_running(tmp_path):
                    pass
        with _no_sync_running(tmp_path):
            pass

        assert (tmp_path / "sync.lock").is_file()
        assert not list(tmp_path.glob("sync.probe.*"))

    def test_a_probe_that_cannot_be_deleted_still_lets_a_reset_run(
        self, tmp_path: Path, caplog
    ) -> None:
        real_unlink = Path.unlink

        def _unlink(path: Path, missing_ok: bool = False) -> None:
            if path.name.startswith("sync.probe."):
                raise PermissionError("Access is denied")
            real_unlink(path, missing_ok=missing_ok)

        ran = []
        with (
            mock.patch.object(Path, "unlink", _unlink),
            caplog.at_level(logging.WARNING, logger="lilbee.runtime.lock"),
            _no_sync_running(tmp_path),
        ):
            ran.append("reset")

        assert ran == ["reset"]
        assert _lock_warnings(caplog) == 1
        assert "Could not delete the lock probe" in caplog.text
        assert len(list(tmp_path.glob("sync.probe.*"))) == 1


@dataclass
class _DrivenWait:
    """One wait for the sync lock, driven over *outcomes*, each "busy" or "taken".

    Attempt number *set_cancel_at* sets *cancel* as it ends. Each attempt records
    its seconds and the busy error it raised.
    """

    outcomes: list[str]
    cancel: threading.Event | None = None
    set_cancel_at: int = -1
    seconds: list[float] = field(default_factory=list)
    raised: list[FileLockTimeout] = field(default_factory=list)

    def run(self) -> None:
        for attempt in _sync_lock_wait(self.cancel):
            with attempt:
                self._make(attempt.seconds)

    def _make(self, seconds: float) -> None:
        number = len(self.seconds)
        self.seconds.append(seconds)
        if self.cancel is not None and number == self.set_cancel_at:
            self.cancel.set()
        if self.outcomes[number] == "busy":
            self.raised.append(FileLockTimeout("sync.lock"))
            raise self.raised[-1]


# Where a test ends a wait that a wrong rule would keep making attempts for.
_ATTEMPTS_OF_A_WRONG_WAIT = 3


class TestSyncLockWait:
    """One rule for every wait for a side of the sync lock: slices and cancel."""

    def test_a_wait_with_no_limit_tries_in_slices_until_an_attempt_takes_the_lock(self) -> None:
        wait = _DrivenWait(["busy", "busy", "taken"])

        wait.run()

        assert wait.seconds == [_SYNC_TAKE_SLICE_S] * 3
        assert len(wait.raised) == 2

    @pytest.mark.parametrize(
        ("outcomes", "set_cancel_at"),
        [
            (["busy", "busy", "taken"], 1),
            (["busy", "taken", "taken"], 1),
            (["taken", "taken"], 0),
        ],
        ids=["during the wait", "at the attempt that takes the lock", "at a free first attempt"],
    )
    def test_a_cancel_set_at_an_attempt_ends_the_wait_after_that_attempt(
        self, outcomes: list[str], set_cancel_at: int
    ) -> None:
        wait = _DrivenWait(outcomes, threading.Event(), set_cancel_at)

        with pytest.raises(SyncMarkWaitCancelledError):
            wait.run()

        assert len(wait.seconds) == set_cancel_at + 1, "the wait tried again after the cancel"

    def test_a_cancel_set_before_the_wait_ends_it_after_one_attempt(self) -> None:
        cancel = threading.Event()
        cancel.set()
        wait = _DrivenWait(["taken", "taken"], cancel)

        with pytest.raises(SyncMarkWaitCancelledError):
            wait.run()

        assert len(wait.seconds) == 1

    def test_an_error_that_is_not_a_busy_lock_ends_the_wait(self) -> None:
        tried: list[float] = []

        with pytest.raises(sqlite3.DatabaseError, match="not a database"):
            for attempt in islice(_sync_lock_wait(None), _ATTEMPTS_OF_A_WRONG_WAIT):
                with attempt:
                    tried.append(attempt.seconds)
                    raise sqlite3.DatabaseError("file is not a database")

        assert len(tried) == 1

    def test_a_busy_error_that_replaced_a_task_cancel_ends_the_wait_as_that_cancel(self) -> None:
        tried: list[float] = []
        busy = FileLockTimeout("sync.lock")
        busy.__context__ = asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            for attempt in islice(_sync_lock_wait(None), _ATTEMPTS_OF_A_WRONG_WAIT):
                with attempt:
                    tried.append(attempt.seconds)
                    raise busy

        assert len(tried) == 1


@pytest.mark.skipif(sys.platform == "win32", reason="ENOLCK comes from POSIX flock")
@pytest.mark.usefixtures("no_lock_daemon_for_file_locks")
class TestFileLocksWithoutLockDaemon:
    """A file lock the filesystem refuses with ENOLCK names the directory and the cause."""

    def test_the_store_write_lock_names_the_directory(self, tmp_path: Path) -> None:
        lancedb_dir = tmp_path / "lancedb"
        ran = []
        with pytest.raises(LockingUnsupportedError) as caught, write_lock(lancedb_dir, timeout=2):
            ran.append("write")

        assert ran == []
        assert str(caught.value).startswith(
            f"{lancedb_dir} is on a filesystem that does not support file locking."
        )
        assert isinstance(caught.value.__cause__, OSError)

    def test_the_scope_lock_names_the_shared_folder(self, tmp_path: Path) -> None:
        scope = tmp_path / "shared-root"
        with pytest.raises(LockingUnsupportedError) as caught:
            acquire_scope_lock(scope, tmp_path / "data", timeout=1)

        assert str(caught.value) == (
            f"{scope} is on a filesystem that does not support file locking. "
            "Move it to a local disk."
        )

    def test_the_server_lock_names_the_directory(self, tmp_path: Path) -> None:
        with pytest.raises(
            LockingUnsupportedError, match="does not support file locking"
        ) as caught:
            acquire_server_lock(tmp_path, timeout=1)

        assert str(tmp_path) in str(caught.value)


def test_a_file_lock_error_other_than_enolck_propagates(tmp_path: Path) -> None:
    refused = OSError(errno.EIO, "Input/output error")
    with (
        mock.patch.object(FileLock, "acquire", side_effect=refused),
        pytest.raises(OSError, match="Input/output error") as caught,
        write_lock(tmp_path / "lancedb", timeout=1),
    ):
        pass

    assert caught.value is refused
