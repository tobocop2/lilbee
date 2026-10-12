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
from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from filelock import FileLock, ReadWriteLock
from filelock import Timeout as FileLockTimeout

from lilbee.core.config import cfg
from lilbee.runtime.lock import (
    LockingUnsupportedError,
    LockTimeoutError,
    ResetRefusedError,
    _lock_path,
    acquire_scope_lock,
    acquire_server_lock,
    no_sync_running,
    read_scope_owner,
    server_lock_path,
    sync_running,
    write_lock,
)
from tests.conftest import reset_held_on_a_thread


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


_RESET_IN_ANOTHER_PROCESS = """
import sys
from pathlib import Path
from lilbee.runtime.lock import ResetRefusedError, no_sync_running
try:
    with no_sync_running(Path(sys.argv[1])):
        pass
except ResetRefusedError:
    sys.exit(3)
"""
_RESET_HOLDS = "reset holds"
_HELD_RESET_IN_ANOTHER_PROCESS = f"""
import sys
from pathlib import Path
from lilbee.runtime.lock import no_sync_running
with no_sync_running(Path(sys.argv[1])):
    print("{_RESET_HOLDS}", flush=True)
    sys.stdin.read()
"""
# A cancelled sync returns within this many seconds, however long the reset lasts.
_CANCEL_BOUND_S = 0.5


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
            with pytest.raises(ResetRefusedError), no_sync_running(tmp_path):
                pass
        with no_sync_running(tmp_path):
            pass

    @pytest.mark.parametrize("reset_held", [reset_held_on_a_thread, _reset_held_in_another_process])
    async def test_a_sync_cancelled_while_it_waits_returns_while_the_reset_still_holds(
        self, tmp_path: Path, reset_held
    ) -> None:
        waits: list[float] = []
        real_acquire = ReadWriteLock.acquire_read

        def _acquire(lock: ReadWriteLock, timeout: float = -1, *, blocking: bool = True) -> None:
            waits.append(timeout)
            real_acquire(lock, timeout, blocking=blocking)

        entered = []

        async def _sync() -> None:
            async with sync_running(tmp_path):
                entered.append("sync")

        async def _take_started() -> None:
            while not waits:
                await asyncio.sleep(0.01)

        before = set(threading.enumerate())
        with reset_held(tmp_path), mock.patch.object(ReadWriteLock, "acquire_read", _acquire):
            sync_task = asyncio.create_task(_sync())
            await asyncio.wait_for(_take_started(), 5)
            sync_task.cancel()
            await asyncio.wait({sync_task}, timeout=2)
            returned = sync_task.done()
            with (
                pytest.raises(ResetRefusedError, match="is running on this library"),
                no_sync_running(tmp_path),
            ):
                pass

        assert returned, "one cancel did not end the wait for the reset"
        assert sync_task.cancelled()
        assert entered == []
        assert waits
        assert all(0 < wait <= _CANCEL_BOUND_S / 2 for wait in waits), waits
        assert _threads_left_since(before) == []
        with no_sync_running(tmp_path):
            pass

    @pytest.mark.parametrize("cancelled_in", ["the wait", "the block"])
    async def test_a_close_error_does_not_replace_the_cancel_of_a_sync(
        self, tmp_path: Path, caplog, cancelled_in: str
    ) -> None:
        entered = asyncio.Event()

        async def _sync() -> None:
            async with sync_running(tmp_path):
                entered.set()
                await asyncio.sleep(30)

        before = set(threading.enumerate())
        waits = cancelled_in == "the wait"
        with (
            reset_held_on_a_thread(tmp_path) if waits else nullcontext(),
            _sync_lock_close_refused(),
            caplog.at_level(logging.WARNING, logger="lilbee.runtime.lock"),
        ):
            sync_task = asyncio.create_task(_sync())
            if waits:
                await asyncio.sleep(_CANCEL_BOUND_S / 2)
            else:
                await asyncio.wait_for(entered.wait(), 5)
            sync_task.cancel()
            await asyncio.wait({sync_task}, timeout=_CANCEL_BOUND_S)

        assert sync_task.cancelled()
        assert entered.is_set() != waits
        assert f"Could not close {tmp_path / 'sync.lock'} (disk I/O error)." in caplog.text
        assert _threads_left_since(before) == []
        with no_sync_running(tmp_path):
            pass

    async def test_a_close_error_after_a_sync_that_ran_is_raised(self, tmp_path: Path) -> None:
        ran = []
        with _sync_lock_close_refused(), pytest.raises(sqlite3.OperationalError, match="disk I/O"):
            async with sync_running(tmp_path):
                ran.append("sync")

        assert ran == ["sync"]
        with no_sync_running(tmp_path):
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
        with no_sync_running(tmp_path):
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
                with no_sync_running(tmp_path):
                    pass

        assert len(still_referenced) == 1
        with no_sync_running(tmp_path):
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
            with no_sync_running(tmp_path):
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

    async def test_a_lock_file_that_is_not_a_database_refuses_a_reset(
        self, tmp_path: Path, caplog
    ) -> None:
        import logging

        junk = b"not a lock database " * 8
        assert not junk.startswith(_SQLITE_HEADER)
        (tmp_path / "sync.lock").write_bytes(junk)
        ran = []
        with caplog.at_level(logging.WARNING, logger="lilbee.runtime.lock"):
            async with sync_running(tmp_path):
                ran.append("sync")
                with (
                    pytest.raises(ResetRefusedError, match="file is not a database") as caught,
                    no_sync_running(tmp_path),
                ):
                    ran.append("reset")

        assert ran == ["sync"]
        assert f"delete {tmp_path / 'sync.lock'}" in str(caught.value)
        assert _lock_warnings(caplog) == 1
        assert "Traceback" not in caplog.text
        (tmp_path / "sync.lock").unlink()
        with no_sync_running(tmp_path):
            ran.append("reset")
        assert ran == ["sync", "reset"]

    async def test_a_lock_file_that_cannot_be_opened_runs_a_sync_and_refuses_a_reset(
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
            async with sync_running(tmp_path):
                ran.append("sync")
            with (
                pytest.raises(ResetRefusedError, match="disk I/O error") as caught,
                no_sync_running(tmp_path),
            ):
                ran.append("reset")

        assert ran == ["sync"]
        assert f"delete {tmp_path / 'sync.lock'}" in str(caught.value)
        assert caught.value.__cause__ is refused
        assert _lock_warnings(caplog) == 1

    async def test_a_lock_that_cannot_be_taken_runs_a_sync_and_refuses_a_reset(
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
            async with sync_running(tmp_path):
                ran.append("sync")
            with (
                pytest.raises(ResetRefusedError, match="disk I/O error") as caught,
                no_sync_running(tmp_path),
            ):
                ran.append("reset")

        assert ran == ["sync"]
        assert f"delete {tmp_path / 'sync.lock'}" in str(caught.value)
        assert _lock_warnings(caplog) == 1
        assert _threads_left_since(before) == []
        with no_sync_running(tmp_path):
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
        with pytest.raises(ResetRefusedError) as caught, no_sync_running(tmp_path):
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
            no_sync_running(tmp_path),
        ):
            ran.append("reset")

        assert ran == ["reset"]

    async def test_a_probe_that_errors_names_the_data_directory_not_the_lock_file(
        self, tmp_path: Path, caplog
    ) -> None:
        real_acquire_write = ReadWriteLock.acquire_write

        def _probe_errors(lock: ReadWriteLock, *args, **kwargs) -> None:
            if Path(lock.lock_file).name.startswith("sync.probe."):
                raise sqlite3.OperationalError("disk I/O error")
            real_acquire_write(lock, *args, **kwargs)

        ran = []
        with (
            mock.patch.object(ReadWriteLock, "acquire_write", _probe_errors),
            caplog.at_level(logging.WARNING, logger="lilbee.runtime.lock"),
        ):
            async with sync_running(tmp_path):
                ran.append("sync")
            with pytest.raises(ResetRefusedError) as caught, no_sync_running(tmp_path):
                ran.append("reset")

        assert ran == ["sync"]
        assert str(caught.value) == (
            f"Cannot tell whether a sync or import is running: {tmp_path} cannot be locked "
            "(disk I/O error)."
        )
        assert _lock_warnings(caplog) == 1

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
            with pytest.raises(ResetRefusedError) as caught, no_sync_running(tmp_path):
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
        ("refusal", "platform", "reported"),
        [
            pytest.param(_LINK_REFUSED, "darwin", _APFS_ADVICE, id="link-on-macos"),
            pytest.param(_LINK_REFUSED, "linux", _LOCAL_DISK_ADVICE, id="link-on-linux"),
            pytest.param(_OPEN_REFUSED, "darwin", str(_OPEN_REFUSED), id="open-not-supported"),
            pytest.param(_READ_ONLY, "darwin", "[Errno 30] Read-only file system", id="read-only"),
        ],
    )
    async def test_a_probe_that_cannot_be_opened_is_deleted_and_reported(
        self, tmp_path: Path, caplog, monkeypatch, refusal: OSError, platform: str, reported: str
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
            async with sync_running(tmp_path):
                ran.append("sync")
            with pytest.raises(ResetRefusedError) as caught, no_sync_running(tmp_path):
                ran.append("reset")

        assert ran == ["sync"]
        assert str(caught.value) == (
            f"Cannot tell whether a sync or import is running: {tmp_path} cannot be locked "
            f"({reported})."
        )
        assert [r.getMessage() for r in caplog.records if r.name == "lilbee.runtime.lock"] == [
            f"Cannot lock {tmp_path} ({reported}); a reset refuses until it can."
        ]
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
                no_sync_running(tmp_path),
            ):
                ran.append("refused reset")
            with no_sync_running(tmp_path):
                ran.append("reset")

        assert ran == ["reset"]
        assert len(probes) == 2

    async def test_a_lockable_mount_leaves_no_probe_behind(self, tmp_path: Path) -> None:
        async with sync_running(tmp_path):
            with pytest.raises(ResetRefusedError, match="is running on this library"):
                with no_sync_running(tmp_path):
                    pass
        with no_sync_running(tmp_path):
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
            no_sync_running(tmp_path),
        ):
            ran.append("reset")

        assert ran == ["reset"]
        assert _lock_warnings(caplog) == 1
        assert "Could not delete the lock probe" in caplog.text
        assert len(list(tmp_path.glob("sync.probe.*"))) == 1


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
