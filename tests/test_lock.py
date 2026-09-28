"""Tests for write locking and file locking."""

import asyncio
import logging
import re
import sqlite3
import threading
import time
from pathlib import Path
from unittest import mock

import pytest
from filelock import _read_write as filelock_read_write
from filelock._read_write import _ForkSafeConnection

from lilbee.core.config import cfg
from lilbee.runtime.lock import (
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


class _NoLockDaemonConnection(_ForkSafeConnection):
    """SQLite on a mount with no lock daemon: every lock reports busy until its busy timeout."""

    unblock = threading.Event()

    def executescript(self, sql_script: str) -> sqlite3.Cursor:
        if "BEGIN" not in sql_script:
            return super().executescript(sql_script)
        pragma = re.search(r"busy_timeout=(\d+)", sql_script)
        (busy_ms,) = (
            self.execute("PRAGMA busy_timeout").fetchone()
            if pragma is None
            else (int(pragma.group(1)),)
        )
        self.unblock.wait(busy_ms / 1000)
        raise sqlite3.OperationalError("database is locked")


@pytest.fixture
def no_lock_daemon():
    """Route filelock's SQLite connections through a filesystem whose locks never succeed."""

    def _connect(database: str, *, factory: type, timeout: float) -> _ForkSafeConnection:
        return real_connect(database, factory=_NoLockDaemonConnection, timeout=timeout)

    real_connect = filelock_read_write._connect
    _NoLockDaemonConnection.unblock.clear()
    with mock.patch.object(filelock_read_write, "_connect", _connect):
        yield
    _NoLockDaemonConnection.unblock.set()


def _lock_warnings(caplog) -> int:
    return sum(r.name == "lilbee.runtime.lock" and r.levelname == "WARNING" for r in caplog.records)


class TestSyncMark:
    """Syncs share the data root's mark; a reset needs it free, in any process."""

    async def test_syncs_share_the_mark(self, tmp_path: Path) -> None:
        async with sync_running(tmp_path), sync_running(tmp_path):
            with pytest.raises(ResetRefusedError), no_sync_running(tmp_path):
                pass
        with no_sync_running(tmp_path):
            pass

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

    @pytest.mark.parametrize("fails_at", ["open", "acquire"])
    async def test_a_filesystem_that_refuses_the_lock_runs_a_sync_and_refuses_a_reset(
        self, tmp_path: Path, caplog, fails_at
    ) -> None:
        import logging
        import sqlite3
        from unittest import mock

        refused = sqlite3.OperationalError("disk I/O error")
        lock = mock.MagicMock()
        lock.acquire_read.side_effect = refused
        lock.acquire_write.side_effect = refused
        factory = (
            mock.MagicMock(side_effect=refused) if fails_at == "open" else lambda *_a, **_k: lock
        )
        ran = []
        with (
            mock.patch("lilbee.runtime.lock.ReadWriteLock", factory),
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
        assert str(tmp_path / "sync.lock") in str(caught.value)
        assert _lock_warnings(caplog) == 1
        assert lock.close.call_count == 2 * (fails_at == "acquire")
        lock.release.assert_not_called()

    async def test_an_error_outside_the_filesystem_class_propagates(self, tmp_path: Path) -> None:
        from unittest import mock

        with (
            mock.patch("lilbee.runtime.lock.ReadWriteLock", side_effect=RuntimeError("bug")),
            pytest.raises(RuntimeError, match="bug"),
        ):
            async with sync_running(tmp_path):
                pass

    @pytest.mark.usefixtures("no_lock_daemon")
    async def test_a_mount_without_locking_runs_a_sync_with_a_warning(
        self, tmp_path: Path, caplog
    ) -> None:
        ran = []
        with caplog.at_level(logging.WARNING, logger="lilbee.runtime.lock"):

            async def _sync() -> None:
                async with sync_running(tmp_path):
                    ran.append("sync")

            try:
                await asyncio.wait_for(_sync(), 5)
            finally:
                _NoLockDaemonConnection.unblock.set()

        assert ran == ["sync"]
        assert _lock_warnings(caplog) == 1
        assert f"{tmp_path} is on a filesystem that does not support file locking" in caplog.text
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
