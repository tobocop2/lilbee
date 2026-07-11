"""Tests for write locking and file locking."""

import threading
import time
from pathlib import Path

import pytest

from lilbee.core.config import cfg
from lilbee.runtime.lock import (
    LockTimeoutError,
    _lock_path,
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
