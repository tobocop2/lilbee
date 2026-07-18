"""The engine's cross-process lifecycle primitives: dirs, build lock, user locks."""

import subprocess
import sys
import threading
from pathlib import Path

import pytest

from lilbee.runtime.engine_lock import (
    build_lock,
    hold_user_lock,
    live_users_exist,
    machine_engine_dir,
    private_engine_dir,
)


class TestEngineDirs:
    def test_env_override_wins(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LILBEE_ENGINE_DIR", str(tmp_path / "slot"))
        assert machine_engine_dir() == tmp_path / "slot"

    def test_default_is_the_per_user_cache_slot(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LILBEE_ENGINE_DIR", raising=False)
        path = machine_engine_dir()
        assert path.name == "engine"
        assert path.parent.name == "lilbee"

    def test_private_dir_lives_under_the_config_root(self, tmp_path: Path):
        assert private_engine_dir(tmp_path) == tmp_path / "data" / "engine"


class TestBuildLock:
    def test_serializes_two_builders(self, tmp_path: Path):
        order: list[str] = []
        first_in = threading.Event()
        release_first = threading.Event()

        def builder(name: str, gate: threading.Event | None) -> None:
            with build_lock(tmp_path):
                order.append(f"{name}-in")
                if gate is not None:
                    first_in.set()
                    gate.wait(timeout=5)
                order.append(f"{name}-out")

        t1 = threading.Thread(target=builder, args=("a", release_first))
        t2 = threading.Thread(target=builder, args=("b", None))
        t1.start()
        first_in.wait(timeout=5)
        t2.start()
        release_first.set()
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert order == ["a-in", "a-out", "b-in", "b-out"]

    def test_creates_the_engine_dir(self, tmp_path: Path):
        missing = tmp_path / "not" / "yet"
        with build_lock(missing):
            assert missing.is_dir()


class TestUserLocks:
    def test_sole_holder_is_last_out(self, tmp_path: Path):
        hold = hold_user_lock(tmp_path)
        assert hold.release_and_check_last() is True

    def test_live_peer_means_not_last(self, tmp_path: Path):
        peer = hold_user_lock(tmp_path, pid=111_111)
        me = hold_user_lock(tmp_path, pid=222_222)
        assert me.release_and_check_last() is False
        assert peer.release_and_check_last() is True

    def test_dead_peer_lock_file_is_cleaned_during_the_check(self, tmp_path: Path):
        dead = tmp_path / "engine-users" / "999999.lock"
        dead.parent.mkdir(parents=True, exist_ok=True)
        dead.touch()  # a lock file whose holder never releases: a dead process
        me = hold_user_lock(tmp_path)
        assert me.release_and_check_last() is True
        assert not dead.exists()

    def test_release_is_idempotent(self, tmp_path: Path):
        hold = hold_user_lock(tmp_path)
        assert hold.release_and_check_last() is True
        assert hold.release_and_check_last() is True

    def test_own_lock_file_is_removed_on_release(self, tmp_path: Path):
        hold = hold_user_lock(tmp_path, pid=333_333)
        own = tmp_path / "engine-users" / "333333.lock"
        assert own.exists()
        hold.release_and_check_last()
        assert not own.exists()

    def test_peer_in_another_process_refuses_the_probe(self, tmp_path: Path):
        """A lock held by a different process reads as a live peer.

        In-process holders short-circuit on the shared singleton instance, so
        the probe's timeout path is exercised only across a real process
        boundary, where the kernel refuses the acquire.
        """
        path = tmp_path / "engine-users" / "424242.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        script = (
            "import sys, time\n"
            "from filelock import FileLock\n"
            "lock = FileLock(sys.argv[1], thread_local=False)\n"  # keep a ref: GC releases
            "lock.acquire()\n"
            "print('held', flush=True)\n"
            "time.sleep(30)\n"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", script, str(path)], stdout=subprocess.PIPE, text=True
        )
        try:
            assert proc.stdout is not None and proc.stdout.readline().strip() == "held"
            assert live_users_exist(tmp_path) is True
            assert path.exists()
        finally:
            proc.kill()
            proc.wait()

    def test_two_holds_in_one_process_share_the_lock(self, tmp_path: Path):
        """Two providers in one process hold the same pid-named lock file.

        On Linux's fcntl locks a second FileLock instance would falsely
        succeed or trip filelock's deadlock detection (the CI integration
        failure); the singleton instance must count holds instead. The file
        survives the first release and only the last release is 'last out'.
        """
        first = hold_user_lock(tmp_path)
        second = hold_user_lock(tmp_path)
        assert first.release_and_check_last() is False  # second still holds
        assert first.path.exists()
        assert second.release_and_check_last() is True
        assert not second.path.exists()

    def test_own_hold_counts_as_a_live_user(self, tmp_path: Path):
        """A process probing a dir where it holds membership must see itself.

        The config-change restart path runs the ladder while this process's
        hold is still in place. The probe uses filelock's default constructor
        while the hold uses thread_local=False on the same path; this pins
        that the probe still refuses (and never deletes) the live lock file
        if a filelock upgrade changes its per-path singleton semantics.
        """
        hold = hold_user_lock(tmp_path)
        assert live_users_exist(tmp_path) is True
        assert hold.path.exists()
        hold.release_and_check_last()

    def test_release_on_another_thread_still_counts_as_last(self, tmp_path: Path):
        """Acquire and release run on different threads in real fronts.

        The fleet builds (and takes membership) on a warm-up thread while
        teardown runs on the signal/exit path. The hold must not mistake its
        own still-held lock for a live peer when the releasing thread differs
        from the acquiring one.
        """
        acquired: list = []

        def acquire() -> None:
            acquired.append(hold_user_lock(tmp_path))

        thread = threading.Thread(target=acquire)
        thread.start()
        thread.join(timeout=5)
        hold = acquired[0]
        assert hold.release_and_check_last() is True
        assert not hold.path.exists()
