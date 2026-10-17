"""The engine reaper: it stops a shared engine once no user holds it, and only then."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pytest
from filelock import FileLock
from filelock import Timeout as FileLockTimeout

from lilbee.providers.fleet import engine_reaper as reaper_mod
from lilbee.providers.fleet import swap_manager as sm
from lilbee.runtime.engine_lock import (
    UserLockHold,
    hold_user_lock,
    reaper_running,
    request_keep_warm,
    try_hold_reaper_lock,
    withdraw_keep_warm,
)


def _record_engine(engine_dir: Path) -> None:
    engine_dir.mkdir(parents=True, exist_ok=True)
    (engine_dir / "llama-swap.state.1.chat.json").write_text("{}", encoding="utf-8")


def _held(engine_dir: Path) -> FileLock:
    lock = try_hold_reaper_lock(engine_dir)
    assert lock is not None
    return lock


@pytest.fixture
def stops(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record stop_engine calls; the real one would signal recorded pids."""
    calls: list[Path] = []

    def _stop(engine_dir: Path) -> list[str]:
        calls.append(engine_dir)
        for state in engine_dir.glob("llama-swap.state.*"):
            state.unlink()
        return []

    monkeypatch.setattr(sm, "stop_engine", _stop)
    return calls


def _sleep_runs(monkeypatch: pytest.MonkeyPatch, *actions) -> list[float]:
    """Run one action per poll sleep, in order; fail if the watch outlasts them."""
    pending = list(actions)
    slept: list[float] = []

    def _sleep(seconds: float) -> None:
        slept.append(seconds)
        assert pending, "the watch kept polling past the expected end"
        pending.pop(0)()

    monkeypatch.setattr(reaper_mod.time, "sleep", _sleep)
    return slept


def test_an_engine_nobody_holds_is_stopped_and_the_watch_ends(tmp_path: Path, stops) -> None:
    _record_engine(tmp_path)
    reaper_lock = _held(tmp_path)  # referenced, so only the watch can release it

    reaper_mod.watch(tmp_path, reaper_lock, poll_s=0)

    assert stops == [tmp_path]
    assert reaper_running(tmp_path) is False


def test_the_engine_lives_while_a_user_holds_it(
    tmp_path: Path, stops, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record_engine(tmp_path)
    user = hold_user_lock(tmp_path, pid=424242)

    def _user_leaves() -> None:
        assert stops == []  # still held on the first poll
        user.release_and_check_last()

    slept = _sleep_runs(monkeypatch, _user_leaves)
    reaper_mod.watch(tmp_path, _held(tmp_path), poll_s=0)

    assert len(slept) == 1
    assert stops == [tmp_path]


def test_a_user_who_binds_while_the_reaper_waits_for_the_lock_keeps_the_engine(
    tmp_path: Path, stops, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recheck under the build lock sees a user who joined after the unlocked poll."""
    _record_engine(tmp_path)
    real = reaper_mod.build_lock
    joined: list[UserLockHold] = []

    def _binder_joins_first(engine_dir: Path):
        if not joined:
            joined.append(hold_user_lock(engine_dir, pid=525252))
        return real(engine_dir)

    def _binder_leaves() -> None:
        assert stops == []
        joined[0].release_and_check_last()

    monkeypatch.setattr(reaper_mod, "build_lock", _binder_joins_first)
    slept = _sleep_runs(monkeypatch, _binder_leaves)
    reaper_mod.watch(tmp_path, _held(tmp_path), poll_s=0)

    assert len(slept) == 1
    assert stops == [tmp_path]


def test_a_keep_warm_optin_holds_the_engine_until_it_is_withdrawn(
    tmp_path: Path, stops, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record_engine(tmp_path)
    request_keep_warm(tmp_path, tmp_path / "install")

    def _withdraw() -> None:
        assert stops == []
        withdraw_keep_warm(tmp_path, tmp_path / "install")

    slept = _sleep_runs(monkeypatch, _withdraw)
    reaper_mod.watch(tmp_path, _held(tmp_path), poll_s=0)

    assert len(slept) == 1  # the opt-in held the engine through one poll
    assert stops == [tmp_path]


def test_a_dir_without_an_engine_ends_the_watch_without_a_stop(tmp_path: Path, stops) -> None:
    request_keep_warm(tmp_path, tmp_path / "install")  # an opt-in alone keeps nothing alive

    reaper_mod.watch(tmp_path, _held(tmp_path), poll_s=0)

    assert stops == []
    assert reaper_running(tmp_path) is False


def test_a_held_build_lock_is_waited_out_not_stepped_over(
    tmp_path: Path, stops, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A build in progress has not taken its membership yet; the reaper must not stop it."""
    _record_engine(tmp_path)
    real = reaper_mod.build_lock
    attempts: list[int] = []

    def _busy_once(engine_dir: Path):
        attempts.append(1)
        if len(attempts) == 1:
            raise FileLockTimeout(str(engine_dir))
        return real(engine_dir)

    monkeypatch.setattr(reaper_mod, "build_lock", _busy_once)
    _sleep_runs(monkeypatch, lambda: None)
    with caplog.at_level(logging.INFO, logger=reaper_mod.__name__):
        reaper_mod.watch(tmp_path, _held(tmp_path), poll_s=0)

    assert len(attempts) == 2
    assert stops == [tmp_path]
    assert "still held" in caplog.text


def test_main_watches_the_dir_named_in_argv(
    tmp_path: Path, stops, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record_engine(tmp_path)
    monkeypatch.setattr(sys, "argv", ["engine_reaper", str(tmp_path)])

    reaper_mod.main()

    assert stops == [tmp_path]


def test_main_leaves_a_dir_another_reaper_watches(
    tmp_path: Path, stops, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record_engine(tmp_path)
    other = _held(tmp_path)
    monkeypatch.setattr(sys, "argv", ["engine_reaper", str(tmp_path)])
    try:
        reaper_mod.main()
    finally:
        other.release()

    assert stops == []


class TestEnsureEngineReaper:
    def test_a_watched_dir_starts_no_second_reaper(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spawned: list[object] = []
        monkeypatch.setattr(sm.subprocess, "Popen", lambda *a, **k: spawned.append(a))
        held = _held(tmp_path)
        try:
            assert sm._ensure_engine_reaper(tmp_path) is True
        finally:
            held.release()
        assert spawned == []

    def test_an_unwatched_dir_starts_a_detached_reaper(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spawned: list[tuple[list[str], dict[str, object]]] = []
        monkeypatch.setattr(sm.subprocess, "Popen", lambda argv, **k: spawned.append((argv, k)))

        assert sm._ensure_engine_reaper(tmp_path) is True

        assert len(spawned) == 1
        argv, kwargs = spawned[0]
        assert argv == [sys.executable, "-m", "lilbee.providers.fleet.engine_reaper", str(tmp_path)]
        assert kwargs["start_new_session"] is True
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert kwargs["stdout"] is subprocess.DEVNULL
        assert kwargs["stderr"] is subprocess.DEVNULL

    def test_a_reaper_that_cannot_start_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def _refuse(*_a: object, **_k: object) -> None:
            raise OSError("no fork")

        monkeypatch.setattr(sm.subprocess, "Popen", _refuse)
        with caplog.at_level(logging.WARNING, logger=sm.__name__):
            assert sm._ensure_engine_reaper(tmp_path) is False
        assert "Could not start the engine reaper" in caplog.text


def test_the_spawned_module_is_the_reaper_module() -> None:
    """The spawn names the module by string; a rename must break here, not in the field."""
    assert reaper_mod.__name__ == sm._REAPER_MODULE
