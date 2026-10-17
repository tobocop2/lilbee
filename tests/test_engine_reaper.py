"""The engine reaper: it stops a shared engine once no user holds it, and only then."""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from filelock import FileLock
from filelock import Timeout as FileLockTimeout
from pytest_httpserver import HTTPServer

from lilbee.core.config.defaults import ENGINE_IDLE_TTL_MINUTES_DEFAULT
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

_UNLOAD_PATH = "/api/models/unload"  # llama-swap's unload-all route


class _FakeReaper:
    """A launched reaper whose exit code is fixed: None while it runs."""

    def __init__(self, *, exit_code: int | None) -> None:
        self._exit_code = exit_code

    def poll(self) -> int | None:
        return self._exit_code


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


class _Clock:
    """A monotonic clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float):
        def _step() -> None:
            self.now += seconds

        return _step


@pytest.fixture
def unload_server(httpserver: HTTPServer, tmp_path: Path) -> HTTPServer:
    """A fake llama-swap proxy recorded as the engine at *tmp_path*; it answers the unload call."""
    httpserver.expect_request(_UNLOAD_PATH, method="POST").respond_with_json({"msg": "ok"})
    tmp_path.mkdir(parents=True, exist_ok=True)
    state = {"pid": 1, "proxy_port": httpserver.port}
    (tmp_path / "llama-swap.state.chat.1.json").write_text(json.dumps(state), encoding="utf-8")
    return httpserver


def _unloads(server: HTTPServer) -> int:
    return sum(1 for request, _response in server.log if request.path == _UNLOAD_PATH)


def _engine_gone(engine_dir: Path):
    def _remove() -> None:
        for state in engine_dir.glob("llama-swap.state.*"):
            state.unlink()

    return _remove


def _watch_keep_warm(
    engine_dir: Path, monkeypatch: pytest.MonkeyPatch, server: HTTPServer, *steps
) -> list[int]:
    """Run the real watch loop, one step per poll sleep; the unload count seen at each sleep."""
    clock = _Clock()
    seen: list[int] = []

    def _record(step):
        def _run() -> None:
            seen.append(_unloads(server))
            step(clock)

        return _run

    _sleep_runs(monkeypatch, *(_record(step) for step in steps))
    reaper_mod.watch(engine_dir, _held(engine_dir), poll_s=0, clock=clock)
    return seen


def _wait(seconds: float):
    return lambda clock: clock.advance(seconds)()


def _then(*actions):
    def _run(clock: _Clock) -> None:
        for action in actions:
            action(clock)

    return _run


def test_a_keep_warm_engine_unloads_once_after_the_idle_time(
    tmp_path: Path, stops, monkeypatch: pytest.MonkeyPatch, unload_server: HTTPServer
) -> None:
    request_keep_warm(tmp_path, tmp_path / "install", 5)

    seen = _watch_keep_warm(
        tmp_path,
        monkeypatch,
        unload_server,
        _wait(299),  # the count starts at the first poll with no user
        _wait(1),
        _wait(600),
        lambda _c: _engine_gone(tmp_path)(),
    )

    assert seen == [0, 0, 1, 1]  # one unload at 300 s, no repeat while unloaded
    assert stops == []  # the engine process stays up


def test_a_user_who_joins_mid_count_cancels_the_unload_and_resets_the_count(
    tmp_path: Path, stops, monkeypatch: pytest.MonkeyPatch, unload_server: HTTPServer
) -> None:
    request_keep_warm(tmp_path, tmp_path / "install", 5)
    holds: list[UserLockHold] = []

    def _joins(_clock: _Clock) -> None:
        holds.append(hold_user_lock(tmp_path, pid=616161))

    def _leaves(_clock: _Clock) -> None:
        holds[0].release_and_check_last()

    seen = _watch_keep_warm(
        tmp_path,
        monkeypatch,
        unload_server,
        _then(_wait(200), _joins),
        _then(_wait(1000), _leaves),  # the user held it past the idle time
        _wait(299),
        _wait(1),
        lambda _c: _engine_gone(tmp_path)(),
    )

    assert seen == [0, 0, 0, 0, 1]  # 300 s after the user left, not after the first idle poll
    assert stops == []


def test_a_zero_marker_never_unloads(
    tmp_path: Path, stops, monkeypatch: pytest.MonkeyPatch, unload_server: HTTPServer
) -> None:
    request_keep_warm(tmp_path, tmp_path / "patient", 0)
    request_keep_warm(tmp_path, tmp_path / "eager", 1)

    seen = _watch_keep_warm(
        tmp_path, monkeypatch, unload_server, _wait(10**6), lambda _c: _engine_gone(tmp_path)()
    )

    assert seen == [0, 0]
    assert stops == []


def test_the_largest_marker_sets_the_idle_time(
    tmp_path: Path, stops, monkeypatch: pytest.MonkeyPatch, unload_server: HTTPServer
) -> None:
    request_keep_warm(tmp_path, tmp_path / "short", 3)
    request_keep_warm(tmp_path, tmp_path / "long", 10)

    seen = _watch_keep_warm(
        tmp_path,
        monkeypatch,
        unload_server,
        _wait(180),
        _wait(419),
        _wait(1),
        lambda _c: _engine_gone(tmp_path)(),
    )

    assert seen == [0, 0, 0, 1]  # 600 s, not 180 s


def test_an_empty_marker_reads_as_the_default_idle_time(
    tmp_path: Path, stops, monkeypatch: pytest.MonkeyPatch, unload_server: HTTPServer
) -> None:
    request_keep_warm(tmp_path, tmp_path / "older-install", 30)
    (marker,) = (tmp_path / "engine-users").glob("*.keep-warm")
    marker.write_text("", encoding="utf-8")  # an opt-in written before markers held minutes
    default_s = ENGINE_IDLE_TTL_MINUTES_DEFAULT * 60

    seen = _watch_keep_warm(
        tmp_path,
        monkeypatch,
        unload_server,
        _wait(default_s - 1),
        _wait(1),
        lambda _c: _engine_gone(tmp_path)(),
    )

    assert seen == [0, 0, 1]


def test_a_failed_unload_is_tried_again_on_the_next_poll(
    tmp_path: Path, stops, monkeypatch: pytest.MonkeyPatch, httpserver: HTTPServer
) -> None:
    httpserver.expect_ordered_request(_UNLOAD_PATH, method="POST").respond_with_data(status=500)
    httpserver.expect_ordered_request(_UNLOAD_PATH, method="POST").respond_with_json({})
    state = {"pid": 1, "proxy_port": httpserver.port}
    (tmp_path / "llama-swap.state.chat.1.json").write_text(json.dumps(state), encoding="utf-8")
    (tmp_path / "llama-swap.state.embed.1.json").write_text("{}", encoding="utf-8")  # unreadable
    request_keep_warm(tmp_path, tmp_path / "install", 1)

    seen = _watch_keep_warm(
        tmp_path,
        monkeypatch,
        httpserver,
        _wait(60),
        _wait(1),
        _wait(1),
        lambda _c: _engine_gone(tmp_path)(),
    )

    assert seen == [0, 1, 2, 2]


def test_an_unreachable_proxy_reads_as_a_failed_unload(tmp_path: Path) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        closed_port = sock.getsockname()[1]
    state = {"pid": 1, "proxy_port": closed_port}
    (tmp_path / "llama-swap.state.chat.1.json").write_text(json.dumps(state), encoding="utf-8")

    assert sm.unload_engine_models(tmp_path) is False


def test_a_keep_warm_optin_holds_the_engine_until_it_is_withdrawn(
    tmp_path: Path, stops, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record_engine(tmp_path)
    request_keep_warm(tmp_path, tmp_path / "install", 5)

    def _withdraw() -> None:
        assert stops == []
        withdraw_keep_warm(tmp_path, tmp_path / "install")

    slept = _sleep_runs(monkeypatch, _withdraw)
    reaper_mod.watch(tmp_path, _held(tmp_path), poll_s=0)

    assert len(slept) == 1  # the opt-in held the engine through one poll
    assert stops == [tmp_path]


def test_a_dir_without_an_engine_ends_the_watch_without_a_stop(tmp_path: Path, stops) -> None:
    request_keep_warm(tmp_path, tmp_path / "install", 5)  # an opt-in alone keeps nothing alive

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


def test_main_claims_the_dir_past_a_users_brief_probe(
    tmp_path: Path, stops, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A starting user probes the reaper lock while the reaper starts; the reaper still claims."""
    _record_engine(tmp_path)
    probe = _held(tmp_path)
    timer = threading.Timer(0.3, probe.release)
    timer.start()
    monkeypatch.setattr(sys, "argv", ["engine_reaper", str(tmp_path)])
    try:
        reaper_mod.main()
    finally:
        timer.join()

    assert stops == [tmp_path]


def test_main_leaves_a_dir_another_reaper_watches(
    tmp_path: Path, stops, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record_engine(tmp_path)
    other = _held(tmp_path)
    monkeypatch.setattr(reaper_mod, "_CLAIM_TIMEOUT_S", 0.1)
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
            assert sm.ensure_engine_reaper(tmp_path) is True
        finally:
            held.release()
        assert spawned == []

    def test_an_unwatched_dir_starts_a_detached_reaper(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spawned: list[tuple[list[str], dict[str, object]]] = []
        claims: list[FileLock] = []

        def _reaper_claims(argv: list[str], **kwargs: object) -> _FakeReaper:
            spawned.append((argv, kwargs))
            claims.append(_held(tmp_path))
            return _FakeReaper(exit_code=None)

        monkeypatch.setattr(sm.subprocess, "Popen", _reaper_claims)
        try:
            assert sm.ensure_engine_reaper(tmp_path) is True
        finally:
            claims[0].release()

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
            assert sm.ensure_engine_reaper(tmp_path) is False
        assert "Could not launch the engine reaper" in caplog.text

    def test_a_reaper_that_exits_without_claiming_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A reaper that dies on import launched fine; only its missing claim shows it."""
        monkeypatch.setattr(sm.subprocess, "Popen", lambda *_a, **_k: _FakeReaper(exit_code=1))
        with caplog.at_level(logging.WARNING, logger=sm.__name__):
            assert sm.ensure_engine_reaper(tmp_path) is False
        assert "did not report running" in caplog.text

    def test_a_reaper_that_never_claims_times_out(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sm.subprocess, "Popen", lambda *_a, **_k: _FakeReaper(exit_code=None))
        monkeypatch.setattr(sm, "_REAPER_READY_TIMEOUT_S", 0.2)
        assert sm.ensure_engine_reaper(tmp_path) is False

    def test_a_reaper_that_lost_the_claim_to_a_live_one_counts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two users ensure at once: the loser's reaper exits, and the winner's lock answers."""
        winner: list[FileLock] = []

        def _loses(*_a: object, **_k: object) -> _FakeReaper:
            winner.append(_held(tmp_path))
            return _FakeReaper(exit_code=0)

        monkeypatch.setattr(sm.subprocess, "Popen", _loses)
        try:
            assert sm.ensure_engine_reaper(tmp_path) is True
        finally:
            winner[0].release()


def test_the_spawned_module_is_the_reaper_module() -> None:
    """The spawn names the module by string; a rename must break here, not in the field."""
    assert reaper_mod.__name__ == sm._REAPER_MODULE
