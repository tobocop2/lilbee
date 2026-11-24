"""Tests for ``lilbee.parent_monitor`` parent-death detection."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from unittest import mock

import psutil
import pytest

from lilbee import parent_monitor
from lilbee.parent_monitor import (
    PARENT_PID_ENV,
    _parent_start_time,
    _same_process,
    _start_identity,
    parse_parent_pid,
    watch_parent_thread,
)


@pytest.fixture(autouse=True)
def clean_env() -> Iterator[None]:
    """Ensure LILBEE_PARENT_PID does not leak between tests."""
    snapshot = os.environ.get(PARENT_PID_ENV)
    os.environ.pop(PARENT_PID_ENV, None)
    yield
    if snapshot is None:
        os.environ.pop(PARENT_PID_ENV, None)
    else:
        os.environ[PARENT_PID_ENV] = snapshot


@pytest.fixture(autouse=True)
def create_time_identity() -> Iterator[None]:
    """Run the identity tests on the create-time path; the Linux path has its own tests."""
    with mock.patch("lilbee.parent_monitor._is_linux", return_value=False):
        yield


class TestParseParentPid:
    def test_returns_none_when_unset(self):
        assert parse_parent_pid() is None

    def test_returns_none_when_empty_string(self):
        assert parse_parent_pid({"LILBEE_PARENT_PID": ""}) is None

    def test_returns_none_for_garbage_string(self, caplog):
        with caplog.at_level("WARNING"):
            assert parse_parent_pid({"LILBEE_PARENT_PID": "not-a-pid"}) is None
        assert "not an integer" in caplog.text

    def test_returns_none_for_zero(self, caplog):
        with caplog.at_level("WARNING"):
            assert parse_parent_pid({"LILBEE_PARENT_PID": "0"}) is None
        assert "non-positive" in caplog.text

    def test_returns_none_for_negative(self, caplog):
        with caplog.at_level("WARNING"):
            assert parse_parent_pid({"LILBEE_PARENT_PID": "-7"}) is None
        assert "non-positive" in caplog.text

    def test_returns_pid_for_valid_input(self):
        assert parse_parent_pid({"LILBEE_PARENT_PID": "12345"}) == 12345

    def test_reads_real_environ_when_no_dict_passed(self):
        os.environ[PARENT_PID_ENV] = "9876"
        try:
            assert parse_parent_pid() == 9876
        finally:
            os.environ.pop(PARENT_PID_ENV, None)


class TestParentIdentity:
    def test_identity_is_create_time_off_linux(self):
        proc = mock.MagicMock()
        proc.create_time.return_value = 7.5
        with mock.patch("lilbee.parent_monitor.psutil.Process", return_value=proc):
            assert _start_identity(123) == 7.5

    def test_is_linux_reads_the_platform(self):
        with mock.patch("lilbee.parent_monitor.sys.platform", "linux"):
            assert parent_monitor._is_linux() is True
        with mock.patch("lilbee.parent_monitor.sys.platform", "darwin"):
            assert parent_monitor._is_linux() is False

    def test_start_time_returns_create_time(self):
        proc = mock.MagicMock()
        proc.create_time.return_value = 100.0
        with mock.patch("lilbee.parent_monitor.psutil.Process", return_value=proc):
            assert _parent_start_time(123) == 100.0

    def test_start_time_none_when_process_gone(self):
        with mock.patch(
            "lilbee.parent_monitor.psutil.Process", side_effect=psutil.NoSuchProcess(123)
        ):
            assert _parent_start_time(123) is None

    def test_start_time_none_when_access_denied(self):
        with mock.patch(
            "lilbee.parent_monitor.psutil.Process", side_effect=psutil.AccessDenied(123)
        ):
            assert _parent_start_time(123) is None

    def test_same_process_true_when_identity_unknown(self):
        assert _same_process(123, None) is True

    def test_same_process_true_when_create_time_matches(self):
        proc = mock.MagicMock()
        proc.create_time.return_value = 100.0
        with mock.patch("lilbee.parent_monitor.psutil.Process", return_value=proc):
            assert _same_process(123, 100.0) is True

    def test_same_process_false_when_pid_recycled(self):
        proc = mock.MagicMock()
        proc.create_time.return_value = 200.0
        with mock.patch("lilbee.parent_monitor.psutil.Process", return_value=proc):
            assert _same_process(123, 100.0) is False

    def test_same_process_false_when_process_vanished(self):
        with mock.patch(
            "lilbee.parent_monitor.psutil.Process", side_effect=psutil.NoSuchProcess(123)
        ):
            assert _same_process(123, 100.0) is False

    def test_same_process_false_when_access_denied(self):
        with mock.patch(
            "lilbee.parent_monitor.psutil.Process", side_effect=psutil.AccessDenied(123)
        ):
            assert _same_process(123, 100.0) is False


_OUR_START = 100.0


def _write_stat(root, pid: int, start_ticks: int, comm: str = "lilbee") -> None:
    """A /proc/<pid>/stat with *start_ticks* in field 22."""
    (root / str(pid)).mkdir(exist_ok=True)
    after_comm = ["S", "1"] + ["0"] * 17 + [str(start_ticks), "0", "0"]
    (root / str(pid) / "stat").write_text(f"{pid} ({comm}) " + " ".join(after_comm) + "\n")


class TestLinuxStartTicks:
    """On Linux the identity is the boot-relative start ticks, which a clock step cannot move."""

    @pytest.fixture(autouse=True)
    def linux(self, tmp_path):
        with (
            mock.patch("lilbee.parent_monitor._is_linux", return_value=True),
            mock.patch.object(parent_monitor, "_PROC_ROOT", tmp_path),
        ):
            yield

    def test_identity_is_field_22_of_proc_stat(self, tmp_path):
        _write_stat(tmp_path, 42, 987654)
        assert _start_identity(42) == 987654.0

    def test_identity_survives_a_comm_with_spaces_and_parens(self, tmp_path):
        _write_stat(tmp_path, 42, 555, comm="a) (b c")
        assert _start_identity(42) == 555.0

    def test_missing_proc_entry_is_no_such_process(self):
        with pytest.raises(psutil.NoSuchProcess):
            _start_identity(42)

    def test_unreadable_proc_entry_is_access_denied(self, tmp_path):
        _write_stat(tmp_path, 42, 1)
        with mock.patch("pathlib.Path.read_text", side_effect=PermissionError):
            with pytest.raises(psutil.AccessDenied):
                _start_identity(42)

    def test_vanished_mid_read_is_no_such_process(self, tmp_path):
        _write_stat(tmp_path, 42, 1)
        with mock.patch("pathlib.Path.read_text", side_effect=ProcessLookupError):
            with pytest.raises(psutil.NoSuchProcess):
                _start_identity(42)

    def test_malformed_stat_is_access_denied(self, tmp_path):
        (tmp_path / "42").mkdir()
        (tmp_path / "42" / "stat").write_text("42 (x) S 1\n")
        with pytest.raises(psutil.AccessDenied):
            _start_identity(42)

    def test_a_clock_step_does_not_fire_the_watcher(self, tmp_path):
        """psutil's create_time moves with btime; the start ticks do not."""
        _write_stat(tmp_path, os.getpid(), 100)
        _write_stat(tmp_path, 123, 50)
        btime = iter([1000.0 + 3 * n for n in range(1000)])  # the wall clock steps each read

        def process(pid: int | None = None) -> mock.MagicMock:
            proc = mock.MagicMock()
            proc.create_time.side_effect = lambda: next(btime)
            return proc

        called = threading.Event()
        with (
            mock.patch("lilbee.parent_monitor.psutil.pid_exists", return_value=True),
            mock.patch("lilbee.parent_monitor.psutil.Process", side_effect=process),
        ):
            watch_parent_thread(123, called.set, poll_interval_secs=0.01)
            assert not called.wait(timeout=0.3), "a clock step was taken for a reused pid"

    def test_a_different_start_tick_fires_the_watcher(self, tmp_path):
        _write_stat(tmp_path, os.getpid(), 100)
        _write_stat(tmp_path, 123, 50)
        called = threading.Event()
        with mock.patch("lilbee.parent_monitor.psutil.pid_exists", return_value=True):
            watch_parent_thread(123, called.set, poll_interval_secs=0.05)
            assert not called.wait(timeout=0.2)
            _write_stat(tmp_path, 123, 60)  # the pid now names another process
            assert called.wait(timeout=2.0), "a reused pid was taken for the parent"

    def test_started_after_us_compares_start_ticks(self, tmp_path):
        _write_stat(tmp_path, os.getpid(), 100)
        _write_stat(tmp_path, 123, 200)
        called = threading.Event()
        with mock.patch("lilbee.parent_monitor.psutil.pid_exists", return_value=True):
            watch_parent_thread(123, called.set, poll_interval_secs=60)
            assert called.wait(timeout=2.0)


def _processes(parent_starts: list[float]) -> mock.MagicMock:
    """A psutil.Process stand-in: this process started at _OUR_START, the parent at each poll."""
    starts = iter(parent_starts)

    def process(pid: int | None = None) -> mock.MagicMock:
        proc = mock.MagicMock()
        proc.create_time.return_value = _OUR_START if pid == os.getpid() else next(starts)
        return proc

    return mock.MagicMock(side_effect=process)


class TestWatchParentThread:
    def test_fires_at_once_when_the_pid_holds_a_process_started_after_us(self):
        """The parent died and its pid was reused before the watch began."""
        called = threading.Event()
        with (
            mock.patch("lilbee.parent_monitor.psutil.pid_exists", return_value=True),
            mock.patch("lilbee.parent_monitor.psutil.Process", _processes([200.0, 200.0])),
        ):
            watch_parent_thread(123, called.set, poll_interval_secs=60)
            assert called.wait(timeout=2.0), "a reused pid was taken for the parent"

    def test_keeps_watching_a_grandparent_that_started_before_us(self):
        """The parent pid may name an editor or a console-script wrapper, not the direct parent."""
        called = threading.Event()
        with (
            mock.patch("lilbee.parent_monitor.psutil.pid_exists", return_value=True),
            mock.patch("lilbee.parent_monitor.psutil.Process", _processes([50.0] * 1000)),
        ):
            watch_parent_thread(123, called.set, poll_interval_secs=0.01)
            assert not called.wait(timeout=0.3)

    def test_fires_when_pid_is_recycled_while_watched(self):
        # PID stays alive, but create-time changes: a different process took it.
        called = threading.Event()
        with (
            mock.patch("lilbee.parent_monitor.psutil.pid_exists", return_value=True),
            mock.patch("lilbee.parent_monitor.psutil.Process", _processes([50.0, 50.0, 200.0])),
        ):
            watch_parent_thread(123, called.set, poll_interval_secs=0)
            assert called.wait(timeout=2.0), "on_death callback never fired"

    def test_invokes_callback_when_pid_disappears(self):
        events = [True, True, False]
        lock = threading.Lock()

        def fake_pid_exists(_: int) -> bool:
            with lock:
                return events.pop(0) if events else False

        called = threading.Event()
        with mock.patch("lilbee.parent_monitor.psutil.pid_exists", side_effect=fake_pid_exists):
            thread = watch_parent_thread(321, lambda: called.set(), poll_interval_secs=0.01)
            assert called.wait(timeout=2.0), "on_death callback never fired"
            thread.join(timeout=1.0)
            assert not thread.is_alive()

    def test_thread_is_daemon(self):
        with mock.patch("lilbee.parent_monitor.psutil.pid_exists", return_value=False):
            thread = watch_parent_thread(111, lambda: None, poll_interval_secs=0)
            thread.join(timeout=1.0)
            assert thread.daemon


class TestMcpMainIntegration:
    def test_main_starts_watcher_when_env_set(self, monkeypatch):
        from lilbee import mcp_server as mcp_mod

        monkeypatch.setenv(PARENT_PID_ENV, "888888")
        with (
            mock.patch.object(mcp_mod, "get_services"),
            mock.patch.object(mcp_mod, "build_mcp_server") as build,
            mock.patch("lilbee.parent_monitor.watch_parent_thread") as watcher,
        ):
            build.return_value.run_stdio_async = mock.AsyncMock()
            mcp_mod.main()
        watcher.assert_called_once()
        build.return_value.run_stdio_async.assert_awaited_once()

    def test_main_skips_watcher_when_env_unset(self, monkeypatch):
        from lilbee import mcp_server as mcp_mod

        monkeypatch.delenv(PARENT_PID_ENV, raising=False)
        with (
            mock.patch.object(mcp_mod, "get_services"),
            mock.patch.object(mcp_mod, "build_mcp_server") as build,
            mock.patch("lilbee.parent_monitor.watch_parent_thread") as watcher,
        ):
            build.return_value.run_stdio_async = mock.AsyncMock()
            mcp_mod.main()
        watcher.assert_not_called()
        build.return_value.run_stdio_async.assert_awaited_once()

    def test_main_logs_when_pre_warm_fails(self, monkeypatch, caplog):
        from lilbee import mcp_server as mcp_mod

        monkeypatch.delenv(PARENT_PID_ENV, raising=False)
        with (
            mock.patch.object(mcp_mod, "get_services", side_effect=RuntimeError("no provider")),
            mock.patch.object(mcp_mod, "build_mcp_server") as build,
            caplog.at_level("DEBUG", logger="lilbee.mcp_server"),
        ):
            build.return_value.run_stdio_async = mock.AsyncMock()
            mcp_mod.main()
        assert "MCP pre-warm failed" in caplog.text
        build.return_value.run_stdio_async.assert_awaited_once()


def test_mcp_parent_death_releases_services_before_hard_exit(monkeypatch):
    """os._exit skips atexit, so the handler must release engine membership itself."""
    import lilbee.mcp_server as mcp_mod

    order: list[str] = []
    monkeypatch.setattr("lilbee.mcp_server.reset_services", lambda: order.append("reset"))
    monkeypatch.setattr(mcp_mod.os, "_exit", lambda code: order.append(f"exit:{code}"))
    mcp_mod._exit_on_parent_death()
    assert order == ["reset", "exit:0"]


def test_mcp_parent_death_time_boxes_a_wedged_cleanup(monkeypatch):
    """A peer holding an engine build lock must not keep the orphan alive.

    The release runs on a daemon thread joined with a deadline; if it wedges, the
    watchdog still exits promptly (its one contract), leaving the engine to the
    peers' reap and idle TTL.
    """
    import time

    import lilbee.mcp_server as mcp_mod

    monkeypatch.setattr(mcp_mod, "_PARENT_DEATH_CLEANUP_S", 0.2)
    started = threading.Event()

    def _wedged() -> None:
        started.set()
        time.sleep(30)  # a peer holds the build lock; cleanup cannot finish

    monkeypatch.setattr("lilbee.mcp_server.reset_services", _wedged)
    exited: list[int] = []
    monkeypatch.setattr(mcp_mod.os, "_exit", lambda code: exited.append(code))

    start = time.monotonic()
    mcp_mod._exit_on_parent_death()

    assert started.is_set()  # cleanup was attempted
    assert exited == [0]  # but the watchdog exited anyway
    assert time.monotonic() - start < 5  # promptly, not after the 30s wedge
