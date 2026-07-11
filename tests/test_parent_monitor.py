"""Tests for ``lilbee.parent_monitor`` parent-death detection."""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Iterator
from unittest import mock

import psutil
import pytest

from lilbee.parent_monitor import (
    PARENT_PID_ENV,
    _parent_start_time,
    _same_process,
    parse_parent_pid,
    watch_parent_async,
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


class TestWatchParentAsync:
    async def test_invokes_callback_when_pid_disappears(self):
        events = [True, True, False]

        def fake_pid_exists(_: int) -> bool:
            return events.pop(0)

        called: list[bool] = []
        with mock.patch("lilbee.parent_monitor.psutil.pid_exists", side_effect=fake_pid_exists):
            await watch_parent_async(123, lambda: called.append(True), poll_interval_secs=0)
        assert called == [True]

    async def test_returns_immediately_when_pid_already_dead(self):
        called: list[bool] = []
        with mock.patch("lilbee.parent_monitor.psutil.pid_exists", return_value=False):
            await watch_parent_async(456, lambda: called.append(True), poll_interval_secs=0)
        assert called == [True]

    async def test_can_be_cancelled_while_polling(self):
        with mock.patch("lilbee.parent_monitor.psutil.pid_exists", return_value=True):
            task = asyncio.create_task(
                watch_parent_async(789, lambda: None, poll_interval_secs=0.05)
            )
            await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    async def test_fires_when_pid_is_recycled(self):
        # PID stays alive, but create-time changes: a different process took it.
        proc = mock.MagicMock()
        proc.create_time.side_effect = [100.0, 100.0, 200.0]
        called: list[bool] = []
        with (
            mock.patch("lilbee.parent_monitor.psutil.pid_exists", return_value=True),
            mock.patch("lilbee.parent_monitor.psutil.Process", return_value=proc),
        ):
            await watch_parent_async(123, lambda: called.append(True), poll_interval_secs=0)
        assert called == [True]


class TestWatchParentThread:
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


class TestRunServerIntegration:
    """Verify _run_server schedules a parent-death watcher when env var set."""

    @pytest.fixture()
    def fake_server(self):
        srv = mock.MagicMock()
        srv.servers = []
        srv.should_exit = False

        async def main_loop() -> None:
            return None

        async def startup() -> None:
            return None

        async def shutdown() -> None:
            return None

        srv.main_loop = main_loop
        srv.startup = startup
        srv.shutdown = shutdown
        return srv

    @pytest.fixture()
    def fake_config(self):
        cfg = mock.MagicMock()
        cfg.loaded = True
        return cfg

    async def test_no_watcher_task_when_env_unset(self, fake_server, fake_config, monkeypatch):
        from lilbee.cli.commands import servers as commands

        monkeypatch.delenv(PARENT_PID_ENV, raising=False)
        with (
            mock.patch.object(
                commands, "port_file", return_value=mock.MagicMock(exists=lambda: False)
            ),
            mock.patch("lilbee.parent_monitor.watch_parent_async") as watcher,
        ):
            await commands._run_server(fake_server, fake_config, "127.0.0.1")
        watcher.assert_not_called()

    async def test_schedules_watcher_when_env_set(
        self, fake_server, fake_config, monkeypatch, tmp_path
    ):
        from lilbee.cli.commands import servers as commands

        monkeypatch.setenv(PARENT_PID_ENV, "999999")
        port_file = tmp_path / "server.port"
        watcher = mock.AsyncMock()

        with (
            mock.patch.object(commands, "port_file", return_value=port_file),
            mock.patch("lilbee.parent_monitor.watch_parent_async", new=watcher),
        ):
            await commands._run_server(fake_server, fake_config, "127.0.0.1")

        watcher.assert_called_once()
        assert watcher.call_args.args[0] == 999999

        # Invoke the on_death callback to verify it flips should_exit.
        on_death = watcher.call_args.args[1]
        on_death()
        assert fake_server.should_exit is True

    async def test_loads_config_when_not_loaded(
        self, fake_server, fake_config, monkeypatch, tmp_path
    ):
        from lilbee.cli.commands import servers as commands

        fake_config.loaded = False
        monkeypatch.delenv(PARENT_PID_ENV, raising=False)
        port_file = tmp_path / "server.port"
        with mock.patch.object(commands, "port_file", return_value=port_file):
            await commands._run_server(fake_server, fake_config, "127.0.0.1")
        fake_config.load.assert_called_once()


class TestMcpMainIntegration:
    def test_main_starts_watcher_when_env_set(self, monkeypatch):
        from lilbee import mcp_server as mcp_mod

        monkeypatch.setenv(PARENT_PID_ENV, "888888")
        with (
            mock.patch.object(mcp_mod, "get_services"),
            mock.patch.object(mcp_mod.mcp, "run"),
            mock.patch("lilbee.parent_monitor.watch_parent_thread") as watcher,
        ):
            mcp_mod.main()
        watcher.assert_called_once()

    def test_main_skips_watcher_when_env_unset(self, monkeypatch):
        from lilbee import mcp_server as mcp_mod

        monkeypatch.delenv(PARENT_PID_ENV, raising=False)
        with (
            mock.patch.object(mcp_mod, "get_services"),
            mock.patch.object(mcp_mod.mcp, "run"),
            mock.patch("lilbee.parent_monitor.watch_parent_thread") as watcher,
        ):
            mcp_mod.main()
        watcher.assert_not_called()

    def test_main_logs_when_pre_warm_fails(self, monkeypatch, caplog):
        from lilbee import mcp_server as mcp_mod

        monkeypatch.delenv(PARENT_PID_ENV, raising=False)
        with (
            mock.patch.object(mcp_mod, "get_services", side_effect=RuntimeError("no provider")),
            mock.patch.object(mcp_mod.mcp, "run"),
            caplog.at_level("DEBUG", logger="lilbee.mcp_server"),
        ):
            mcp_mod.main()
        assert "MCP pre-warm failed" in caplog.text
