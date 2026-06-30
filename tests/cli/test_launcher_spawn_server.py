"""Tests for ``lilbee.cli.launchers.server.spawn_server`` stdout/stderr wiring.

``subprocess.Popen`` is patched so no real ``lilbee serve`` process is spawned;
the tests assert how the child's stdout/stderr are routed (DEVNULL in quiet
mode, a size-capped log file otherwise).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest import mock

import httpx
import pytest

from lilbee.cli.launchers import server as server_mod
from lilbee.core.config import cfg


@pytest.fixture()
def _capture_popen():
    """Patch Popen to record the stdout/stderr kwargs without spawning."""
    captured: dict = {}

    def fake_popen(cmd, *, stdout, stderr):
        captured["cmd"] = cmd
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        return mock.MagicMock()

    with mock.patch.object(server_mod.subprocess, "Popen", side_effect=fake_popen):
        yield captured


def test_quiet_mode_routes_output_to_devnull(monkeypatch, _capture_popen) -> None:
    monkeypatch.setenv("LILBEE_LAUNCHER_SERVE_QUIET", "1")
    server_mod.spawn_server(8080)
    assert _capture_popen["stdout"] == subprocess.DEVNULL
    assert _capture_popen["stderr"] == subprocess.DEVNULL


def test_non_quiet_writes_to_log_file(monkeypatch, tmp_path: Path, _capture_popen) -> None:
    monkeypatch.delenv("LILBEE_LAUNCHER_SERVE_QUIET", raising=False)
    monkeypatch.setattr(cfg, "data_dir", tmp_path)
    server_mod.spawn_server(8080)
    log_path = tmp_path / "logs" / "launcher-serve.log"
    assert log_path.exists()  # the log dir + file were created
    # stdout is an open binary file handle, stderr is folded into it.
    assert _capture_popen["stdout"].name == str(log_path)
    assert _capture_popen["stderr"] == subprocess.STDOUT
    _capture_popen["stdout"].close()


def test_oversized_log_is_truncated_before_reopen(
    monkeypatch, tmp_path: Path, _capture_popen
) -> None:
    """A pre-existing log past the 5 MB cap is unlinked, so the session starts fresh."""
    monkeypatch.delenv("LILBEE_LAUNCHER_SERVE_QUIET", raising=False)
    monkeypatch.setattr(cfg, "data_dir", tmp_path)
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "launcher-serve.log"
    log_path.write_bytes(b"x" * (5 * 1024 * 1024 + 1))  # one byte over the cap

    server_mod.spawn_server(8080)

    # The oversized file was unlinked then reopened empty (append mode, nothing
    # written by the patched Popen), so the stale firehose did not survive.
    assert log_path.stat().st_size == 0
    _capture_popen["stdout"].close()


def test_win32_cmd_wrapper_falls_back_to_sys_executable(
    monkeypatch, tmp_path: Path, _capture_popen
) -> None:
    """When shutil.which returns a .cmd path on win32, spawn_server falls back to
    sys.executable -m lilbee (finding #5: PermissionError on .cmd with shell=False)."""
    monkeypatch.delenv("LILBEE_LAUNCHER_SERVE_QUIET", raising=False)
    monkeypatch.setattr(cfg, "data_dir", tmp_path)
    monkeypatch.setattr(server_mod.shutil, "which", lambda _name: r"C:\Python\Scripts\lilbee.cmd")
    monkeypatch.setattr(server_mod.sys, "platform", "win32")
    server_mod.spawn_server(8080)
    cmd = _capture_popen["cmd"]
    assert cmd[0] == server_mod.sys.executable
    assert cmd[1:3] == ["-m", "lilbee"]
    _capture_popen["stdout"].close()


def test_non_cmd_bin_on_win32_is_used_directly(
    monkeypatch, tmp_path: Path, _capture_popen
) -> None:
    """An .exe on win32 (not .cmd) is passed directly without the fallback."""
    monkeypatch.delenv("LILBEE_LAUNCHER_SERVE_QUIET", raising=False)
    monkeypatch.setattr(cfg, "data_dir", tmp_path)
    monkeypatch.setattr(
        server_mod.shutil, "which", lambda _name: r"C:\Python\Scripts\lilbee.exe"
    )
    monkeypatch.setattr(server_mod.sys, "platform", "win32")
    server_mod.spawn_server(8080)
    cmd = _capture_popen["cmd"]
    assert cmd[0] == r"C:\Python\Scripts\lilbee.exe"
    _capture_popen["stdout"].close()


def test_log_unlink_permission_error_falls_through_to_append(
    monkeypatch, tmp_path: Path, _capture_popen
) -> None:
    """When unlink raises PermissionError (Windows held-open file), spawn_server
    falls through to append mode rather than crashing (finding #6)."""
    monkeypatch.delenv("LILBEE_LAUNCHER_SERVE_QUIET", raising=False)
    monkeypatch.setattr(cfg, "data_dir", tmp_path)
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "launcher-serve.log"
    log_path.write_bytes(b"x" * (5 * 1024 * 1024 + 1))  # over cap so unlink is attempted

    original_unlink = Path.unlink

    def _fail_unlink(self, *, missing_ok: bool = False) -> None:
        if self == log_path:
            raise PermissionError("file in use")
        original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", _fail_unlink)
    # spawn_server must not raise; it falls through to append mode.
    server_mod.spawn_server(8080)
    # The file still exists (not truncated because unlink was denied).
    assert log_path.exists()
    _capture_popen["stdout"].close()


class TestHealthProbes:
    """chat_ready / served_chat_ctx / wait_for_chat_warm read /api/health."""

    def test_chat_ready_false_on_non_ok_status(self, monkeypatch) -> None:
        monkeypatch.setattr(
            server_mod.httpx, "get", lambda *_a, **_k: httpx.Response(503, text="loading")
        )
        assert server_mod.chat_ready(8080) is False

    def test_health_ok_true_on_non_json_200_body(self, monkeypatch) -> None:
        """A 200 with a non-JSON body still counts as healthy (empty parsed body)."""
        monkeypatch.setattr(
            server_mod.httpx, "get", lambda *_a, **_k: httpx.Response(200, text="OK")
        )
        assert server_mod.health_ok(8080) is True
        assert server_mod.chat_ready(8080) is False  # no chat_ready field in body

    def test_served_chat_ctx_none_on_transport_error(self, monkeypatch) -> None:
        def _boom(*_a, **_k):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(server_mod.httpx, "get", _boom)
        assert server_mod.served_chat_ctx(8080) is None

    def test_served_chat_ctx_none_on_non_ok_status(self, monkeypatch) -> None:
        monkeypatch.setattr(
            server_mod.httpx, "get", lambda *_a, **_k: httpx.Response(500, text="err")
        )
        assert server_mod.served_chat_ctx(8080) is None

    def test_served_chat_ctx_reads_chat_ctx(self, monkeypatch) -> None:
        monkeypatch.setattr(
            server_mod.httpx, "get", lambda *_a, **_k: httpx.Response(200, json={"chat_ctx": 4096})
        )
        assert server_mod.served_chat_ctx(8080) == 4096

    def test_wait_for_chat_warm_returns_true_immediately_when_ready(self, monkeypatch) -> None:
        monkeypatch.setattr(server_mod, "chat_ready", lambda _p: True)
        assert server_mod.wait_for_chat_warm(8080, timeout_s=0.1) is True

    def test_wait_for_chat_warm_polls_until_ready(self, monkeypatch) -> None:
        # Not ready on entry, then ready on the next poll: exercises the loop body
        # (sleep + re-probe) rather than the immediate-return fast path.
        calls = iter([False, False, True])
        monkeypatch.setattr(server_mod, "chat_ready", lambda _p: next(calls))
        monkeypatch.setattr(server_mod.time, "sleep", lambda _s: None)
        assert server_mod.wait_for_chat_warm(8080, timeout_s=5.0) is True

    def test_wait_for_chat_warm_times_out(self, monkeypatch) -> None:
        monkeypatch.setattr(server_mod, "chat_ready", lambda _p: False)
        monkeypatch.setattr(server_mod.time, "sleep", lambda _s: None)
        assert server_mod.wait_for_chat_warm(8080, timeout_s=0.0) is False


class TestEnsureServerRunningRetries:
    """A port stolen between free_port() and bind gets a fresh port on retry."""

    @staticmethod
    def _proc() -> mock.MagicMock:
        proc = mock.MagicMock()
        proc.poll.return_value = None
        return proc

    def test_retries_with_a_fresh_port_then_succeeds(self, monkeypatch) -> None:
        first, second = self._proc(), self._proc()
        monkeypatch.setattr(
            server_mod, "running_server_session", mock.MagicMock(side_effect=[None, ("tok", 2222)])
        )
        monkeypatch.setattr(server_mod, "free_port", mock.MagicMock(side_effect=[1111, 2222]))
        spawn = mock.MagicMock(side_effect=[first, second])
        monkeypatch.setattr(server_mod, "spawn_server", spawn)
        monkeypatch.setattr(
            server_mod, "wait_for_health", mock.MagicMock(side_effect=[False, True])
        )

        session, spawned = server_mod.ensure_server_running()

        assert session == ("tok", 2222)
        assert spawned is second
        assert spawn.call_args_list == [mock.call(1111), mock.call(2222)]
        first.terminate.assert_called_once()
        second.terminate.assert_not_called()

    def test_gives_up_after_bounded_attempts(self, monkeypatch) -> None:
        import typer

        procs = [self._proc() for _ in range(server_mod._SPAWN_ATTEMPTS)]
        monkeypatch.setattr(server_mod, "running_server_session", lambda: None)
        monkeypatch.setattr(server_mod, "free_port", mock.MagicMock(side_effect=[1, 2, 3]))
        spawn = mock.MagicMock(side_effect=procs)
        monkeypatch.setattr(server_mod, "spawn_server", spawn)
        monkeypatch.setattr(server_mod, "wait_for_health", lambda _p: False)

        with pytest.raises(typer.Exit) as excinfo:
            server_mod.ensure_server_running()

        assert excinfo.value.exit_code == 1
        assert spawn.call_count == server_mod._SPAWN_ATTEMPTS
        for proc in procs:
            proc.terminate.assert_called_once()

    def test_honors_configured_server_port(self, monkeypatch) -> None:
        import typer

        monkeypatch.setattr(cfg, "server_port", 8080)
        monkeypatch.setattr(server_mod, "running_server_session", lambda: None)
        monkeypatch.setattr(server_mod, "free_port", mock.MagicMock(side_effect=[1, 2, 3]))
        seen: list[int] = []
        monkeypatch.setattr(server_mod, "_spawn_and_wait", lambda port: seen.append(port) or None)
        with pytest.raises(typer.Exit):
            server_mod.ensure_server_running()
        # The pinned port is used every attempt so a persisted URL stays valid.
        assert seen == [8080] * server_mod._SPAWN_ATTEMPTS

    def test_falls_back_to_free_port_when_server_port_unset(self, monkeypatch) -> None:
        import typer

        monkeypatch.setattr(cfg, "server_port", 0)
        monkeypatch.setattr(server_mod, "running_server_session", lambda: None)
        monkeypatch.setattr(server_mod, "free_port", mock.MagicMock(side_effect=[7, 8, 9]))
        seen: list[int] = []
        monkeypatch.setattr(server_mod, "_spawn_and_wait", lambda port: seen.append(port) or None)
        with pytest.raises(typer.Exit):
            server_mod.ensure_server_running()
        assert seen == [7, 8, 9]
