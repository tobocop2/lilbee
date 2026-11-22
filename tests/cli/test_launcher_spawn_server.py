"""Tests for ``lilbee.cli.launchers.server.spawn_server`` stdout/stderr wiring.

``spawn_bound_child`` is patched so no real ``lilbee serve`` process is spawned;
the tests assert how the child's stdout/stderr are routed (DEVNULL in quiet
mode, a size-capped log file otherwise).
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

import httpx
import pytest

from lilbee.cli.launchers import server as server_mod
from lilbee.core.config import cfg
from tests._mock_effects import repeat_last
from tests.conftest import clean_env

# A stand-in for `lilbee serve`: it answers the health probe and writes the session
# files the launcher reads, and records every stop signal it gets.
_FAKE_SERVE = """\
import http.server
import json
import os
import signal
import sys

record = os.environ["FAKE_SERVE_RECORD"]
data = os.path.join(os.environ["LILBEE_DATA"], "data")


def note(signum, _frame):
    with open(record, "a", encoding="utf-8") as handle:
        handle.write(signal.Signals(signum).name + "\\n")
    if signum == signal.SIGTERM:
        sys.exit(0)


class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"chat_ready": False}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


signal.signal(signal.SIGINT, note)
signal.signal(signal.SIGTERM, note)
with open(os.environ["FAKE_SERVE_PID"], "w", encoding="utf-8") as handle:
    handle.write(str(os.getpid()))
port = int(sys.argv[sys.argv.index("--port") + 1])
httpd = http.server.HTTPServer(("127.0.0.1", port), Health)
os.makedirs(data, exist_ok=True)
with open(os.path.join(data, "server.json"), "w", encoding="utf-8") as handle:
    json.dump({"token": "fake-token"}, handle)
with open(os.path.join(data, "server.port"), "w", encoding="utf-8") as handle:
    handle.write(str(port))
httpd.serve_forever()
"""

# A stand-in for the agent client: it says when it runs, then waits for the Ctrl-C.
_FAKE_CLIENT = """\
import os
import time

with open(os.environ["FAKE_CLIENT_PID"], "w", encoding="utf-8") as handle:
    handle.write(str(os.getpid()))
print("client ready", flush=True)
time.sleep(60)
"""

# The launcher: the real run_launcher, over the fake serve and the fake client.
_LAUNCHER = """\
import os

from lilbee.cli.launchers.launcher import run_launcher


class FakeClientLauncher:
    name = "fake-client"
    install_hint = "unused"

    def find_binary(self):
        return os.environ["FAKE_CLIENT"]

    def prepare(self, *, token, port, model_refs):
        return [], dict(os.environ)


try:
    run_launcher(FakeClientLauncher())
except KeyboardInterrupt:
    print("interrupted", flush=True)
"""


def _write_script(path: Path, source: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{source}", encoding="utf-8")
    path.chmod(0o755)
    return path


def _start_launcher(tmp_path: Path) -> subprocess.Popen[str]:
    """Start a launcher, in a process group of its own as a terminal job, up to the client."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_script(bin_dir / "lilbee", _FAKE_SERVE)
    client = _write_script(tmp_path / "fake-client", _FAKE_CLIENT)
    env = {
        **clean_env(tmp_path),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "LILBEE_LAUNCHER_SERVE_QUIET": "1",
        "FAKE_CLIENT": str(client),
        "FAKE_CLIENT_PID": str(tmp_path / "client.pid"),
        "FAKE_SERVE_RECORD": str(tmp_path / "signals.txt"),
        "FAKE_SERVE_PID": str(tmp_path / "serve.pid"),
    }
    launcher = subprocess.Popen(
        [sys.executable, "-c", _LAUNCHER],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        start_new_session=True,
    )
    assert launcher.stdout is not None
    try:
        for line in launcher.stdout:
            if line.strip() == "client ready":
                return launcher
    except BaseException:
        # A timeout here leaves the launcher and its fakes running; they are ours to end.
        os.killpg(launcher.pid, signal.SIGKILL)
        _end_fakes(tmp_path)
        raise
    stdout, stderr = launcher.communicate()
    _end_fakes(tmp_path)
    raise AssertionError(f"the launcher ended before the client ran: {stdout}{stderr}")


def _signals_seen(tmp_path: Path) -> list[str]:
    record = tmp_path / "signals.txt"
    return record.read_text(encoding="utf-8").split() if record.exists() else []


def _end_fakes(tmp_path: Path) -> None:
    """Stop the fake serve and client the test left running; they are this test's own."""
    for name in ("serve.pid", "client.pid"):
        pid_file = tmp_path / name
        if pid_file.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(pid_file.read_text(encoding="utf-8")), signal.SIGKILL)


@pytest.fixture()
def _capture_popen():
    """Patch the bound spawn to record the stdout/stderr kwargs without spawning."""
    captured: dict = {}

    def fake_popen(cmd, *, stdout, stderr, env=None, **popen_kwargs):
        captured["cmd"] = cmd
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        captured["env"] = env
        captured["popen_kwargs"] = popen_kwargs
        return mock.MagicMock()

    with mock.patch.object(server_mod, "spawn_bound_child", side_effect=fake_popen):
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


def test_non_cmd_bin_on_win32_is_used_directly(monkeypatch, tmp_path: Path, _capture_popen) -> None:
    """An .exe on win32 (not .cmd) is passed directly without the fallback."""
    monkeypatch.delenv("LILBEE_LAUNCHER_SERVE_QUIET", raising=False)
    monkeypatch.setattr(cfg, "data_dir", tmp_path)
    monkeypatch.setattr(server_mod.shutil, "which", lambda _name: r"C:\Python\Scripts\lilbee.exe")
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


def test_the_server_command_runs_the_lilbee_binary(monkeypatch, _capture_popen) -> None:
    monkeypatch.setenv("LILBEE_LAUNCHER_SERVE_QUIET", "1")
    monkeypatch.setattr(
        server_mod.shutil, "which", lambda name: "/opt/bin/lilbee" if name == "lilbee" else None
    )

    server_mod.spawn_server(8080)

    assert _capture_popen["cmd"] == ["/opt/bin/lilbee", "serve", "--port", "8080"]


def test_the_server_command_falls_back_to_the_module_form(monkeypatch, _capture_popen) -> None:
    monkeypatch.setenv("LILBEE_LAUNCHER_SERVE_QUIET", "1")
    monkeypatch.setattr(server_mod.shutil, "which", lambda _name: None)

    server_mod.spawn_server(8080)

    assert _capture_popen["cmd"] == [sys.executable, "-m", "lilbee", "serve", "--port", "8080"]


@pytest.mark.parametrize("overrides", [None, {"A": "1"}], ids=["no-overrides", "overrides"])
def test_serve_gets_a_session_of_its_own_on_posix_with_or_without_env_overrides(
    monkeypatch, _capture_popen, overrides
) -> None:
    monkeypatch.setenv("LILBEE_LAUNCHER_SERVE_QUIET", "1")

    server_mod.spawn_server(8080, env_overrides=overrides)

    assert _capture_popen["popen_kwargs"] == {"start_new_session": os.name == "posix"}


def test_the_log_directory_is_created_with_its_parents(
    monkeypatch, tmp_path: Path, _capture_popen
) -> None:
    monkeypatch.delenv("LILBEE_LAUNCHER_SERVE_QUIET", raising=False)
    monkeypatch.setattr(cfg, "data_dir", tmp_path / "a" / "b")

    server_mod.spawn_server(8080)

    assert (tmp_path / "a" / "b" / "logs" / "launcher-serve.log").exists()
    _capture_popen["stdout"].close()


def test_a_log_exactly_at_the_cap_is_kept(monkeypatch, tmp_path: Path, _capture_popen) -> None:
    monkeypatch.delenv("LILBEE_LAUNCHER_SERVE_QUIET", raising=False)
    monkeypatch.setattr(cfg, "data_dir", tmp_path)
    log_path = tmp_path / "logs" / "launcher-serve.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_bytes(b"x" * (5 * 1024 * 1024))

    server_mod.spawn_server(8080)

    assert log_path.stat().st_size == 5 * 1024 * 1024
    _capture_popen["stdout"].close()


class TestHealthProbes:
    """chat_ready / served_chat_ctx / wait_for_chat_warm read /api/health."""

    @pytest.fixture(autouse=True)
    def _token(self, monkeypatch):
        """/api/health needs the token like every other route; these probes read
        it from server.json, which does not exist under test."""
        monkeypatch.setattr(server_mod, "_session_token", lambda: "t")

    def test_no_probe_is_attempted_before_the_token_exists(self, monkeypatch) -> None:
        """A probe racing startup finds no server.json, which means no server
        yet; firing a request that can only 401 is pointless."""
        monkeypatch.setattr(server_mod, "_session_token", lambda: None)

        def _fail(*_a, **_k):
            raise AssertionError("probed without a token")

        monkeypatch.setattr(server_mod.httpx, "get", _fail)
        assert server_mod.health_ok(8080) is False
        assert server_mod.chat_ready(8080) is False
        assert server_mod.served_chat_ctx(8080) is None

    def test_the_probe_sends_the_bearer_token(self, monkeypatch) -> None:
        seen: dict[str, object] = {}

        def _capture(url, **kwargs):
            seen.update(kwargs)
            return httpx.Response(200, json={"status": "ok"})

        monkeypatch.setattr(server_mod.httpx, "get", _capture)
        assert server_mod.health_ok(8080) is True
        assert seen["headers"] == {"Authorization": "Bearer t"}

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
            server_mod,
            "running_server_session",
            mock.MagicMock(side_effect=repeat_last(None, ("tok", 2222))),
        )
        monkeypatch.setattr(
            server_mod, "free_port", mock.MagicMock(side_effect=repeat_last(1111, 2222))
        )
        spawn = mock.MagicMock(side_effect=repeat_last(first, second))
        monkeypatch.setattr(server_mod, "spawn_server", spawn)
        monkeypatch.setattr(
            server_mod, "wait_for_health", mock.MagicMock(side_effect=repeat_last(False, True))
        )

        session, spawned = server_mod.ensure_server_running()

        assert session == ("tok", 2222)
        assert spawned is second
        assert spawn.call_args_list == [
            mock.call(1111, env_overrides=None),
            mock.call(2222, env_overrides=None),
        ]
        first.terminate.assert_called_once()
        second.terminate.assert_not_called()

    def test_gives_up_after_bounded_attempts(self, monkeypatch) -> None:
        import typer

        procs = [self._proc() for _ in range(server_mod._SPAWN_ATTEMPTS)]
        monkeypatch.setattr(server_mod, "running_server_session", lambda: None)
        monkeypatch.setattr(
            server_mod, "free_port", mock.MagicMock(side_effect=repeat_last(1, 2, 3))
        )
        spawn = mock.MagicMock(side_effect=procs)
        monkeypatch.setattr(server_mod, "spawn_server", spawn)
        monkeypatch.setattr(server_mod, "wait_for_health", lambda _p, **_kw: False)

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
        monkeypatch.setattr(
            server_mod, "free_port", mock.MagicMock(side_effect=repeat_last(1, 2, 3))
        )
        seen: list[int] = []
        monkeypatch.setattr(
            server_mod,
            "_spawn_and_wait",
            lambda port, **_kw: seen.append(port) or server_mod._FailedStart(None, ""),
        )
        with pytest.raises(typer.Exit):
            server_mod.ensure_server_running()
        # The pinned port is used every attempt so a persisted URL stays valid.
        assert seen == [8080] * server_mod._SPAWN_ATTEMPTS

    def test_falls_back_to_free_port_when_server_port_unset(self, monkeypatch) -> None:
        import typer

        monkeypatch.setattr(cfg, "server_port", 0)
        monkeypatch.setattr(server_mod, "running_server_session", lambda: None)
        monkeypatch.setattr(
            server_mod, "free_port", mock.MagicMock(side_effect=repeat_last(7, 8, 9))
        )
        seen: list[int] = []
        monkeypatch.setattr(
            server_mod,
            "_spawn_and_wait",
            lambda port, **_kw: seen.append(port) or server_mod._FailedStart(None, ""),
        )
        with pytest.raises(typer.Exit):
            server_mod.ensure_server_running()
        assert seen == [7, 8, 9]


class TestServerExitAtStartup:
    """A server that exits while the launcher waits ends the wait and shows its log."""

    @staticmethod
    def _proc(poll_result: int | None) -> mock.MagicMock:
        proc = mock.MagicMock()
        proc.poll.return_value = poll_result
        return proc

    @staticmethod
    def _forbid(*_args, **_kwargs):
        raise AssertionError("the launcher kept waiting on an exited server")

    def test_wait_for_health_returns_false_at_once_when_the_server_exited(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(server_mod, "health_ok", self._forbid)
        monkeypatch.setattr(server_mod.time, "sleep", self._forbid)

        assert server_mod.wait_for_health(8080, 60.0, server=self._proc(1)) is False

    def test_wait_for_health_still_polls_a_live_server(self, monkeypatch) -> None:
        health = mock.MagicMock(side_effect=[False, True])
        sleep = mock.MagicMock()
        monkeypatch.setattr(server_mod, "health_ok", health)
        monkeypatch.setattr(server_mod.time, "sleep", sleep)

        assert server_mod.wait_for_health(8080, 60.0, server=self._proc(None)) is True
        assert health.call_args_list == [mock.call(8080), mock.call(8080)]
        sleep.assert_called_once_with(server_mod._SERVER_POLL_INTERVAL_S)

    def test_wait_for_health_with_no_time_left_never_probes(self, monkeypatch) -> None:
        health = mock.MagicMock(return_value=True)
        monkeypatch.setattr(server_mod, "health_ok", health)
        monkeypatch.setattr(server_mod.time, "monotonic", lambda: 100.0)

        assert server_mod.wait_for_health(8080, 0.0) is False
        health.assert_not_called()

    def test_wait_for_health_stops_when_the_server_exits_mid_wait(self, monkeypatch) -> None:
        proc = mock.MagicMock()
        proc.poll.side_effect = [None, 3]
        monkeypatch.setattr(server_mod, "health_ok", lambda _p: False)
        monkeypatch.setattr(server_mod.time, "sleep", lambda _s: None)

        assert server_mod.wait_for_health(8080, 5.0, server=proc) is False
        assert proc.poll.call_count == 2

    def _start_failing_launch(
        self, monkeypatch, tmp_path: Path, *, writes: bool, status: int | None
    ) -> Path:
        """Set up a launch whose every spawn exits; each spawn appends one line when *writes*."""
        log_path = tmp_path / "logs" / "launcher-serve.log"
        log_path.parent.mkdir(parents=True)
        log_path.write_bytes(b"older session: stale error\n")
        monkeypatch.setattr(cfg, "data_dir", tmp_path)
        monkeypatch.setattr(cfg, "server_port", 0)
        monkeypatch.setattr(server_mod, "running_server_session", lambda: None)
        monkeypatch.setattr(server_mod, "free_port", lambda: 4321)
        spawns: list[int] = []

        def fake_spawn(_port, **_kw):
            spawns.append(1)
            if writes:
                with log_path.open("ab") as handle:
                    handle.write(f"attempt {len(spawns)}: build error\n".encode())
            return self._proc(status)

        monkeypatch.setattr(server_mod, "spawn_server", fake_spawn)
        monkeypatch.setattr(server_mod, "health_ok", self._forbid)
        monkeypatch.setattr(server_mod.time, "sleep", self._forbid)
        return log_path

    def test_launcher_shows_only_what_the_last_spawn_wrote(
        self, monkeypatch, tmp_path: Path, capsys
    ) -> None:
        import typer

        monkeypatch.delenv("LILBEE_LAUNCHER_SERVE_QUIET", raising=False)
        log_path = self._start_failing_launch(monkeypatch, tmp_path, writes=True, status=2)

        with pytest.raises(typer.Exit):
            server_mod.ensure_server_running()

        err = capsys.readouterr().err
        assert "failed to start on port 4321" in err
        assert f"Last lines of {log_path}" in err
        assert f"attempt {server_mod._SPAWN_ATTEMPTS}: build error" in err
        assert "attempt 1: build error" not in err
        assert "stale error" not in err

    def test_launcher_says_so_when_the_server_exits_without_output(
        self, monkeypatch, tmp_path: Path, capsys
    ) -> None:
        import typer

        monkeypatch.delenv("LILBEE_LAUNCHER_SERVE_QUIET", raising=False)
        log_path = self._start_failing_launch(monkeypatch, tmp_path, writes=False, status=7)

        with pytest.raises(typer.Exit):
            server_mod.ensure_server_running()

        err = capsys.readouterr().err
        assert f"The server exited with status 7; it wrote nothing to {log_path}." in err
        assert "stale error" not in err

    def test_launcher_reports_the_exit_status_alone_in_quiet_mode(
        self, monkeypatch, tmp_path: Path, capsys
    ) -> None:
        import typer

        monkeypatch.setenv("LILBEE_LAUNCHER_SERVE_QUIET", "1")
        self._start_failing_launch(monkeypatch, tmp_path, writes=True, status=3)

        with pytest.raises(typer.Exit):
            server_mod.ensure_server_running()

        err = capsys.readouterr().err
        assert (
            "The server exited with status 3; its output is off (LILBEE_LAUNCHER_SERVE_QUIET)"
            in err
        )
        assert "Last lines" not in err
        assert "build error" not in err

    def test_launcher_adds_nothing_when_the_server_hangs_instead_of_exiting(
        self, monkeypatch, tmp_path: Path, capsys
    ) -> None:
        import typer

        monkeypatch.delenv("LILBEE_LAUNCHER_SERVE_QUIET", raising=False)
        self._start_failing_launch(monkeypatch, tmp_path, writes=False, status=None)
        monkeypatch.setattr(server_mod, "wait_for_health", lambda _p, **_kw: False)

        with pytest.raises(typer.Exit):
            server_mod.ensure_server_running()

        err = capsys.readouterr().err
        assert "failed to start on port 4321" in err
        assert "exited" not in err
        assert "Last lines" not in err

    def test_log_since_reads_only_after_the_start_offset(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.delenv("LILBEE_LAUNCHER_SERVE_QUIET", raising=False)
        monkeypatch.setattr(cfg, "data_dir", tmp_path)
        log_path = tmp_path / "logs" / "launcher-serve.log"
        log_path.parent.mkdir(parents=True)
        log_path.write_bytes(b"old\n")
        start = server_mod._serve_log_size()
        with log_path.open("ab") as handle:
            handle.write(b"new\n")

        assert start == 4
        assert server_mod._serve_log_since(start) == "new"

    def test_log_since_reads_a_log_truncated_after_the_offset_from_its_start(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("LILBEE_LAUNCHER_SERVE_QUIET", raising=False)
        monkeypatch.setattr(cfg, "data_dir", tmp_path)
        log_path = tmp_path / "logs" / "launcher-serve.log"
        log_path.parent.mkdir(parents=True)
        log_path.write_bytes(b"fresh\n")

        assert server_mod._serve_log_since(10_000) == "fresh"

    def test_log_since_bounds_the_read_on_a_large_log(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.delenv("LILBEE_LAUNCHER_SERVE_QUIET", raising=False)
        monkeypatch.setattr(cfg, "data_dir", tmp_path)
        log_path = tmp_path / "logs" / "launcher-serve.log"
        log_path.parent.mkdir(parents=True)
        log_path.write_text("".join(f"line {n:06d}\n" for n in range(200_000)), encoding="utf-8")
        reads: list[int] = []
        real = log_path.open("rb")

        def read_and_record() -> bytes:
            data = real.read()
            reads.append(len(data))
            return data

        spy = mock.MagicMock(wraps=real)
        spy.__enter__.return_value = spy
        spy.__exit__.return_value = False
        spy.read.side_effect = read_and_record
        try:
            with monkeypatch.context() as patch:
                patch.setattr(Path, "open", lambda self, *a, **k: spy)
                tail = server_mod._serve_log_since(0)
        finally:
            real.close()

        assert log_path.stat().st_size > 10 * 16 * 1024
        assert reads == [16 * 1024]
        lines = tail.splitlines()
        assert len(lines) == 20
        assert lines[-1] == "line 199999"

    def test_log_since_is_empty_without_a_log(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.delenv("LILBEE_LAUNCHER_SERVE_QUIET", raising=False)
        monkeypatch.setattr(cfg, "data_dir", tmp_path)

        assert server_mod._serve_log_size() == 0
        assert server_mod._serve_log_since(0) == ""

    def test_spawn_and_wait_hands_the_child_to_the_wait(self, monkeypatch) -> None:
        proc = self._proc(None)
        monkeypatch.setattr(server_mod, "spawn_server", lambda port, **_kw: proc)
        wait = mock.MagicMock(return_value=True)
        monkeypatch.setattr(server_mod, "wait_for_health", wait)

        assert server_mod._spawn_and_wait(4321) is proc
        wait.assert_called_once_with(4321, server=proc)

    def test_spawn_and_wait_returns_the_exit_status_of_a_dead_child(self, monkeypatch) -> None:
        proc = self._proc(5)
        monkeypatch.setattr(server_mod, "spawn_server", lambda port, **_kw: proc)
        monkeypatch.setattr(server_mod, "wait_for_health", lambda _p, **_kw: False)
        monkeypatch.setattr(server_mod, "_serve_log_since", lambda _start: "boom")

        assert server_mod._spawn_and_wait(4321) == server_mod._FailedStart(5, "boom")

    def test_spawn_and_wait_passes_the_overrides_and_shows_the_status(self, monkeypatch) -> None:
        spawn = mock.MagicMock(return_value=self._proc(None))
        status = mock.MagicMock()
        monkeypatch.setattr(server_mod, "spawn_server", spawn)
        monkeypatch.setattr(server_mod, "wait_for_health", lambda _p, **_kw: True)
        monkeypatch.setattr(server_mod.console, "status", status)

        server_mod._spawn_and_wait(4321, env_overrides={"A": "1"})

        spawn.assert_called_once_with(4321, env_overrides={"A": "1"})
        status.assert_called_once_with("Starting lilbee server on port 4321...")

    def test_log_size_is_zero_in_quiet_mode_even_with_a_log(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("LILBEE_LAUNCHER_SERVE_QUIET", "1")
        monkeypatch.setattr(cfg, "data_dir", tmp_path)
        log_path = tmp_path / "logs" / "launcher-serve.log"
        log_path.parent.mkdir(parents=True)
        log_path.write_bytes(b"stale\n")

        assert server_mod._serve_log_size() == 0

    def test_log_since_replaces_bytes_that_are_not_utf8(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.delenv("LILBEE_LAUNCHER_SERVE_QUIET", raising=False)
        monkeypatch.setattr(cfg, "data_dir", tmp_path)
        log_path = tmp_path / "logs" / "launcher-serve.log"
        log_path.parent.mkdir(parents=True)
        log_path.write_bytes(b"ok\n\xff\xfe boom\n")

        assert server_mod._serve_log_since(0) == "ok\n�� boom"

    def test_a_healthy_server_is_reused_after_probing_its_own_port(self, monkeypatch) -> None:
        health = mock.MagicMock(return_value=True)
        monkeypatch.setattr(server_mod, "running_server_session", lambda: ("tok", 1234))
        monkeypatch.setattr(server_mod, "health_ok", health)

        assert server_mod.ensure_server_running() == (("tok", 1234), None)
        health.assert_called_once_with(1234)

    def test_the_overrides_reach_the_spawn(self, monkeypatch) -> None:
        proc = self._proc(None)
        spawn = mock.MagicMock(return_value=proc)
        monkeypatch.setattr(cfg, "server_port", 0)
        monkeypatch.setattr(server_mod, "running_server_session", lambda: None)
        monkeypatch.setattr(server_mod, "free_port", lambda: 4321)
        monkeypatch.setattr(server_mod, "_spawn_and_wait", spawn)
        monkeypatch.setattr(server_mod, "_session_for_spawned", lambda _spawned: ("tok", 4321))

        session, spawned = server_mod.ensure_server_running(env_overrides={"A": "1"})

        assert (session, spawned) == (("tok", 4321), proc)
        spawn.assert_called_once_with(4321, env_overrides={"A": "1"})

    def test_the_failure_line_is_red_on_stderr(self, monkeypatch) -> None:
        import typer

        secho = mock.MagicMock()
        monkeypatch.setattr(cfg, "server_port", 0)
        monkeypatch.setattr(server_mod, "running_server_session", lambda: None)
        monkeypatch.setattr(server_mod, "free_port", lambda: 4321)
        monkeypatch.setattr(
            server_mod, "_spawn_and_wait", lambda _port, **_kw: server_mod._FailedStart(None, "")
        )
        monkeypatch.setattr(server_mod.typer, "secho", secho)

        with pytest.raises(typer.Exit):
            server_mod.ensure_server_running()

        secho.assert_called_once_with(
            "lilbee server failed to start on port 4321.", err=True, fg=typer.colors.RED
        )


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no process groups to signal")
def test_a_ctrl_c_to_the_launcher_group_leaves_the_stop_to_the_launcher(tmp_path: Path) -> None:
    """A terminal Ctrl-C reaches the launcher and client only; serve gets one SIGTERM."""
    launcher = _start_launcher(tmp_path)
    try:
        os.killpg(launcher.pid, signal.SIGINT)
        stdout, stderr = launcher.communicate(timeout=30)
    finally:
        _end_fakes(tmp_path)

    assert launcher.returncode == 0, stderr
    assert stdout.split()[-1:] == ["interrupted"], stderr
    assert _signals_seen(tmp_path) == ["SIGTERM"]


@pytest.mark.skipif(sys.platform == "win32", reason="Windows ends the child through a job object")
def test_serve_still_stops_when_the_launcher_is_killed(tmp_path: Path) -> None:
    """Outside the launcher's group, serve still gets SIGTERM when the launcher dies hard."""
    launcher = _start_launcher(tmp_path)
    try:
        os.kill(launcher.pid, signal.SIGKILL)
        launcher.wait(timeout=30)
        deadline = time.monotonic() + 10
        while not _signals_seen(tmp_path) and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        # The orphaned client holds the launcher's output pipes open until it ends.
        _end_fakes(tmp_path)
        launcher.communicate(timeout=30)

    assert _signals_seen(tmp_path) == ["SIGTERM"]
