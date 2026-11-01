"""A real server process ends after a real SIGINT while an engine start is stuck."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="SIGINT is POSIX here")

# The fake write bound is longer than the shutdown bound, so the signal lands with
# the request open. Exit needs both bounds; the budget is far below the rejected
# fix's 6 min 17 s.
_WRITE_BOUND_S = 2
_SHUTDOWN_BOUND_S = 1
_EXIT_BUDGET_S = 20.0
_WAIT_FOR_SERVER_S = 30.0
_INTERRUPTED_STATUS = 130
_STILL_STARTING = f"The engines are still starting after {_WRITE_BOUND_S} s."

_PROBE = f"""
import signal, sys, threading
import typer
from lilbee.app import engine_service
from lilbee.cli.commands import servers
from lilbee.core.config import cfg
from lilbee.providers.fleet import provider as fleet

fleet._FLEET_WAIT_S = {_WRITE_BOUND_S}
fleet._SHUTDOWN_WORK_WAIT_S = {_SHUTDOWN_BOUND_S}
cfg.allow_http_engines = True
# A runner in the background starts this process with SIGINT ignored; a shell does not.
signal.signal(signal.SIGINT, signal.default_int_handler)

def _stuck(self, root):
    print("stuck", flush=True)
    threading.Event().wait()

fleet.FleetProvider._acquire_engine = _stuck
join = fleet.FleetProvider._join_fleet_work

def _joined(self, work, *, bounded=True):
    if bounded:
        print("joined", flush=True)
    join(self, work, bounded=bounded)

fleet.FleetProvider._join_fleet_work = _joined
teardown = fleet.FleetProvider._release_engines

def _announced(self, **kwargs):
    teardown(self, **kwargs)
    if self._shut_down:
        print("teardown", flush=True)

fleet.FleetProvider._release_engines = _announced

def _local():
    from lilbee.app.services import get_services
    return get_services().provider._local

def _start_the_stuck_build():
    _local()._ensure_fleet(bounded=False)

# No route reads a fleet on an empty data dir, so this route's worker reads it.
engine_service.roles = lambda: _local()._ensure_fleet()

threading.Thread(target=_start_the_stuck_build, daemon=True).start()
try:
    servers.serve(host="127.0.0.1", port=0, data_dir=None, use_global=False)
except typer.Exit as exit_:
    print(f"exit {{exit_.exit_code}}", flush=True)
except KeyboardInterrupt:
    # The interpreter's own status for an unhandled interrupt varies. A normal exit
    # still joins threads and runs atexit, which is what this test proves.
    print("interrupted", flush=True)
    sys.exit(130)
"""


class _Server:
    """The probe process and the lines it printed, read on a thread."""

    def __init__(self, home: Path) -> None:
        env = {
            **os.environ,
            "LILBEE_DATA": str(home),
            "HOME": str(home),
            "PYTHONUNBUFFERED": "1",
        }
        self.process = subprocess.Popen(
            [sys.executable, "-c", _PROBE],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
        )
        self.lines: list[str] = []
        self._changed = threading.Condition()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            with self._changed:
                self.lines.append(line.strip())
                self._changed.notify_all()

    def wait_for_line(self, start: str) -> str:
        with self._changed:
            found = self._changed.wait_for(
                lambda: any(line.startswith(start) for line in self.lines), _WAIT_FOR_SERVER_S
            )
            assert found, f"no line {start!r} in {self.lines}"
            return next(line for line in self.lines if line.startswith(start))

    def port(self) -> int:
        return int(self.wait_for_line("Listening on").rsplit(":", 1)[1])

    def stop_and_time_the_exit(self) -> float:
        """Send SIGINT; return the seconds until the process ended."""
        sent = time.monotonic()
        self.process.send_signal(signal.SIGINT)
        try:
            self.process.wait(timeout=_EXIT_BUDGET_S)
        except subprocess.TimeoutExpired:
            pytest.fail(f"the process was alive {_EXIT_BUDGET_S} s after SIGINT: {self.lines[-5:]}")
        self._reader.join(_EXIT_BUDGET_S)
        return time.monotonic() - sent


@pytest.fixture
def server(tmp_path: Path) -> Iterator[_Server]:
    """The probe server; a failure at any point leaves no process behind."""
    started = _Server(tmp_path)
    yield started
    if started.process.poll() is None:
        started.process.kill()
    started.process.wait()


def _request(
    home: Path, port: int, method: str, path: str, body: dict[str, str] | None
) -> httpx.Response:
    """Send one request with the token the server wrote under *home*."""
    token = json.loads(next(home.rglob("server.json")).read_text(encoding="utf-8"))["token"]
    return httpx.request(
        method,
        f"http://127.0.0.1:{port}{path}",
        json=body,
        headers={"Authorization": f"Bearer {token}"},
        timeout=_EXIT_BUDGET_S,
    )


def _assert_ended_by_the_signal(server: _Server, elapsed: float) -> None:
    assert elapsed < _EXIT_BUDGET_S
    assert server.process.returncode == _INTERRUPTED_STATUS, "\n".join(server.lines)
    assert "interrupted" in server.lines
    assert server.lines.count("teardown") == 1, "\n".join(server.lines)


def _sigint_with_a_request_open(
    server: _Server, home: Path, method: str, path: str, body: dict[str, str] | None
) -> tuple[float, httpx.Response]:
    """Open a request that joins the stuck build, then send SIGINT."""
    port = server.port()
    server.wait_for_line("stuck")
    answer: list[httpx.Response] = []
    client = threading.Thread(
        target=lambda: answer.append(_request(home, port, method, path, body)), daemon=True
    )
    client.start()
    server.wait_for_line("joined")
    elapsed = server.stop_and_time_the_exit()
    client.join(_EXIT_BUDGET_S)
    assert not client.is_alive()
    return elapsed, answer[0]


def test_a_server_with_a_queued_write_ends_after_sigint(server: _Server, tmp_path: Path) -> None:
    elapsed, answer = _sigint_with_a_request_open(
        server, tmp_path, "PUT", "/api/engines/preferred", {"engine": "llama.cpp"}
    )
    _assert_ended_by_the_signal(server, elapsed)
    assert answer.status_code == 503
    assert answer.json()["detail"] == (
        "The engine choice is saved, but the running server could not apply it: "
        f"{_STILL_STARTING}. Restart lilbee to use the choice."
    )


def test_a_server_with_an_open_read_ends_after_sigint(server: _Server, tmp_path: Path) -> None:
    elapsed, answer = _sigint_with_a_request_open(
        server, tmp_path, "GET", "/api/engines/roles", None
    )
    _assert_ended_by_the_signal(server, elapsed)
    assert answer.status_code == 500


def test_a_server_with_no_write_ends_after_sigint(server: _Server) -> None:
    server.port()
    server.wait_for_line("stuck")
    elapsed = server.stop_and_time_the_exit()
    _assert_ended_by_the_signal(server, elapsed)
