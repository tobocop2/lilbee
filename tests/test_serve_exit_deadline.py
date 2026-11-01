"""The whole shutdown of ``lilbee serve`` has one deadline, proved on a fresh interpreter."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import tomllib
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

_FAILURE_CEILING_S = 30.0
_FIXED_PATH = "/usr/bin:/bin"
_DEADLINE_EXIT_STATUS = 1
_SIGINT_EXIT_STATUS = 130
_WRITTEN_KEY = "temperature"
_OLD_VALUE = 0.25
_NEW_VALUE = 0.75

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal delivery")

_SERVE_PROBE = """\
import os, sys, threading
import lilbee.app.services as services
import lilbee.cli.commands.servers as servers
import lilbee.core.security as security
import lilbee.server.app as app_mod
import lilbee.server.handlers.rag as rag
from lilbee.cli.app import app

def say(line):
    print(line, flush=True)

never = threading.Event()
servers._EXIT_DEADLINE_S = float(os.environ["X_DEADLINE"])
if os.environ.get("X_GRACE"):
    servers._GRACEFUL_SHUTDOWN_S = float(os.environ["X_GRACE"])

class Searcher:
    def search(self, q, top_k=5, chunk_type=None):
        say("HELD")
        held_on_the_engine(never)
        return []

def held_on_the_engine(event):
    event.wait()

class Services:
    searcher = Searcher()

def no_engines():
    raise RuntimeError("no engines in this probe")

app_mod.get_services = no_engines
rag.get_services = lambda: Services()

def stuck_teardown():
    say("HELD")
    never.wait()

teardown = os.environ.get("X_TEARDOWN")
if teardown == "release":
    app_mod.reset_services = never.set
elif teardown == "stuck":
    app_mod.reset_services = stuck_teardown
    services.reset_services = stuck_teardown

if os.environ.get("X_HOLD_REPLACE"):
    real_replace = security.os.replace
    def held_in_the_write(src, dst):
        if not str(dst).endswith("config.toml"):
            real_replace(src, dst)
            return
        say("HELD")
        servers.arm_exit_deadline(servers._EXIT_DEADLINE_S)
        never.wait()
    security.os.replace = held_in_the_write

sys.argv = ["lilbee", "serve", "--host", "127.0.0.1", "--port", "0"]
app()
"""

_DIES_IN_THE_WRITE_PROBE = """\
import os, sys
from pathlib import Path
import lilbee.core.security as security
from lilbee.core import settings
root = Path(os.environ["LILBEE_DATA"])
def die_at(*_args):
    os._exit(9)
if sys.argv[1] == "before-sync":
    security.os.fsync = die_at
else:
    security.os.replace = die_at
settings.save(root, {"temperature": 0.75})
"""


def _child_env(home: Path) -> dict[str, str]:
    return {**os.environ, "PATH": _FIXED_PATH, "LILBEE_DATA": str(home), "HOME": str(home)}


class _Serve:
    """A real ``lilbee serve`` in a fresh interpreter, with its output read line by line."""

    def __init__(self, home: Path, *, preexec: Callable[[], None] | None = None, **env: str):
        self.home = home
        self.process = subprocess.Popen(
            [sys.executable, "-u", "-c", _SERVE_PROBE],
            env={**_child_env(home / "data"), "HF_HUB_OFFLINE": "1", **env},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            preexec_fn=preexec,
        )
        # A kill at the failure ceiling turns a blocked read of the output into an EOF.
        self.ceiling = threading.Timer(_FAILURE_CEILING_S, self.process.kill)
        self.ceiling.daemon = True
        self.ceiling.start()
        self.port = int(self.read_until("Listening on").rsplit(":", 1)[1])
        self.client: socket.socket | None = None

    def read_until(self, needle: str) -> str:
        """Block until the server prints a line holding *needle*; EOF is a failure."""
        assert self.process.stdout is not None
        seen = []
        for line in self.process.stdout:
            if needle in line:
                return line.strip()
            seen.append(line)
        raise AssertionError(f"the server ended before it printed {needle!r}: {seen[-12:]}")

    def request(self, method: str, path: str, body: str = "") -> socket.socket:
        """Send a request with the server's token and return the open connection."""
        token_file = next((self.home / "data").rglob("server.json"))
        token = json.loads(token_file.read_text(encoding="utf-8"))["token"]
        connection = socket.create_connection(("127.0.0.1", self.port), timeout=_FAILURE_CEILING_S)
        connection.sendall(
            f"{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            f"Authorization: Bearer {token}\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n\r\n{body}".encode()
        )
        return connection

    def hold_a_read(self) -> None:
        """Open a search and block until its worker sits on the fake engine."""
        self.client = self.request("GET", "/api/search?q=x")
        self.read_until("HELD")

    def wait(self) -> int:
        """Read the rest of the output and return the exit status, under a failure ceiling."""
        self.process.communicate(timeout=_FAILURE_CEILING_S)
        return self.process.returncode

    def fault_log(self) -> str:
        """Return the text of the server fault log."""
        return next(self.home.rglob("server-fault.log")).read_text(encoding="utf-8")


@pytest.fixture
def start(tmp_path: Path) -> Iterator[Callable[..., _Serve]]:
    """Start servers that a failing test cannot leave behind."""
    started: list[_Serve] = []

    def _start(**kwargs: object) -> _Serve:
        server = _Serve(tmp_path, **kwargs)  # type: ignore[arg-type]
        started.append(server)
        return server

    yield _start
    for server in started:
        server.ceiling.cancel()
        if server.process.poll() is None:
            server.process.kill()
        server.process.communicate()


def _ignore_sigint() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _assert_the_deadline_ended_it(server: _Serve) -> None:
    assert server.wait() == _DEADLINE_EXIT_STATUS
    fault_log = server.fault_log()
    assert "Timeout (" in fault_log
    assert "in held_on_the_engine" in fault_log


@posix_only
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM], ids=["sigint", "sigterm"])
def test_a_signal_with_a_read_blocked_ends_at_the_deadline(
    start: Callable[..., _Serve], signum: signal.Signals
) -> None:
    server = start(X_DEADLINE="1")
    server.hold_a_read()
    server.process.send_signal(signum)
    _assert_the_deadline_ended_it(server)


def test_the_shutdown_route_with_a_read_blocked_ends_at_the_deadline(
    start: Callable[..., _Serve],
) -> None:
    server = start(X_DEADLINE="1")
    server.hold_a_read()
    server.request("POST", "/api/shutdown")
    _assert_the_deadline_ended_it(server)


@posix_only
def test_an_inherited_ignore_of_sigint_does_not_hide_it(start: Callable[..., _Serve]) -> None:
    server = start(X_DEADLINE="20", preexec=_ignore_sigint)
    server.process.send_signal(signal.SIGINT)
    assert server.wait() == _SIGINT_EXIT_STATUS


def test_the_grace_lets_a_clean_exit_reach_the_lifespan_teardown(
    start: Callable[..., _Serve],
) -> None:
    server = start(X_DEADLINE="20", X_GRACE="0.1", X_TEARDOWN="release")
    server.hold_a_read()
    server.request("POST", "/api/shutdown")
    assert server.wait() == 0


@posix_only
def test_a_second_ctrl_c_leaves_a_stuck_teardown_before_the_deadline(
    start: Callable[..., _Serve],
) -> None:
    server = start(X_DEADLINE="20", X_TEARDOWN="stuck")
    server.process.send_signal(signal.SIGINT)
    server.read_until("HELD")
    server.process.send_signal(signal.SIGINT)
    assert server.wait() == _SIGINT_EXIT_STATUS
    assert "Timeout (" not in server.fault_log()


@posix_only
def test_a_write_held_at_the_deadline_never_answers_200_and_leaves_the_old_file(
    start: Callable[..., _Serve], tmp_path: Path
) -> None:
    config = tmp_path / "data" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text(f"{_WRITTEN_KEY} = {_OLD_VALUE}\n", encoding="utf-8")
    server = start(X_DEADLINE="1", X_HOLD_REPLACE="1")
    client = server.request("PATCH", "/api/config", json.dumps({_WRITTEN_KEY: _NEW_VALUE}))
    server.read_until("HELD")
    assert server.wait() == _DEADLINE_EXIT_STATUS
    assert "Timeout (" in server.fault_log()
    answer = client.recv(200)
    assert b" 200 " not in answer
    assert tomllib.loads(config.read_text(encoding="utf-8"))[_WRITTEN_KEY] == _OLD_VALUE


@pytest.mark.parametrize("phase", ["before-sync", "before-replace"])
def test_a_process_that_dies_in_the_save_leaves_the_old_whole_file(
    tmp_path: Path, phase: str
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(f"{_WRITTEN_KEY} = {_OLD_VALUE}\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-c", _DIES_IN_THE_WRITE_PROBE, phase],
        env=_child_env(tmp_path),
        capture_output=True,
        encoding="utf-8",
        timeout=_FAILURE_CEILING_S,
    )
    assert result.returncode == 9, result.stderr
    assert tomllib.loads(config.read_text(encoding="utf-8")) == {_WRITTEN_KEY: _OLD_VALUE}


@posix_only
def test_a_closed_stderr_and_a_closed_fault_descriptor_still_exit_at_the_deadline(
    tmp_path: Path,
) -> None:
    probe = (
        "import faulthandler, os, sys, threading\n"
        "from lilbee.cli.commands.serve_logging import _fault_log, arm_exit_deadline\n"
        "_fault_log.enable()\n"
        "os.close(_fault_log._handle.fileno())\n"
        "os.close(2)\n"
        "sys.stderr = None\n"
        "arm_exit_deadline(1.0)\n"
        "threading.Event().wait()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env=_child_env(tmp_path),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=_FAILURE_CEILING_S,
    )
    assert result.returncode == _DEADLINE_EXIT_STATUS
