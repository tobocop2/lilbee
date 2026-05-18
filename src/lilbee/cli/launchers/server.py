"""Server-lifecycle helpers shared by every ``lilbee launch <client>`` command."""

from __future__ import annotations

import logging
import shutil
import socket
import subprocess
import sys
import time

import httpx
import typer

from lilbee.cli.app import console
from lilbee.cli.commands.agent_config import running_server_session

log = logging.getLogger(__name__)

_LOCAL_HOST = "127.0.0.1"
_SERVER_BOOT_TIMEOUT_S = 60.0
_SERVER_POLL_INTERVAL_S = 0.5
_HEALTH_PROBE_TIMEOUT_S = 2.0
_HTTP_OK = 200
_TERMINATE_GRACE_S = 10
_KILL_GRACE_S = 5


def free_port() -> int:
    """Return an unused TCP port on the loopback interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((_LOCAL_HOST, 0))
        return int(s.getsockname()[1])


def health_ok(port: int) -> bool:
    """Single-shot ``/api/health`` probe; True iff a 200 comes back fast."""
    try:
        resp = httpx.get(f"http://{_LOCAL_HOST}:{port}/api/health", timeout=_HEALTH_PROBE_TIMEOUT_S)
    except httpx.HTTPError:
        return False
    return resp.status_code == _HTTP_OK


def wait_for_health(port: int, timeout_s: float = _SERVER_BOOT_TIMEOUT_S) -> bool:
    """Poll ``/api/health`` until it answers 200 or *timeout_s* elapses."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if health_ok(port):
            return True
        time.sleep(_SERVER_POLL_INTERVAL_S)
    return False


def spawn_server(port: int) -> subprocess.Popen[bytes]:
    """Spawn ``lilbee serve --port <port>`` as a background subprocess.

    Prefers the ``lilbee`` binary on PATH so frozen builds (Nuitka standalone)
    spawn the binary directly. Falls back to ``sys.executable -m lilbee`` for
    pip / editable installs where the entry point shims to the same form.
    """
    lilbee_bin = shutil.which("lilbee")
    cmd = (
        [lilbee_bin, "serve", "--port", str(port)]
        if lilbee_bin is not None
        else [sys.executable, "-m", "lilbee", "serve", "--port", str(port)]
    )
    # Only caller-controlled value is the validated integer port; no shell.
    return subprocess.Popen(  # noqa: S603
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_spawned_server(proc: subprocess.Popen[bytes]) -> None:
    """Terminate *proc* gracefully, escalating to kill if it ignores SIGTERM."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=_TERMINATE_GRACE_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=_KILL_GRACE_S)


def ensure_server_running() -> tuple[tuple[str, int], subprocess.Popen[bytes] | None]:
    """Return ``(session, spawned_proc)`` for a usable lilbee server.

    Reuses an already-running server when its session files are healthy.
    Otherwise spawns a fresh server on a free port. The returned ``spawned_proc``
    is ``None`` when an existing server was reused; the caller is responsible
    for stopping a spawned process when it is done with it.
    """
    existing = running_server_session()
    if existing is not None and health_ok(existing[1]):
        return existing, None
    chosen_port = free_port()
    console.print(f"Starting lilbee server on port {chosen_port}...")
    spawned = spawn_server(chosen_port)
    if not wait_for_health(chosen_port):
        stop_spawned_server(spawned)
        typer.secho(
            f"lilbee server failed to start on port {chosen_port}; check the logs.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    session = running_server_session()
    if session is None:
        stop_spawned_server(spawned)
        typer.secho(
            "lilbee server started but did not write a session file; cannot continue.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    return session, spawned
