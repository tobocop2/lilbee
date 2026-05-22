"""Server-lifecycle helpers shared by every ``lilbee launch <client>`` command."""

from __future__ import annotations

import json
import logging
import shutil
import socket
import subprocess
import sys
import time

import httpx
import typer

from lilbee.app.services import get_services
from lilbee.catalog.types import ModelTask
from lilbee.cli.app import console
from lilbee.cli.commands.servers import port_file
from lilbee.server.auth import server_json_path

log = logging.getLogger(__name__)

LOOPBACK = "127.0.0.1"
"""Loopback address used for launcher-spawned sessions and the URLs we hand to clients."""

_SERVER_BOOT_TIMEOUT_S = 60.0
_SERVER_POLL_INTERVAL_S = 0.5
_HEALTH_PROBE_TIMEOUT_S = 2.0
_HTTP_OK = 200
_TERMINATE_GRACE_S = 10
_KILL_GRACE_S = 5


def running_server_session() -> tuple[str, int] | None:
    """Return ``(token, port)`` for a server already running on this machine, else None."""
    session_path = server_json_path()
    port_path = port_file()
    if not session_path.exists() or not port_path.exists():
        return None
    try:
        data = json.loads(session_path.read_text(encoding="utf-8"))
        token = data.get("token")
        port = int(port_path.read_text(encoding="utf-8").strip())
    except (json.JSONDecodeError, OSError, ValueError):
        return None
    if not isinstance(token, str) or not token:
        return None
    return token, port


def installed_chat_model_refs() -> list[str]:
    """Return sorted refs for every chat-task model in the registry."""
    registry = get_services().registry
    return sorted(m.ref for m in registry.list_installed() if m.task == ModelTask.CHAT)


def free_port() -> int:
    """Return an unused TCP port on the loopback interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((LOOPBACK, 0))
        return int(s.getsockname()[1])


def health_ok(port: int) -> bool:
    """Single-shot ``/api/health`` probe; True iff a 200 comes back fast."""
    try:
        resp = httpx.get(f"http://{LOOPBACK}:{port}/api/health", timeout=_HEALTH_PROBE_TIMEOUT_S)
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

    Stdout/stderr go to ``cfg.data_dir / "logs" / "launcher-serve.log"`` (size
    capped at 5 MB) so a crash mid-session leaves a trace instead of disappearing.
    Set ``LILBEE_LAUNCHER_SERVE_QUIET=1`` to restore the previous DEVNULL behavior.
    """
    import os

    from lilbee.core.config import cfg

    lilbee_bin = shutil.which("lilbee")
    cmd = (
        [lilbee_bin, "serve", "--port", str(port)]
        if lilbee_bin is not None
        else [sys.executable, "-m", "lilbee", "serve", "--port", str(port)]
    )

    if os.environ.get("LILBEE_LAUNCHER_SERVE_QUIET"):
        stdout: object = subprocess.DEVNULL
        stderr: object = subprocess.DEVNULL
    else:
        log_dir = cfg.data_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "launcher-serve.log"
        # Truncate when the file passes 5 MB so a long-lived session doesn't
        # accumulate the chat-completion firehose into the data dir indefinitely.
        if log_path.exists() and log_path.stat().st_size > 5 * 1024 * 1024:
            log_path.unlink()
        log_file = log_path.open("ab")
        stdout = log_file
        stderr = subprocess.STDOUT

    # Only caller-controlled value is the validated integer port; no shell.
    return subprocess.Popen(  # noqa: S603
        cmd,
        stdout=stdout,
        stderr=stderr,
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
