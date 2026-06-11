"""Server-lifecycle helpers shared by every ``lilbee launch <client>`` command."""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from typing import IO

import httpx
import typer

from lilbee.app.services import get_services
from lilbee.catalog.types import ModelTask
from lilbee.cli.app import console
from lilbee.cli.commands.servers import port_file
from lilbee.core.config import cfg
from lilbee.modelhub.registry import ModelRegistry
from lilbee.providers.fleet.swap_config import cold_load_timeout_s
from lilbee.server.auth import server_json_path

log = logging.getLogger(__name__)

LOOPBACK = "127.0.0.1"
"""Loopback address used for launcher-spawned sessions and the URLs we hand to clients."""

_SERVER_BOOT_TIMEOUT_S = 60.0
_SERVER_POLL_INTERVAL_S = 0.5
# Floor on the cold model-load wait; chat_warm_budget_s() scales it up with the weights.
_WARM_TIMEOUT_S = 600.0
_HEALTH_PROBE_TIMEOUT_S = 2.0
_HTTP_OK = 200
_TERMINATE_GRACE_S = 10
_KILL_GRACE_S = 5
# Spawn attempts; free_port()'s released probe port can be stolen before the server binds.
_SPAWN_ATTEMPTS = 3


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


def chat_ready(port: int) -> bool:
    """Single-shot probe: True iff ``/api/health`` reports the chat engine warm."""
    try:
        resp = httpx.get(f"http://{LOOPBACK}:{port}/api/health", timeout=_HEALTH_PROBE_TIMEOUT_S)
    except httpx.HTTPError:
        return False
    if resp.status_code != _HTTP_OK:
        return False
    return bool(resp.json().get("chat_ready", False))


def served_chat_ctx(port: int) -> int | None:
    """The chat window ``/api/health`` reports, or None if unknown/unreachable.

    A launcher passes this to the client so it trims history to the model's
    actual window instead of overflowing on a long agentic session.
    """
    try:
        resp = httpx.get(f"http://{LOOPBACK}:{port}/api/health", timeout=_HEALTH_PROBE_TIMEOUT_S)
    except httpx.HTTPError:
        return None
    if resp.status_code != _HTTP_OK:
        return None
    ctx = resp.json().get("chat_ctx")
    return ctx if isinstance(ctx, int) and ctx > 0 else None


def chat_warm_budget_s() -> float:
    """Warm wait scaled to the chat model's on-disk weights at the engine's cold-load rate."""
    try:
        shards = ModelRegistry(cfg.models_dir).shard_paths(str(cfg.chat_model))
    except (KeyError, ValueError):
        return _WARM_TIMEOUT_S
    total_bytes = sum(shard.stat().st_size for shard in shards)
    return max(_WARM_TIMEOUT_S, float(cold_load_timeout_s(total_bytes)))


def wait_for_chat_warm(port: int, timeout_s: float | None = None) -> bool:
    """Block until the chat model is loaded, showing a warming indicator.

    The server warms the chat role on a background thread at startup, so a client
    launched the instant the HTTP port binds would otherwise hit an
    apparently-dead stream during the cold model load. Returns True once the
    chat engine reports ready, or False if the budget (weights-scaled via
    :func:`chat_warm_budget_s` unless given) elapses first; the caller proceeds
    either way, so a still-loading model just warms on the first call.
    """
    if timeout_s is None:
        timeout_s = chat_warm_budget_s()
    if chat_ready(port):
        return True
    deadline = time.monotonic() + timeout_s
    with console.status("Warming the chat model..."):
        while time.monotonic() < deadline:
            if chat_ready(port):
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
    lilbee_bin = shutil.which("lilbee")
    cmd = (
        [lilbee_bin, "serve", "--port", str(port)]
        if lilbee_bin is not None
        else [sys.executable, "-m", "lilbee", "serve", "--port", str(port)]
    )

    if os.environ.get("LILBEE_LAUNCHER_SERVE_QUIET"):
        stdout: int | IO[bytes] = subprocess.DEVNULL
        stderr: int | IO[bytes] = subprocess.DEVNULL
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
    last_port = 0
    for _ in range(_SPAWN_ATTEMPTS):
        last_port = free_port()
        spawned = _spawn_and_wait(last_port)
        if spawned is not None:
            return _session_for_spawned(spawned), spawned
    typer.secho(
        f"lilbee server failed to start on port {last_port}; check the logs.",
        err=True,
        fg=typer.colors.RED,
    )
    raise typer.Exit(1)


def _spawn_and_wait(port: int) -> subprocess.Popen[bytes] | None:
    """Spawn a server on *port* and wait for health; None when it never comes up."""
    console.print(f"Starting lilbee server on port {port}...")
    spawned = spawn_server(port)
    if wait_for_health(port):
        return spawned
    stop_spawned_server(spawned)
    return None


def _session_for_spawned(spawned: subprocess.Popen[bytes]) -> tuple[str, int]:
    """Read the session a freshly-healthy server wrote, stopping it when missing."""
    session = running_server_session()
    if session is None:
        stop_spawned_server(spawned)
        typer.secho(
            "lilbee server started but did not write a session file; cannot continue.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    return session
