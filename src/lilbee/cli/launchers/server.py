"""Server-lifecycle helpers shared by every ``lilbee launch <client>`` command."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import IO, NamedTuple

import httpx
import typer

from lilbee.cli.app import console
from lilbee.cli.commands.servers import port_file
from lilbee.cli.launchers.warm_render import render_warm
from lilbee.core.config import cfg
from lilbee.modelhub.registry import ModelRegistry
from lilbee.parent_monitor import PARENT_PID_ENV
from lilbee.providers.fleet.child_guard import spawn_bound_child
from lilbee.providers.fleet.read_rate import measured_read_rate
from lilbee.providers.fleet.swap_config import cold_load_timeout_s
from lilbee.providers.roles import WorkerRole
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
_HEALTH_PATH = "/api/health"
_TERMINATE_GRACE_S = 10
_KILL_GRACE_S = 5
# Spawn attempts; free_port()'s released probe port can be stolen before the server binds.
_SPAWN_ATTEMPTS = 3
_SERVE_LOG_TAIL_LINES = 20
_SERVE_LOG_TAIL_BYTES = 16 * 1024


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


def free_port() -> int:
    """Return an unused TCP port on the loopback interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((LOOPBACK, 0))
        return int(s.getsockname()[1])


def _session_token() -> str | None:
    """The bearer token from server.json, or None if it is not readable yet."""
    try:
        data = json.loads(server_json_path().read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None
    token = data.get("token")
    return token if isinstance(token, str) and token else None


def _probe_health(port: int) -> dict[str, object] | None:
    """GET ``/api/health`` once; return the parsed body on 200, else None.

    The single place the probe URL, timeout, error handling, and status check
    live, so the three public probes below stay consistent.

    Health needs the token like every other route: it reports the chat
    engine's last error, which carries model paths and loader failures. The
    token is re-read per attempt rather than captured once, because these
    probes poll a server that is still starting and server.json does not exist
    until its lifespan has run. No token yet means no server yet, which is the
    same answer a refused connection gives.
    """
    token = _session_token()
    if token is None:
        return None
    try:
        resp = httpx.get(
            f"http://{LOOPBACK}:{port}{_HEALTH_PATH}",
            timeout=_HEALTH_PROBE_TIMEOUT_S,
            headers={"Authorization": f"Bearer {token}"},
        )
    except httpx.HTTPError:
        return None
    if resp.status_code != _HTTP_OK:
        return None
    try:
        body = resp.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def health_ok(port: int) -> bool:
    """Single-shot ``/api/health`` probe; True iff a 200 comes back fast."""
    return _probe_health(port) is not None


def wait_for_health(
    port: int,
    timeout_s: float = _SERVER_BOOT_TIMEOUT_S,
    *,
    server: subprocess.Popen[bytes] | None = None,
) -> bool:
    """Poll ``/api/health`` until it answers 200, *timeout_s* elapses, or *server* exits."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if server is not None and server.poll() is not None:
            return False
        if health_ok(port):
            return True
        time.sleep(_SERVER_POLL_INTERVAL_S)
    return False


def chat_ready(port: int) -> bool:
    """Single-shot probe: True iff ``/api/health`` reports the chat engine warm."""
    body = _probe_health(port)
    return bool(body and body.get("chat_ready", False))


def served_chat_ctx(port: int) -> int | None:
    """The chat window ``/api/health`` reports, or None if unknown/unreachable.

    A launcher passes this to the client so it trims history to the model's
    actual window instead of overflowing on a long agentic session.
    """
    body = _probe_health(port)
    if body is None:
        return None
    ctx = body.get("chat_ctx")
    return ctx if isinstance(ctx, int) and ctx > 0 else None


def planned_chat_ctx() -> int | None:
    """The per-slot window the fleet will serve the configured chat model, or None.

    Mirrors the fleet's own single-GPU chat sizing, so it answers before the
    engine is up: a ``cfg.num_ctx`` pin clamped to the trained window as the launch
    clamps it, else the same
    :func:`resolve_chat_ctx` against the same budget the fleet sizes with, which
    is the memory the GPU reports rather than the host's (see
    :func:`lilbee.providers.fleet.planning.plan_sizing_budget`).

    A tensor-split chat is sized by the fleet against per-device headroom
    instead, so this can over-report there; it is only a fallback for a chat
    engine that is not up yet, and the served window wins once it is. A
    remote-served chat model has no local window to compute.
    """
    from lilbee.app.engine_choice import resolve_engines
    from lilbee.providers.base import ProviderError
    from lilbee.providers.engine_params import resolve_chat_ctx, resolve_model_path
    from lilbee.providers.fleet.planning import plan_sizing_budget
    from lilbee.providers.fleet.planning import planned_chat_ctx as pinned_chat_ctx
    from lilbee.providers.model_ref import parse_model_ref

    ref = str(cfg.chat_model)
    if not ref or not parse_model_ref(ref).is_local:
        return None
    try:
        path = resolve_model_path(ref)
        engine = resolve_engines(cfg).for_role(WorkerRole.CHAT).adapter
        meta = engine.model_meta(path)
        if cfg.num_ctx is not None:
            return pinned_chat_ctx(path, meta)
        return resolve_chat_ctx(
            path,
            meta,
            engine=engine,
            available_bytes=plan_sizing_budget(engine),
        )
    except (ProviderError, OSError, ValueError):
        # Sizing needs the model file and its GGUF header. Without them the trained
        # window is unknown: a pin stands as given, an unpinned window stays unknown.
        log.debug("planned_chat_ctx failed for %s", ref, exc_info=True)
        return cfg.num_ctx


def client_chat_ctx(port: int) -> int | None:
    """The chat window to advertise to a launched client, warning when it is small.

    The chat role builds lazily, so a launcher that hands off before the engine
    is warm gets nothing from ``/api/health``; fall back to the window the fleet
    plans to serve rather than leaving the client with no window at all. A window
    below ``cfg.chat_n_ctx_target`` means the host could not back what was asked
    for, which changes how much history an agent can keep, so say so.
    """
    ctx = served_chat_ctx(port)
    if ctx is None:
        ctx = planned_chat_ctx()
    if ctx is not None and ctx < cfg.chat_n_ctx_target:
        typer.secho(
            f"Warning: the chat model is served with a {ctx:,}-token context, below the "
            f"configured chat_n_ctx_target of {cfg.chat_n_ctx_target:,}. Either the model "
            "was trained on a smaller window, or its weights leave too little of the "
            "memory budget for the KV cache. A longer-context model, a smaller "
            "quantization, or a higher gpu_memory_fraction raises it.",
            err=True,
            fg=typer.colors.YELLOW,
        )
    return ctx


def chat_warm_budget_s() -> float:
    """Warm wait scaled to the chat model's on-disk weights at the engine's cold-load rate.

    A split model sums its shards; a single file, which resolves to its blob with no
    shard beside it, is its own size. The volume's read rate is measured on the file
    the engine opens first.
    """
    registry = ModelRegistry(cfg.models_dir)
    try:
        first = registry.resolve(str(cfg.chat_model))
        shards = registry.shard_paths(str(cfg.chat_model)) or [first]
    except (KeyError, ValueError):
        return _WARM_TIMEOUT_S
    total_bytes = sum(shard.stat().st_size for shard in shards)
    rate = measured_read_rate(first)
    return max(_WARM_TIMEOUT_S, float(cold_load_timeout_s(total_bytes, WorkerRole.CHAT, rate)))


def wait_for_chat_warm(port: int, timeout_s: float | None = None) -> bool:
    """Block until the chat model is loaded, showing granular warm progress.

    The server warms the chat role on a background thread at startup, so a client
    launched the instant the HTTP port binds would otherwise hit an
    apparently-dead stream during the cold model load. Streams ``/api/warm/stream``
    to render a real read-phase byte bar then an engine-load spinner; falls back
    to a plain readiness poll when that stream can't be opened.
    Returns True once the chat engine reports ready, or False if the budget
    (weights-scaled via :func:`chat_warm_budget_s` unless given) elapses first;
    the caller proceeds either way, so a still-loading model just warms on the
    first call.
    """
    if timeout_s is None:
        timeout_s = chat_warm_budget_s()
    if chat_ready(port):
        return True
    streamed = render_warm(f"http://{LOOPBACK}:{port}", timeout_s)
    if streamed is not None:
        # The stream ran (ready, error, or its own timeout); don't double-spend
        # the budget on a second poll. The caller proceeds on False regardless.
        return streamed
    return _poll_chat_ready(port, timeout_s)


def _poll_chat_ready(port: int, timeout_s: float) -> bool:
    """Fallback warm wait when the progress stream is unavailable: poll readiness."""
    deadline = time.monotonic() + timeout_s
    with console.status("Warming the chat model..."):
        while time.monotonic() < deadline:
            if chat_ready(port):
                return True
            time.sleep(_SERVER_POLL_INTERVAL_S)
    return False


def serve_log_path() -> Path:
    """The file a launcher-spawned server writes its stdout and stderr to."""
    return cfg.data_dir / "logs" / "launcher-serve.log"


def _serve_log_size() -> int:
    """The byte size of the server log, 0 when quiet mode or no log exists."""
    if os.environ.get("LILBEE_LAUNCHER_SERVE_QUIET"):
        return 0
    try:
        return serve_log_path().stat().st_size
    except OSError:
        return 0


def _serve_log_since(start: int) -> str:
    """The last lines the server wrote after byte *start*, empty when it wrote none."""
    if os.environ.get("LILBEE_LAUNCHER_SERVE_QUIET"):
        return ""
    try:
        with serve_log_path().open("rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            # A log truncated at spawn is smaller than the recorded start.
            first = 0 if size < start else start
            handle.seek(max(first, size - _SERVE_LOG_TAIL_BYTES))
            raw = handle.read()
    except OSError:
        return ""
    lines = raw.decode("utf-8", errors="replace").splitlines()[-_SERVE_LOG_TAIL_LINES:]
    return "\n".join(lines)


class _FailedStart(NamedTuple):
    """Why one spawned server did not become healthy."""

    exit_status: int | None
    output: str


def spawn_server(
    port: int, *, env_overrides: dict[str, str] | None = None
) -> subprocess.Popen[bytes]:
    """Spawn ``lilbee serve --port <port>`` as a background subprocess.

    Prefers the ``lilbee`` binary on PATH so frozen builds (Nuitka standalone)
    spawn the binary directly. Falls back to ``sys.executable -m lilbee`` for
    pip / editable installs where the entry point shims to the same form.

    ``env_overrides`` are layered onto the inherited environment for the child
    (e.g. ``LILBEE_CHAT_N_CTX_TARGET`` to size the served window for a launched
    agent); ``None`` inherits the parent environment unchanged.

    Stdout/stderr go to ``cfg.data_dir / "logs" / "launcher-serve.log"`` (size
    capped at 5 MB) so a crash mid-session leaves a trace instead of disappearing.
    Set ``LILBEE_LAUNCHER_SERVE_QUIET=1`` to restore the previous DEVNULL behavior.
    """
    lilbee_bin = shutil.which("lilbee")
    # On Windows, pip/uv may install a ``lilbee.cmd`` wrapper instead of a bare
    # executable. Popen(shell=False) raises PermissionError on .cmd files, so
    # fall through to the sys.executable -m lilbee form in that case.
    _bin_is_cmd = sys.platform == "win32" and (
        lilbee_bin is not None and lilbee_bin.lower().endswith(".cmd")
    )
    cmd = (
        [lilbee_bin, "serve", "--port", str(port)]
        if lilbee_bin is not None and not _bin_is_cmd
        else [sys.executable, "-m", "lilbee", "serve", "--port", str(port)]
    )

    log_file: IO[bytes] | None = None
    if os.environ.get("LILBEE_LAUNCHER_SERVE_QUIET"):
        stdout: int | IO[bytes] = subprocess.DEVNULL
        stderr: int | IO[bytes] = subprocess.DEVNULL
    else:
        log_path = serve_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate when the file passes 5 MB so a long-lived session doesn't
        # accumulate the chat-completion firehose into the data dir indefinitely.
        # On Windows the file may still be held open by a previous session, so
        # fall through to append mode when unlink is denied.
        if log_path.exists() and log_path.stat().st_size > 5 * 1024 * 1024:
            with contextlib.suppress(OSError):
                log_path.unlink()
        log_file = log_path.open("ab")
        stdout = log_file
        stderr = subprocess.STDOUT

    # LILBEE_PARENT_PID arms serve's parent-death watcher, so a hard-killed
    # launcher (whose finally never runs) does not orphan serve holding server_lock.
    child_env = {**os.environ, **(env_overrides or {}), PARENT_PID_ENV: str(os.getpid())}

    # Windows: a group of its own stops a console Ctrl-C reaching serve and lets
    # CTRL_BREAK target serve alone.
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    try:
        # The launcher's one SIGTERM must be serve's first signal, so the clean stop
        # runs instead of the second-signal force quit. A session of its own keeps a
        # terminal Ctrl-C to the launcher's group away from serve. The watcher alone
        # stops serve when the launcher dies: the kernel death signal is sent again
        # each time a launcher thread exits, so it can be a second SIGTERM.
        return spawn_bound_child(
            cmd,
            bind_lifetime=False,
            stdout=stdout,
            stderr=stderr,
            env=child_env,
            start_new_session=os.name == "posix",
            creationflags=creationflags,
        )
    finally:
        # Popen dups the fd into the child; the parent's handle is no longer
        # needed and would otherwise leak for the launcher's whole lifetime.
        if log_file is not None:
            log_file.close()


def stop_spawned_server(proc: subprocess.Popen[bytes]) -> None:
    """Terminate *proc* gracefully, escalating to kill if it ignores SIGTERM."""
    if proc.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            # terminate() is TerminateProcess there: serve's handlers never run. CTRL_BREAK
            # reaches the one process group spawn_server made for serve and nothing else.
            # It raises OSError when the launcher has no console (pythonw, a service).
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
        proc.wait(timeout=_TERMINATE_GRACE_S)
    except (OSError, subprocess.TimeoutExpired):
        proc.kill()
        try:
            proc.wait(timeout=_KILL_GRACE_S)
        except subprocess.TimeoutExpired:
            log.warning(
                "Server process %d was not reaped within %ds after SIGKILL",
                proc.pid,
                _KILL_GRACE_S,
            )


def ensure_server_running(
    *, env_overrides: dict[str, str] | None = None
) -> tuple[tuple[str, int], subprocess.Popen[bytes] | None]:
    """Return ``(session, spawned_proc)`` for a usable lilbee server.

    Reuses an already-running server when its session files are healthy.
    Otherwise spawns a fresh server on a free port. The returned ``spawned_proc``
    is ``None`` when an existing server was reused; the caller is responsible
    for stopping a spawned process when it is done with it.

    ``env_overrides`` reach a freshly spawned child (e.g. a launcher sizing the
    served window); a reused server keeps whatever window it booted with.
    """
    existing = running_server_session()
    if existing is not None and health_ok(existing[1]):
        return existing, None
    last_port = 0
    failed = _FailedStart(None, "")
    # A server that exits is retried like one that hangs: a stolen port looks the same.
    for _ in range(_SPAWN_ATTEMPTS):
        # Honor a user-pinned port so a persisted agent config keeps a valid URL;
        # fall back to a free port when unset (0).
        last_port = cfg.server_port or free_port()
        spawned = _spawn_and_wait(last_port, env_overrides=env_overrides)
        if not isinstance(spawned, _FailedStart):
            return _session_for_spawned(spawned), spawned
        failed = spawned
    typer.secho(
        f"lilbee server failed to start on port {last_port}.",
        err=True,
        fg=typer.colors.RED,
    )
    _report_failed_start(failed)
    raise typer.Exit(1)


def _report_failed_start(failed: _FailedStart) -> None:
    """Print how the last spawned server ended and what it wrote to the log."""
    log_path = serve_log_path()
    if failed.output:
        typer.echo(f"Last lines of {log_path}:\n{failed.output}", err=True)
    elif failed.exit_status is not None:
        where = (
            "its output is off (LILBEE_LAUNCHER_SERVE_QUIET)"
            if os.environ.get("LILBEE_LAUNCHER_SERVE_QUIET")
            else f"it wrote nothing to {log_path}"
        )
        typer.echo(f"The server exited with status {failed.exit_status}; {where}.", err=True)


def _spawn_and_wait(
    port: int, *, env_overrides: dict[str, str] | None = None
) -> subprocess.Popen[bytes] | _FailedStart:
    """Spawn a server on *port* and wait for health; the failure when it exits or hangs."""
    log_start = _serve_log_size()
    spawned = spawn_server(port, env_overrides=env_overrides)
    with console.status(f"Starting lilbee server on port {port}..."):
        healthy = wait_for_health(port, server=spawned)
    if healthy:
        return spawned
    exit_status = spawned.poll()
    stop_spawned_server(spawned)
    return _FailedStart(exit_status, _serve_log_since(log_start))


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
