"""Cross-role helpers for worker subprocesses.

Bootstraps the per-role workers (embed, chat, rerank, vision) and runs
the recv loop. Health pings and graceful shutdown live on a dedicated
daemon thread so a long inference call cannot starve liveness or stop
the parent from terminating the worker.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from lilbee.providers.worker.transport import RoleConfig, WorkerRole
from lilbee.providers.worker.transport_pipe import _serialize_exception
from lilbee.providers.worker.wire_kinds import WireKind

log = logging.getLogger(__name__)

# How often the main data loop wakes from poll() to re-check the shutdown
# flag. 100 ms is well under any user-visible cancel budget but rare enough
# not to show up as CPU noise.
_DATA_POLL_INTERVAL_S = 0.1

#: Subdirectory of ``cfg.data_root`` where per-role worker logs land.
#: Shared with :mod:`lilbee.providers.worker.transport_pipe` so the parent's
#: :class:`WorkerCrashError` points at the exact file the worker wrote.
WORKER_LOGS_DIR_NAME = "logs"


def redirect_stdio_to_devnull() -> None:  # pragma: no cover - subprocess fd swap
    """Send stdout/stderr to /dev/null so llama-cpp's C-level prints stay quiet."""
    devnull_fd = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    sys.stdout = open(os.devnull, "w")  # noqa: SIM115
    sys.stderr = open(os.devnull, "w")  # noqa: SIM115


def configure_worker_logging(role: WorkerRole) -> None:
    """Append worker logs to ``$LILBEE_DATA/logs/worker-<role>.log``.

    ``LILBEE_DATA`` is canonicalized at cfg construction
    (:func:`lilbee.core.config.model._build_cfg`) and at every write
    site that switches the data root, so the env is the single source
    of truth.
    """
    data_dir = os.environ.get("LILBEE_DATA")
    if not data_dir:
        return
    logs_dir = os.path.join(data_dir, WORKER_LOGS_DIR_NAME)
    with contextlib.suppress(OSError):
        os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"worker-{role}.log")
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


class Reply:
    """Sender for the in-flight call's response frames on the data pipe.

    Handlers receive a Reply instance and emit response frames via
    ``reply.send(kind, payload)``. The parent's :class:`PipeChannel`
    holds the call lock for the full request/reply or request/stream
    window, so a frame sent here belongs to the call the worker is
    currently servicing by construction.
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def send(self, kind: WireKind, payload: Any) -> None:
        """Send one response frame on the data pipe."""
        self._conn.send((kind, payload))


@dataclass
class WorkerLoopState:
    """Per-loop state shared by the dispatcher and the role's handlers.

    Holds the role's lazy-loaded session. Each handler receives a
    reference and pulls its model via ``state.session``.
    """

    session: Any


KindHandler = Callable[[Reply, Any, WorkerLoopState], None]
"""Per-role handler signature: ``(reply, payload, state) -> None``.

Handlers reach their role-specific session via ``state.session`` and
emit response frames via ``reply.send(kind, payload)``.
"""


def run_worker(
    data_conn: Any,
    health_conn: Any,
    abort_flag: Any,
    role_config: RoleConfig,
    *,
    session_factory: Callable[[RoleConfig, Any], Any],
    kind_handlers: dict[WireKind, KindHandler],
) -> None:
    """Bootstrap stdio + logging, then run the recv loop until shutdown.

    The control plane (ping, shutdown) travels on the health pipe and is
    served by a daemon thread. The main loop polls the data pipe with a
    short timeout so the shutdown flag set by the heartbeat thread fires
    promptly even when no data frame is pending. Unknown data-pipe kinds
    reply with a serialized ``ValueError``.
    """
    redirect_stdio_to_devnull()
    configure_worker_logging(role_config.role)
    log.info(
        "%s worker online (pid=%s, model=%s)",
        role_config.role,
        os.getpid(),
        role_config.model_path,
    )
    state = WorkerLoopState(session=session_factory(role_config, abort_flag))
    shutdown_event = threading.Event()
    heartbeat = _start_heartbeat_thread(health_conn, role_config.role, shutdown_event)
    try:
        while not shutdown_event.is_set():
            if not _handle_data_frame(
                data_conn,
                state,
                kind_handlers,
                role_config.role,
                shutdown_event,
            ):
                break
    finally:
        shutdown_event.set()
        state.session.close()
        with contextlib.suppress(Exception):
            data_conn.close()
        with contextlib.suppress(Exception):
            health_conn.close()
        heartbeat.join(timeout=1.0)


def _start_heartbeat_thread(
    health_conn: Any, role: WorkerRole, shutdown_event: threading.Event
) -> threading.Thread:
    """Spawn the daemon thread that owns the health pipe."""
    thread = threading.Thread(
        target=_heartbeat_loop,
        args=(health_conn, role, shutdown_event),
        name=f"lilbee-worker-{role}-heartbeat",
        daemon=True,
    )
    thread.start()
    return thread


def _heartbeat_loop(health_conn: Any, role: WorkerRole, shutdown_event: threading.Event) -> None:
    """Serve ping/shutdown on the health pipe until the parent closes it.

    On SHUTDOWN, sets the shutdown event so the main thread exits its
    poll loop within ``_DATA_POLL_INTERVAL_S`` regardless of whether a
    data frame is pending. The ACK reply is best-effort: the parent's
    close() suppresses pipe errors so a torn-down health connection
    does not surface as a noisy supervisor warning.
    """
    while True:
        try:
            kind, _ = health_conn.recv()
        except (EOFError, OSError):
            shutdown_event.set()
            return
        if kind == WireKind.PING:
            try:
                health_conn.send((WireKind.PONG, None))
            except (BrokenPipeError, OSError):
                shutdown_event.set()
                return
            continue
        if kind == WireKind.SHUTDOWN:
            with contextlib.suppress(BrokenPipeError, OSError):
                health_conn.send((WireKind.ACK, None))
            shutdown_event.set()
            return
        log.warning("%s worker dropped unexpected health-pipe kind %r", role, kind)


def _handle_data_frame(
    data_conn: Any,
    state: WorkerLoopState,
    kind_handlers: dict[WireKind, KindHandler],
    role: WorkerRole,
    shutdown_event: threading.Event,
) -> bool:
    """Read and dispatch one data-pipe frame. Return False to stop the loop.

    Uses ``poll()`` with a short timeout so a shutdown flag set by the
    heartbeat thread takes effect within one poll interval even when the
    data pipe is idle. A long-running handler (a multi-second chat stream)
    is interrupted only after the handler returns; the join in
    ``PipeChannel.close`` falls back to ``terminate()`` past the close
    timeout for that case.
    """
    while not data_conn.poll(_DATA_POLL_INTERVAL_S):
        if shutdown_event.is_set():
            return False
    try:
        kind, payload = data_conn.recv()
    except EOFError:
        return False
    reply = Reply(data_conn)
    handler = kind_handlers.get(kind)
    if handler is not None:
        handler(reply, payload, state)
        return True
    try:
        raise ValueError(f"{role} worker received unknown kind {kind!r}")
    except ValueError as exc:
        reply.send(WireKind.ERROR, _serialize_exception(exc))
    return True


__all__ = [
    "KindHandler",
    "Reply",
    "WorkerLoopState",
    "configure_worker_logging",
    "redirect_stdio_to_devnull",
    "run_worker",
]
