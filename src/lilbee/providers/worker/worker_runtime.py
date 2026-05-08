"""Cross-role helpers for worker subprocesses.

Bootstraps the per-role workers (embed, chat, rerank, vision) and runs
the recv loop. Health pings live on a dedicated daemon thread so a long
inference call cannot starve heartbeats.
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

from lilbee.providers.worker.transport import RoleConfig
from lilbee.providers.worker.transport_pipe import _serialize_exception
from lilbee.providers.worker.wire_kinds import WireKind

log = logging.getLogger(__name__)

_CONTROL_CALL_ID = 0
"""Sentinel call-id for shutdown/ack/ping/pong frames."""


def redirect_stdio_to_devnull() -> None:  # pragma: no cover - subprocess fd swap
    """Send stdout/stderr to /dev/null so llama-cpp's C-level prints stay quiet."""
    devnull_fd = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    sys.stdout = open(os.devnull, "w")  # noqa: SIM115
    sys.stderr = open(os.devnull, "w")  # noqa: SIM115


def configure_worker_logging(role: str) -> None:
    """Append worker logs to ``$LILBEE_DATA/logs/worker-<role>.log`` if set."""
    data_dir = os.environ.get("LILBEE_DATA")
    if not data_dir:
        return
    logs_dir = os.path.join(data_dir, "logs")
    with contextlib.suppress(OSError):
        os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"worker-{role}.log")
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


class Reply:
    """Per-call sender bound to one ``call_id`` on the data pipe.

    Handlers receive a Reply instance and emit response frames via
    ``reply.send(KIND, payload)``. The call_id is injected transparently so
    handlers do not need to thread it through their internal logic.
    """

    def __init__(self, conn: Any, call_id: int) -> None:
        self._conn = conn
        self._call_id = call_id

    def send(self, kind: WireKind, payload: Any) -> None:
        """Send one response frame tagged with this Reply's call_id."""
        self._conn.send((self._call_id, kind, payload))


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

    Health pings travel on a separate pipe owned by a daemon thread that
    answers ping → pong without depending on the data-frame handler. The
    main loop reads frames as ``(call_id, kind, payload)`` and dispatches
    role-specific kinds via *kind_handlers*. Unknown kinds reply with a
    serialized ``ValueError``.
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
    heartbeat = _start_heartbeat_thread(health_conn, role_config.role)
    try:
        while _handle_data_frame(data_conn, state, kind_handlers, role_config.role):
            pass
    finally:
        state.session.close()
        with contextlib.suppress(Exception):
            data_conn.close()
        with contextlib.suppress(Exception):
            health_conn.close()
        heartbeat.join(timeout=1.0)


def _start_heartbeat_thread(health_conn: Any, role: str) -> threading.Thread:
    """Spawn the daemon thread that owns the health pipe."""
    thread = threading.Thread(
        target=_heartbeat_loop,
        args=(health_conn, role),
        name=f"lilbee-worker-{role}-heartbeat",
        daemon=True,
    )
    thread.start()
    return thread


def _heartbeat_loop(health_conn: Any, role: str) -> None:
    """Respond to ping → pong on the health pipe until the parent closes it."""
    while True:
        try:
            _call_id, kind, _ = health_conn.recv()
        except (EOFError, OSError):
            return
        if kind == WireKind.PING:
            try:
                health_conn.send((_CONTROL_CALL_ID, WireKind.PONG, None))
            except (BrokenPipeError, OSError):
                return
            continue
        log.warning("%s worker dropped unexpected health-pipe kind %r", role, kind)


def _handle_data_frame(
    data_conn: Any,
    state: WorkerLoopState,
    kind_handlers: dict[WireKind, KindHandler],
    role: str,
) -> bool:
    """Read and dispatch one data-pipe frame. Return False to stop the loop."""
    try:
        call_id, kind, payload = data_conn.recv()
    except EOFError:
        return False
    if kind == WireKind.SHUTDOWN:
        data_conn.send((call_id, WireKind.ACK, None))
        return False
    reply = Reply(data_conn, call_id)
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
