"""Cross-role helpers for worker subprocesses.

Bootstraps the per-role workers (embed, chat, rerank, vision) and runs
the recv loop with shared ping/shutdown handling via :func:`run_worker`.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import wait
from typing import Any

from lilbee.providers.worker.transport import RoleConfig
from lilbee.providers.worker.transport_pipe import _serialize_exception
from lilbee.providers.worker.wire_kinds import (
    ACK_KIND,
    ERROR_KIND,
    PING_KIND,
    PONG_KIND,
    SHUTDOWN_KIND,
)

log = logging.getLogger(__name__)

_POLL_TIMEOUT_S = 0.5
"""Bounded poll so the worker can react to shutdown within a tick instead
of blocking forever on bare recv."""


def redirect_stdio_to_devnull() -> None:  # pragma: no cover - subprocess fd swap
    """Send stdout/stderr to /dev/null so llama-cpp's C-level prints stay quiet.

    The pool transport speaks pickle over a pipe; nothing the worker
    process writes to fd 1 or fd 2 is ever consumed by the parent.
    Carries ``# pragma: no cover`` because closing fds 1/2 inside the
    pytest-runner process would deadlock pytest-xdist.
    """
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


@dataclass
class WorkerLoopState:
    """Per-loop state shared by the dispatcher and the role's handlers.

    Holds the role's lazy-loaded session. Each handler receives a
    reference and pulls its model via ``state.session``.
    """

    session: Any


KindHandler = Callable[[Any, Any, WorkerLoopState], None]
"""Per-role handler signature: ``(data_conn, payload, state) -> None``.

Handlers reach their role-specific session via ``state.session``.
"""


def run_worker(
    data_conn: Any,
    health_conn: Any,
    abort_flag: Any,
    role_config: RoleConfig,
    *,
    session_factory: Callable[[RoleConfig, Any], Any],
    kind_handlers: dict[str, KindHandler],
) -> None:
    """Bootstrap stdio + logging, then run the recv loop until shutdown.

    Two pipes per worker: ``data_conn`` carries call/stream/shutdown,
    ``health_conn`` carries ping/pong. The loop multiplexes the two via
    :func:`multiprocessing.connection.wait` so neither stalls the other.
    Built-in kinds (ping on health, shutdown on data) are handled here;
    *kind_handlers* maps role-specific kinds (e.g. ``"embed"``, ``"chat"``)
    to the function that processes one such request. Unknown kinds reply
    with a serialized ``ValueError``.
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
    try:
        while True:
            ready = wait([data_conn, health_conn], timeout=_POLL_TIMEOUT_S)
            if not ready:
                continue
            if data_conn in ready and not _handle_data_frame(
                data_conn, state, kind_handlers, role_config.role
            ):
                return
            if health_conn in ready:
                _handle_health_frame(health_conn, role_config.role)
    finally:
        # session.close() swallows its own teardown errors per role-specific
        # contract. conn.close() can raise if the parent already tore down.
        state.session.close()
        with contextlib.suppress(Exception):
            data_conn.close()
        with contextlib.suppress(Exception):
            health_conn.close()


def _handle_data_frame(
    data_conn: Any,
    state: WorkerLoopState,
    kind_handlers: dict[str, KindHandler],
    role: str,
) -> bool:
    """Read and dispatch one data-pipe frame. Return False to stop the loop."""
    try:
        kind, payload = data_conn.recv()
    except EOFError:
        return False
    if kind == SHUTDOWN_KIND:
        data_conn.send((ACK_KIND, None))
        return False
    handler = kind_handlers.get(kind)
    if handler is not None:
        handler(data_conn, payload, state)
        return True
    try:
        raise ValueError(f"{role} worker received unknown kind {kind!r}")
    except ValueError as exc:
        data_conn.send((ERROR_KIND, _serialize_exception(exc)))
    return True


def _handle_health_frame(health_conn: Any, role: str) -> None:
    """Read one health-pipe frame; reply pong on ping, log + drop on anything else."""
    try:
        kind, _ = health_conn.recv()
    except EOFError:
        return
    if kind == PING_KIND:
        health_conn.send((PONG_KIND, None))
        return
    log.warning("%s worker dropped unexpected health-pipe kind %r", role, kind)


__all__ = [
    "KindHandler",
    "WorkerLoopState",
    "configure_worker_logging",
    "redirect_stdio_to_devnull",
    "run_worker",
]
