"""Cross-role helpers for worker subprocesses.

Bootstraps the per-role workers (embed, chat, rerank, vision) and runs
the recv loop with shared ping/shutdown handling via :func:`run_worker`.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
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

    ``stream_in_flight`` is flipped on for the emission window of a
    streaming response (via :func:`stream_window`); the dispatcher
    silently drops pings dispatched during that window so no pong frame
    interleaves with the stream output. Combined with the parent-side
    ``stream`` reader filter that consumes orphan pongs and the parent
    accessor's ``ping`` skip when the channel reports in-flight work,
    this completes the worker-pool defence against bb-ubnm.
    """

    session: Any
    stream_in_flight: bool = False


@contextmanager
def stream_window(state: WorkerLoopState) -> Iterator[None]:
    """Mark *state* as actively streaming for the duration of the block.

    The flag is always cleared on exit so a streaming error does not
    leave the worker permanently ping-deaf.
    """
    state.stream_in_flight = True
    try:
        yield
    finally:
        state.stream_in_flight = False


KindHandler = Callable[[Any, Any, WorkerLoopState], None]
"""Per-role handler signature: ``(conn, payload, state) -> None``.

Handlers reach their role-specific session via ``state.session``.
"""


def run_worker(
    conn: Any,
    abort_flag: Any,
    role_config: RoleConfig,
    *,
    session_factory: Callable[[RoleConfig, Any], Any],
    kind_handlers: dict[str, KindHandler],
) -> None:
    """Bootstrap stdio + logging, then run the recv loop until shutdown.

    Built-in kinds (ping/shutdown) are handled here; *kind_handlers*
    maps role-specific kinds (e.g. ``"embed"``, ``"chat"``) to the
    function that processes one such request. Unknown kinds reply with
    a serialized ``ValueError``. The worker's session is built once via
    *session_factory* and closed in the ``finally`` clause.
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
            if not conn.poll(timeout=_POLL_TIMEOUT_S):
                continue
            try:
                kind, payload = conn.recv()
            except EOFError:
                return
            if not _dispatch_kind(conn, kind, payload, state, kind_handlers, role_config.role):
                return
    finally:
        # session.close() swallows its own teardown errors per role-specific
        # contract. conn.close() can raise if the parent already tore down.
        state.session.close()
        with contextlib.suppress(Exception):
            conn.close()


def _dispatch_kind(
    conn: Any,
    kind: str,
    payload: Any,
    state: WorkerLoopState,
    kind_handlers: dict[str, KindHandler],
    role: str,
) -> bool:
    """Handle one request. Return False to stop the worker loop."""
    if kind == SHUTDOWN_KIND:
        conn.send((ACK_KIND, None))
        return False
    if kind == PING_KIND:
        if state.stream_in_flight:
            # Drop the ping rather than emit a pong frame the parent's
            # stream consumer would read out of band. The next tick re-pings.
            return True
        conn.send((PONG_KIND, None))
        return True
    handler = kind_handlers.get(kind)
    if handler is not None:
        handler(conn, payload, state)
        return True
    try:
        raise ValueError(f"{role} worker received unknown kind {kind!r}")
    except ValueError as exc:
        conn.send((ERROR_KIND, _serialize_exception(exc)))
    return True


__all__ = [
    "KindHandler",
    "WorkerLoopState",
    "configure_worker_logging",
    "redirect_stdio_to_devnull",
    "run_worker",
    "stream_window",
]
