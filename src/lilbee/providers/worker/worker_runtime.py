"""Cross-role helpers for worker subprocesses.

Each per-role worker (embed, chat, rerank, vision) bootstraps its
process the same way: silence the C-level stdio that llama-cpp emits
and append worker logs to ``$LILBEE_DATA/logs/worker-<role>.log`` if
the env var is set. Centralizing these here keeps the per-role files
focused on inference semantics and avoids cross-imports of
``embed_worker``'s private helpers from chat/rerank/vision.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys


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


__all__ = ["configure_worker_logging", "redirect_stdio_to_devnull"]
