"""Thread-safe helpers for posting from @work(thread=True) workers to the main thread.

Textual's call_from_thread raises OSError when the app's message queue
has already been closed during shutdown. Since workers run in daemon
threads, they can outlive the app. This module provides a drop-in
wrapper that silently drops calls when the app is gone.
"""

from __future__ import annotations

import logging
from typing import Any

from textual.dom import DOMNode

log = logging.getLogger(__name__)


def call_from_thread(node: DOMNode, fn: Any, *args: Any, **kwargs: Any) -> None:
    """Post *fn* to the main thread via the app.

    Drops the call (does not crash the worker) when the target node's app
    is no longer reachable, e.g. during shutdown or after a screen was
    replaced. Logs at warning level because a silent drop masks real
    bugs: a previous iteration of this project lost hours to progress
    updates that were being swallowed here without a trace.
    """
    try:
        node.app.call_from_thread(fn, *args, **kwargs)
    except Exception as exc:
        log.warning(
            "call_from_thread dropped %s: %s",
            getattr(fn, "__name__", fn),
            exc,
        )
