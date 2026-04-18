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
    replaced. Logs at debug so the drop is discoverable without leaking
    warning text into the TUI render (textual's log handler routes
    stderr into the rendered frame). Long-running workers that must
    survive a screen switch should own their state on the app
    (TaskBarController pattern in widgets/task_bar.py) rather than
    relying on this wrapper.
    """
    try:
        node.app.call_from_thread(fn, *args, **kwargs)
    except Exception as exc:
        log.debug(
            "call_from_thread dropped %s: %s",
            getattr(fn, "__name__", fn),
            exc,
        )
