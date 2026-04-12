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
    """Post *fn* to the main thread via the app, dropping silently on shutdown."""
    try:
        node.app.call_from_thread(fn, *args, **kwargs)
    except Exception:
        log.debug(
            "call_from_thread dropped (app shutting down): %s",
            getattr(fn, "__name__", fn),
        )
