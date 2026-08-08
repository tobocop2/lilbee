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
    # Resolving the app is guarded separately from running *fn*. Textual's
    # ``node.app`` reads a contextvar that is unset in a plain thread and then
    # walks ``node._parent``, which raises AttributeError on a node whose
    # MessagePump state is gone: seen in CI as "'ChatScreen' object has no
    # attribute '_MessagePump__parent'" from a worker outliving its screen.
    # That is the same "app is gone" case this wrapper exists for, but it cannot
    # be folded into the except below: an AttributeError raised *inside* fn must
    # still propagate, which is what the docstring promises.
    try:
        app = node.app
    except (AttributeError, RuntimeError) as exc:
        log.debug("call_from_thread found no app for %s: %s", getattr(fn, "__name__", fn), exc)
        return
    try:
        app.call_from_thread(fn, *args, **kwargs)
    except (OSError, RuntimeError) as exc:
        # Only the shutdown signals: OSError when the message queue is closed,
        # RuntimeError (incl. NoActiveAppError) when the app is no longer running.
        # A genuine exception raised inside *fn* propagates so the bug surfaces
        # instead of being silently swallowed.
        log.debug(
            "call_from_thread dropped %s: %s",
            getattr(fn, "__name__", fn),
            exc,
        )
