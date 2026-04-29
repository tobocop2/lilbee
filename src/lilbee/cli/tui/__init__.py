"""Textual TUI for lilbee -- full-screen interactive knowledge base."""

from __future__ import annotations

import logging
import os
import sys

from lilbee.cli.sync import shutdown_executor
from lilbee.core.services import reset_services


def _silence_stderr_log_handlers() -> None:
    """Drop stderr/stdout-streaming log handlers before mounting the TUI. (bb-82ce)

    TODO bb-pmyi: stripping handlers loses logs for the session. The proper
    fix routes lilbee.* loggers through Textual's RichLog widget or a
    rotating ~/.lilbee/tui.log so users debugging a TUI session still get
    their logs.

    FileHandler (a StreamHandler subclass whose .stream is the log file)
    is skipped explicitly: only stderr/stdout handlers share an fd with
    the TUI render target.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if not isinstance(handler, logging.StreamHandler):
            continue
        if isinstance(handler, logging.FileHandler):
            continue
        if handler.stream in (sys.stderr, sys.stdout):
            root.removeHandler(handler)


def run_tui(*, auto_sync: bool = False, initial_view: str | None = None) -> None:
    """Launch the full-screen Textual TUI.

    *initial_view* deep-links to a named view (e.g. ``"Catalog"``) after
    the default chat screen is mounted. Used by ``lilbee model browse``.
    """
    from lilbee.cli.tui.app import LilbeeApp

    _silence_stderr_log_handlers()

    app = LilbeeApp(auto_sync=auto_sync, initial_view=initial_view)
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            shutdown_executor()
            reset_services()
        except (KeyboardInterrupt, Exception):
            # Rapid Ctrl+C during shutdown: force exit immediately
            os._exit(1)
