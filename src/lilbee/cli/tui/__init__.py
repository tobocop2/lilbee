"""Textual TUI for lilbee -- full-screen interactive knowledge base."""

from __future__ import annotations

import logging
import sys

from lilbee.cli.tui.log_routing import setup_tui_log_file
from lilbee.core.services import reset_services


def _silence_stderr_log_handlers() -> None:
    """Drop stderr/stdout-streaming log handlers before mounting the TUI. (bb-82ce)

    lilbee logs route to ``cfg.data_root/logs/tui.log`` for the duration of
    the TUI session via :func:`setup_tui_log_file`. FileHandler instances
    are skipped explicitly here so that file route survives.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if not isinstance(handler, logging.StreamHandler):
            continue
        if isinstance(handler, logging.FileHandler):
            continue
        if handler.stream in (sys.stderr, sys.stdout):
            root.removeHandler(handler)


def run_tui(*, initial_view: str | None = None) -> None:
    """Launch the full-screen Textual TUI.

    *initial_view* deep-links to a named view (e.g. ``"Catalog"``) after
    the default chat screen is mounted. Used by ``lilbee model browse``.
    """
    # heavy: cli.sync transitively imports ingest -> store -> pyarrow (~1.8s
    # cold-start on bare runners); only needed at TUI shutdown. (bb-oae5)
    from lilbee.cli.sync import shutdown_executor
    from lilbee.cli.tui.app import LilbeeApp

    setup_tui_log_file()
    _silence_stderr_log_handlers()

    app = LilbeeApp(initial_view=initial_view)
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_executor()
        reset_services()
