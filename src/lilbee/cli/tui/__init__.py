"""Textual TUI for lilbee -- full-screen interactive knowledge base."""

from __future__ import annotations

import logging
import os
import sys

from lilbee.cli.sync import shutdown_executor
from lilbee.services import reset_services


def _silence_stderr_log_handlers() -> None:
    """Drop stderr/stdout-streaming log handlers before mounting the TUI.

    Textual writes its UI to stderr; any parallel writer to the same fd
    corrupts the screen (bb-82ce: 'WARNING lilbee.concepts: Concept graph
    disabled: spaCy model unavailable' bleeding into the bottom-bar
    box-drawing characters). The CLI entrypoint installs a basicConfig
    stderr StreamHandler at WARNING level; remove it for the duration of
    the TUI session. Textual users can still see lilbee logs via
    ``lilbee --log-level=DEBUG`` against a non-TUI subcommand.

    Targets ``StreamHandler`` whose ``.stream`` is sys.stderr / sys.stdout.
    ``FileHandler`` (a ``StreamHandler`` subclass whose ``.stream`` is the
    open log file) is skipped: file writes don't share an fd with the
    TUI render target.
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
            # Rapid Ctrl+C during shutdown — force exit immediately
            os._exit(1)
