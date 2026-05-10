"""Textual TUI for lilbee -- full-screen interactive knowledge base."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from lilbee.app.services import reset_services
from lilbee.cli.tui.log_routing import setup_tui_log_file


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


def _redirect_native_stderr_to(log_path: Path) -> int | None:
    """Redirect fd 2 to *log_path* so C-level stderr writes don't reach the TUI.

    The Python ``logging.StreamHandler(stderr)`` strip in
    ``_silence_stderr_log_handlers`` only catches Python-level handlers.
    Native dependencies (kreuzberg's vendored tesseract emits
    ``"Detected N diacritics"`` during OCR; pdfium/poppler also chatter)
    write straight to fd 2, which Textual then renders as garbage on top
    of the alternate-screen buffer.

    Returns the saved original fd so the caller can restore it on
    teardown, or ``None`` when the redirect couldn't be installed (in
    which case the caller leaves stderr alone).
    """
    try:
        log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    except OSError:
        return None
    saved = os.dup(2)
    os.dup2(log_fd, 2)
    os.close(log_fd)
    return saved


def _restore_native_stderr(saved_fd: int | None) -> None:
    """Undo ``_redirect_native_stderr_to``; safe to call when ``saved_fd`` is None."""
    if saved_fd is None:
        return
    os.dup2(saved_fd, 2)
    os.close(saved_fd)


def run_tui(*, initial_view: str | None = None) -> None:
    """Launch the full-screen Textual TUI.

    *initial_view* deep-links to a named view (e.g. ``"Catalog"``) after
    the default chat screen is mounted. Used by ``lilbee model browse``.
    """
    # heavy: cli.sync transitively imports ingest -> store -> pyarrow (~1.8s
    # cold-start on bare runners); only needed at TUI shutdown. (bb-oae5)
    from lilbee.cli.sync import shutdown_executor
    from lilbee.cli.tui.app import LilbeeApp

    log_path = setup_tui_log_file()
    _silence_stderr_log_handlers()
    saved_stderr_fd = _redirect_native_stderr_to(log_path)

    app = LilbeeApp(initial_view=initial_view)
    try:
        app.run()
    except KeyboardInterrupt:
        pass  # Ctrl-C exits the TUI; cleanup runs in the finally block
    finally:
        _restore_native_stderr(saved_stderr_fd)
        shutdown_executor()
        reset_services()
