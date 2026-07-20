"""Textual TUI for lilbee -- full-screen interactive knowledge base."""

from __future__ import annotations

import io
import logging
import os
import sys
from dataclasses import dataclass
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


@dataclass
class _StderrRedirect:
    """State captured by ``_redirect_native_stderr_to`` for restoration on exit."""

    saved_fd: int
    saved_sys_stderr: object
    saved_sys_dunder_stderr: object


def _redirect_native_stderr_to(log_path: Path) -> _StderrRedirect | None:
    """Send native fd-2 writes to *log_path* without breaking Textual's render.

    Textual's Linux/macOS driver writes its alternate-screen ANSI to
    ``sys.__stderr__`` (see
    ``textual.drivers.linux_driver.LinuxDriver.write``). Native deps
    like xberg's vendored tesseract write directly to fd 2 and leak
    onto the same buffer, e.g. "Detected N diacritics", which corrupts
    the TUI.

    Strategy: dup the original fd 2 to a saved fd, repoint
    ``sys.__stderr__`` and ``sys.stderr`` at that saved fd so Textual
    keeps drawing to the real terminal, then dup2 fd 2 itself to the
    log file so any fd-2 writer (xberg, tesseract, poppler) lands
    in ``tui.log`` instead of on top of the screen.
    """
    try:
        log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    except OSError:
        return None
    saved_fd = os.dup(2)
    terminal_stderr = io.TextIOWrapper(
        io.FileIO(saved_fd, "w", closefd=False),
        encoding=sys.stderr.encoding or "utf-8",
        errors="replace",
        write_through=True,
    )
    saved_sys_stderr = sys.stderr
    saved_sys_dunder_stderr = sys.__stderr__
    sys.stderr = terminal_stderr
    sys.__stderr__ = terminal_stderr  # type: ignore[misc]
    os.dup2(log_fd, 2)
    os.close(log_fd)
    return _StderrRedirect(
        saved_fd=saved_fd,
        saved_sys_stderr=saved_sys_stderr,
        saved_sys_dunder_stderr=saved_sys_dunder_stderr,
    )


def _restore_native_stderr(redirect: _StderrRedirect | None) -> None:
    """Undo ``_redirect_native_stderr_to``; safe to call when *redirect* is None."""
    if redirect is None:
        return
    os.dup2(redirect.saved_fd, 2)
    os.close(redirect.saved_fd)
    sys.stderr = redirect.saved_sys_stderr  # type: ignore[assignment]
    sys.__stderr__ = redirect.saved_sys_dunder_stderr  # type: ignore[misc,assignment]


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
    stderr_redirect = _redirect_native_stderr_to(log_path)

    app = LilbeeApp(initial_view=initial_view)
    try:
        app.run()
    except KeyboardInterrupt:
        pass  # Ctrl-C exits the TUI; cleanup runs in the finally block
    finally:
        _restore_native_stderr(stderr_redirect)
        shutdown_executor()
        reset_services()
