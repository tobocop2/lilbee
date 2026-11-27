"""Textual TUI for lilbee -- full-screen interactive knowledge base."""

from __future__ import annotations

import asyncio
import io
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from lilbee.app.services import (
    exit_code_of,
    exit_when_stragglers_would_hang,
    loop_takes_signals,
    reset_services_on_exit,
)
from lilbee.cli.log_routing import detach_rotating_file_handler
from lilbee.cli.tui.log_routing import setup_tui_log_file
from lilbee.core.security import OWNER_ONLY_MODE
from lilbee.runtime.asyncio_loop import run_without_join

if TYPE_CHECKING:
    from lilbee.cli.tui.app import LilbeeApp


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


# xterm private mode 1036 (metaSendsEscape): Meta+key arrives as ESC plus the key, which
# Textual reads, instead of one 8-bit byte that its UTF-8 decoder rejects. ``s`` saves the mode
# and ``r`` restores it. Terminals that do not know the mode ignore all three sequences.
_META_SENDS_ESCAPE_SAVE_AND_SET = "\x1b[?1036s\x1b[?1036h"
_META_SENDS_ESCAPE_RESTORE = "\x1b[?1036r"


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
        log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, OWNER_ONLY_MODE)
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


def _write_to_terminal(sequence: str) -> None:
    """Write *sequence* to the terminal; a stream that is not a terminal gets nothing."""
    stream = sys.__stderr__
    if stream is None or not stream.isatty():
        return
    try:
        stream.write(sequence)
        stream.flush()
    except (OSError, ValueError):
        pass  # The terminal is gone (hangup, closed stream); there is nothing to set or restore.


def _prestart_mp_resource_tracker() -> None:
    """Start the multiprocessing resource tracker while stderr has a real descriptor.

    Textual replaces ``sys.stderr`` with a stream whose ``fileno()`` is -1, and a
    tracker launched after that crashes with ``bad value(s) in fds_to_keep``.
    Frozen builds run it too: ``__main__._dispatch_frozen_child`` intercepts the
    tracker's ``-c`` reinvocation. Windows has no tracker.
    """
    if sys.platform == "win32":
        return
    try:
        from multiprocessing import resource_tracker

        resource_tracker.ensure_running()
    except (OSError, RuntimeError, ValueError, ImportError):
        # Best effort: the first worker spawn surfaces a real error at call time.
        pass


async def _run_app(app: LilbeeApp) -> None:
    """Run *app* while its loop takes the routed signals.

    A plain handler raises the exit inside whatever task step runs, such as a
    timer's tick, and that task keeps it as an exception nobody reads. A loop
    callback runs between steps, so the exit leaves the loop and no task holds it.
    """
    with loop_takes_signals(asyncio.get_running_loop()):
        await app.run_async()


def run_tui(*, initial_view: str | None = None) -> None:
    """Launch the full-screen Textual TUI.

    *initial_view* deep-links to a named view (e.g. ``"Catalog"``) after
    the default chat screen is mounted. Used by ``lilbee model browse``.
    A non-zero ``return_code`` of the app ends the process with that status, and a
    signal's SystemExit keeps its status through the forced exit for stuck threads.
    """
    # heavy: cli.sync transitively imports ingest -> store -> pyarrow (~1.8s
    # cold-start on bare runners); only needed at TUI shutdown. (bb-oae5)
    from lilbee.cli.sync import shutdown_executor
    from lilbee.cli.tui.app import LilbeeApp

    _prestart_mp_resource_tracker()
    log_path = setup_tui_log_file()
    _silence_stderr_log_handlers()
    stderr_redirect = _redirect_native_stderr_to(log_path)

    app = LilbeeApp(initial_view=initial_view)
    status = 0
    _write_to_terminal(_META_SENDS_ESCAPE_SAVE_AND_SET)
    try:
        # Textual's run would use asyncio.run, which joins a stuck worker with no bound on
        # 3.11; this leaves it to the straggler exit below.
        run_without_join(_run_app(app))
        # None means the app never reached its exit.
        status = app.return_code or 0
    except KeyboardInterrupt:
        pass  # Ctrl-C exits the TUI; cleanup runs in the finally block
    except SystemExit as stop:
        # A SIGTERM ends the run with 128 + signum; the codes map as Python maps them.
        status = exit_code_of(stop)
        raise
    finally:
        _write_to_terminal(_META_SENDS_ESCAPE_RESTORE)
        try:
            # Tasks stop through their own cancel while the executors and services live.
            app.task_bar.stop_all()
            _restore_native_stderr(stderr_redirect)
            shutdown_executor()
            reset_services_on_exit()
            exit_when_stragglers_would_hang(status)
        finally:
            # The tui.log handler lives to the end of the session, the straggler wait included.
            detach_rotating_file_handler(log_path)
    if status:
        raise SystemExit(status)
