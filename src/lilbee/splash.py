"""Splash animation lifecycle — starts and stops the animation subprocess.

The animation itself lives in ``_splash_runner.py`` (stdlib-only, zero lilbee
imports). This module manages the subprocess, pipe-based IPC, and cleanup.

IPC uses an OS pipe: parent holds the write end, child polls the read end.
When the parent closes the write end (or dies), the child sees EOF and exits.
This guarantees no orphan processes — the OS closes the pipe on parent death.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import subprocess
import sys
from dataclasses import dataclass

_SPLASH_FD_ENV = "_LILBEE_SPLASH_FD"

_SHOW_CURSOR = "\033[?25h"

_STOP_TIMEOUT = 3.0


@dataclass
class SplashHandle:
    """Opaque handle returned by ``start()`` for use with ``stop()``."""

    process: subprocess.Popen[bytes]
    write_fd: int


_active_handle: SplashHandle | None = None


def _should_skip() -> bool:
    """Return True when the splash animation should be suppressed."""
    if not os.isatty(2):
        return True
    return bool(os.environ.get("LILBEE_NO_SPLASH", ""))


def start() -> SplashHandle | None:
    """Launch the splash animation subprocess.

    Returns a handle for ``stop()``, or None if the splash was skipped.
    The caller must eventually call ``stop(handle)`` to clean up.
    """
    global _active_handle

    if _should_skip():
        return None

    read_fd, write_fd = os.pipe()

    if sys.platform == "win32":
        os.set_inheritable(read_fd, True)

    proc = subprocess.Popen(
        [sys.executable, "-m", "lilbee._splash_runner", str(read_fd)],
        close_fds=False,
        stderr=None,
        stdout=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )

    os.close(read_fd)

    os.environ[_SPLASH_FD_ENV] = str(write_fd)

    handle = SplashHandle(process=proc, write_fd=write_fd)
    _active_handle = handle

    atexit.register(_atexit_cleanup)

    return handle


def stop(handle: SplashHandle | None) -> None:
    """Stop the splash animation and wait for the subprocess to exit."""
    global _active_handle

    if handle is None:
        return

    _close_write_fd(handle.write_fd)

    try:
        handle.process.wait(timeout=_STOP_TIMEOUT)
    except subprocess.TimeoutExpired:
        handle.process.kill()
        handle.process.wait(timeout=1.0)

    os.environ.pop(_SPLASH_FD_ENV, None)

    _active_handle = None

    _restore_cursor()


def dismiss() -> None:
    """Signal the splash to stop from the TUI side.

    Called by the chat screen's ``on_show()`` to dismiss the splash once
    the TUI is ready to paint. Reads the pipe fd from the environment
    variable and closes it — the subprocess sees EOF and exits.
    """
    fd_str = os.environ.pop(_SPLASH_FD_ENV, None)
    if fd_str is None:
        return
    _close_write_fd(int(fd_str))


def _close_write_fd(fd: int) -> None:
    """Close a pipe write fd, ignoring errors if already closed."""
    with contextlib.suppress(OSError):
        os.close(fd)


def _restore_cursor() -> None:
    """Belt-and-suspenders cursor restore on stderr."""
    try:
        sys.stderr.write(_SHOW_CURSOR)
        sys.stderr.flush()
    except OSError:
        pass


def _atexit_cleanup() -> None:
    """Last-resort cleanup if stop() was never called."""
    if _active_handle is not None:
        stop(_active_handle)
