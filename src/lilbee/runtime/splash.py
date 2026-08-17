"""Splash animation lifecycle: starts and stops the animation subprocess.

The animation itself lives in ``_splash_runner.py`` (stdlib-only, zero lilbee
imports). This module manages the subprocess, pipe-based IPC, and cleanup.

IPC uses an OS pipe: parent holds the write end, child polls the read end.
When the parent closes the write end (or dies), the child sees EOF and exits.
This guarantees no orphan processes: the OS closes the pipe on parent death.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

from lilbee.runtime._splash_runner import TAKEOVER_BYTE

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


def _spawn_args(read_fd: int) -> tuple[str, dict[str, Any]]:
    """Per-platform argv reference and Popen kwargs for the pipe's read end.

    POSIX passes the fd itself; pass_fds keeps only read_fd open in the child
    (close_fds=False would leak all open descriptors, including any held by
    libraries). subprocess cannot pass fds on Windows, so the child gets the
    pipe's OS handle number on argv instead and reopens it as a fd
    (msvcrt.open_osfhandle in the runner; inherited handles keep their value
    in the child). handle_list restricts inheritance to exactly this handle,
    the same no-leak guarantee pass_fds gives on POSIX.
    """
    if sys.platform == "win32":  # pragma: no cover - Windows-only, covered on the Windows CI leg
        import msvcrt

        handle = msvcrt.get_osfhandle(read_fd)
        os.set_handle_inheritable(handle, True)
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.lpAttributeList = {"handle_list": [handle]}
        return str(handle), {"startupinfo": startupinfo}
    os.set_inheritable(read_fd, True)
    return str(read_fd), {"pass_fds": (read_fd,)}


def start() -> SplashHandle | None:
    """Launch the splash animation subprocess.
    Returns a handle for ``stop()``, or None if the splash was skipped.
    The caller must eventually call ``stop(handle)`` to clean up.
    """
    global _active_handle

    if _should_skip():
        return None

    read_fd, write_fd = os.pipe()
    pipe_ref, spawn_kwargs = _spawn_args(read_fd)

    # Trusted: sys.executable is this interpreter, module path is static,
    # the one runtime value (pipe_ref) derives from an int from os.pipe().
    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, "-m", "lilbee.runtime._splash_runner", pipe_ref],
        stderr=None,
        stdout=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        **spawn_kwargs,
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
    Called once the TUI is ready to paint. Writes the takeover byte so the
    subprocess exits without touching the terminal (its frame clear and
    cursor-show would land on the Textual alt-screen and leave a visible
    cursor for the whole session), then closes the pipe, waits for the
    subprocess, and clears the active handle so ``atexit`` does not re-run
    ``stop()``.
    """
    global _active_handle

    fd_str = os.environ.pop(_SPLASH_FD_ENV, None)
    if fd_str is not None:
        _signal_takeover(int(fd_str))
        _close_write_fd(int(fd_str))

    handle = _active_handle
    if handle is None:
        return
    _active_handle = None

    # Signal and close the write end. This may double-signal/close the same
    # fd that the env-var path already handled; both helpers suppress
    # OSError so that is harmless.
    # No _restore_cursor() here: we are inside Textual's alt-screen,
    # where writing cursor-show would produce a visible artifact.
    _signal_takeover(handle.write_fd)
    _close_write_fd(handle.write_fd)
    try:
        handle.process.wait(timeout=_STOP_TIMEOUT)
    except subprocess.TimeoutExpired:
        handle.process.kill()
        handle.process.wait(timeout=1.0)


def _close_write_fd(fd: int) -> None:
    """Close a pipe write fd, ignoring errors if already closed."""
    with contextlib.suppress(OSError):
        os.close(fd)


def _signal_takeover(fd: int) -> None:
    """Send the TUI-takeover byte, ignoring errors if the pipe is gone."""
    with contextlib.suppress(OSError):
        os.write(fd, TAKEOVER_BYTE)


def _restore_cursor() -> None:
    """Belt-and-suspenders cursor restore on stderr."""
    try:
        sys.stderr.write(_SHOW_CURSOR)
        sys.stderr.flush()
    except OSError:
        pass  # stderr may be closed during interpreter shutdown


def _atexit_cleanup() -> None:
    """Last-resort cleanup if stop() was never called."""
    if _active_handle is not None:
        stop(_active_handle)
