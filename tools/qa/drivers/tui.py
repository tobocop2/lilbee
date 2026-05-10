"""Cross-platform TUI driver: PTY (pywinpty / pty.openpty + subprocess) + pyte Screen.

Streaming-output assertions go to the SSE layer, not pyte. This driver asserts
on visible state at stable points (e.g. after a chat round-trip completes).

POSIX backend: pty.openpty() for the master/slave pair, then subprocess.Popen
with the slave as stdin/stdout/stderr and start_new_session=True. subprocess
uses os.posix_spawn() when conditions are right (Python 3.11+, no preexec_fn,
default close_fds), which avoids the multi-threaded fork issue that
ptyprocess and pty.fork() exhibit. Lets pytest-xdist + threaded MCP/HTTP
fixtures coexist with TuiSession spawns without the deadlock risk.

Windows backend: pywinpty (ConPTY). Single-threaded fork is not a concern
on Windows; pywinpty uses CreateProcess under the hood.
"""

from __future__ import annotations

import contextlib
import os
import signal
import struct
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Self

import pyte

# Posix-only stdlib modules. Windows takes the pywinpty branch below and
# never touches these; importing them at module top breaks `import drivers.tui`
# on the Windows lane.
if sys.platform != "win32":
    import fcntl
    import pty
    import termios

_DEFAULT_COLS = 120
_DEFAULT_ROWS = 40
_READ_CHUNK_BYTES = 65536
_DEFAULT_POLL_INTERVAL = 0.2
_TERM_GRACE_SECONDS = 1.0


class _PosixPty:
    """POSIX PTY child via subprocess.Popen + pty.openpty().

    Avoids os.forkpty() so we don't trip the Python 3.14 multi-threaded
    DeprecationWarning and the underlying deadlock risk. subprocess.Popen
    uses posix_spawn() under the hood on Python 3.11+ when no preexec_fn
    is set and close_fds defaults to True.

    Mirrors enough of pywinpty.PtyProcess for the TuiSession code below
    to use a single API: classmethod spawn(), write(bytes), read(int),
    pid attribute, fd attribute, isalive(), terminate().
    """

    def __init__(self, master_fd: int, proc: subprocess.Popen[bytes]) -> None:
        self._master_fd = master_fd
        self._proc = proc

    @classmethod
    def spawn(
        cls,
        cmd: list[str],
        *,
        dimensions: tuple[int, int] = (_DEFAULT_ROWS, _DEFAULT_COLS),
        env: Mapping[str, str] | None = None,
    ) -> _PosixPty:
        rows, cols = dimensions
        master_fd, slave_fd = pty.openpty()
        try:
            fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
            popen_env = dict(env) if env is not None else None
            proc = subprocess.Popen(
                cmd,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                start_new_session=True,
                close_fds=True,
                env=popen_env,
            )
        except BaseException:
            os.close(master_fd)
            os.close(slave_fd)
            raise
        os.close(slave_fd)
        return cls(master_fd, proc)

    @property
    def pid(self) -> int:
        return self._proc.pid

    @property
    def fd(self) -> int:
        return self._master_fd

    def write(self, data: bytes) -> int:
        return os.write(self._master_fd, data)

    def read(self, size: int) -> bytes:
        return os.read(self._master_fd, size)

    def isalive(self) -> bool:
        return self._proc.poll() is None

    def terminate(self, force: bool = False) -> None:
        with contextlib.suppress(ProcessLookupError, OSError):
            self._proc.terminate()
        if not force:
            return
        try:
            self._proc.wait(timeout=_TERM_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError, OSError):
                self._proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                self._proc.wait(timeout=_TERM_GRACE_SECONDS)

    def close_master(self) -> None:
        with contextlib.suppress(OSError):
            os.close(self._master_fd)


if sys.platform == "win32":
    # `winpty` is a Windows-only dep and not on the import path of mypy
    # runs on POSIX hosts; `import-not-found` is the correct silence here.
    from winpty import PtyProcess  # type: ignore[import-not-found]
else:
    # The two backends share the same `pid`/`fd`/`write`/`read`/`isalive`/
    # `terminate` shape that `TuiSession` uses, but mypy can't model the
    # platform-conditional binding as a type alias without a Protocol the
    # winpty stub doesn't satisfy. Live with the assignment-misc silence.
    PtyProcess = _PosixPty  # type: ignore[misc,assignment]


def _is_pid_alive(pid: int) -> bool:
    """Non-blocking POSIX liveness via WNOHANG waitpid + kill(0)."""
    try:
        wpid, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return False
    if wpid == pid:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, OSError):
        return False
    return True


def _safe_isalive_winpty(proc: PtyProcess) -> bool:
    """Wrap ``proc.isalive()`` with the OSError/ValueError swallow that
    pywinpty occasionally raises mid-teardown (the underlying ConPTY
    socket can be closed before isalive's poll lands)."""
    try:
        return bool(proc.isalive())
    except (OSError, ValueError):
        return False


class TuiSession:
    """Drive a TUI binary in a real PTY and assert against the rendered text grid."""

    def __init__(
        self,
        cmd: list[str],
        *,
        cols: int = _DEFAULT_COLS,
        rows: int = _DEFAULT_ROWS,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._screen = pyte.Screen(cols, rows)
        self._stream = pyte.ByteStream()
        self._stream.attach(self._screen)
        self._proc = PtyProcess.spawn(
            cmd,
            dimensions=(rows, cols),
            env=dict(env) if env is not None else None,
        )
        # PTY read() is blocking by default on both backends. POSIX exposes a
        # raw fd; pywinpty exposes a socket via .fileobj (it uses ConPTY +
        # an AF_INET loopback socket internally). Force both into non-blocking
        # mode so a quiet TUI doesn't stall the harness; _drain_once swallows
        # BlockingIOError.
        if sys.platform == "win32":
            with contextlib.suppress(AttributeError, OSError):
                self._proc.fileobj.setblocking(False)
        else:
            with contextlib.suppress(AttributeError, OSError):
                os.set_blocking(self._proc.fd, False)

    def send(self, keys: str) -> None:
        """Write keys to the PTY input. Use '\\r' to submit a line.

        pywinpty.PtyProcess.write expects str on Windows; the POSIX backend
        takes bytes. Pick the right type per backend.
        """
        if sys.platform == "win32":
            self._proc.write(keys)
        else:
            self._proc.write(keys.encode())

    def _drain_once(self) -> None:
        try:
            chunk = self._proc.read(_READ_CHUNK_BYTES)
        except (BlockingIOError, TimeoutError, EOFError, OSError):
            return
        if not chunk:
            return
        payload = chunk if isinstance(chunk, bytes) else chunk.encode("utf-8", "replace")
        self._stream.feed(payload)

    def text(self) -> str:
        """Return the currently-visible screen as one searchable string."""
        self._drain_once()
        return "\n".join(self._screen.display)

    def visible(self) -> list[str]:
        """Return only the currently-visible rows."""
        self._drain_once()
        return list(self._screen.display)

    def cursor(self) -> tuple[int, int]:
        """Return (row, col) of the cursor on the visible screen."""
        self._drain_once()
        return self._screen.cursor.y, self._screen.cursor.x

    def wait_for(
        self,
        needle: str,
        *,
        timeout: float = 30.0,
        poll: float = _DEFAULT_POLL_INTERVAL,
        case_sensitive: bool = False,
    ) -> None:
        deadline = time.monotonic() + timeout
        match = needle if case_sensitive else needle.lower()
        while time.monotonic() < deadline:
            haystack = self.text() if case_sensitive else self.text().lower()
            if match in haystack:
                return
            time.sleep(poll)
        raise AssertionError(
            f"timeout waiting for {needle!r} after {timeout:.1f}s. visible:\n"
            + "\n".join(self.visible())
        )

    def screenshot(self, path: Path) -> None:
        path.write_text(self.text(), encoding="utf-8")

    def is_alive(self) -> bool:
        """Non-blocking liveness check."""
        if sys.platform == "win32":
            return _safe_isalive_winpty(self._proc)
        return _is_pid_alive(self._proc.pid)

    def close(self) -> None:
        """Terminate without hanging on a stubborn TUI."""
        if sys.platform == "win32":
            with contextlib.suppress(Exception):
                self._proc.terminate(force=True)
            return
        pid = self._proc.pid
        with contextlib.suppress(ProcessLookupError, OSError):
            os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + _TERM_GRACE_SECONDS
        while time.monotonic() < deadline:
            try:
                wpid, _ = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                break
            if wpid == pid:
                break
            time.sleep(0.1)
        else:
            with contextlib.suppress(ProcessLookupError, OSError):
                os.kill(pid, signal.SIGKILL)
            with contextlib.suppress(ChildProcessError, OSError):
                os.waitpid(pid, 0)
        # Close the master fd on the POSIX backend so the PTY's resources
        # don't leak across tests.
        if isinstance(self._proc, _PosixPty):
            self._proc.close_master()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
