"""Cross-platform TUI driver: PTY (pywinpty/ptyprocess) + pyte Screen.

Streaming-output assertions go to the SSE layer, not pyte. This driver asserts
on visible state at stable points (e.g. after a chat round-trip completes).
"""

from __future__ import annotations

import contextlib
import os
import signal
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Self

import pyte

if sys.platform == "win32":
    from winpty import PtyProcess  # type: ignore[import-not-found]
else:
    from ptyprocess import PtyProcess  # type: ignore[import-not-found]


_DEFAULT_COLS = 120
_DEFAULT_ROWS = 40
_READ_CHUNK_BYTES = 65536
_DEFAULT_POLL_INTERVAL = 0.2


def worker_port_offset() -> int:
    """Translate PYTEST_XDIST_WORKER (gw0/gw1/.../master) into a port offset."""
    raw = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
    if raw == "master" or not raw.startswith("gw"):
        return 0
    return int(raw.removeprefix("gw"))


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

        pywinpty.PtyProcess.write expects str on Windows; ptyprocess.PtyProcess.write
        expects bytes on POSIX. Pick the right type per backend.
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
        """Non-blocking liveness check. Bypasses ptyprocess.isalive on POSIX
        because that blocks on os.waitpid once self.terminated flips True."""
        if sys.platform == "win32":
            return _safe_isalive_winpty(self._proc)
        pid = getattr(self._proc, "pid", None)
        if pid is None:
            return False
        return _is_pid_alive(pid)

    def close(self) -> None:
        """Terminate without hanging on a stubborn TUI.

        ptyprocess.terminate gates on isalive(), which can block on os.waitpid
        when the TUI's worker tree is still tearing down. We bypass that by
        signalling the pid directly via os.kill on POSIX and letting pywinpty
        do its TerminateProcess on Windows.
        """
        if sys.platform == "win32":
            with contextlib.suppress(Exception):
                self._proc.terminate(force=True)
            return
        pid = getattr(self._proc, "pid", None)
        if pid is None:
            return
        with contextlib.suppress(ProcessLookupError, OSError):
            os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            try:
                wpid, _ = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                return
            if wpid == pid:
                return
            time.sleep(0.1)
        with contextlib.suppress(ProcessLookupError, OSError):
            os.kill(pid, signal.SIGKILL)
        with contextlib.suppress(ChildProcessError, OSError):
            os.waitpid(pid, 0)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def lilbee_env(data_dir: Path, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build a deterministic environment for spawning lilbee under QA."""
    env = os.environ.copy()
    env["LILBEE_DATA"] = str(data_dir)
    env["LILBEE_NO_SPLASH"] = "1"
    env["LILBEE_LOG_LEVEL"] = "WARNING"
    if extra:
        env.update(extra)
    return env
