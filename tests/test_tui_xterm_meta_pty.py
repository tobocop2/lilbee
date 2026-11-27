"""The real TUI in a terminal that behaves as xterm does for Meta keys (POSIX only)."""

from __future__ import annotations

import fcntl
import os
import pty
import re
import signal
import struct
import subprocess
import sys
import termios
import threading
import time
from pathlib import Path

import pytest

from conftest import clean_env
from tests._signals import child_preexec

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Windows has no pty")

_MODE_SEQUENCE = re.compile(rb"\x1b\[\?1036([hlsr])")
_ANSI = re.compile(
    r"\x1b\[[0-9;?<>=]*[ -/]*[@-~]|\x1b[()][A-Z0-9]|\x1b[=>]|\x1b\][^\x07\x1b]*(\x07|\x1b\\)"
)
_CTRL_P = b"\x10"
_ESCAPE = b"\x1b"
_REPLACEMENT_CHARACTER = "�"
# The signal handlers of the lilbee command, so that a SIGTERM leaves through run_tui's cleanup.
_CHILD = """
from lilbee.app.services import install_engine_lifecycle_hooks, mark_interactive_session
from lilbee.cli.tui import run_tui

install_engine_lifecycle_hooks()
mark_interactive_session()
run_tui()
"""
_STEP_TIMEOUT_S = 40.0
_ESCAPE_GAP_S = 0.5


class _XtermModel:
    """A pseudo terminal that holds metaSendsEscape as xterm does, by the ctlseqs table."""

    def __init__(self, argv: list[str], env: dict[str, str]) -> None:
        controller, terminal = pty.openpty()
        fcntl.ioctl(terminal, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
        self._controller = controller
        self._written: list[bytes] = []
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        self.child = subprocess.Popen(
            argv,
            env=env,
            stdin=terminal,
            stdout=terminal,
            stderr=terminal,
            start_new_session=True,
            preexec_fn=child_preexec(),
        )
        os.close(terminal)

    def _read(self) -> None:
        while True:
            try:
                chunk = os.read(self._controller, 65536)
            except OSError:  # Linux: the child closed the terminal
                return
            if not chunk:
                return
            self._written.append(chunk)

    def output(self) -> bytes:
        return b"".join(self._written)

    def meta_sends_escape(self) -> bool:
        """Replay the mode sequences the app wrote: set, reset, save and restore."""
        mode = False
        saved = False
        for match in _MODE_SEQUENCE.finditer(self.output()):
            op = match.group(1)
            if op == b"h":
                mode = True
            elif op == b"l":
                mode = False
            elif op == b"s":
                saved = mode
            else:
                mode = saved
        return mode

    def type_meta(self, key: bytes) -> bytes:
        """Type Meta+key as xterm sends it in the current mode; return the bytes sent."""
        sent = _ESCAPE + key if self.meta_sends_escape() else bytes([key[0] | 0x80])
        os.write(self._controller, sent)
        return sent

    def type(self, data: bytes) -> bytes:
        os.write(self._controller, data)
        return data

    def wait_for(self, text: str, start: int, timeout: float = _STEP_TIMEOUT_S) -> bool:
        """Return whether *text* shows on the screen text written after offset *start*."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            shown = _ANSI.sub("", self.output()[start:].decode("utf-8", "replace"))
            if text in shown:
                return True
            if self.child.poll() is not None:
                return False
            time.sleep(0.1)
        return False

    def palette_opens(self) -> bool:
        """Ctrl+P opens the command palette only while the input thread lives."""
        start = len(self.output())
        self.type(_CTRL_P)
        opened = self.wait_for("Search for commands", start)
        self.type(_ESCAPE)
        time.sleep(_ESCAPE_GAP_S)  # A key within the escape delay would join this Escape.
        return opened

    def stop(self) -> None:
        if self.child.poll() is None:
            self.child.send_signal(signal.SIGTERM)
        try:
            self.child.wait(timeout=_STEP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self.child.kill()
            self.child.wait()
        self._reader.join(timeout=5)
        os.close(self._controller)


def _boot(tmp_path: Path) -> _XtermModel:
    env = clean_env(tmp_path)
    env["TERM"] = "xterm-256color"
    env["LANG"] = "C.UTF-8"
    argv = [sys.executable, "-c", _CHILD]
    terminal = _XtermModel(argv, env)
    # The first screen draws once the app runs; the palette answers once the input thread reads.
    deadline = time.monotonic() + _STEP_TIMEOUT_S * 2
    while not terminal.palette_opens():
        assert time.monotonic() < deadline, "the TUI never opened the command palette"
        assert terminal.child.poll() is None, "the TUI exited while starting"
    return terminal


@pytest.mark.timeout(300)
def test_meta_keys_survive_in_xterm_model(tmp_path: Path) -> None:
    terminal = _boot(tmp_path)
    try:
        assert terminal.meta_sends_escape(), "the app did not ask for ESC-prefixed meta"
        sent: list[bytes] = []
        for send in (
            lambda: terminal.type_meta(b"a"),
            lambda: terminal.type_meta(b"\r"),
            lambda: terminal.type(_ESCAPE + b"a"),
            lambda: terminal.type("é日".encode()),
        ):
            data = send()
            sent.append(data)
            assert terminal.palette_opens(), f"the TUI stopped reading after {data!r}"
            assert terminal.child.poll() is None, f"the TUI exited after {data!r}"
        assert sent[:2] == [b"\x1ba", b"\x1b\r"]
        shown = terminal.output().decode("utf-8", "replace")
        assert "UnicodeDecodeError" not in shown
        assert _REPLACEMENT_CHARACTER not in shown
    finally:
        terminal.stop()
    modes = _MODE_SEQUENCE.findall(terminal.output())
    assert not terminal.meta_sends_escape(), f"the app left ESC-prefixed meta mode on: {modes}"
    assert terminal.child.returncode == 128 + signal.SIGTERM
