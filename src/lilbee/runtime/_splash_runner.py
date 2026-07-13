"""Standalone splash animation process: stdlib plus the shared wordmark.

Launched as a subprocess by ``splash.start()``. Reads a pipe fd from argv
and animates until the pipe signals EOF (parent closed its write end, or
parent died). This guarantees no orphan/zombie animation processes.
"""

from __future__ import annotations

import contextlib
import os
import select
import signal
import sys
import time
from collections.abc import Callable
from enum import IntEnum

from lilbee.runtime.bee_logo import (
    BEE_LINES,
    LOGO_WIDTH,
    ROSE_BRIGHT_XTERM,
    ROSE_DIM_XTERM,
    ROSE_MID_XTERM,
    xterm_fg,
)

HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
CLEAR_LINE = "\033[2K"
MOVE_UP = "\033[A"

ROSE_BRIGHT = xterm_fg(ROSE_BRIGHT_XTERM)
ROSE_MID = xterm_fg(ROSE_MID_XTERM)
ROSE_DIM = xterm_fg(ROSE_DIM_XTERM)
RESET = "\033[0m"

FRAME_INTERVAL = 0.15
STARTUP_DELAY = 0.08
POLL_INTERVAL = 0.01

# Knight-rider bar uses three falloff steps after the bright head.
_BAR_FALLOFF_DENSE = 1
_BAR_FALLOFF_LIGHT = 2

# Subprocess entry point expects exactly ``python -m ... <pipe_fd>`` (script name + 1 arg).
_EXPECTED_ARGV_LEN = 2

# Sent down the pipe by ``splash.dismiss()`` when the TUI takes over the
# terminal: the child must exit without writing anything (no frame clear, no
# cursor-show), because every byte would land on Textual's alt-screen and the
# cursor-show would leave a visible cursor for the whole TUI session.
TAKEOVER_BYTE = b"T"


class PipeSignal(IntEnum):
    """What the control pipe currently says the child should do."""

    OPEN = 0
    CLOSED = 1
    TAKEOVER = 2


COLOR_SEQUENCE = [ROSE_BRIGHT, ROSE_MID, ROSE_DIM, ROSE_MID]


def apply_color(line: str, color: str) -> str:
    """Apply color to non-empty parts of a line."""
    if not line.strip():
        return line
    return color + line + RESET


def build_logo_frames() -> list[list[str]]:
    """Pre-create 4 color-pulsed versions of the logo."""
    return [[apply_color(line, color) for line in BEE_LINES] for color in COLOR_SEQUENCE]


def build_knight_rider_frames() -> list[str]:
    """Build a Knight Rider bar sweeping the full logo width and back."""
    frames: list[str] = []
    sweep_range = LOGO_WIDTH - 1
    total_frames = sweep_range * 2

    for pos in range(total_frames):
        head_pos = pos if pos < sweep_range else (total_frames - pos)

        bar = ""
        for i in range(LOGO_WIDTH):
            dist = abs(i - head_pos)
            if dist == 0:
                bar += ROSE_BRIGHT + "\u2593" + RESET
            elif dist == _BAR_FALLOFF_DENSE:
                bar += ROSE_DIM + "\u2592" + RESET
            elif dist == _BAR_FALLOFF_LIGHT:
                bar += ROSE_DIM + "\u2591" + RESET
            else:
                bar += " "
        frames.append(bar)

    return frames


def left_pad() -> int:
    """Columns needed to centre the wordmark, matching the C bootstrap's formula.

    The bootstrap frame this animation repaints in place is centred with
    ``(columns - LILBEE_LOGO_WIDTH) / 2``; diverging here would draw the two
    stages at different offsets and break the one-continuous-logo illusion.
    """
    try:
        columns = os.get_terminal_size(2).columns
    except OSError:
        return 0
    return max((columns - LOGO_WIDTH) // 2, 0)


def render_frame(logo_lines: list[str], loading_bar: str, pad: int = 0) -> bytes:
    """Build a single frame as raw bytes for os.write()."""
    margin = " " * pad
    all_lines = [margin + line for line in logo_lines]
    all_lines += ["", f"{margin}  {loading_bar}"]
    return ("\n".join(all_lines) + "\n").encode()


def move_up_and_clear(n: int) -> bytes:
    """ANSI sequence to move cursor up n lines and clear each one."""
    return ((MOVE_UP + CLEAR_LINE) * n).encode()


def clear_screen(frame_height: int) -> bytes:
    """Erase the splash frame area and restore the cursor to the top.

    Uses line-by-line clear (move-up + erase) instead of ``\\033[2J\\033[H``
    so the subprocess never writes a cursor-home escape. A cursor-home
    would land on the Textual alt-screen if the TUI starts before the
    subprocess has finished, leaving a stuck cursor artifact at (0,0).
    """
    return move_up_and_clear(frame_height) + SHOW_CURSOR.encode()


def _read_signal(pipe_fd: int) -> PipeSignal:
    """Read one byte: EOF/error means CLOSED, the takeover byte means TAKEOVER."""
    try:
        data = os.read(pipe_fd, 1)
    except OSError:
        return PipeSignal.CLOSED
    if data == TAKEOVER_BYTE:
        return PipeSignal.TAKEOVER
    return PipeSignal.CLOSED if len(data) == 0 else PipeSignal.OPEN


def _poll_pipe_win32(pipe_fd: int) -> PipeSignal:  # pragma: no cover  Windows-only
    """Win32 pipe poll using PeekNamedPipe."""
    import ctypes
    import msvcrt

    try:
        handle = msvcrt.get_osfhandle(pipe_fd)  # type: ignore[attr-defined]
    except OSError:
        return PipeSignal.CLOSED  # bad fd, pipe is gone
    avail = ctypes.c_ulong(0)
    if not ctypes.windll.kernel32.PeekNamedPipe(  # type: ignore[attr-defined]
        handle, None, 0, None, ctypes.byref(avail), None
    ):
        return PipeSignal.CLOSED
    if avail.value == 0:
        return PipeSignal.OPEN
    return _read_signal(pipe_fd)


def _poll_pipe_posix(pipe_fd: int) -> PipeSignal:  # pragma: no cover  POSIX-only
    """POSIX pipe poll using select."""
    try:
        readable, _, _ = select.select([pipe_fd], [], [], 0)
    except (ValueError, OSError):
        return PipeSignal.CLOSED
    if not readable:
        return PipeSignal.OPEN
    return _read_signal(pipe_fd)


def poll_pipe(pipe_fd: int) -> PipeSignal:
    """Check the control pipe without blocking."""
    if sys.platform == "win32":
        return _poll_pipe_win32(pipe_fd)  # pragma: no cover  Windows-only
    return _poll_pipe_posix(pipe_fd)  # pragma: no cover  POSIX-only


def animation_loop(pipe_fd: int) -> None:
    """Run the animation, exiting when the pipe signals EOF or takeover.

    A plain EOF (``splash.stop()``, parent death) clears the frame and
    restores the cursor so the shell gets a clean terminal back. A takeover
    byte (``splash.dismiss()``) means Textual owns the terminal: exit without
    writing a single byte more.
    """
    fd = 2  # stderr

    logo_frames = build_logo_frames()
    knight_frames = build_knight_rider_frames()
    pad = left_pad()
    frame_height = len(BEE_LINES) + 2

    got_signal = False
    pipe_signal = PipeSignal.OPEN

    if sys.platform != "win32":  # pragma: no cover - POSIX-only SIGTERM handler

        def handle_term(signum: int, frame: object) -> None:
            nonlocal got_signal
            got_signal = True

        signal.signal(signal.SIGTERM, handle_term)

    def should_stop() -> bool:
        nonlocal pipe_signal
        if pipe_signal is PipeSignal.OPEN:
            pipe_signal = poll_pipe(pipe_fd)
        return got_signal or pipe_signal is not PipeSignal.OPEN

    for _ in range(int(STARTUP_DELAY / POLL_INTERVAL)):
        if should_stop():
            return  # nothing drawn yet, nothing to clean up
        time.sleep(POLL_INTERVAL)

    try:
        os.write(fd, HIDE_CURSOR.encode())
        _animate_frames(fd, logo_frames, knight_frames, pad, frame_height, should_stop)
    except OSError:
        pass  # parent closed the splash pipe; just stop drawing
    finally:
        if pipe_signal is not PipeSignal.TAKEOVER:
            with contextlib.suppress(OSError):
                os.write(fd, clear_screen(frame_height))


def _animate_frames(
    fd: int,
    logo_frames: list[list[str]],
    knight_frames: list[str],
    pad: int,
    frame_height: int,
    should_stop: Callable[[], bool],
) -> None:
    """Draw pulse/sweep frames until *should_stop* reports a stop condition."""
    frame_idx = 0
    while not should_stop():
        logo = logo_frames[frame_idx % len(logo_frames)]
        knight = knight_frames[frame_idx % len(knight_frames)]
        os.write(fd, render_frame(logo, knight, pad))

        for _ in range(int(FRAME_INTERVAL / POLL_INTERVAL)):
            if should_stop():
                break
            time.sleep(POLL_INTERVAL)

        if not should_stop():
            os.write(fd, move_up_and_clear(frame_height))  # pragma: no cover

        frame_idx += 1


def main() -> None:
    """Entry point when run as ``python -m lilbee.runtime._splash_runner <pipe_fd>``."""
    if len(sys.argv) != _EXPECTED_ARGV_LEN:
        sys.exit(1)

    try:
        pipe_fd = int(sys.argv[1])
    except ValueError:
        sys.exit(1)

    try:
        animation_loop(pipe_fd)
    finally:
        with contextlib.suppress(OSError):
            os.close(pipe_fd)


if __name__ == "__main__":
    main()
