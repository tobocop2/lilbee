"""Standalone splash animation process — zero lilbee imports, stdlib only.

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

HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
CLEAR_LINE = "\033[2K"
MOVE_UP = "\033[A"

AMBER_BRIGHT = "\033[38;5;214m"
AMBER_MID = "\033[38;5;172m"
AMBER_DIM = "\033[38;5;94m"
RESET = "\033[0m"

FRAME_INTERVAL = 0.15
STARTUP_DELAY = 0.08
POLL_INTERVAL = 0.01

BEE_LINES = [
    "                                                       ",
    "@@@       @@@  @@@       @@@@@@@   @@@@@@@@  @@@@@@@@  ",
    "@@@       @@@  @@@       @@@@@@@@  @@@@@@@@  @@@@@@@@  ",
    "@@@       @@@  @@@       @@!  @@@  @@!       @@!       ",
    "@!       !@!  !@!       !@   @!@  !@!       !@!       ",
    "@!!       !!@  @!!       @!@!@!@   @!!!:!    @!!!:!    ",
    "!!!       !!!  !!!       !!!@!!!!  !!!!!:    !!!!!:    ",
    "!!:       !!:  !!:       !!:  !!!  !!:       !!:       ",
    " :!:      :!:   :!:      :!:  !:!  :!:       :!:       ",
    " :: ::::   ::   :: ::::   :: ::::   :: ::::   :: ::::  ",
    ": :: : :  :    : :: : :  :: : ::   : :: ::   : :: ::   ",
    "                                                       ",
]

LOGO_WIDTH = len(BEE_LINES[1])

COLOR_SEQUENCE = [AMBER_BRIGHT, AMBER_MID, AMBER_DIM, AMBER_MID]


def apply_color(line: str, color: str) -> str:
    """Apply color to non-empty parts of a line."""
    if not line.strip():
        return line
    return color + line + RESET


def build_logo_frames() -> list[list[str]]:
    """Pre-create 4 color-pulsed versions of the logo."""
    return [[apply_color(line, color) for line in BEE_LINES] for color in COLOR_SEQUENCE]


def build_knight_rider_frames() -> list[str]:
    """Build 22-frame Knight Rider bar spanning the full logo width."""
    frames: list[str] = []
    sweep_range = LOGO_WIDTH - 1
    total_frames = sweep_range * 2

    for pos in range(total_frames):
        head_pos = pos if pos < sweep_range else (total_frames - pos)

        bar = ""
        for i in range(LOGO_WIDTH):
            dist = abs(i - head_pos)
            if dist == 0:
                bar += AMBER_BRIGHT + "\u2593" + RESET
            elif dist == 1:
                bar += AMBER_DIM + "\u2592" + RESET
            elif dist == 2:
                bar += AMBER_DIM + "\u2591" + RESET
            else:
                bar += " "
        frames.append(bar)

    return frames


def render_frame(logo_lines: list[str], loading_bar: str) -> bytes:
    """Build a single frame as raw bytes for os.write()."""
    all_lines = [*logo_lines, "", f"  {loading_bar}"]
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


def _read_eof(pipe_fd: int) -> bool:
    """Try to read one byte — returns True if EOF, False if data available."""
    try:
        return len(os.read(pipe_fd, 1)) == 0
    except OSError:
        return True


def pipe_closed(pipe_fd: int) -> bool:
    """Check if the pipe has been closed (EOF) without blocking."""
    if sys.platform == "win32":
        import ctypes
        import msvcrt

        handle = msvcrt.get_osfhandle(pipe_fd)
        avail = ctypes.c_ulong(0)
        if not ctypes.windll.kernel32.PeekNamedPipe(
            handle, None, 0, None, ctypes.byref(avail), None
        ):
            return True
        if avail.value == 0:
            return False
        return _read_eof(pipe_fd)
    if sys.platform != "win32":
        try:
            readable, _, _ = select.select([pipe_fd], [], [], 0)
        except (ValueError, OSError):
            return True
        if not readable:
            return False
        return _read_eof(pipe_fd)
    return True  # pragma: no cover


def animation_loop(pipe_fd: int) -> None:
    """Run the animation, exiting when the pipe signals EOF."""
    fd = 2  # stderr

    logo_frames = build_logo_frames()
    knight_frames = build_knight_rider_frames()
    frame_height = len(BEE_LINES) + 2

    got_signal = False

    if sys.platform != "win32":

        def handle_term(signum: int, frame: object) -> None:
            nonlocal got_signal
            got_signal = True

        signal.signal(signal.SIGTERM, handle_term)

    for _ in range(int(STARTUP_DELAY / POLL_INTERVAL)):
        if got_signal or pipe_closed(pipe_fd):
            return
        time.sleep(POLL_INTERVAL)

    try:
        os.write(fd, HIDE_CURSOR.encode())
        frame_idx = 0
        knight_idx = 0

        while not got_signal and not pipe_closed(pipe_fd):
            logo = logo_frames[frame_idx % len(logo_frames)]
            knight = knight_frames[knight_idx % len(knight_frames)]
            rendered = render_frame(logo, knight)
            os.write(fd, rendered)

            for _ in range(int(FRAME_INTERVAL / POLL_INTERVAL)):
                if got_signal or pipe_closed(pipe_fd):
                    break
                time.sleep(POLL_INTERVAL)

            if not got_signal and not pipe_closed(pipe_fd):
                os.write(fd, move_up_and_clear(frame_height))  # pragma: no cover

            frame_idx += 1
            knight_idx += 1
    except OSError:
        pass
    finally:
        with contextlib.suppress(OSError):
            os.write(fd, clear_screen(frame_height))


def main() -> None:
    """Entry point when run as ``python -m lilbee._splash_runner <pipe_fd>``."""
    if len(sys.argv) != 2:
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
