"""Startup splash animation — shows ASCII logo while heavy imports load.

This module intentionally uses ONLY the Python standard library so it loads
instantly and can fork an animation process before any heavy dependencies.
"""

from __future__ import annotations

import os
import signal
import time

_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"
_CLEAR_LINE = "\033[2K"
_MOVE_UP = "\033[A"

_READY_FILE = "lilbee-splash-ready"

# ---------------------------------------------------------------------------
# Logo in poison font style (embedded, no external dependency)
# ---------------------------------------------------------------------------
_BEE_LINES = [
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

# ---------------------------------------------------------------------------
# Logo color pulsing frames (4 intensities)
# Bright amber -> dim -> faint -> dim (cycles)
_AMBER_PULSE_BRIGHT = "\033[38;5;214m"  # bright
_AMBER_PULSE_MID = "\033[38;5;172m"  # mid
_AMBER_PULSE_DIM = "\033[38;5;94m"  # dim
_AMBER_RESET = "\033[0m"


def _apply_color(line: str, color: str) -> str:
    """Apply color to non-empty parts of a line."""
    if not line.strip():
        return line
    return color + line + _AMBER_RESET


# Pre-create 4 color-pulsed versions of the logo
_LOGO_FRAMES: list[list[str]] = []
_color_sequence = [
    _AMBER_PULSE_BRIGHT,
    _AMBER_PULSE_MID,
    _AMBER_PULSE_DIM,
    _AMBER_PULSE_MID,
]
for color in _color_sequence:
    frame = [_apply_color(line, color) for line in _BEE_LINES]
    _LOGO_FRAMES.append(frame)

# ---------------------------------------------------------------------------
# Knight Rider bar — amber gradient (CRT style from site colors)
# Site colors: #ffb000 (bright), #996a00 (dim), #4d3500 (faint)
# Using ANSI 256 colors for the gradient effect
# ---------------------------------------------------------------------------
_AMBER_BRIGHT = "\033[38;5;214m"  # bright amber
_AMBER_DIM = "\033[38;5;172m"  # dim amber
_AMBER_FAINT = "\033[38;5;94m"  # faint amber
_AMBER_RESET = "\033[0m"

# Knight Rider bar frames (22 frames, oscillates left-right-left)
# Uses ▓ (head), ▒ (trail), ░ (fade), space (empty)
_KNIGHT_RIDER_FRAMES: list[str] = []
_LOGO_WIDTH = 55  # match logo width
_BAR_PADDING = 0  # left padding to center under logo

for pos in range(22):
    bar_width = _LOGO_WIDTH - _BAR_PADDING
    head_pos = pos if pos < 11 else 21 - pos  # 0-10 then 10-0

    bar = " " * _BAR_PADDING
    for i in range(bar_width):
        if i < head_pos - 2:
            bar += " "  # empty before trail
        elif i == head_pos - 2:
            bar += _AMBER_FAINT + "░" + _AMBER_RESET  # fade
        elif i == head_pos - 1:
            bar += _AMBER_DIM + "▒" + _AMBER_RESET  # trail
        elif i == head_pos:
            bar += _AMBER_BRIGHT + "▓" + _AMBER_RESET  # head
        elif i == head_pos + 1:
            bar += _AMBER_DIM + "▒" + _AMBER_RESET  # trail
        elif i == head_pos + 2:
            bar += _AMBER_FAINT + "░" + _AMBER_RESET  # fade
        else:
            bar += " "
    _KNIGHT_RIDER_FRAMES.append(bar)


_FRAME_INTERVAL = 0.15
_STARTUP_DELAY = 0.08


def _pick_frames() -> tuple[list[str], list[str], list[str], list[str]]:
    """Return logo frames with color pulsing animation."""
    return (
        _LOGO_FRAMES[0],
        _LOGO_FRAMES[1],
        _LOGO_FRAMES[2],
        _LOGO_FRAMES[3],
    )


def _should_skip() -> bool:
    """Return True when the splash animation should be suppressed."""
    if not os.isatty(2):
        return True
    return bool(os.environ.get("LILBEE_NO_SPLASH", ""))


def _render_frame(lines: list[str], loading_text: str) -> bytes:
    """Build a single frame as raw bytes for os.write()."""
    all_lines = [*lines, "", f"  {loading_text}"]
    return ("\n".join(all_lines) + "\n").encode()


def _move_up_and_clear(n: int) -> bytes:
    """ANSI sequence to move cursor up n lines and clear each one."""
    return ((_MOVE_UP + _CLEAR_LINE) * n).encode()


def _clear_frame(height: int) -> bytes:
    """Full terminal reset for clean handoff to TUI/Textual."""
    return b"\033[2J\033[H\033[?25h"


def _is_ready(ready_file: str | None) -> bool:
    """Check if the ready file exists (signals TUI is ready)."""
    if ready_file is None:
        return False
    try:
        return os.path.exists(ready_file)
    except OSError:
        return False


def _animation_loop(
    frame0: list[str],
    frame1: list[str],
    frame2: list[str],
    frame3: list[str],
    ready_file: str | None = None,
) -> None:
    """Run the animation in a forked child process. Exits via os._exit()."""
    fd = 2  # stderr  # pragma: no cover
    frames = [frame0, frame1, frame2, frame3]  # pragma: no cover
    frame_height = len(frame0) + 2  # pragma: no cover

    got_signal = False  # pragma: no cover

    def _handle_stop(signum: int, frame: object) -> None:  # pragma: no cover
        nonlocal got_signal
        got_signal = True

    signal.signal(signal.SIGUSR1, _handle_stop)  # pragma: no cover

    # Delay before showing anything — fast warm starts skip the splash entirely
    for _ in range(int(_STARTUP_DELAY / 0.01)):  # pragma: no cover
        if got_signal or _is_ready(ready_file):
            os._exit(0)
        time.sleep(0.01)

    try:  # pragma: no cover
        os.write(fd, _HIDE_CURSOR.encode())
        frame_idx = 0
        knight_idx = 0
        while not got_signal and not _is_ready(ready_file):
            knight_frame = _KNIGHT_RIDER_FRAMES[knight_idx % len(_KNIGHT_RIDER_FRAMES)]
            rendered = _render_frame(frames[frame_idx % 4], knight_frame)
            os.write(fd, rendered)
            # Wait, checking signal and ready file frequently
            for _ in range(int(_FRAME_INTERVAL / 0.01)):
                if got_signal or _is_ready(ready_file):
                    break
                time.sleep(0.01)
            if not got_signal and not _is_ready(ready_file):
                os.write(fd, _move_up_and_clear(frame_height))
            frame_idx += 1
            knight_idx += 1
    except OSError:  # pragma: no cover
        pass
    finally:
        try:  # noqa: SIM105 — avoid importing contextlib in forked child  # pragma: no cover
            os.write(fd, _clear_frame(frame_height))
        except OSError:  # pragma: no cover
            pass
        os._exit(0)  # pragma: no cover


def start(ready_file: str | None = None) -> int:
    """Fork and start the splash animation in a child process.

    Args:
        ready_file: Path to a file that, when created/written to, signals
            the splash should stop. Pass None to run for fixed duration.

    Returns the child PID, or 0 if the splash was skipped.
    The caller must eventually call ``stop(pid)`` to clean up.
    """
    if _should_skip():
        return 0

    frame0, frame1, frame2, frame3 = _pick_frames()

    if not hasattr(os, "fork"):  # pragma: no cover
        # Windows: fall back to threading
        return _start_threaded(frame0, frame1, frame2, frame3, ready_file)

    pid = os.fork()
    if pid == 0:  # pragma: no cover
        # Child — run animation and never return
        _animation_loop(frame0, frame1, frame2, frame3, ready_file)  # pragma: no cover
        os._exit(0)  # safety net  # pragma: no cover
    return pid


def stop(pid: int) -> None:
    """Stop the splash animation and wait for cleanup."""
    if pid == 0:
        return

    if pid < 0:  # pragma: no cover
        _stop_threaded(pid)
        return

    if not hasattr(os, "fork"):  # pragma: no cover
        _stop_threaded(pid)
        return

    try:
        os.kill(pid, signal.SIGUSR1)
    except OSError:  # pragma: no cover
        return
    # Wait for child to finish clearing the terminal
    import contextlib

    with contextlib.suppress(ChildProcessError):
        os.waitpid(pid, 0)


# ---------------------------------------------------------------------------
# Threading fallback for Windows (no os.fork)
# ---------------------------------------------------------------------------
_thread_stop_flag: bool = False
_thread_obj: object = None


def _start_threaded(  # pragma: no cover
    frame0: list[str],
    frame1: list[str],
    frame2: list[str],
    frame3: list[str],
    ready_file: str | None = None,
) -> int:
    """Start animation in a daemon thread. Returns a sentinel PID of -1."""
    import threading

    global _thread_stop_flag, _thread_obj
    _thread_stop_flag = False

    def _loop() -> None:  # pragma: no cover
        fd = 2
        frames = [frame0, frame1, frame2, frame3]
        frame_height = len(frame0) + 2

        # Startup delay
        for _ in range(int(_STARTUP_DELAY / 0.01)):  # pragma: no cover
            if _thread_stop_flag or _is_ready(ready_file):
                return
            time.sleep(0.01)
        try:  # pragma: no cover
            os.write(fd, _HIDE_CURSOR.encode())
            frame_idx = 0
            knight_idx = 0
            while not _thread_stop_flag and not _is_ready(ready_file):
                knight_frame = _KNIGHT_RIDER_FRAMES[knight_idx % len(_KNIGHT_RIDER_FRAMES)]
                rendered = _render_frame(frames[frame_idx % 4], knight_frame)
                os.write(fd, rendered)
                for _ in range(int(_FRAME_INTERVAL / 0.01)):
                    if _thread_stop_flag or _is_ready(ready_file):
                        break
                    time.sleep(0.01)
                if not _thread_stop_flag and not _is_ready(ready_file):
                    os.write(fd, _move_up_and_clear(frame_height))
                frame_idx += 1
                knight_idx += 1
        except OSError:  # pragma: no cover
            pass
        finally:
            try:  # noqa: SIM105 — keep stdlib-only in animation thread  # pragma: no cover
                os.write(fd, _clear_frame(frame_height))
            except OSError:  # pragma: no cover
                pass

    t = threading.Thread(target=_loop, daemon=True)
    _thread_obj = t
    t.start()
    return -1


def _stop_threaded(pid: int) -> None:  # pragma: no cover
    """Stop the threaded animation."""
    global _thread_stop_flag
    _thread_stop_flag = True
    if _thread_obj is not None and hasattr(_thread_obj, "join"):
        _thread_obj.join(timeout=2.0)  # type: ignore[union-attr]
