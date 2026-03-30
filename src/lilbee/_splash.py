"""Startup splash animation — shows an ASCII bee while heavy imports load.

This module intentionally uses ONLY the Python standard library so it loads
instantly and can fork an animation process before any heavy dependencies.
"""

from __future__ import annotations

import os
import signal
import sys
import time

_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"
_CLEAR_LINE = "\033[2K"
_MOVE_UP = "\033[A"

# ---------------------------------------------------------------------------
# Bee art — large version from site/index.html (lines 71-108)
# ---------------------------------------------------------------------------
_BEE_BODY = [
    "                 %%%%%%  %%%%%%%%%            ::::::::",
    "              %%%%%ZZZZ%%%%%%   %%%ZZZZ     ::::::::::         ::::::",
    "             %%%ZZZZZ%%%%%%%%%%%%%%ZZZZZZ  :::::::::::    :::::::::::::::::",
    "             ZZZ%ZZZ%%%%%%%%%%%%%%%ZZZZZZZ::::::::::***:::::::::::::::::::::",
    "          ZZZ%ZZZZZZ%%%%%%%%%%%%%%ZZZZZZZZZ::::::***:::::::::::::::::::::::",
    "        ZZZ%ZZZZZZZZZZ%%%%%%%%%%ZZZZZZ%ZZZZ:::***:::::::::::::::::::::::",
    "       ZZ%ZZZZZZZZZZZZZZZZZZZZZZZ%%%%% %ZZZ:**::::::::::::::::::::::",
    "      ZZ%ZZZZZZZZZZZZZZZZZZZ%%%%% | | %ZZZ *:::::::::::::::::::",
    "      Z%ZZZZZZZZZZZZZZZ%%%%%%%%%%%%%%%ZZZ::::::::::::::::::::::::::",
    "       ZZZZZZZZZZZ%%%%%ZZZZZZZZZZZZZZZZZ%%%%:::ZZZZ:::::::::::::::::",
    "         ZZZZ%%%%%ZZZZZZZZZZZZZZZZZZ%%%%%ZZZ%%ZZZ%ZZ%%*:::::::::::",
    "            ZZZZZZZZZZZZZZZZZZ%%%%%%%%%ZZZZZZZZZZ%ZZ%:::*:::::::",
    "            *:::%%%%%%%%%%%%%%%%%%%%%%%ZZZZZZZZZZ%%%*::::*::::",
    "          *:::::::%%%%%%%%%%%%%%%%%%%%%%%ZZZZZ%%      *:::Z",
    "         **:ZZZZ:::%%%%%%%%%%%%%%%%%%%%%%%%%%%ZZ      ZZZZZ",
    "        *:ZZZZZZZ       %%%%%%%%%%%%%%%%%%%%%ZZZZ    ZZZZZZZ",
    "       *::::ZZZZZZ         %%%%%%%%%%%%%%%ZZZZZZZ      ZZZ",
    "        *::ZZZZZZ           Z%%%%%%%%%%%ZZZZZZZ%%",
    "          ZZZZ              ZZZZZZZZZZZZZZZZ%%%%%",
    "                           %%%ZZZZZZZZZZZ%%%%%%%%",
    "                          Z%%%%%%%%%%%%%%%%%%%%%",
    "                          ZZ%%%%%%%%%%%%%%%%%%%",
    "                          %ZZZZZZZZZZZZZZZZZZZ",
    "                          %%ZZZZZZZZZZZZZZZZZ",
    "                           %%%%%%%%%%%%%%%%",
    "                            %%%%%%%%%%%%%",
    "                             %%%%%%%%%",
    "                              ZZZZ",
    "                              ZZZ",
    "                             ZZ",
    "                            Z",
]

_WINGS_SPREAD = [
    "     %           %",
    "         %           %",
    "            %           %",
    "               %          %",
    "                 %          %",
    "                   %          %                   :::",
    "                    %          %                ::::::",
]

_WINGS_TUCKED = [
    "            %    %",
    "              %    %",
    "                %    %",
    "                 %    %",
    "                  %    %",
    "                   %    %                         :::",
    "                    %    %                      ::::::",
]

_LOADING_DOTS = ["loading", "loading.", "loading..", "loading..."]

# Small fallback bee for narrow/short terminals
_BEE_SMALL_BODY = [
    "     _{    }_",
    "    / /o  o\\ \\",
    "    \\  \\__/  /",
    "     '-____-'",
]

_SMALL_WINGS_SPREAD = [
    "      \\)  (/",
]

_SMALL_WINGS_TUCKED = [
    "       |  |",
]

# Minimum terminal size for the large bee
_MIN_COLS_LARGE = 70
_MIN_ROWS_LARGE = 42

_FRAME_INTERVAL = 0.15
_STARTUP_DELAY = 0.08


def _get_terminal_size() -> tuple[int, int]:
    """Return (columns, rows) of the terminal, with safe fallback."""
    try:
        cols, rows = os.get_terminal_size(2)  # stderr fd
        return cols, rows
    except (OSError, ValueError):
        return 80, 24


def _pick_frames() -> tuple[list[str], list[str]]:
    """Choose large or small bee based on terminal size."""
    cols, rows = _get_terminal_size()
    if cols >= _MIN_COLS_LARGE and rows >= _MIN_ROWS_LARGE:
        frame0 = _WINGS_SPREAD + _BEE_BODY
        frame1 = _WINGS_TUCKED + _BEE_BODY
    else:
        frame0 = _SMALL_WINGS_SPREAD + _BEE_SMALL_BODY
        frame1 = _SMALL_WINGS_TUCKED + _BEE_SMALL_BODY
    return frame0, frame1


def _should_skip() -> bool:
    """Return True when the splash animation should be suppressed."""
    if not os.isatty(2):
        return True
    if os.environ.get("LILBEE_NO_SPLASH", ""):
        return True
    argv = sys.argv[1:]
    skip_flags = {"--version", "-V", "--help", "-h", "help"}
    if any(arg in skip_flags for arg in argv):
        return True
    return bool(argv and argv[0] in ("--install-completion", "--show-completion"))


def _render_frame(lines: list[str], loading_text: str) -> bytes:
    """Build a single frame as raw bytes for os.write()."""
    all_lines = [*lines, "", f"  {loading_text}"]
    return ("\n".join(all_lines) + "\n").encode()


def _move_up_and_clear(n: int) -> bytes:
    """ANSI sequence to move cursor up n lines and clear each one."""
    return ((_MOVE_UP + _CLEAR_LINE) * n).encode()


def _clear_frame(height: int) -> bytes:
    """ANSI to erase an entire rendered frame and restore cursor."""
    parts = [_MOVE_UP + _CLEAR_LINE for _ in range(height)]
    parts.append(_CLEAR_LINE + "\r" + _SHOW_CURSOR)
    return "".join(parts).encode()


def _animation_loop(frame0: list[str], frame1: list[str]) -> None:
    """Run the animation in a forked child process. Exits via os._exit()."""
    fd = 2  # stderr
    frames = [frame0, frame1]
    # +2 for the blank line and loading text line
    frame_height = len(frame0) + 2

    got_signal = False

    def _handle_stop(signum: int, frame: object) -> None:
        nonlocal got_signal
        got_signal = True

    signal.signal(signal.SIGUSR1, _handle_stop)

    # Delay before showing anything — fast warm starts skip the splash entirely
    for _ in range(int(_STARTUP_DELAY / 0.01)):
        if got_signal:
            os._exit(0)
        time.sleep(0.01)

    try:
        os.write(fd, _HIDE_CURSOR.encode())
        frame_idx = 0
        dot_idx = 0
        while not got_signal:
            rendered = _render_frame(frames[frame_idx % 2], _LOADING_DOTS[dot_idx % 4])
            os.write(fd, rendered)
            # Wait, checking signal frequently
            for _ in range(int(_FRAME_INTERVAL / 0.01)):
                if got_signal:
                    break
                time.sleep(0.01)
            if not got_signal:
                os.write(fd, _move_up_and_clear(frame_height))
            frame_idx += 1
            dot_idx += 1
    except OSError:
        pass
    finally:
        try:  # noqa: SIM105 — avoid importing contextlib in forked child
            os.write(fd, _clear_frame(frame_height))
        except OSError:
            pass
        os._exit(0)


def start() -> int:
    """Fork and start the splash animation in a child process.

    Returns the child PID, or 0 if the splash was skipped.
    The caller must eventually call ``stop(pid)`` to clean up.
    """
    if _should_skip():
        return 0

    frame0, frame1 = _pick_frames()

    if not hasattr(os, "fork"):
        # Windows: fall back to threading
        return _start_threaded(frame0, frame1)

    pid = os.fork()
    if pid == 0:
        # Child — run animation and never return
        _animation_loop(frame0, frame1)
        os._exit(0)  # safety net
    return pid


def stop(pid: int) -> None:
    """Stop the splash animation and wait for cleanup."""
    if pid == 0:
        return

    if pid < 0:
        _stop_threaded(pid)
        return

    if not hasattr(os, "fork"):
        _stop_threaded(pid)
        return

    try:
        os.kill(pid, signal.SIGUSR1)
    except OSError:
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


def _start_threaded(frame0: list[str], frame1: list[str]) -> int:
    """Start animation in a daemon thread. Returns a sentinel PID of -1."""
    import threading

    global _thread_stop_flag, _thread_obj
    _thread_stop_flag = False

    def _loop() -> None:
        fd = 2
        frames = [frame0, frame1]
        frame_height = len(frame0) + 2
        # Startup delay
        for _ in range(int(_STARTUP_DELAY / 0.01)):
            if _thread_stop_flag:
                return
            time.sleep(0.01)
        try:
            os.write(fd, _HIDE_CURSOR.encode())
            frame_idx = 0
            dot_idx = 0
            while not _thread_stop_flag:
                rendered = _render_frame(frames[frame_idx % 2], _LOADING_DOTS[dot_idx % 4])
                os.write(fd, rendered)
                for _ in range(int(_FRAME_INTERVAL / 0.01)):
                    if _thread_stop_flag:
                        break
                    time.sleep(0.01)
                if not _thread_stop_flag:
                    os.write(fd, _move_up_and_clear(frame_height))
                frame_idx += 1
                dot_idx += 1
        except OSError:
            pass
        finally:
            try:  # noqa: SIM105 — keep stdlib-only in animation thread
                os.write(fd, _clear_frame(frame_height))
            except OSError:
                pass

    t = threading.Thread(target=_loop, daemon=True)
    _thread_obj = t
    t.start()
    return -1


def _stop_threaded(pid: int) -> None:
    """Stop the threaded animation."""
    global _thread_stop_flag
    _thread_stop_flag = True
    if _thread_obj is not None and hasattr(_thread_obj, "join"):
        _thread_obj.join(timeout=2.0)  # type: ignore[union-attr]
