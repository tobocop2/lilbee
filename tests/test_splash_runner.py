"""Tests for _splash_runner.py: animation subprocess utilities."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest


def test_apply_color_non_empty():
    from lilbee.runtime._splash_runner import RESET, ROSE_BRIGHT, apply_color

    result = apply_color("hello", ROSE_BRIGHT)
    assert result == ROSE_BRIGHT + "hello" + RESET


def test_apply_color_empty_line():
    from lilbee.runtime._splash_runner import apply_color

    assert apply_color("   ", "color") == "   "


def test_build_logo_frames():
    from lilbee.runtime._splash_runner import COLOR_SEQUENCE, build_logo_frames

    frames = build_logo_frames()
    assert len(frames) == len(COLOR_SEQUENCE)
    assert all(isinstance(f, list) for f in frames)


def test_build_knight_rider_frames():
    from lilbee.runtime._splash_runner import LOGO_WIDTH, build_knight_rider_frames

    frames = build_knight_rider_frames()
    assert len(frames) == (LOGO_WIDTH - 1) * 2


def test_render_frame():
    from lilbee.runtime._splash_runner import render_frame

    result = render_frame(["line1", "line2"], "bar")
    assert isinstance(result, bytes)
    assert b"line1" in result
    assert b"bar" in result


def test_render_frame_centres_every_row_by_the_same_margin():
    from lilbee.runtime._splash_runner import render_frame

    lines = render_frame(["line1", "line2"], "bar", pad=4).decode().splitlines()
    assert lines[0] == "    line1"
    assert lines[1] == "    line2"
    assert lines[-1] == "      bar"  # margin plus the bar's own two-space indent


def test_left_pad_matches_the_bootstrap_formula(monkeypatch):
    import os

    from lilbee.runtime import _splash_runner
    from lilbee.runtime.bee_logo import LOGO_WIDTH

    monkeypatch.setattr(os, "get_terminal_size", lambda fd=1: os.terminal_size((121, 40)))
    assert _splash_runner.left_pad() == (121 - LOGO_WIDTH) // 2


def test_left_pad_is_zero_when_the_terminal_size_is_unknown(monkeypatch):
    import os

    from lilbee.runtime import _splash_runner

    def _raise(fd=1):
        raise OSError("not a tty")

    monkeypatch.setattr(os, "get_terminal_size", _raise)
    assert _splash_runner.left_pad() == 0


def test_move_up_and_clear():
    from lilbee.runtime._splash_runner import move_up_and_clear

    result = move_up_and_clear(3)
    assert isinstance(result, bytes)
    assert result.count(b"\033[A") == 3


def test_clear_screen():
    """clear_screen erases the splash frame without cursor-home.

    Uses move-up-and-clear instead of ``\\033[2J\\033[H`` so the subprocess
    never writes a cursor-home escape into Textual's alt-screen.
    Restores cursor visibility so non-TUI exits leave the terminal clean.
    """
    from lilbee.runtime._splash_runner import clear_screen

    result = clear_screen(5)
    assert b"\033[A" in result  # move-up sequences
    assert b"\033[2K" in result  # clear-line sequences
    assert b"\033[?25h" in result  # cursor restore
    assert b"\033[2J" not in result  # no full-screen clear
    assert b"\033[H" not in result  # no cursor home


def test_poll_pipe_returns_closed_on_eof():
    """poll_pipe reports CLOSED when the read end gets EOF."""
    r, w = os.pipe()
    os.close(w)  # close write end -> read gets EOF
    from lilbee.runtime._splash_runner import PipeSignal, poll_pipe

    assert poll_pipe(r) is PipeSignal.CLOSED
    os.close(r)


def test_poll_pipe_returns_open_when_open():
    """poll_pipe reports OPEN when the pipe is still open."""
    r, w = os.pipe()
    from lilbee.runtime._splash_runner import PipeSignal, poll_pipe

    assert poll_pipe(r) is PipeSignal.OPEN
    os.close(w)
    os.close(r)


def test_poll_pipe_open_with_unrelated_data():
    """poll_pipe reports OPEN when a non-takeover byte is available."""
    r, w = os.pipe()
    os.write(w, b"x")
    from lilbee.runtime._splash_runner import PipeSignal, poll_pipe

    assert poll_pipe(r) is PipeSignal.OPEN
    os.close(w)
    os.close(r)


def test_poll_pipe_returns_takeover_on_takeover_byte():
    """poll_pipe reports TAKEOVER when the parent sends the takeover byte."""
    r, w = os.pipe()
    from lilbee.runtime._splash_runner import TAKEOVER_BYTE, PipeSignal, poll_pipe

    os.write(w, TAKEOVER_BYTE)
    assert poll_pipe(r) is PipeSignal.TAKEOVER
    os.close(w)
    os.close(r)


def test_poll_pipe_returns_closed_on_bad_fd():
    """A closed fd reports CLOSED instead of raising EBADF (matches POSIX branch)."""
    r, w = os.pipe()
    os.close(w)
    os.close(r)
    from lilbee.runtime._splash_runner import PipeSignal, poll_pipe

    assert poll_pipe(r) is PipeSignal.CLOSED


def test_read_signal_with_bad_fd():
    """_read_signal reports CLOSED when os.read raises OSError."""
    from lilbee.runtime._splash_runner import PipeSignal, _read_signal

    assert _read_signal(-1) is PipeSignal.CLOSED


@pytest.mark.skipif(sys.platform == "win32", reason="select-based path is Unix-only")
def test_poll_pipe_select_error_returns_closed():
    """poll_pipe reports CLOSED when select raises."""
    from lilbee.runtime._splash_runner import PipeSignal, poll_pipe

    with patch("select.select", side_effect=ValueError("bad fd")):
        assert poll_pipe(-1) is PipeSignal.CLOSED


@pytest.mark.skipif(sys.platform == "win32", reason="select-based path is Unix-only")
@patch("os.read", side_effect=OSError("bad fd"))
@patch("select.select", return_value=([42], [], []))
def test_poll_pipe_read_error_returns_closed(_mock_select: object, _mock_read: object):
    """poll_pipe reports CLOSED when os.read raises after select succeeds."""
    from lilbee.runtime._splash_runner import PipeSignal, poll_pipe

    assert poll_pipe(42) is PipeSignal.CLOSED


def test_animation_loop_exits_on_closed_pipe():
    """animation_loop exits cleanly when pipe is already closed."""
    r, w = os.pipe()
    os.close(w)

    from lilbee.runtime._splash_runner import animation_loop

    animation_loop(r)
    os.close(r)


@patch("lilbee.runtime._splash_runner.STARTUP_DELAY", 0)
@patch("lilbee.runtime._splash_runner.FRAME_INTERVAL", 0.003)
@patch("lilbee.runtime._splash_runner.POLL_INTERVAL", 0.001)
@patch("time.sleep")
def test_animation_loop_renders_one_full_frame(_mock_sleep: object):
    """animation_loop renders at least one frame with move_up_and_clear."""
    from lilbee.runtime._splash_runner import PipeSignal, animation_loop

    call_count = 0

    def mock_poll_pipe(_fd: int) -> PipeSignal:
        nonlocal call_count
        call_count += 1
        # OPEN for the outer while (call 1), then CLOSED on the 2nd inner
        # loop iteration (call 3) to exercise the inner break path
        return PipeSignal.CLOSED if call_count >= 3 else PipeSignal.OPEN

    written: list[bytes] = []

    def mock_write(fd: int, data: bytes) -> int:
        written.append(data)
        return len(data)

    with (
        patch("lilbee.runtime._splash_runner.poll_pipe", side_effect=mock_poll_pipe),
        patch("os.write", side_effect=mock_write),
    ):
        animation_loop(0)

    # Should have written: HIDE_CURSOR, frame, then the final clear_screen
    assert len(written) >= 3
    assert b"\033[?25h" in written[-1]  # EOF exit restores the cursor


@patch("lilbee.runtime._splash_runner.STARTUP_DELAY", 0)
@patch("lilbee.runtime._splash_runner.FRAME_INTERVAL", 0.003)
@patch("lilbee.runtime._splash_runner.POLL_INTERVAL", 0.001)
@patch("time.sleep")
def test_animation_loop_takeover_writes_nothing_after_signal(_mock_sleep: object):
    """A TUI takeover exits without any frame clear or cursor-show write."""
    from lilbee.runtime._splash_runner import TAKEOVER_BYTE, animation_loop

    r, w = os.pipe()
    written: list[bytes] = []

    def mock_write(fd: int, data: bytes) -> int:
        written.append(data)
        return len(data)

    os.write(w, TAKEOVER_BYTE)
    with patch("os.write", side_effect=mock_write):
        animation_loop(r)
    os.close(w)
    os.close(r)

    joined = b"".join(written)
    assert b"\033[?25h" not in joined  # no cursor-show onto the alt-screen
    assert b"\033[2K" not in joined  # no line clears onto the alt-screen


def test_animation_loop_takeover_before_first_frame_writes_nothing():
    """A takeover during the startup delay exits before anything is drawn."""
    from lilbee.runtime._splash_runner import TAKEOVER_BYTE, animation_loop

    r, w = os.pipe()
    os.write(w, TAKEOVER_BYTE)

    written: list[bytes] = []

    def mock_write(fd: int, data: bytes) -> int:
        written.append(data)
        return len(data)

    with patch("os.write", side_effect=mock_write):
        animation_loop(r)
    os.close(w)
    os.close(r)

    assert written == []


@patch("lilbee.runtime._splash_runner.STARTUP_DELAY", 0)
@patch("lilbee.runtime._splash_runner.FRAME_INTERVAL", 0)
@patch("os.write", side_effect=OSError("broken"))
def test_animation_loop_handles_write_error(_mock_write: object):
    """animation_loop handles OSError during rendering."""
    from lilbee.runtime._splash_runner import animation_loop

    r, w = os.pipe()
    os.close(w)
    animation_loop(r)
    os.close(r)


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM not catchable on Windows")
@patch("lilbee.runtime._splash_runner.STARTUP_DELAY", 0)
@patch("lilbee.runtime._splash_runner.FRAME_INTERVAL", 0.05)
@patch("lilbee.runtime._splash_runner.POLL_INTERVAL", 0.001)
def test_animation_loop_exits_on_sigterm():
    """animation_loop exits when SIGTERM is received (covers line 140)."""
    import signal
    import threading

    from lilbee.runtime._splash_runner import animation_loop

    r, w = os.pipe()

    def send_sigterm():
        import time

        time.sleep(0.02)
        os.kill(os.getpid(), signal.SIGTERM)

    t = threading.Thread(target=send_sigterm)
    t.start()

    with patch("os.write", return_value=0):
        animation_loop(r)

    t.join()
    os.close(w)
    os.close(r)


@patch("lilbee.runtime._splash_runner.STARTUP_DELAY", 0.01)
@patch("lilbee.runtime._splash_runner.POLL_INTERVAL", 0.001)
def test_animation_loop_startup_delay_with_open_pipe():
    """animation_loop sleeps during startup delay when pipe is open."""
    from lilbee.runtime._splash_runner import animation_loop

    r, w = os.pipe()

    # Close pipe after a short delay to let startup delay loop run
    import threading

    def close_later():
        import time

        time.sleep(0.02)
        os.close(w)

    t = threading.Thread(target=close_later)
    t.start()

    with patch("os.write", return_value=0):
        animation_loop(r)

    t.join()
    os.close(r)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows poll_pipe path")
def test_poll_pipe_windows_path():
    """poll_pipe uses os.read on Windows."""
    from lilbee.runtime._splash_runner import PipeSignal, poll_pipe

    r, w = os.pipe()
    os.close(w)
    assert poll_pipe(r) is PipeSignal.CLOSED
    os.close(r)


def test_main_guard():
    """__main__ guard calls main()."""
    import runpy

    with (
        patch("lilbee.runtime._splash_runner.main"),
        pytest.raises(SystemExit),
    ):
        # Remove from sys.modules after patch setup (which imports it)
        # so runpy doesn't warn about pre-existing module
        saved = sys.modules.pop("lilbee.runtime._splash_runner", None)
        try:
            runpy.run_module("lilbee.runtime._splash_runner", run_name="__main__")
        finally:
            if saved is not None:
                sys.modules["lilbee.runtime._splash_runner"] = saved


def test_main_missing_args():
    """main exits with code 1 when no pipe_fd argument."""
    from lilbee.runtime._splash_runner import main

    with patch("sys.argv", ["_splash_runner"]), pytest.raises(SystemExit, match="1"):
        main()


def test_main_invalid_fd():
    """main exits with code 1 when pipe_fd is not an integer."""
    from lilbee.runtime._splash_runner import main

    with patch("sys.argv", ["_splash_runner", "abc"]), pytest.raises(SystemExit, match="1"):
        main()


def test_main_valid_fd():
    """main runs animation_loop with a valid pipe reference.

    argv carries what the parent passes: the fd on POSIX, the pipe's OS
    handle on Windows (fds do not survive process boundaries there).
    """
    r, w = os.pipe()
    os.close(w)

    if sys.platform == "win32":
        import msvcrt

        pipe_ref = str(msvcrt.get_osfhandle(r))
    else:
        pipe_ref = str(r)

    from lilbee.runtime._splash_runner import main

    with patch("sys.argv", ["_splash_runner", pipe_ref]):
        main()


def test_animation_loop_sleeps_during_startup_then_exits(monkeypatch):
    """The startup poll loop sleeps while the pipe is open, then returns once it
    closes. Covers the poll/sleep path deterministically; before this the line
    was hit only by the timing-dependent e2e subprocess test."""
    import lilbee.runtime._splash_runner as sr

    sleeps: list[float] = []
    polls = {"n": 0}

    def fake_poll(_fd: int) -> sr.PipeSignal:
        polls["n"] += 1
        return sr.PipeSignal.CLOSED if polls["n"] > 1 else sr.PipeSignal.OPEN

    monkeypatch.setattr(sr.time, "sleep", lambda interval: sleeps.append(interval))
    monkeypatch.setattr(sr, "poll_pipe", fake_poll)
    sr.animation_loop(0)
    assert sleeps == [sr.POLL_INTERVAL]


def test_open_pipe_ref_returns_the_fd_on_posix(monkeypatch):
    """POSIX passes the fd through unchanged; only Windows converts a handle."""
    monkeypatch.setattr("sys.platform", "linux")
    from lilbee.runtime._splash_runner import _open_pipe_ref

    assert _open_pipe_ref(7) == 7
