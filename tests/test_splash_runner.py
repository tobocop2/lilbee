"""Tests for _splash_runner.py — animation subprocess utilities."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest


def test_apply_color_non_empty():
    from lilbee._splash_runner import AMBER_BRIGHT, RESET, apply_color

    result = apply_color("hello", AMBER_BRIGHT)
    assert result == AMBER_BRIGHT + "hello" + RESET


def test_apply_color_empty_line():
    from lilbee._splash_runner import apply_color

    assert apply_color("   ", "color") == "   "


def test_build_logo_frames():
    from lilbee._splash_runner import COLOR_SEQUENCE, build_logo_frames

    frames = build_logo_frames()
    assert len(frames) == len(COLOR_SEQUENCE)
    assert all(isinstance(f, list) for f in frames)


def test_build_knight_rider_frames():
    from lilbee._splash_runner import LOGO_WIDTH, build_knight_rider_frames

    frames = build_knight_rider_frames()
    assert len(frames) == (LOGO_WIDTH - 1) * 2


def test_render_frame():
    from lilbee._splash_runner import render_frame

    result = render_frame(["line1", "line2"], "bar")
    assert isinstance(result, bytes)
    assert b"line1" in result
    assert b"bar" in result


def test_move_up_and_clear():
    from lilbee._splash_runner import move_up_and_clear

    result = move_up_and_clear(3)
    assert isinstance(result, bytes)
    assert result.count(b"\033[A") == 3


def test_clear_screen():
    from lilbee._splash_runner import clear_screen

    result = clear_screen()
    assert b"\033[2J" in result
    assert b"\033[?25h" in result


def test_pipe_closed_returns_true_on_eof():
    """pipe_closed returns True when the read end gets EOF."""
    r, w = os.pipe()
    os.close(w)  # close write end -> read gets EOF
    from lilbee._splash_runner import pipe_closed

    assert pipe_closed(r) is True
    os.close(r)


def test_pipe_closed_returns_false_when_open():
    """pipe_closed returns False when pipe is still open."""
    r, w = os.pipe()
    from lilbee._splash_runner import pipe_closed

    assert pipe_closed(r) is False
    os.close(w)
    os.close(r)


def test_pipe_closed_with_data_available():
    """pipe_closed returns False when data is written but pipe not closed."""
    r, w = os.pipe()
    os.write(w, b"x")
    from lilbee._splash_runner import pipe_closed

    assert pipe_closed(r) is False
    os.close(w)
    os.close(r)


def test_read_eof_with_bad_fd():
    """_read_eof returns True when os.read raises OSError."""
    from lilbee._splash_runner import _read_eof

    assert _read_eof(-1) is True


@pytest.mark.skipif(sys.platform == "win32", reason="select-based path is Unix-only")
def test_pipe_closed_select_error_returns_true():
    """pipe_closed returns True when select raises."""
    from lilbee._splash_runner import pipe_closed

    with patch("select.select", side_effect=ValueError("bad fd")):
        assert pipe_closed(-1) is True


@pytest.mark.skipif(sys.platform == "win32", reason="select-based path is Unix-only")
@patch("os.read", side_effect=OSError("bad fd"))
@patch("select.select", return_value=([42], [], []))
def test_pipe_closed_read_error_returns_true(_mock_select: object, _mock_read: object):
    """pipe_closed returns True when os.read raises after select succeeds."""
    from lilbee._splash_runner import pipe_closed

    assert pipe_closed(42) is True


def test_animation_loop_exits_on_closed_pipe():
    """animation_loop exits cleanly when pipe is already closed."""
    r, w = os.pipe()
    os.close(w)

    from lilbee._splash_runner import animation_loop

    animation_loop(r)
    os.close(r)


@patch("lilbee._splash_runner.STARTUP_DELAY", 0)
@patch("lilbee._splash_runner.FRAME_INTERVAL", 0.003)
@patch("lilbee._splash_runner.POLL_INTERVAL", 0.001)
@patch("time.sleep")
def test_animation_loop_renders_one_full_frame(_mock_sleep: object):
    """animation_loop renders at least one frame with move_up_and_clear."""
    from lilbee._splash_runner import animation_loop

    call_count = 0

    def mock_pipe_closed(_fd: int) -> bool:
        nonlocal call_count
        call_count += 1
        # Let startup delay pass (returns False), one full frame render,
        # then close on the second frame check
        return call_count > 20

    written: list[bytes] = []

    def mock_write(fd: int, data: bytes) -> int:
        written.append(data)
        return len(data)

    with (
        patch("lilbee._splash_runner.pipe_closed", side_effect=mock_pipe_closed),
        patch("os.write", side_effect=mock_write),
    ):
        animation_loop(0)

    # Should have written: HIDE_CURSOR, frame, move_up_and_clear, possibly more
    assert len(written) >= 3


@patch("lilbee._splash_runner.STARTUP_DELAY", 0)
@patch("lilbee._splash_runner.FRAME_INTERVAL", 0)
@patch("os.write", side_effect=OSError("broken"))
def test_animation_loop_handles_write_error(_mock_write: object):
    """animation_loop handles OSError during rendering."""
    from lilbee._splash_runner import animation_loop

    r, w = os.pipe()
    os.close(w)
    animation_loop(r)
    os.close(r)


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM not catchable on Windows")
@patch("lilbee._splash_runner.STARTUP_DELAY", 0)
@patch("lilbee._splash_runner.FRAME_INTERVAL", 0.05)
@patch("lilbee._splash_runner.POLL_INTERVAL", 0.001)
def test_animation_loop_exits_on_sigterm():
    """animation_loop exits when SIGTERM is received (covers line 140)."""
    import signal
    import threading

    from lilbee._splash_runner import animation_loop

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


@patch("lilbee._splash_runner.STARTUP_DELAY", 0.01)
@patch("lilbee._splash_runner.POLL_INTERVAL", 0.001)
def test_animation_loop_startup_delay_with_open_pipe():
    """animation_loop sleeps during startup delay when pipe is open."""
    from lilbee._splash_runner import animation_loop

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


@pytest.mark.skipif(sys.platform != "win32", reason="Windows pipe_closed path")
def test_pipe_closed_windows_path():
    """pipe_closed uses os.read on Windows."""
    from lilbee._splash_runner import pipe_closed

    r, w = os.pipe()
    os.close(w)
    assert pipe_closed(r) is True
    os.close(r)


def test_main_guard():
    """__main__ guard calls main()."""
    import runpy

    with (
        patch("lilbee._splash_runner.main"),
        pytest.raises(SystemExit),
    ):
        # Remove from sys.modules after patch setup (which imports it)
        # so runpy doesn't warn about pre-existing module
        saved = sys.modules.pop("lilbee._splash_runner", None)
        try:
            runpy.run_module("lilbee._splash_runner", run_name="__main__")
        finally:
            if saved is not None:
                sys.modules["lilbee._splash_runner"] = saved


def test_main_missing_args():
    """main exits with code 1 when no pipe_fd argument."""
    from lilbee._splash_runner import main

    with patch("sys.argv", ["_splash_runner"]), pytest.raises(SystemExit, match="1"):
        main()


def test_main_invalid_fd():
    """main exits with code 1 when pipe_fd is not an integer."""
    from lilbee._splash_runner import main

    with patch("sys.argv", ["_splash_runner", "abc"]), pytest.raises(SystemExit, match="1"):
        main()


def test_main_valid_fd():
    """main runs animation_loop with a valid pipe fd."""
    r, w = os.pipe()
    os.close(w)

    from lilbee._splash_runner import main

    with patch("sys.argv", ["_splash_runner", str(r)]):
        main()
