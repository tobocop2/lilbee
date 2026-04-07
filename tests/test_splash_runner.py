"""Tests for _splash_runner.py — animation subprocess utilities."""

from __future__ import annotations

import contextlib
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


@pytest.mark.skipif(sys.platform == "win32", reason="select-based path is Unix-only")
def test_pipe_closed_select_error_returns_true():
    """pipe_closed returns True when select raises."""
    from lilbee._splash_runner import pipe_closed

    with patch("select.select", side_effect=ValueError("bad fd")):
        assert pipe_closed(-1) is True


def test_animation_loop_exits_on_closed_pipe():
    """animation_loop exits cleanly when pipe is already closed."""
    r, w = os.pipe()
    os.close(w)

    from lilbee._splash_runner import animation_loop

    animation_loop(r)
    os.close(r)


@patch("lilbee._splash_runner.STARTUP_DELAY", 0)
@patch("lilbee._splash_runner.FRAME_INTERVAL", 0)
@patch("lilbee._splash_runner.POLL_INTERVAL", 0.001)
def test_animation_loop_renders_frames():
    """animation_loop renders at least one frame before pipe closes."""

    from lilbee._splash_runner import animation_loop

    r, w = os.pipe()
    writes: list[bytes] = []

    original_write = os.write

    def capture_write(fd: int, data: bytes) -> int:
        if fd == 2:
            writes.append(data)
            # Close pipe after first frame to stop the loop
            if len(writes) >= 2:
                with contextlib.suppress(OSError):
                    os.close(w)
        return original_write(fd, data)

    with patch("os.write", side_effect=capture_write):
        animation_loop(r)

    with contextlib.suppress(OSError):
        os.close(r)
    assert len(writes) >= 1


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
