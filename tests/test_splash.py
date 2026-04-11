"""Tests for the splash animation lifecycle and runner modules."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from lilbee._splash_runner import (
    BEE_LINES,
    LOGO_WIDTH,
    apply_color,
    build_knight_rider_frames,
    build_logo_frames,
    clear_screen,
    move_up_and_clear,
    render_frame,
)
from lilbee.splash import (
    _SPLASH_FD_ENV,
    _should_skip,
    dismiss,
    start,
    stop,
)


class TestShouldSkip:
    def test_not_tty(self) -> None:
        with patch("lilbee.splash.os.isatty", return_value=False):
            assert _should_skip() is True

    def test_env_var_set(self) -> None:
        with (
            patch("lilbee.splash.os.isatty", return_value=True),
            patch.dict(os.environ, {"LILBEE_NO_SPLASH": "1"}),
        ):
            assert _should_skip() is True

    def test_tty_no_env_var(self) -> None:
        with (
            patch("lilbee.splash.os.isatty", return_value=True),
            patch.dict(os.environ, {}, clear=True),
        ):
            assert _should_skip() is False


class TestStartStop:
    def test_start_returns_none_when_skipped(self) -> None:
        with patch("lilbee.splash._should_skip", return_value=True):
            assert start() is None

    def test_stop_none_is_noop(self) -> None:
        stop(None)

    def test_start_and_stop(self) -> None:
        with patch("lilbee.splash._should_skip", return_value=False):
            handle = start()
            assert handle is not None
            assert handle.process.poll() is None  # still running
            assert _SPLASH_FD_ENV in os.environ
            stop(handle)
            assert handle.process.poll() is not None  # exited
            assert _SPLASH_FD_ENV not in os.environ

    def test_start_sets_env_var(self) -> None:
        with patch("lilbee.splash._should_skip", return_value=False):
            handle = start()
            assert handle is not None
            assert _SPLASH_FD_ENV in os.environ
            fd_str = os.environ[_SPLASH_FD_ENV]
            assert int(fd_str) == handle.write_fd
            stop(handle)


class TestDismiss:
    def test_dismiss_no_env_var(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            dismiss()  # should not raise

    def test_dismiss_closes_fd_and_clears_env(self) -> None:
        read_fd, write_fd = os.pipe()
        os.close(read_fd)
        os.environ[_SPLASH_FD_ENV] = str(write_fd)
        dismiss()
        assert _SPLASH_FD_ENV not in os.environ
        with pytest.raises(OSError):
            os.close(write_fd)  # already closed

    def test_dismiss_tolerates_already_closed_fd(self) -> None:
        read_fd, write_fd = os.pipe()
        os.close(read_fd)
        os.close(write_fd)
        os.environ[_SPLASH_FD_ENV] = str(write_fd)
        dismiss()  # should not raise
        assert _SPLASH_FD_ENV not in os.environ


class TestLogoFrames:
    def test_four_pulse_frames(self) -> None:
        frames = build_logo_frames()
        assert len(frames) == 4

    def test_logo_has_content(self) -> None:
        frames = build_logo_frames()
        assert any("@" in line for line in frames[0])

    def test_logo_color_pulsing(self) -> None:
        frames = build_logo_frames()
        assert "\033[38;5;214m" in frames[0][1]  # bright
        assert "\033[38;5;94m" in frames[2][1]  # dim


class TestKnightRiderFrames:
    def test_frame_count(self) -> None:
        frames = build_knight_rider_frames()
        assert len(frames) > 0
        assert len(frames) % 2 == 0  # symmetric oscillation

    def test_frames_contain_bar_chars(self) -> None:
        frames = build_knight_rider_frames()
        for frame in frames:
            assert "\u2593" in frame or "\u2592" in frame or "\u2591" in frame

    def test_frames_oscillate(self) -> None:
        frames = build_knight_rider_frames()
        half = len(frames) // 2
        head_positions = []
        for frame in frames:
            stripped = ""
            for ch in frame:
                if ch in ("\u2593", "\u2592", "\u2591", " "):
                    stripped += ch
            pos = stripped.find("\u2593")
            head_positions.append(pos)
        first_half = head_positions[:half]
        second_half = head_positions[half:]
        assert first_half[-1] > first_half[0]  # sweeps right
        assert second_half[-1] < second_half[0]  # sweeps left

    def test_bar_spans_logo_width(self) -> None:
        frames = build_knight_rider_frames()
        sweep_range = LOGO_WIDTH - 1
        assert len(frames) == sweep_range * 2


class TestApplyColor:
    def test_empty_line_unchanged(self) -> None:
        assert apply_color("   ", "\033[38;5;214m") == "   "

    def test_colored_line(self) -> None:
        result = apply_color("hello", "\033[38;5;214m")
        assert result.startswith("\033[38;5;214m")
        assert result.endswith("\033[0m")
        assert "hello" in result


class TestRenderFrame:
    def test_output_bytes(self) -> None:
        lines = ["line1", "line2"]
        result = render_frame(lines, "loading...")
        assert isinstance(result, bytes)
        text = result.decode()
        assert "line1" in text
        assert "line2" in text
        assert "loading..." in text


class TestClearScreen:
    def test_contains_escape_codes(self) -> None:
        result = clear_screen()
        text = result.decode()
        assert "\033[2J" in text
        assert "\033[H" in text
        assert "\033[?25h" in text


class TestMoveUpAndClear:
    def test_repeat_count(self) -> None:
        result = move_up_and_clear(3)
        text = result.decode()
        assert text.count("\033[A") == 3
        assert text.count("\033[2K") == 3


class TestBeeLines:
    def test_logo_width_constant(self) -> None:
        assert len(BEE_LINES[1]) == LOGO_WIDTH

    def test_logo_lines_count(self) -> None:
        assert len(BEE_LINES) == 12


class TestStopTimeout:
    def test_stop_kills_on_timeout(self) -> None:
        """stop() kills the process when it doesn't exit within timeout."""
        import subprocess
        from unittest.mock import MagicMock

        from lilbee.splash import SplashHandle, stop

        mock_proc = MagicMock()
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 3), None]
        handle = SplashHandle(process=mock_proc, write_fd=-1)

        with patch("lilbee.splash._close_write_fd"):
            stop(handle)

        mock_proc.kill.assert_called_once()
        assert mock_proc.wait.call_count == 2


class TestRestoreCursor:
    def test_restore_cursor_oserror(self) -> None:
        """_restore_cursor handles OSError from stderr."""
        from lilbee.splash import _restore_cursor

        with patch("sys.stderr") as mock_stderr:
            mock_stderr.write.side_effect = OSError("broken pipe")
            _restore_cursor()  # should not raise


class TestAtexitCleanup:
    def test_atexit_calls_stop(self) -> None:
        """_atexit_cleanup calls stop when handle is active."""
        import lilbee.splash as splash_mod
        from lilbee.splash import _atexit_cleanup

        mock_handle = object()
        original = splash_mod._active_handle
        splash_mod._active_handle = mock_handle
        try:
            with patch("lilbee.splash.stop") as mock_stop:
                _atexit_cleanup()
                mock_stop.assert_called_once_with(mock_handle)
        finally:
            splash_mod._active_handle = original

    def test_atexit_noop_when_none(self) -> None:
        """_atexit_cleanup is a no-op when no active handle."""
        import lilbee.splash as splash_mod
        from lilbee.splash import _atexit_cleanup

        original = splash_mod._active_handle
        splash_mod._active_handle = None
        try:
            with patch("lilbee.splash.stop") as mock_stop:
                _atexit_cleanup()
                mock_stop.assert_not_called()
        finally:
            splash_mod._active_handle = original
