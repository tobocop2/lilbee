"""Tests for the startup splash animation module."""

from __future__ import annotations

import os
import time
from unittest.mock import patch

import pytest

from lilbee.splash import (
    _clear_frame,
    _move_up_and_clear,
    _pick_frames,
    _render_frame,
    _should_skip,
    _start_threaded,
    _stop_threaded,
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

    def test_env_var_empty(self) -> None:
        with (
            patch("lilbee.splash.os.isatty", return_value=True),
            patch.dict(os.environ, {"LILBEE_NO_SPLASH": ""}, clear=False),
        ):
            assert _should_skip() is False

    @pytest.mark.parametrize("flag", ["--version", "-V", "--help", "-h", "help"])
    def test_skip_flags(self, flag: str) -> None:
        with (
            patch("lilbee.splash.os.isatty", return_value=True),
            patch.dict(os.environ, {}, clear=False),
            patch("lilbee.splash.sys.argv", ["lilbee", flag]),
        ):
            assert _should_skip() is True

    @pytest.mark.parametrize("flag", ["--install-completion", "--show-completion"])
    def test_skip_completion_flags(self, flag: str) -> None:
        with (
            patch("lilbee.splash.os.isatty", return_value=True),
            patch.dict(os.environ, {}, clear=False),
            patch("lilbee.splash.sys.argv", ["lilbee", flag]),
        ):
            assert _should_skip() is True

    def test_normal_command_not_skipped(self) -> None:
        with (
            patch("lilbee.splash.os.isatty", return_value=True),
            patch.dict(os.environ, {}, clear=False),
            patch("lilbee.splash.sys.argv", ["lilbee", "status"]),
        ):
            assert _should_skip() is False

    def test_no_args_not_skipped(self) -> None:
        with (
            patch("lilbee.splash.os.isatty", return_value=True),
            patch.dict(os.environ, {}, clear=False),
            patch("lilbee.splash.sys.argv", ["lilbee"]),
        ):
            assert _should_skip() is False


class TestPickFrames:
    def test_returns_4_logo_frames(self) -> None:
        frame0, frame1, frame2, frame3 = _pick_frames()
        # Logo has 12 lines (poison font style)
        assert len(frame0) == 12
        # Should return 4 frames for color pulsing animation
        assert len(frame1) == 12
        assert len(frame2) == 12
        assert len(frame3) == 12
        # Check it's the lilbee logo
        assert any("lilbee" in line.lower() or "@@" in line for line in frame0)


class TestStartWithReadyFile:
    @pytest.mark.skipif(not hasattr(os, "fork"), reason="No os.fork on this platform")
    def test_start_with_ready_file(self) -> None:
        """Verify start() accepts ready_file parameter."""
        with (
            patch("lilbee.splash._should_skip", return_value=False),
            patch("lilbee.splash._STARTUP_DELAY", 0.0),
        ):
            pid = start(ready_file="/tmp/test-ready")
            assert pid > 0
            stop(pid)


class TestRenderFrame:
    def test_output_bytes(self) -> None:
        lines = ["line1", "line2"]
        result = _render_frame(lines, "loading...")
        assert isinstance(result, bytes)
        text = result.decode()
        assert "line1" in text
        assert "line2" in text
        assert "loading..." in text

    def test_includes_blank_line(self) -> None:
        lines = ["art"]
        result = _render_frame(lines, "loading").decode()
        parts = result.split("\n")
        # art, blank, loading text, trailing newline
        assert parts[0] == "art"
        assert parts[1] == ""
        assert "loading" in parts[2]


class TestAnsiHelpers:
    def test_move_up_and_clear(self) -> None:
        result = _move_up_and_clear(3)
        assert isinstance(result, bytes)
        text = result.decode()
        assert text.count("\033[A") == 3
        assert text.count("\033[2K") == 3

    def test_clear_frame(self) -> None:
        result = _clear_frame(2)
        text = result.decode()
        assert "\033[?25h" in text  # show cursor
        assert "\033[2J" in text  # clear screen
        assert "\033[H" in text  # home cursor

    def test_move_up_zero(self) -> None:
        result = _move_up_and_clear(0)
        assert result == b""


class TestStartStop:
    def test_start_returns_zero_when_skipped(self) -> None:
        with patch("lilbee.splash._should_skip", return_value=True):
            assert start() == 0

    def test_stop_zero_is_noop(self) -> None:
        stop(0)

    def test_stop_negative_pid_doesnt_crash(self) -> None:
        """Negative PIDs other than -1 are handled by threaded fallback."""
        stop(-999)

    def test_stop_nonexistent_pid(self) -> None:
        stop(999999)

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="No os.fork on this platform")
    def test_stop_with_no_fork_attr(self) -> None:
        """When os.fork doesn't exist, stop uses threaded fallback."""
        with (
            patch("lilbee.splash.hasattr", return_value=False),
            patch("lilbee.splash._stop_threaded") as mock,
        ):
            stop(123)
            mock.assert_called_once_with(123)


class TestForkAnimation:
    @pytest.mark.skipif(not hasattr(os, "fork"), reason="No os.fork on this platform")
    def test_start_stop_fork(self) -> None:
        """Verify the fork-based animation starts and stops cleanly."""
        with (
            patch("lilbee.splash._should_skip", return_value=False),
            patch("lilbee.splash._STARTUP_DELAY", 0.0),
        ):
            pid = start()
            assert pid > 0
            time.sleep(0.05)
            stop(pid)

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="No os.fork on this platform")
    def test_animation_writes_to_stderr(self) -> None:
        """Verify the child writes animation frames to stderr."""
        r_fd, w_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            # Child: redirect stderr to pipe, write one frame
            os.dup2(w_fd, 2)
            os.close(r_fd)
            os.close(w_fd)
            from lilbee.splash import _BEE_LINES, _render_frame

            os.write(2, _render_frame(_BEE_LINES, "loading..."))
            os._exit(0)
        else:
            os.close(w_fd)
            os.waitpid(pid, 0)
            output = os.read(r_fd, 4096).decode()
            os.close(r_fd)
            assert "@" in output or "lilbee" in output.lower()


class TestThreadedFallback:
    def test_start_stop_threaded(self) -> None:
        """Verify the threaded fallback starts and stops cleanly."""
        import lilbee.splash as mod

        with patch("lilbee.splash._STARTUP_DELAY", 0.0):
            frame0 = mod._BEE_LINES
            frame1 = mod._BEE_LINES
            pid = _start_threaded(frame0, frame1, frame0, frame1)
            assert pid == -1
            time.sleep(0.05)
            _stop_threaded(pid)

    def test_start_threaded_with_ready_file(self, tmp_path: pytest.TempPathFactory) -> None:
        """Verify threaded fallback respects ready file."""
        import lilbee.splash as mod

        ready_file = tmp_path / "ready"
        ready_file.touch()

        with patch("lilbee.splash._STARTUP_DELAY", 0.0):
            frame0 = mod._BEE_LINES
            frame1 = mod._BEE_LINES
            pid = _start_threaded(frame0, frame1, frame0, frame1, ready_file=str(ready_file))
            assert pid == -1
            time.sleep(0.1)
            _stop_threaded(pid)

    def test_start_uses_threading_when_no_fork(self) -> None:
        """On platforms without os.fork, start() falls back to threading."""
        pass  # Covered by TestNoForkFallback

    def test_stop_threaded_without_thread(self) -> None:
        """_stop_threaded with no active thread should not raise."""
        import lilbee.splash as mod

        mod._thread_obj = None
        _stop_threaded(-1)


class TestNoForkFallback:
    def test_start_falls_back_to_threading(self) -> None:
        """When os.fork is absent, start() uses threaded fallback."""
        with (
            patch("lilbee.splash._should_skip", return_value=False),
            patch("lilbee.splash._STARTUP_DELAY", 0.0),
            patch("lilbee.splash.hasattr", return_value=False),
            patch("lilbee.splash._start_threaded", return_value=-1) as mock_threaded,
        ):
            pid = start()
            assert pid == -1
            mock_threaded.assert_called_once()

    def test_stop_falls_back_to_threading(self) -> None:
        """When pid is -1 (threaded sentinel), stop() uses threaded fallback."""
        with patch("lilbee.splash._stop_threaded") as mock_stop:
            stop(-1)
            mock_stop.assert_called_once_with(-1)


class TestLogoFrames:
    def test_four_pulse_frames(self) -> None:
        from lilbee.splash import _LOGO_FRAMES

        assert len(_LOGO_FRAMES) == 4

    def test_logo_has_poison_style(self) -> None:
        frame0 = _pick_frames()[0]
        assert any("@" in line for line in frame0)

    def test_logo_color_pulsing(self) -> None:
        from lilbee.splash import _LOGO_FRAMES

        assert "\033[38;5;214m" in _LOGO_FRAMES[0][1]
        assert "\033[38;5;94m" in _LOGO_FRAMES[2][1]


class TestKnightRiderFrames:
    def test_frame_count(self) -> None:
        from lilbee.splash import _KNIGHT_RIDER_FRAMES

        assert len(_KNIGHT_RIDER_FRAMES) == 22

    def test_frames_contain_ansi(self) -> None:
        from lilbee.splash import _KNIGHT_RIDER_FRAMES

        for frame in _KNIGHT_RIDER_FRAMES:
            assert "\033[" in frame
            assert "▓" in frame or "▒" in frame or "░" in frame

    def test_frames_oscillate(self) -> None:
        from lilbee.splash import _KNIGHT_RIDER_FRAMES

        head_positions = []
        for frame in _KNIGHT_RIDER_FRAMES:
            pos = frame.find("▓")
            head_positions.append(pos)

        first_half = head_positions[:11]
        second_half = head_positions[11:]
        assert max(first_half) >= min(first_half)
        assert max(second_half) <= max(first_half)
        assert head_positions[0] == head_positions[-1]


class TestIsReady:
    def test_returns_false_when_none(self) -> None:
        from lilbee.splash import _is_ready

        assert _is_ready(None) is False

    def test_returns_false_when_file_not_exists(self, tmp_path: pytest.TempPathFactory) -> None:
        from lilbee.splash import _is_ready

        fake_file = tmp_path / "nonexistent_ready_file"
        assert _is_ready(str(fake_file)) is False

    def test_returns_true_when_file_exists(self, tmp_path: pytest.TempPathFactory) -> None:
        from lilbee.splash import _is_ready

        ready_file = tmp_path / "ready_file"
        ready_file.touch()
        assert _is_ready(str(ready_file)) is True

    def test_returns_false_on_oserror(self, tmp_path: pytest.TempPathFactory) -> None:
        """_is_ready should return False on OSError (e.g., permission denied)."""
        from lilbee.splash import _is_ready

        with patch("lilbee.splash.os.path.exists", side_effect=OSError("permission denied")):
            assert _is_ready("/some/path") is False


class TestReadyFileConstant:
    def test_ready_file_constant(self) -> None:
        from lilbee.cli.tui.app import _READY_FILE as APP_READY
        from lilbee.launcher import _READY_FILE as LAUNCHER_READY
        from lilbee.splash import _READY_FILE

        assert _READY_FILE == "lilbee-splash-ready"
        assert LAUNCHER_READY == "lilbee-splash-ready"
        assert APP_READY == "lilbee-splash-ready"
        assert _READY_FILE == LAUNCHER_READY == APP_READY
