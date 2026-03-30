"""Tests for the startup splash animation module."""

from __future__ import annotations

import os
import time
from unittest.mock import patch

import pytest

from lilbee._splash import (
    _LOADING_DOTS,
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
        with patch("lilbee._splash.os.isatty", return_value=False):
            assert _should_skip() is True

    def test_env_var_set(self) -> None:
        with (
            patch("lilbee._splash.os.isatty", return_value=True),
            patch.dict(os.environ, {"LILBEE_NO_SPLASH": "1"}),
        ):
            assert _should_skip() is True

    def test_env_var_empty(self) -> None:
        with (
            patch("lilbee._splash.os.isatty", return_value=True),
            patch.dict(os.environ, {"LILBEE_NO_SPLASH": ""}, clear=False),
        ):
            assert _should_skip() is False

    @pytest.mark.parametrize("flag", ["--version", "-V", "--help", "-h", "help"])
    def test_skip_flags(self, flag: str) -> None:
        with (
            patch("lilbee._splash.os.isatty", return_value=True),
            patch.dict(os.environ, {}, clear=False),
            patch("lilbee._splash.sys.argv", ["lilbee", flag]),
        ):
            assert _should_skip() is True

    @pytest.mark.parametrize("flag", ["--install-completion", "--show-completion"])
    def test_skip_completion_flags(self, flag: str) -> None:
        with (
            patch("lilbee._splash.os.isatty", return_value=True),
            patch.dict(os.environ, {}, clear=False),
            patch("lilbee._splash.sys.argv", ["lilbee", flag]),
        ):
            assert _should_skip() is True

    def test_normal_command_not_skipped(self) -> None:
        with (
            patch("lilbee._splash.os.isatty", return_value=True),
            patch.dict(os.environ, {}, clear=False),
            patch("lilbee._splash.sys.argv", ["lilbee", "status"]),
        ):
            assert _should_skip() is False

    def test_no_args_not_skipped(self) -> None:
        with (
            patch("lilbee._splash.os.isatty", return_value=True),
            patch.dict(os.environ, {}, clear=False),
            patch("lilbee._splash.sys.argv", ["lilbee"]),
        ):
            assert _should_skip() is False


class TestPickFrames:
    def test_large_terminal(self) -> None:
        with patch("lilbee._splash._get_terminal_size", return_value=(120, 50)):
            frame0, _frame1 = _pick_frames()
            # Large bee: 7 wing lines + 31 body lines
            assert len(frame0) == 38
            assert any("ZZZZ" in line for line in frame0)

    def test_small_terminal(self) -> None:
        with patch("lilbee._splash._get_terminal_size", return_value=(60, 20)):
            frame0, _frame1 = _pick_frames()
            # Small bee: 1 wing line + 4 body lines
            assert len(frame0) == 5
            assert "\\)" in frame0[0]


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
        assert text.count("\033[A") == 2

    def test_move_up_zero(self) -> None:
        result = _move_up_and_clear(0)
        assert result == b""


class TestStartStop:
    def test_start_returns_zero_when_skipped(self) -> None:
        with patch("lilbee._splash._should_skip", return_value=True):
            assert start() == 0

    def test_stop_zero_is_noop(self) -> None:
        stop(0)

    def test_stop_negative_pid_doesnt_crash(self) -> None:
        """Negative PIDs other than -1 are handled by threaded fallback."""
        stop(-999)

    def test_stop_nonexistent_pid(self) -> None:
        stop(999999)


class TestForkAnimation:
    @pytest.mark.skipif(not hasattr(os, "fork"), reason="No os.fork on this platform")
    def test_start_stop_fork(self) -> None:
        """Verify the fork-based animation starts and stops cleanly."""
        with (
            patch("lilbee._splash._should_skip", return_value=False),
            patch("lilbee._splash._get_terminal_size", return_value=(60, 20)),
            patch("lilbee._splash._STARTUP_DELAY", 0.0),
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
            from lilbee._splash import _BEE_SMALL_BODY, _SMALL_WINGS_SPREAD

            frame0 = _SMALL_WINGS_SPREAD + _BEE_SMALL_BODY
            from lilbee._splash import _render_frame

            os.write(2, _render_frame(frame0, "loading..."))
            os._exit(0)
        else:
            os.close(w_fd)
            os.waitpid(pid, 0)
            output = os.read(r_fd, 4096).decode()
            os.close(r_fd)
            assert "\\)" in output or "o" in output


class TestThreadedFallback:
    def test_start_stop_threaded(self) -> None:
        """Verify the threaded fallback starts and stops cleanly."""
        import lilbee._splash as mod

        with (
            patch("lilbee._splash._get_terminal_size", return_value=(60, 20)),
            patch("lilbee._splash._STARTUP_DELAY", 0.0),
        ):
            frame0 = mod._SMALL_WINGS_SPREAD + mod._BEE_SMALL_BODY
            frame1 = mod._SMALL_WINGS_TUCKED + mod._BEE_SMALL_BODY
            pid = _start_threaded(frame0, frame1)
            assert pid == -1
            time.sleep(0.05)
            _stop_threaded(pid)

    def test_start_uses_threading_when_no_fork(self) -> None:
        """On platforms without os.fork, start() falls back to threading."""
        pass  # Covered by TestNoForkFallback

    def test_stop_threaded_without_thread(self) -> None:
        """_stop_threaded with no active thread should not raise."""
        import lilbee._splash as mod

        mod._thread_obj = None
        _stop_threaded(-1)


class TestNoForkFallback:
    def test_start_falls_back_to_threading(self) -> None:
        """When os.fork is absent, start() uses threaded fallback."""
        with (
            patch("lilbee._splash._should_skip", return_value=False),
            patch("lilbee._splash._get_terminal_size", return_value=(60, 20)),
            patch("lilbee._splash._STARTUP_DELAY", 0.0),
            patch("lilbee._splash.hasattr", return_value=False),
            patch("lilbee._splash._start_threaded", return_value=-1) as mock_threaded,
        ):
            pid = start()
            assert pid == -1
            mock_threaded.assert_called_once()

    def test_stop_falls_back_to_threading(self) -> None:
        """When pid is -1 (threaded sentinel), stop() uses threaded fallback."""
        with patch("lilbee._splash._stop_threaded") as mock_stop:
            stop(-1)
            mock_stop.assert_called_once_with(-1)


class TestGetTerminalSize:
    def test_returns_tuple(self) -> None:
        from lilbee._splash import _get_terminal_size

        cols, rows = _get_terminal_size()
        assert isinstance(cols, int)
        assert isinstance(rows, int)

    def test_fallback_on_error(self) -> None:
        from lilbee._splash import _get_terminal_size

        with patch("lilbee._splash.os.get_terminal_size", side_effect=OSError):
            cols, rows = _get_terminal_size()
            assert cols == 80
            assert rows == 24

    def test_fallback_on_value_error(self) -> None:
        from lilbee._splash import _get_terminal_size

        with patch("lilbee._splash.os.get_terminal_size", side_effect=ValueError):
            cols, rows = _get_terminal_size()
            assert cols == 80
            assert rows == 24


class TestLoadingDots:
    def test_four_phases(self) -> None:
        assert len(_LOADING_DOTS) == 4
        assert _LOADING_DOTS[0] == "loading"
        assert _LOADING_DOTS[-1] == "loading..."
