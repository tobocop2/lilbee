"""Tests for the splash animation lifecycle and runner modules."""

from __future__ import annotations

import io
import os
from unittest.mock import MagicMock, patch

import lilbee.runtime.splash as splash_mod
from lilbee.runtime._splash_runner import (
    BEE_LINES,
    LOGO_WIDTH,
    apply_color,
    build_knight_rider_frames,
    build_logo_frames,
    clear_screen,
    move_up_and_clear,
    render_frame,
)
from lilbee.runtime.splash import (
    SplashHandle,
    _should_skip,
    dismiss,
    start,
    stop,
)
from tests._pipe_eof import pipe_reaches_eof


class TestShouldSkip:
    def test_not_tty(self) -> None:
        with patch("lilbee.runtime.splash.os.isatty", return_value=False):
            assert _should_skip() is True

    def test_env_var_set(self) -> None:
        with (
            patch("lilbee.runtime.splash.os.isatty", return_value=True),
            patch.dict(os.environ, {"LILBEE_NO_SPLASH": "1"}),
        ):
            assert _should_skip() is True

    def test_tty_no_env_var(self) -> None:
        with (
            patch("lilbee.runtime.splash.sys.platform", "linux"),
            patch("lilbee.runtime.splash.os.isatty", return_value=True),
            patch.dict(os.environ, {}, clear=True),
        ):
            assert _should_skip() is False

    def test_win32_tty_does_not_skip(self) -> None:
        # Windows passes the pipe as an inheritable OS handle instead of an
        # fd, so the splash runs there like everywhere else.
        with (
            patch("lilbee.runtime.splash.sys.platform", "win32"),
            patch("lilbee.runtime.splash.os.isatty", return_value=True),
            patch.dict(os.environ, {}, clear=True),
        ):
            assert _should_skip() is False


class TestStartStop:
    def test_start_returns_none_when_skipped(self) -> None:
        with patch("lilbee.runtime.splash._should_skip", return_value=True):
            assert start() is None

    def test_stop_none_is_noop(self) -> None:
        stop(None)

    def test_start_and_stop(self) -> None:
        env_before = dict(os.environ)
        with patch("lilbee.runtime.splash._should_skip", return_value=False):
            handle = start()
            assert handle is not None
            assert handle.process.poll() is None  # still running
            assert dict(os.environ) == env_before
            stop(handle)
            assert handle.process.poll() is not None  # exited
            assert dict(os.environ) == env_before


def _arm(handle: object) -> object:
    """Make *handle* the active splash and return the previous one."""
    import lilbee.runtime.splash as splash_mod

    original = splash_mod._active_handle
    splash_mod._active_handle = handle  # type: ignore[assignment]
    return original


def _start_with_known_write_fd() -> tuple[SplashHandle, int]:
    """Start a splash with no child process and return it with its write-end number."""
    real_pipe = os.pipe
    pipe_ends: list[int] = []

    def recording_pipe() -> tuple[int, int]:
        ends = real_pipe()
        pipe_ends.extend(ends)
        return ends

    with (
        patch("lilbee.runtime.splash._should_skip", return_value=False),
        patch("lilbee.runtime.splash.subprocess.Popen"),
        patch("lilbee.runtime.splash.atexit.register"),
        patch("lilbee.runtime.splash.os.pipe", side_effect=recording_pipe),
    ):
        handle = start()
    assert handle is not None
    _, write_fd = pipe_ends
    return handle, write_fd


class TestDismiss:
    def test_dismiss_no_handle(self) -> None:
        original = _arm(None)
        try:
            dismiss()  # should not raise
        finally:
            _arm(original)

    def test_dismiss_closes_fd_and_clears_handle(self) -> None:
        read_fd, write_fd = os.pipe()
        pipe = io.FileIO(write_fd, "wb")  # held here, so garbage collection cannot close it
        original = _arm(SplashHandle(process=MagicMock(), pipe=pipe))
        try:
            dismiss()
            assert splash_mod._active_handle is None
            assert pipe_reaches_eof(read_fd)
        finally:
            pipe.close()
            os.close(read_fd)
            _arm(original)

    def test_dismiss_sends_takeover_byte_before_closing(self) -> None:
        """dismiss() tells the child the TUI owns the terminal, then EOFs."""
        from lilbee.runtime._splash_runner import TAKEOVER_BYTE

        read_fd, write_fd = os.pipe()
        original = _arm(SplashHandle(process=MagicMock(), pipe=io.FileIO(write_fd, "wb")))
        try:
            dismiss()
            assert os.read(read_fd, 2) == TAKEOVER_BYTE
            assert os.read(read_fd, 1) == b""  # write end closed after the byte
        finally:
            os.close(read_fd)
            _arm(original)

    def test_dismiss_tolerates_an_already_closed_pipe(self) -> None:
        import lilbee.runtime.splash as splash_mod

        read_fd, write_fd = os.pipe()
        os.close(read_fd)
        pipe = io.FileIO(write_fd, "wb")
        pipe.close()
        original = _arm(SplashHandle(process=MagicMock(), pipe=pipe))
        try:
            dismiss()  # should not raise
            assert splash_mod._active_handle is None
        finally:
            _arm(original)

    def test_second_dismiss_leaves_a_reused_descriptor_open(self) -> None:
        """A dismiss after the pipe is closed never closes the number's next owner."""
        handle, splash_write_fd = _start_with_known_write_fd()
        new_read, new_write = os.pipe()
        original = _arm(handle)
        try:
            dismiss()
            os.dup2(new_read, splash_write_fd)
            _arm(handle)
            dismiss()
            os.fstat(splash_write_fd)  # raises EBADF if dismiss closed it
        finally:
            _arm(original)
            for fd in (splash_write_fd, new_read, new_write):
                os.close(fd)

    def test_dismiss_writes_no_environment(self) -> None:

        env_before = dict(os.environ)
        read_fd, write_fd = os.pipe()
        os.close(read_fd)
        original = _arm(SplashHandle(process=MagicMock(), pipe=io.FileIO(write_fd, "wb")))
        try:
            dismiss()
            assert dict(os.environ) == env_before
        finally:
            _arm(original)

    def test_dismiss_waits_for_process_and_clears_handle(self) -> None:
        """dismiss() waits for the subprocess and clears _active_handle
        so atexit does not re-run stop() while Textual owns the terminal.
        """
        import lilbee.runtime.splash as splash_mod

        original = splash_mod._active_handle
        try:
            with patch("lilbee.runtime.splash._should_skip", return_value=False):
                handle = start()
                assert handle is not None
                splash_mod._active_handle = handle

                dismiss()

                assert splash_mod._active_handle is None
                assert handle.process.poll() is not None  # exited
        finally:
            splash_mod._active_handle = original

    def test_dismiss_kills_on_timeout(self) -> None:
        """dismiss() kills the subprocess when it doesn't exit within timeout."""
        import subprocess

        import lilbee.runtime.splash as splash_mod

        original = splash_mod._active_handle
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 3), None]
        handle = SplashHandle(process=mock_proc, pipe=MagicMock())
        splash_mod._active_handle = handle
        try:
            dismiss()
            mock_proc.kill.assert_called_once()
            assert splash_mod._active_handle is None
        finally:
            splash_mod._active_handle = original


class TestLogoFrames:
    def test_four_pulse_frames(self) -> None:
        frames = build_logo_frames()
        assert len(frames) == 4

    def test_logo_has_content(self) -> None:
        frames = build_logo_frames()
        assert any("@" in line for line in frames[0])

    def test_logo_color_pulsing(self) -> None:
        frames = build_logo_frames()
        assert "\033[38;5;217m" in frames[0][1]  # bright
        assert "\033[38;5;95m" in frames[2][1]  # dim


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
        assert apply_color("   ", "\033[38;5;217m") == "   "

    def test_colored_line(self) -> None:
        result = apply_color("hello", "\033[38;5;217m")
        assert result.startswith("\033[38;5;217m")
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
    def test_uses_line_erase_not_cursor_home(self) -> None:
        result = clear_screen(5)
        text = result.decode()
        assert "\033[A" in text  # move-up
        assert "\033[2K" in text  # clear-line
        assert "\033[?25h" in text  # cursor restore
        assert "\033[2J" not in text  # no full-screen clear
        assert "\033[H" not in text  # no cursor home


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
    def test_second_stop_leaves_a_reused_descriptor_open(self) -> None:
        """A stop after the pipe is closed never closes the number's next owner."""
        original = splash_mod._active_handle
        handle, splash_write_fd = _start_with_known_write_fd()
        new_read, new_write = os.pipe()
        try:
            stop(handle)
            os.dup2(new_read, splash_write_fd)
            stop(handle)
            os.fstat(splash_write_fd)  # raises EBADF if stop closed it
        finally:
            splash_mod._active_handle = original
            for fd in (splash_write_fd, new_read, new_write):
                os.close(fd)

    def test_stop_kills_on_timeout(self) -> None:
        """stop() kills the process when it doesn't exit within timeout."""
        import subprocess
        from unittest.mock import MagicMock

        mock_proc = MagicMock()
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 3), None]
        handle = SplashHandle(process=mock_proc, pipe=MagicMock())

        stop(handle)

        mock_proc.kill.assert_called_once()
        assert mock_proc.wait.call_count == 2


class TestRestoreCursor:
    def test_restore_cursor_oserror(self) -> None:
        """_restore_cursor handles OSError from stderr."""
        from lilbee.runtime.splash import _restore_cursor

        with patch("sys.stderr") as mock_stderr:
            mock_stderr.write.side_effect = OSError("broken pipe")
            _restore_cursor()  # should not raise


class TestAtexitCleanup:
    def test_atexit_calls_stop(self) -> None:
        """_atexit_cleanup calls stop when handle is active."""
        import lilbee.runtime.splash as splash_mod
        from lilbee.runtime.splash import _atexit_cleanup

        mock_handle = object()
        original = splash_mod._active_handle
        splash_mod._active_handle = mock_handle
        try:
            with patch("lilbee.runtime.splash.stop") as mock_stop:
                _atexit_cleanup()
                mock_stop.assert_called_once_with(mock_handle)
        finally:
            splash_mod._active_handle = original

    def test_atexit_noop_when_none(self) -> None:
        """_atexit_cleanup is a no-op when no active handle."""
        import lilbee.runtime.splash as splash_mod
        from lilbee.runtime.splash import _atexit_cleanup

        original = splash_mod._active_handle
        splash_mod._active_handle = None
        try:
            with patch("lilbee.runtime.splash.stop") as mock_stop:
                _atexit_cleanup()
                mock_stop.assert_not_called()
        finally:
            splash_mod._active_handle = original
