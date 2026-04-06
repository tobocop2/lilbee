"""Tests for the startup splash animation module."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

from lilbee.splash import (
    _clear_frame,
    _move_up_and_clear,
    _pick_frames,
    _render_frame,
    _should_skip,
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


class TestStartStop:
    def test_start_returns_zero_when_skipped(self) -> None:
        with patch("lilbee.splash._should_skip", return_value=True):
            assert start() == 0

    def test_stop_zero_is_noop(self) -> None:
        stop(0)

    @pytest.mark.skipif(sys.platform == "win32", reason="os.fork not available on Windows")
    def test_start_stops_work(self) -> None:
        """Verify the animation starts and stops cleanly."""
        with (
            patch("lilbee.splash._should_skip", return_value=False),
            patch("lilbee.splash._STARTUP_DELAY", 0.0),
        ):
            pid = start()
            assert pid > 0
            stop(pid)

    @pytest.mark.skipif(sys.platform == "win32", reason="os.fork not available on Windows")
    def test_start_with_ready_file(self, tmp_path: pytest.TempPathFactory) -> None:
        """Verify start() accepts ready_file parameter."""
        ready_file = tmp_path / "ready"
        ready_file.touch()

        with (
            patch("lilbee.splash._should_skip", return_value=False),
            patch("lilbee.splash._STARTUP_DELAY", 0.0),
        ):
            pid = start(ready_file=str(ready_file))
            assert pid > 0
            stop(pid)


class TestPickFrames:
    def test_returns_4_logo_frames(self) -> None:
        frame0, frame1, frame2, frame3 = _pick_frames()
        assert len(frame0) == 12
        assert len(frame1) == 12
        assert len(frame2) == 12
        assert len(frame3) == 12
        assert any("@" in line for line in frame0)


class TestRenderFrame:
    def test_output_bytes(self) -> None:
        lines = ["line1", "line2"]
        result = _render_frame(lines, "loading...")
        assert isinstance(result, bytes)
        text = result.decode()
        assert "line1" in text
        assert "line2" in text
        assert "loading..." in text


class TestClearFrame:
    def test_contains_escape_codes(self) -> None:
        result = _clear_frame(2)
        text = result.decode()
        assert "\033[2J" in text
        assert "\033[H" in text
        assert "\033[?25h" in text


class TestMoveUpAndClear:
    def test_repeat_count(self) -> None:
        result = _move_up_and_clear(3)
        text = result.decode()
        assert text.count("\033[A") == 3
        assert text.count("\033[2K") == 3


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
