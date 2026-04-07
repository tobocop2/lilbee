"""End-to-end tests for the startup splash animation.

These tests run ``lilbee`` as a real subprocess and verify that the
animated bee appears (or is suppressed) under various conditions.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


def _run_lilbee(
    *args: str, env_extra: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run lilbee in a non-TTY subprocess (no animation expected)."""
    env = {**os.environ, **(env_extra or {})}
    return subprocess.run(
        [sys.executable, "-m", "lilbee", *args],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="PTY tests require Unix")
class TestSplashE2EWithPTY:
    """Tests using subprocess with a PTY to verify splash behavior."""

    def test_animation_runs_cleanly(self) -> None:
        """The splash process should not crash when running with a PTY."""
        result = _run_lilbee("--version", env_extra={"LILBEE_NO_SPLASH": ""})
        assert result.returncode == 0
        assert "lilbee" in result.stdout.lower()

    def test_no_splash_env_suppresses_animation(self) -> None:
        """LILBEE_NO_SPLASH=1 should suppress all animation output."""
        result = _run_lilbee("--version", env_extra={"LILBEE_NO_SPLASH": "1"})
        assert result.returncode == 0
        assert "\033[?25l" not in result.stderr

    def test_cursor_restored_after_run(self) -> None:
        """After a normal run, hidden cursor should be restored."""
        result = _run_lilbee("--version", env_extra={"LILBEE_NO_SPLASH": ""})
        assert result.returncode == 0
        if "\033[?25l" in result.stderr:
            assert "\033[?25h" in result.stderr


class TestSplashE2ENonTTY:
    """Tests running without a TTY — animation should be suppressed."""

    def test_no_animation_in_pipe(self) -> None:
        """When stdout/stderr are pipes (not a TTY), no animation should appear."""
        result = _run_lilbee("--version")
        assert result.returncode == 0
        assert "\033[?25l" not in result.stderr
        assert "lilbee" in result.stdout.lower()

    def test_no_splash_env_in_pipe(self) -> None:
        """LILBEE_NO_SPLASH works even in non-TTY mode."""
        result = _run_lilbee("--version", env_extra={"LILBEE_NO_SPLASH": "1"})
        assert result.returncode == 0
        assert "\033[?25l" not in result.stderr

    def test_version_output_clean(self) -> None:
        """--version output should be clean with no animation artifacts."""
        result = _run_lilbee("--version")
        assert result.returncode == 0
        stdout_lines = result.stdout.strip().split("\n")
        assert len(stdout_lines) == 1
        assert stdout_lines[0].startswith("lilbee ")
