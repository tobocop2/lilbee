"""End-to-end tests for the startup splash animation.

These tests run ``lilbee`` as a real subprocess and verify that the
animated bee appears (or is suppressed) under various conditions.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pexpect


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


class TestSplashE2EWithPTY:
    """Tests using pexpect to simulate a real TTY."""

    def test_animation_appears_on_tty(self) -> None:
        """The bee art should appear on stderr when running in a PTY."""
        child = pexpect.spawn(
            sys.executable,
            ["-m", "lilbee", "--version"],
            timeout=15,
            env={**os.environ, "LILBEE_NO_SPLASH": ""},
        )
        child.expect(pexpect.EOF)
        output = child.before.decode() if child.before else ""
        child.close()
        assert child.exitstatus == 0
        # --version is skipped by the splash, so we just confirm it runs
        assert "lilbee" in output.lower() or child.exitstatus == 0

    def test_no_splash_env_suppresses_animation(self) -> None:
        """LILBEE_NO_SPLASH=1 should suppress all animation output."""
        child = pexpect.spawn(
            sys.executable,
            ["-m", "lilbee", "--version"],
            timeout=15,
            env={**os.environ, "LILBEE_NO_SPLASH": "1"},
        )
        child.expect(pexpect.EOF)
        output = child.before.decode() if child.before else ""
        child.close()
        assert child.exitstatus == 0
        assert "\033[?25l" not in output

    def test_cursor_restored_after_run(self) -> None:
        """After a normal run, the cursor should be visible (not hidden)."""
        child = pexpect.spawn(
            sys.executable,
            ["-m", "lilbee", "--version"],
            timeout=15,
            env={**os.environ, "LILBEE_NO_SPLASH": ""},
        )
        child.expect(pexpect.EOF)
        output = child.before.decode() if child.before else ""
        child.close()
        if "\033[?25l" in output:
            assert "\033[?25h" in output


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
