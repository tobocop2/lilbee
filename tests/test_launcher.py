"""Tests for the ``lilbee`` console-script entry point in ``runtime.launcher``."""

from __future__ import annotations

import io
import os
import sys
from unittest.mock import Mock

import pytest

import lilbee.runtime.launcher as launcher
from lilbee.runtime.launcher import _COMPLETION_ENV_SUFFIX


@pytest.fixture
def _no_completion_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip any ``_*_COMPLETE`` vars the surrounding shell may have set."""
    for key in list(os.environ):
        if key.startswith("_") and key.endswith(_COMPLETION_ENV_SUFFIX):
            monkeypatch.delenv(key, raising=False)


def _patch_app_and_splash(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock, Mock]:
    fake_app = Mock(name="cli.app")
    start_spy = Mock(name="splash.start", return_value=None)
    stop_spy = Mock(name="splash.stop")
    monkeypatch.setattr("lilbee.cli.app", fake_app)
    monkeypatch.setattr("lilbee.runtime.splash.start", start_spy)
    monkeypatch.setattr("lilbee.runtime.splash.stop", stop_spy)
    return fake_app, start_spy, stop_spy


def test_bare_invocation_shows_splash(
    monkeypatch: pytest.MonkeyPatch, _no_completion_env: None
) -> None:
    fake_app, start_spy, _ = _patch_app_and_splash(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lilbee"])

    launcher.main()

    start_spy.assert_called_once()
    fake_app.assert_called_once()


def test_chat_subcommand_shows_splash(
    monkeypatch: pytest.MonkeyPatch, _no_completion_env: None
) -> None:
    _, start_spy, _ = _patch_app_and_splash(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lilbee", "chat"])

    launcher.main()

    start_spy.assert_called_once()


def test_non_interactive_subcommand_skips_splash(
    monkeypatch: pytest.MonkeyPatch, _no_completion_env: None
) -> None:
    fake_app, start_spy, _ = _patch_app_and_splash(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lilbee", "--version"])

    launcher.main()

    start_spy.assert_not_called()
    fake_app.assert_called_once()


def test_shell_completion_skips_splash(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``_LILBEE_COMPLETE`` invocation (zsh/bash tab completion) must not
    start the splash animation even though its argv is empty."""
    fake_app, start_spy, _ = _patch_app_and_splash(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lilbee"])
    monkeypatch.setenv("_LILBEE_COMPLETE", "zsh_complete")

    launcher.main()

    start_spy.assert_not_called()
    fake_app.assert_called_once()


def test_app_import_failure_stops_splash(
    monkeypatch: pytest.MonkeyPatch, _no_completion_env: None
) -> None:
    """If the heavy ``lilbee.cli`` import fails, the splash must be stopped
    before the exception propagates so the terminal isn't left mid-animation."""
    _, start_spy, stop_spy = _patch_app_and_splash(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["lilbee"])

    class _Boom:
        def __getattr__(self, name: str) -> object:
            raise RuntimeError("heavy import failed")

    monkeypatch.setitem(sys.modules, "lilbee.cli", _Boom())

    with pytest.raises(RuntimeError, match="heavy import failed"):
        launcher.main()

    start_spy.assert_called_once()
    stop_spy.assert_called_once()


def test_keyboard_interrupt_restores_cursor_and_exits_130(
    monkeypatch: pytest.MonkeyPatch, _no_completion_env: None
) -> None:
    """Ctrl-C out of the TUI must re-show the cursor and exit with code 130."""
    fake_app, _, _ = _patch_app_and_splash(monkeypatch)
    fake_app.side_effect = KeyboardInterrupt
    monkeypatch.setattr(sys, "argv", ["lilbee"])

    with pytest.raises(SystemExit) as exc:
        launcher.main()

    assert exc.value.code == 130


class TestForceUtf8Stdio:
    """``_force_utf8_stdio`` makes stdio UTF-8 without crashing on odd streams."""

    def test_reconfigures_both_streams_to_utf8(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out = Mock(spec=io.TextIOWrapper)
        err = Mock(spec=io.TextIOWrapper)
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", err)
        launcher._force_utf8_stdio()
        for stream in (out, err):
            stream.reconfigure.assert_called_once_with(encoding="utf-8", errors="backslashreplace")

    def test_non_textiowrapper_stream_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A StringIO redirect / capture shim isn't a TextIOWrapper; must be a no-op.
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        monkeypatch.setattr(sys, "stderr", io.StringIO())
        launcher._force_utf8_stdio()

    def test_reconfigure_error_is_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bad = Mock(spec=io.TextIOWrapper)
        bad.reconfigure.side_effect = ValueError("stream detached")
        monkeypatch.setattr(sys, "stdout", bad)
        monkeypatch.setattr(sys, "stderr", Mock(spec=io.TextIOWrapper))
        launcher._force_utf8_stdio()
