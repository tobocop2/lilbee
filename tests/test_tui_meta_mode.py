"""run_tui asks the terminal for ESC-prefixed meta keys and restores the mode on exit."""

from __future__ import annotations

import io
import sys
from collections.abc import Callable
from unittest import mock

import pytest

from lilbee.cli.tui import run_tui
from lilbee.cli.tui.app import LilbeeApp

SAVE_AND_SET = "\x1b[?1036s\x1b[?1036h"
RESTORE = "\x1b[?1036r"


class _Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


class _GoneTerminal(_Terminal):
    def write(self, text: str) -> int:
        raise OSError(5, "Input/output error")


class _ClosedTerminal(_Terminal):
    def write(self, text: str) -> int:
        raise ValueError("I/O operation on closed file.")


def _run(
    monkeypatch: pytest.MonkeyPatch,
    stream: io.StringIO | None,
    run_async: mock.Mock,
    at_straggler_exit: Callable[[int], None] | None = None,
) -> mock.Mock:
    """Run run_tui with *stream* as the terminal; return the mock of the final forced exit.

    *at_straggler_exit* runs in place of that exit, with the status it gets.
    """
    monkeypatch.setattr(sys, "__stderr__", stream)
    with (
        mock.patch("lilbee.cli.tui._redirect_native_stderr_to", return_value=None),
        mock.patch.object(LilbeeApp, "run_async", run_async),
        mock.patch.object(LilbeeApp, "return_code", None),
        mock.patch("lilbee.cli.sync.shutdown_executor"),
        mock.patch("lilbee.cli.tui.reset_services_on_exit"),
        mock.patch(
            "lilbee.cli.tui.exit_when_stragglers_would_hang", side_effect=at_straggler_exit
        ) as straggler,
    ):
        run_tui()
    return straggler


def test_start_writes_save_then_enable(monkeypatch: pytest.MonkeyPatch) -> None:
    terminal = _Terminal()
    seen_during_run: list[str] = []
    run_async = mock.AsyncMock(side_effect=lambda: seen_during_run.append(terminal.getvalue()))

    _run(monkeypatch, terminal, run_async)

    assert seen_during_run == [SAVE_AND_SET]
    assert terminal.getvalue() == SAVE_AND_SET + RESTORE


def test_exit_writes_restore_after_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    terminal = _Terminal()
    run_async = mock.AsyncMock(side_effect=RuntimeError("the app crashed"))

    with pytest.raises(RuntimeError, match="the app crashed"):
        _run(monkeypatch, terminal, run_async)

    assert terminal.getvalue() == SAVE_AND_SET + RESTORE


def test_exit_writes_restore_after_ctrl_c(monkeypatch: pytest.MonkeyPatch) -> None:
    terminal = _Terminal()
    run_async = mock.AsyncMock(side_effect=KeyboardInterrupt)

    _run(monkeypatch, terminal, run_async)

    assert terminal.getvalue() == SAVE_AND_SET + RESTORE


def test_non_tty_writes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = io.StringIO()

    _run(monkeypatch, pipe, mock.AsyncMock())

    assert pipe.getvalue() == ""


def test_no_terminal_stream_still_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    straggler = _run(monkeypatch, None, mock.AsyncMock())

    straggler.assert_called_once_with(0)


@pytest.mark.parametrize("terminal", [_GoneTerminal, _ClosedTerminal], ids=["gone", "closed"])
def test_a_terminal_that_is_gone_does_not_stop_the_exit(
    monkeypatch: pytest.MonkeyPatch, terminal: type[_Terminal]
) -> None:
    straggler = _run(monkeypatch, terminal(), mock.AsyncMock())

    straggler.assert_called_once_with(0)


def test_exit_writes_restore_before_the_straggler_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    # The straggler exit can end the process, so a restore written after it never arrives.
    terminal = _Terminal()
    seen_at_straggler_exit: list[str] = []

    _run(
        monkeypatch,
        terminal,
        mock.AsyncMock(),
        at_straggler_exit=lambda _status: seen_at_straggler_exit.append(terminal.getvalue()),
    )

    assert seen_at_straggler_exit == [SAVE_AND_SET + RESTORE]
