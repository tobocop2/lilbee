"""The stall diagnostics ask for a privileged py-spy dump only when the owner opts in."""

from __future__ import annotations

import os
import subprocess
import threading
import time

import pytest

from tests import _stall_diagnostics as diagnostics

_DENIED_TEXT = "Permission Denied: Try running again with elevated permissions"
_STACK_TEXT = "test_sleep (test_sleeper.py:5)"
_PY_SPY_PATH = os.path.abspath("/opt/tools/py-spy")
_PID = 4242
_EXIT_BOUND_S = 2.0


class _Runner:
    """Stands in for subprocess.run: plain calls are denied, sudo calls succeed or fail."""

    def __init__(self, *, sudo_returncode: int = 0, plain_error: str = _DENIED_TEXT) -> None:
        self.commands: list[list[str]] = []
        self._sudo_returncode = sudo_returncode
        self._plain_error = plain_error

    def __call__(self, command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        if command[0] == "sudo":
            return subprocess.CompletedProcess(command, self._sudo_returncode, _STACK_TEXT, "")
        return subprocess.CompletedProcess(command, 1, "", self._plain_error)


def _dump(monkeypatch: pytest.MonkeyPatch, runner: _Runner, *, sudo: str | None) -> str:
    monkeypatch.setattr(diagnostics.subprocess, "run", runner)
    if sudo is None:
        monkeypatch.delenv(diagnostics.SUDO_ENV, raising=False)
    else:
        monkeypatch.setenv(diagnostics.SUDO_ENV, sudo)
    return diagnostics._py_spy_dump(_PID, _PY_SPY_PATH)


def test_a_denied_dump_without_the_opt_in_says_how_to_enable_sudo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _Runner()
    text = _dump(monkeypatch, runner, sudo=None)
    assert _DENIED_TEXT in text
    assert f"set {diagnostics.SUDO_ENV}=1 to retry with sudo -n" in text
    assert [command[0] for command in runner.commands] == [_PY_SPY_PATH]


def test_a_denied_dump_with_the_opt_in_retries_with_sudo_and_the_absolute_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _Runner()
    text = _dump(monkeypatch, runner, sudo="1")
    assert _STACK_TEXT in text
    assert [command[:3] for command in runner.commands] == [
        [_PY_SPY_PATH, "dump", "--pid"],
        ["sudo", "-n", _PY_SPY_PATH],
    ]


def test_a_failed_sudo_retry_reports_the_sudo_error(monkeypatch: pytest.MonkeyPatch) -> None:
    text = _dump(monkeypatch, _Runner(sudo_returncode=1), sudo="1")
    assert _STACK_TEXT not in text
    assert "exited 1 with no output" in text


@pytest.mark.parametrize("sudo", [None, "1"])
def test_a_dump_that_fails_for_another_reason_never_tries_sudo(
    monkeypatch: pytest.MonkeyPatch, sudo: str | None
) -> None:
    runner = _Runner(plain_error="Error: no such process")
    text = _dump(monkeypatch, runner, sudo=sudo)
    assert "no such process" in text
    assert diagnostics.SUDO_ENV not in text
    assert len(runner.commands) == 1


@pytest.mark.parametrize("sudo", ["0", "yes", ""])
def test_only_the_value_one_turns_the_sudo_retry_on(
    monkeypatch: pytest.MonkeyPatch, sudo: str
) -> None:
    runner = _Runner()
    _dump(monkeypatch, runner, sudo=sudo)
    assert len(runner.commands) == 1


def test_a_missing_py_spy_is_named() -> None:
    assert "not on PATH" in diagnostics._py_spy_dump(_PID, None)


def test_a_py_spy_that_cannot_start_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        raise OSError("exec format error")

    monkeypatch.setattr(diagnostics.subprocess, "run", refuse)
    assert "exec format error" in diagnostics._py_spy_dump(_PID, _PY_SPY_PATH)


def test_a_run_that_ends_in_time_does_not_wait_for_a_snapshot_in_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    def slow_report() -> str:
        started.set()
        release.wait(30)
        return "late"

    monkeypatch.setattr(diagnostics, "process_tree_report", slow_report)
    try:
        begin = time.monotonic()
        with diagnostics.StallWatch(0.0):
            assert started.wait(5)
        assert time.monotonic() - begin < _EXIT_BOUND_S
    finally:
        release.set()
