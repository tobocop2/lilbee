"""The browser launch recorder counts each browser once, whatever state its processes are in."""

from __future__ import annotations

from pathlib import Path

import psutil
import pytest

from tests import _browser_launches
from tests._browser_launches import BrowserLaunches

SHELL = Path("/opt/chrome/headless_shell")


class _Child:
    def __init__(self, pid: int, cmdline: list[str]) -> None:
        self.pid = pid
        self._cmdline = cmdline

    def create_time(self) -> float:
        return 1.0

    def exe(self) -> str:
        return str(SHELL)

    def cmdline(self) -> list[str]:
        return self._cmdline


class _Parent:
    def __init__(self, children: list[_Child]) -> None:
        self._children = children

    def children(self, recursive: bool) -> list[_Child]:
        return self._children


def _scan(monkeypatch: pytest.MonkeyPatch, children: list[_Child]) -> list[list[str]]:
    monkeypatch.setattr(psutil, "Process", lambda: _Parent(children))
    monkeypatch.setattr(_browser_launches, "_same_file", lambda left, right: True)
    launches = BrowserLaunches(SHELL)
    launches._scan()
    return launches.command_lines


def test_a_browser_and_its_helpers_count_as_one_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    browser = _Child(1, [str(SHELL), "--headless"])
    helper = _Child(2, [str(SHELL), "--type=renderer"])
    assert _scan(monkeypatch, [browser, helper]) == [[str(SHELL), "--headless"]]


def test_a_process_whose_command_line_is_gone_is_not_a_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux reads an exiting process's command line as empty, so a helper then looks bare."""
    browser = _Child(1, [str(SHELL), "--headless"])
    exiting_helper = _Child(2, [])
    assert _scan(monkeypatch, [browser, exiting_helper]) == [[str(SHELL), "--headless"]]
