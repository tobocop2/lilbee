"""Fixtures for the vendor smi tests: a real child process that prints and then fails."""

from __future__ import annotations

import os
import stat
from collections.abc import Generator
from pathlib import Path

import pytest

from lilbee.providers.fleet.gpu_backends import base as base_mod
from tests.providers.fleet.gpu_backends import FakeTool


@pytest.fixture(autouse=True)
def _fresh_exit_warnings() -> Generator[None]:
    """Each test sees the first failing exit of a tool, whatever ran before it."""
    base_mod._warned_exits.clear()
    yield
    base_mod._warned_exits.clear()


@pytest.fixture
def fake_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeTool:
    """Put a tool on PATH that prints ``stdout`` and exits with ``code``."""
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")

    def _make(name: str, stdout: str, code: int) -> None:
        printed = bin_dir / f"{name}.out"
        printed.write_text(stdout)
        script = bin_dir / name
        script.write_text(f'#!/bin/sh\ncat "{printed}"\nexit {code}\n')
        script.chmod(script.stat().st_mode | stat.S_IXUSR)

    return _make
