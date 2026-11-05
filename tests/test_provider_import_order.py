"""A fresh interpreter imports each provider-layer entry point first without a cycle."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "lilbee.providers.fleet.engine_reaper",
        "lilbee.providers.fleet.swap_manager",
        "lilbee.providers.roles",
        "lilbee.providers.reload_keys",
    ],
)
def test_module_imports_first_in_fresh_interpreter(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 0, result.stderr
