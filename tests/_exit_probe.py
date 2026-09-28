"""Run a script in a fresh interpreter and assert it exits, bounded by a timeout."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_EXIT_BUDGET_S = 30.0


def assert_probe_exits(probe: str, home: Path, expected_stdout: str) -> None:
    """Fail with ``TimeoutExpired`` when *probe*'s interpreter would not exit."""
    env = {**os.environ, "LILBEE_DATA": str(home), "HOME": str(home)}
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env=env,
        capture_output=True,
        encoding="utf-8",
        timeout=_EXIT_BUDGET_S,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected_stdout
