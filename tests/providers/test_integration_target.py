"""The integration target runs the directory of tests that need the client library."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_integration_target_runs_the_integration_directory() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    match = re.search(r"^test-integration:\n\t(.+)$", makefile, re.MULTILINE)
    assert match, "Makefile has no test-integration target"
    assert "tests/integration/" in match.group(1).split()
