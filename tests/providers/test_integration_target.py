"""The integration target runs the directory of tests that need the client library."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_integration_target_runs_the_integration_directory() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    match = re.search(r"^test-integration:\n\t(.+)$", makefile, re.MULTILINE)
    assert match, "Makefile has no test-integration target"
    assert "tests/integration/" in match.group(1).split()


def test_check_runs_pytest_on_the_preset_replay() -> None:
    replay = "tests/integration/test_preset_fixtures.py"
    dry_run = subprocess.run(
        ["make", "-n", "check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    commands = [line.split() for line in dry_run.stdout.splitlines()]
    assert any(words[:3] == ["uv", "run", "pytest"] and replay in words for words in commands), (
        "make check does not run pytest on the preset replay"
    )
    assert (ROOT / replay).is_file()
