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


def _makefile() -> str:
    return (ROOT / "Makefile").read_text(encoding="utf-8")


def test_check_runs_the_preset_replay_target() -> None:
    match = re.search(r"^check:(.+?)(?:##.*)?$", _makefile(), re.MULTILINE)
    assert match, "Makefile has no check target"
    assert "test-presets" in match.group(1).split()


def test_preset_target_runs_the_replay_file() -> None:
    match = re.search(r"^test-presets:\n\t(.+)$", _makefile(), re.MULTILINE)
    assert match, "Makefile has no test-presets target"
    assert "tests/integration/test_preset_fixtures.py" in match.group(1).split()
    assert (ROOT / "tests/integration/test_preset_fixtures.py").is_file()
