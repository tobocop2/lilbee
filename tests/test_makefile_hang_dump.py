"""The Makefile owns the hang watchdog seconds, and no test recipe can run without them."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import tests.test_formats as formats

ROOT = Path(__file__).resolve().parent.parent
SECONDS = "LILBEE_TEST_HANG_DUMP_S"
FOLDER = "LILBEE_TEST_HANG_DUMP_DIR"

needs_make = pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")


def _env() -> dict[str, str]:
    """The environment of a developer who set neither variable."""
    return {key: value for key, value in os.environ.items() if key not in (SECONDS, FOLDER)}


def _make(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", *args], cwd=ROOT, capture_output=True, encoding="utf-8", env=_env(), check=False
    )


@needs_make
def test_every_recipe_inherits_the_watchdog_variables(tmp_path: Path) -> None:
    """A recipe sees the variables the way pytest does: through its environment."""
    probe = tmp_path / "probe.mk"
    probe.write_text(f"show:\n\t@echo $${SECONDS} $${FOLDER}\n", encoding="utf-8")
    shown = _make("-f", "Makefile", "-f", str(probe), "show")
    assert shown.returncode == 0, shown.stderr
    seconds, folder = shown.stdout.split()
    assert int(seconds) > 0
    assert folder


@needs_make
def test_an_empty_value_stops_make_instead_of_switching_the_watchdog_off() -> None:
    stopped = _make("-n", "test", f"{SECONDS}=")
    assert stopped.returncode != 0
    assert "the hang watchdog would be off" in stopped.stderr


def test_ci_keeps_no_copy_of_the_seconds() -> None:
    """One owner: a second number in CI could drift from the Makefile's."""
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert not re.search(rf"^\s*{SECONDS}:", workflow, re.MULTILINE)
    assert re.search(rf"^\s*{FOLDER}:", workflow, re.MULTILINE)


def test_the_code_sync_tests_use_the_shared_parser_cache() -> None:
    """A cold parser download per test once took five workers down together."""
    marks = [mark for mark in formats.TestSyncCode.pytestmark if mark.name == "usefixtures"]
    assert [mark.args for mark in marks] == [("code_parsers_ready",)]
