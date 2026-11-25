"""The Makefile test recipe arms the hang watchdog with the same setting as CI."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CI_SECONDS = re.compile(r'^\s*LILBEE_TEST_HANG_DUMP_S:\s*"(\d+)"', re.MULTILINE)
RECIPE_SECONDS = re.compile(r"LILBEE_TEST_HANG_DUMP_S=(\S*)")
RECIPE_DIR = re.compile(r"LILBEE_TEST_HANG_DUMP_DIR=(\S*)")


def _ci_seconds() -> str:
    workflow = ROOT / ".github" / "workflows" / "ci.yml"
    found = CI_SECONDS.findall(workflow.read_text(encoding="utf-8"))
    assert len(found) == 1, "CI must set LILBEE_TEST_HANG_DUMP_S once"
    return found[0]


@pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
def test_the_test_recipe_arms_the_watchdog_with_the_ci_seconds() -> None:
    recipe = subprocess.run(
        ["make", "-n", "test"], cwd=ROOT, capture_output=True, encoding="utf-8", check=True
    ).stdout
    seconds = RECIPE_SECONDS.search(recipe)
    folder = RECIPE_DIR.search(recipe)
    assert seconds is not None and seconds.group(1) == _ci_seconds(), recipe
    assert folder is not None and folder.group(1), recipe


def test_the_makefile_holds_no_second_copy_of_the_seconds() -> None:
    """The Makefile reads the number from CI; a literal there could drift from it."""
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert not re.search(r"LILBEE_TEST_HANG_DUMP_S=\d", makefile)
