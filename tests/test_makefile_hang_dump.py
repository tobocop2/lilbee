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


def _make(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", *args],
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        env=_env() if env is None else env,
        check=False,
    )


UNIT_RECIPES = ("test", "test-ci", "test-ci-split", "test-ci-serial", "test-ci-forked")
# The watchdog ignores a test's own timeout, so these must not run under it.
LONG_RECIPES = ("test-integration", "test-presets")


def _seen_by(recipe: str, tmp_path: Path, env: dict[str, str] | None = None) -> str:
    """The watchdog variables as the recipe's own environment holds them.

    A prerequisite inherits the target-specific variables of its target. The ``+`` prefix
    runs its line under ``-n``, so nothing else of the recipe runs.
    """
    probe = tmp_path / "probe.mk"
    probe.write_text(
        f"probe-{recipe}:\n\t+@echo PROBE $${SECONDS}/$${FOLDER}\n{recipe}: probe-{recipe}\n",
        encoding="utf-8",
    )
    shown = _make("-n", "-f", "Makefile", "-f", str(probe), recipe, env=env)
    assert shown.returncode == 0, shown.stderr
    (line,) = [line for line in shown.stdout.splitlines() if line.startswith("PROBE ")]
    return line.removeprefix("PROBE ")


@needs_make
@pytest.mark.parametrize("recipe", UNIT_RECIPES)
def test_every_recipe_inherits_the_watchdog_variables(recipe: str, tmp_path: Path) -> None:
    """A unit recipe sees the variables the way pytest does: through its environment."""
    seconds, folder = _seen_by(recipe, tmp_path).split("/", 1)
    assert seconds == "90"
    assert folder


@needs_make
@pytest.mark.parametrize("recipe", LONG_RECIPES)
def test_the_long_recipes_do_not_get_the_watchdog(recipe: str, tmp_path: Path) -> None:
    assert _seen_by(recipe, tmp_path) == "/"


@needs_make
@pytest.mark.parametrize("recipe", LONG_RECIPES)
def test_a_value_from_the_shell_does_not_reach_the_long_recipes(recipe: str, tmp_path: Path) -> None:
    shell = {**_env(), SECONDS: "7", FOLDER: "/from/the/shell"}
    assert _seen_by(recipe, tmp_path, env=shell) == "/"


@needs_make
@pytest.mark.parametrize("recipe", UNIT_RECIPES)
def test_an_empty_value_stops_make_instead_of_switching_the_watchdog_off(recipe: str) -> None:
    stopped = _make("-n", recipe, env={**_env(), SECONDS: ""})
    assert stopped.returncode != 0
    assert "the hang watchdog would be off" in stopped.stderr


@needs_make
@pytest.mark.parametrize("target", (*LONG_RECIPES, "lint"))
def test_an_empty_value_does_not_stop_the_other_targets(target: str) -> None:
    shown = _make("-n", target, env={**_env(), SECONDS: ""})
    assert shown.returncode == 0, shown.stderr
    assert "the hang watchdog would be off" not in shown.stderr


def test_ci_keeps_no_copy_of_the_seconds() -> None:
    """One owner: a second number in CI could drift from the Makefile's."""
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert not re.search(rf"^\s*{SECONDS}:", workflow, re.MULTILINE)
    assert re.search(rf"^\s*{FOLDER}:", workflow, re.MULTILINE)


def test_the_code_sync_tests_use_the_shared_parser_cache() -> None:
    """A cold parser download per test once took five workers down together."""
    marks = [mark for mark in formats.TestSyncCode.pytestmark if mark.name == "usefixtures"]
    assert [mark.args for mark in marks] == [("code_parsers_ready",)]
