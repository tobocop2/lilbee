"""The suite's xdist flags end the run when a worker dies mid-run."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUN_BUDGET_S = 60.0
# Coverage flags are irrelevant to the toy suite. --forked moves the test into a child
# process, so the child dies and the xdist worker does not; the worker is what is under test.
_DROPPED_FLAG_PREFIXES = ("--cov", "--forked")
_DIST_FLAG = "--dist"
_RULE_LINE = re.compile(r"^([A-Za-z][\w.-]*):(?!=)")
_CHILD_ENV_PREFIXES = ("PYTEST_", "COV_CORE_")
_PYTEST_WORD = "pytest"
_MAKE_RECURSION_ENV_VARS = ("MAKEFLAGS", "MAKELEVEL")
_CRASH_NODEID = "test_toy.py::test_worker_dies"
_CRASH_REPORT = "crashed while running"
_EXIT_TESTS_FAILED = 1
_MARKER_ENV = "TOY_DIED_ONCE"
_TOY_SUITE = """
import os
import pathlib

import pytest


@pytest.mark.parametrize("index", range(200))
def test_before(index):
    pass


def test_worker_dies():
    # Dies once, like a timeout, so a rerun of this test on another worker passes.
    marker = pathlib.Path(os.environ["TOY_DIED_ONCE"])
    if not marker.exists():
        marker.touch()
        os._exit(1)


@pytest.mark.parametrize("index", range(2))
def test_after(index):
    pass
"""


def _make(*args: str) -> str:
    """Run make in the repo root, outside any parent make, and return stdout."""
    make_env = {k: v for k, v in os.environ.items() if k not in _MAKE_RECURSION_ENV_VARS}
    return subprocess.run(
        ["make", "--no-print-directory", *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        encoding="utf-8",
        env=make_env,
    ).stdout


def _targets_running_pytest() -> list[str]:
    """Makefile targets whose recipe calls pytest, read from make's rule database."""
    targets: dict[str, None] = {}
    current = ""
    for line in _make("-pRrq").splitlines():
        rule = _RULE_LINE.match(line)
        if rule:
            current = rule.group(1)
        elif not line:
            current = ""
        elif line.startswith("\t") and _PYTEST_WORD in line.split():
            targets[current] = None
    return [target for target in targets if target]


def _pytest_args(target: str) -> list[str]:
    """The pytest arguments the target's expanded recipe passes, without coverage or --forked."""
    recipe = _make("-n", target)
    pytest_line = next(line for line in recipe.splitlines() if _PYTEST_WORD in shlex.split(line))
    words = shlex.split(pytest_line)
    after_pytest = words[words.index(_PYTEST_WORD) + 1 :]
    return [word for word in after_pytest if not word.startswith(_DROPPED_FLAG_PREFIXES)]


def _xdist_targets() -> dict[str, list[str]]:
    """Every pytest target that distributes work, with its arguments."""
    args_by_target = {target: _pytest_args(target) for target in _targets_running_pytest()}
    return {t: args for t, args in args_by_target.items() if _DIST_FLAG in args}


_XDIST_TARGETS = _xdist_targets()


def test_the_makefile_has_xdist_targets_to_check() -> None:
    assert {"test", "test-ci", "test-ci-forked"} <= set(_XDIST_TARGETS)
    assert "test-ci-serial" not in _XDIST_TARGETS


@pytest.mark.timeout(_RUN_BUDGET_S * 2)
@pytest.mark.parametrize("target", sorted(_XDIST_TARGETS))
def test_a_dead_worker_fails_the_run_instead_of_hanging_it(target: str, tmp_path: Path) -> None:
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "test_toy.py").write_text(_TOY_SUITE, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith(_CHILD_ENV_PREFIXES)}
    env[_MARKER_ENV] = str(tmp_path / "died-once")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *_XDIST_TARGETS[target]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        encoding="utf-8",
        timeout=_RUN_BUDGET_S,
    )
    assert result.returncode == _EXIT_TESTS_FAILED, result.stdout + result.stderr
    assert _CRASH_REPORT in result.stdout
    assert _CRASH_NODEID in result.stdout
