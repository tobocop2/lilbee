"""The suite's xdist flags end the run when a worker dies mid-run."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CI_TARGET = "test-ci"
_RUN_BUDGET_S = 60.0
_DROPPED_FLAG_PREFIX = "--cov"
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


def _ci_xdist_args() -> list[str]:
    """The pytest arguments the CI recipe passes, without the coverage flags."""
    make_env = {k: v for k, v in os.environ.items() if k not in _MAKE_RECURSION_ENV_VARS}
    recipe = subprocess.run(
        ["make", "--no-print-directory", "-n", _CI_TARGET],
        cwd=_REPO_ROOT,
        capture_output=True,
        encoding="utf-8",
        check=True,
        env=make_env,
    ).stdout
    pytest_line = next(line for line in recipe.splitlines() if _PYTEST_WORD in shlex.split(line))
    words = shlex.split(pytest_line)
    after_pytest = words[words.index(_PYTEST_WORD) + 1 :]
    return [word for word in after_pytest if not word.startswith(_DROPPED_FLAG_PREFIX)]


@pytest.mark.timeout(_RUN_BUDGET_S * 2)
def test_a_dead_worker_fails_the_run_instead_of_hanging_it(tmp_path: Path) -> None:
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "test_toy.py").write_text(_TOY_SUITE, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith(_CHILD_ENV_PREFIXES)}
    env[_MARKER_ENV] = str(tmp_path / "died-once")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *_ci_xdist_args()],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        encoding="utf-8",
        timeout=_RUN_BUDGET_S,
    )
    assert result.returncode == _EXIT_TESTS_FAILED, result.stdout + result.stderr
    assert _CRASH_REPORT in result.stdout
    assert _CRASH_NODEID in result.stdout
