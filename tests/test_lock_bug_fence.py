"""The suite's fence: a lock bug error that a test did not expect fails that test."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_CHILD_TESTS = """
import pytest

from lilbee.runtime.lock import LockBugError, LockOrderError


def _swallow():
    try:
        raise LockOrderError("planted bug")
    except LockBugError:
        pass


def test_swallowed():
    _swallow()


def test_expected():
    with pytest.raises(LockOrderError, match="planted bug"):
        raise LockOrderError("planted bug")


@pytest.mark.parametrize("kind", [Exception, BaseException])
def test_broad(kind):
    with pytest.raises(kind, match="planted bug"):
        raise LockOrderError("planted bug")


def test_base_class():
    with pytest.raises(LockBugError, match="planted bug"):
        raise LockOrderError("planted bug")


def test_in_a_tuple():
    with pytest.raises((ValueError, LockOrderError), match="planted bug"):
        raise LockOrderError("planted bug")


@pytest.mark.lock_bug_expected
def test_marked():
    _swallow()


def test_clean():
    pass
"""
_OUTCOME = re.compile(r"^(PASSED|FAILED|ERROR) \S*::(test_[\w\[\]]+)", re.MULTILINE)
# Coverage of the parent run stays out of the child run.
_COVERAGE_ENV_PREFIX = "COV_CORE_"


@pytest.fixture(scope="module")
def child_run(tmp_path_factory) -> subprocess.CompletedProcess[str]:
    """The outcome of a pytest run of small tests under the root conftest."""
    child = tmp_path_factory.mktemp("fence") / "test_child.py"
    child.write_text(_CHILD_TESTS, encoding="utf-8")
    env = {
        key: value for key, value in os.environ.items() if not key.startswith(_COVERAGE_ENV_PREFIX)
    }
    command = [sys.executable, "-m", "pytest", "-p", "tests.conftest", "-p", "no:cacheprovider"]
    command += ["-c", str(_REPO / "pyproject.toml"), "--rootdir", str(_REPO), "-q", "-rA"]
    return subprocess.run(
        [*command, str(child)],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


@pytest.fixture(scope="module")
def outcomes(child_run) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for outcome, name in _OUTCOME.findall(child_run.stdout):
        found.setdefault(name, set()).add(outcome)
    assert set(found) == {
        "test_swallowed",
        "test_expected",
        "test_broad[Exception]",
        "test_broad[BaseException]",
        "test_base_class",
        "test_in_a_tuple",
        "test_marked",
        "test_clean",
    }, child_run.stdout + child_run.stderr
    return found


def test_a_swallowed_bug_error_fails_its_test(child_run, outcomes):
    assert "ERROR" in outcomes["test_swallowed"]
    assert child_run.returncode == 1
    assert "planted bug" in child_run.stdout


def test_a_bug_error_the_test_expects_passes(outcomes):
    assert outcomes["test_expected"] == {"PASSED"}


@pytest.mark.parametrize("kind", ["Exception", "BaseException"])
def test_a_raises_of_a_broader_class_expects_no_bug_error(outcomes, kind):
    assert "ERROR" in outcomes[f"test_broad[{kind}]"]


def test_a_raises_of_the_base_class_expects_a_subclass(outcomes):
    assert outcomes["test_base_class"] == {"PASSED"}


def test_a_raises_of_a_tuple_with_the_class_expects_it(outcomes):
    assert outcomes["test_in_a_tuple"] == {"PASSED"}


def test_a_marked_test_that_swallows_one_passes(outcomes):
    assert outcomes["test_marked"] == {"PASSED"}


def test_a_test_with_no_bug_error_is_untouched(outcomes):
    assert outcomes["test_clean"] == {"PASSED"}
