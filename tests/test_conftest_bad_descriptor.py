"""The repo conftest must not turn a "Bad file descriptor" error into a pass."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

_REPO_ROOT = Path(__file__).resolve().parent.parent

# The conftest once downgraded this error only when its traceback ended in a
# file named selectors.py, so the shim carries that name and loads by path.
_SHIM = """
import os


def poll():
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    os.close(write_fd)
    os.close(read_fd)
"""

_INNER_TEST = """
import importlib.util
from pathlib import Path


def test_second_close_of_a_descriptor():
    path = Path(__file__).parent / "shim" / "selectors.py"
    spec = importlib.util.spec_from_file_location("shim_selectors", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.poll()
"""


def test_a_bad_descriptor_error_in_a_test_fails_it(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second os.close raises EBADF from selectors.py in the test body, and the test fails."""
    monkeypatch.setenv("PYTHONPATH", str(_REPO_ROOT))
    pytester.mkdir("shim")
    (pytester.path / "shim" / "selectors.py").write_text(_SHIM, encoding="utf-8")
    pytester.makepyfile(test_inner=_INNER_TEST)

    result = pytester.runpytest_subprocess(
        "-p", "tests.conftest", "-p", "no:cacheprovider", "-p", "no:randomly"
    )

    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*Bad file descriptor*"])
