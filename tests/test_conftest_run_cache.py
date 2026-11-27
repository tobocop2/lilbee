"""The run's shared caches add no name beside the workers' temp folders while tests run."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.conftest import RUN_CACHE_DIRNAME, child_pythonpath, make_run_cache, run_cache

_INNER_CONFTEST = """
import pytest

from tests.conftest import make_run_cache, run_cache


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    make_run_cache(session.config)


@pytest.fixture(scope="session")
def filled_cache(tmp_path_factory):
    (run_cache(tmp_path_factory) / "filled").mkdir()
"""

_INNER_HELPERS = """
import time


def names(folder):
    return {p.name for p in folder.iterdir() if not p.name.endswith(".lock")
            and not p.name.startswith("popen-gw")}


def wait(path):
    deadline = time.monotonic() + 60
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert path.exists(), f"the other worker never reached {path.name}"
"""

# One worker watches the run's folder; the other fills a session cache while it watches.
_INNER_WATCH = """
from inner_helpers import names, wait


def test_watch(tmp_path_factory):
    root = tmp_path_factory.getbasetemp().parent
    before = names(root)
    (root / "armed.lock").touch()
    wait(root / "filled.lock")
    assert names(root) == before
"""

_INNER_FILL = """
from inner_helpers import wait


def test_fill(request, tmp_path_factory):
    root = tmp_path_factory.getbasetemp().parent
    wait(root / "armed.lock")
    request.getfixturevalue("filled_cache")
    (root / "filled.lock").touch()
"""


def test_a_cache_filled_on_another_worker_adds_no_name_beside_the_workers(pytester, monkeypatch):
    """The fuzz guard of #1425 watches that folder; a cache fill must not show up in it."""
    repo = Path(__file__).resolve().parent.parent
    monkeypatch.setenv("PYTHONPATH", child_pythonpath(repo, pytester.path))
    pytester.makeconftest(_INNER_CONFTEST)
    pytester.makepyfile(
        inner_helpers=_INNER_HELPERS, test_watch=_INNER_WATCH, test_fill=_INNER_FILL
    )
    result = pytester.runpytest_subprocess(
        "-n", "2", "--dist", "loadfile", "-p", "no:cacheprovider", "--no-cov"
    )
    result.assert_outcomes(passed=2)


def _config(tmp_path: Path, *, worker: bool = False, xdist: bool = False) -> SimpleNamespace:
    config = SimpleNamespace(
        _tmp_path_factory=SimpleNamespace(getbasetemp=lambda: tmp_path / "pytest-0"),
        pluginmanager=SimpleNamespace(has_plugin=lambda name: xdist and name == "dsession"),
    )
    if worker:
        config.workerinput = {}
    return config


@pytest.mark.parametrize(
    ("worker", "xdist", "made"),
    [
        pytest.param(False, True, ["pytest-0/run-cache"], id="xdist-controller-in-its-temp-folder"),
        pytest.param(False, False, ["run-cache"], id="plain-run-beside-its-temp-folder"),
        pytest.param(True, True, [], id="worker-makes-none"),
    ],
)
def test_the_process_that_owns_the_run_makes_the_folder(tmp_path, worker, xdist, made):
    (tmp_path / "pytest-0").mkdir()
    make_run_cache(_config(tmp_path, worker=worker, xdist=xdist))
    found = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob(RUN_CACHE_DIRNAME))
    assert found == made


@pytest.mark.usefixtures("tessdata_ready")
def test_the_tesseract_data_sits_in_the_run_cache(tmp_path_factory):
    folder = run_cache(tmp_path_factory)
    assert folder.is_dir()
    assert Path(os.environ["TESSDATA_PREFIX"]).is_relative_to(folder)
