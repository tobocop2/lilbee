"""A long-lived test worker does not keep the objects of the tests it finished."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import pytest

from tests.conftest import child_pythonpath

pytest_plugins = ["pytester"]

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WARM_UP = 10
_TESTS = 40
# pytest keeps about 180 objects of each finished test here (its reports and the fixture
# finalizers of higher scopes); one kept server app is about 20,000.
_MAX_KEPT_PER_TEST = 300

_COUNTER = f"""
import gc
import json
from pathlib import Path

_done = []
_live = {{}}


def pytest_runtest_logfinish(nodeid, location):
    _done.append(nodeid)
    if len(_done) in ({_WARM_UP}, {_TESTS}):
        gc.collect()
        _live[len(_done)] = len(gc.get_objects())


def pytest_sessionfinish(session):
    Path("live.json").write_text(json.dumps(_live), encoding="utf-8")
"""

_APP_TESTS = f"""
from unittest.mock import MagicMock

import pytest
from litestar.testing import TestClient

from lilbee.server.app import create_app


@pytest.fixture
def services():
    yield MagicMock()


@pytest.mark.parametrize("n", range({_TESTS}))
def test_one_request_to_a_new_app(n, services):
    TestClient(app=create_app()).get("/api/health")
"""


def test_a_finished_test_leaves_its_server_app_to_the_collector(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONPATH", child_pythonpath(_REPO_ROOT))
    pytester.makeconftest(_COUNTER)
    pytester.makepyfile(test_apps=_APP_TESTS)

    # pytest keeps each warning it captures until the session ends, and each new app warns.
    plugins = ["tests.conftest", "no:cacheprovider", "no:randomly", "no:xdist", "no:warnings"]
    result = pytester.runpytest_subprocess(*(arg for name in plugins for arg in ("-p", name)))

    result.assert_outcomes(passed=_TESTS)
    live = json.loads((pytester.path / "live.json").read_text(encoding="utf-8"))
    kept_per_test = (live[str(_TESTS)] - live[str(_WARM_UP)]) / (_TESTS - _WARM_UP)
    assert kept_per_test < _MAX_KEPT_PER_TEST


def test_full_collections_skip_the_objects_of_collection() -> None:
    # The collector lists only the objects it still walks, and collection made this function.
    collected = test_full_collections_skip_the_objects_of_collection
    assert all(obj is not collected for obj in gc.get_objects())
