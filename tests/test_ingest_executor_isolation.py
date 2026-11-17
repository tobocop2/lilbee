"""One test's shared ingest pool must not reach the next test."""

import threading

import pytest

from lilbee.data import offload
from tests.conftest import shutdown_ingest_pool

pytest_plugins = ["pytester"]

_SUB_CONFTEST = "from tests.conftest import _shutdown_ingest_pool  # noqa: F401\n"

_SUB_TESTS = """
import threading

from lilbee.data import offload


def test_first_builds_a_pool():
    offload._ingest_executor().submit(lambda: None).result()
    assert offload._ingest_executor.cache_info().currsize == 1


def test_second_finds_no_pool():
    assert offload._ingest_executor.cache_info().currsize == 0
    assert not [
        t for t in threading.enumerate() if t.name.startswith(offload._POOL_THREAD_PREFIX)
    ]
"""


def _live_ingest_threads() -> list[str]:
    return [
        t.name
        for t in threading.enumerate()
        if t.name.startswith(offload._POOL_THREAD_PREFIX) and t.is_alive()
    ]


def test_teardown_shuts_down_a_pool_that_something_still_references() -> None:
    held = offload._ingest_executor()
    held.submit(lambda: None).result()
    assert offload._ingest_executor.cache_info().currsize == 1
    assert _live_ingest_threads()

    shutdown_ingest_pool()

    assert offload._ingest_executor.cache_info().currsize == 0
    assert _live_ingest_threads() == []


def test_autouse_fixture_removes_the_pool_before_the_next_test(pytester: pytest.Pytester) -> None:
    pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function\n")
    pytester.makeconftest(_SUB_CONFTEST)
    pytester.makepyfile(test_inner=_SUB_TESTS)

    result = pytester.runpytest_inprocess("-p", "no:xdist", "-p", "no:cacheprovider")

    result.assert_outcomes(passed=2)
