"""The repo conftest gives each test the signal handlers that the test before it changed."""

from __future__ import annotations

import signal

_INNER_CONFTEST = """
from tests.conftest import _restore_signal_handlers  # noqa: F401
"""

# The first test does what an in-process serve does: it routes the signals to the
# engine stop and returns. The second test runs next in the same process.
_INNER_TEST = """
import signal

from lilbee.app.services import install_engine_lifecycle_hooks, stop_on_interrupt

_WATCHED = tuple(
    getattr(signal, name) for name in ("SIGINT", "SIGTERM", "SIGHUP") if hasattr(signal, name)
)
_AT_START = {sig: signal.getsignal(sig) for sig in _WATCHED}


def test_a_serve_routes_the_signals_to_the_stop():
    install_engine_lifecycle_hooks()
    stop_on_interrupt()
    assert signal.getsignal(signal.SIGINT) != _AT_START[signal.SIGINT]
    assert signal.getsignal(signal.SIGTERM) != _AT_START[signal.SIGTERM]


def test_the_next_test_gets_the_handlers_back():
    assert {sig: signal.getsignal(sig) for sig in _WATCHED} == _AT_START
"""


def test_a_routed_handler_does_not_reach_the_next_test(pytester):
    """An xdist worker runs many tests, so a handler one test routes must end with that test."""
    # A worker started as a background job ignores SIGINT, and the route keeps that ignore.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    pytester.makeconftest(_INNER_CONFTEST)
    pytester.makepyfile(_INNER_TEST)

    result = pytester.runpytest_inprocess("-p", "no:cacheprovider", "-p", "no:randomly")

    result.assert_outcomes(passed=2)
