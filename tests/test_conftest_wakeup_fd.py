"""The repo conftest fails a test that leaves the signal wakeup descriptor changed."""

from __future__ import annotations

_INNER_CONFTEST = """
from tests.conftest import _test_leaves_the_signal_wakeup_fd_as_it_found_it  # noqa: F401
"""

# The first test sets a socket as the wakeup descriptor and closes it, the leak of #1442.
# The second restores what it found, as the wakeup_taking_loops fixture does; it fails
# with a bad descriptor when the leak reaches it. The third changes and restores.
_INNER_TEST = """
import signal
import socket


def test_a_leaks_a_closed_wakeup_fd():
    a, b = socket.socketpair()
    a.setblocking(False)
    signal.set_wakeup_fd(a.fileno())
    a.close()
    b.close()


def test_b_restores_what_it_found():
    previous_fd = signal.set_wakeup_fd(-1)
    signal.set_wakeup_fd(previous_fd)


def test_c_changes_and_restores():
    a, b = socket.socketpair()
    a.setblocking(False)
    old = signal.set_wakeup_fd(a.fileno())
    signal.set_wakeup_fd(old)
    a.close()
    b.close()
"""


def test_a_leaked_wakeup_fd_fails_the_leaking_test_and_not_the_next(pytester):
    pytester.makeconftest(_INNER_CONFTEST)
    pytester.makepyfile(_INNER_TEST)

    result = pytester.runpytest_inprocess("-p", "no:cacheprovider", "-p", "no:randomly")

    result.assert_outcomes(passed=3, errors=1)
    result.stdout.fnmatch_lines(["*ERROR at teardown of test_a_leaks_a_closed_wakeup_fd*"])
