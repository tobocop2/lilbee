"""The test helper that gives a child process the default SIGHUP and SIGINT."""

from __future__ import annotations

import signal
import subprocess
import sys
from collections.abc import Callable, Iterator

import pytest

from tests._signals import child_preexec

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal disposition")

_REPORT = (
    "import signal;"
    "print(signal.getsignal(signal.SIGHUP) == signal.SIG_IGN,"
    " signal.getsignal(signal.SIGHUP) == signal.SIG_DFL,"
    " signal.getsignal(signal.SIGINT) == signal.SIG_IGN)"
)


@pytest.fixture
def ignored_in_parent() -> Iterator[None]:
    """Ignore SIGHUP and SIGINT here, as a nohup launch from a background job does."""
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGHUP, signal.SIGINT)}
    for sig in saved:
        signal.signal(sig, signal.SIG_IGN)
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def _child_report(preexec: Callable[[], None] | None) -> str:
    done = subprocess.run(
        [sys.executable, "-I", "-c", _REPORT],
        capture_output=True,
        encoding="utf-8",
        timeout=30,
        preexec_fn=preexec,
        check=True,
    )
    return done.stdout.strip()


def test_a_child_inherits_the_ignored_signals_without_the_helper(ignored_in_parent: None) -> None:
    assert _child_report(None) == "True False True"


def test_the_helper_gives_the_child_the_default_sighup_and_a_live_sigint(
    ignored_in_parent: None,
) -> None:
    assert _child_report(child_preexec()) == "False True False"


def test_the_helper_runs_the_extra_setup_after_the_reset(ignored_in_parent: None) -> None:
    def ignore_sighup() -> None:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)

    assert _child_report(child_preexec(ignore_sighup)) == "True False False"
