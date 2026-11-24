"""The launch of a stand-in engine survives a port that someone else takes after it was chosen."""

from __future__ import annotations

import socket
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from tests.conformance import harness
from tests.conformance.test_managed_verdicts import GOOD_PIN

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the stand-in binary is a POSIX executable script"
)


@contextmanager
def _silent_listener() -> Iterator[int]:
    """A bound, listening socket that never answers: the engine cannot bind its port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        yield int(sock.getsockname()[1])


def _ports(monkeypatch: pytest.MonkeyPatch, taken: int, times: int) -> list[int]:
    """Make free_port hand out *taken* for *times* calls, then real free ports."""
    handed: list[int] = []
    real = harness.free_port

    def fake() -> int:
        port = taken if len(handed) < times else real()
        handed.append(port)
        return port

    monkeypatch.setattr(harness, "free_port", fake)
    return handed


def test_launch_retries_on_a_new_port_when_the_engine_cannot_bind(
    make_binary: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    with _silent_listener() as taken:
        handed = _ports(monkeypatch, taken, times=1)
        started = harness.launch(make_binary("retry"), GOOD_PIN, timeout=15.0)
    assert started is not None
    proc, port = started
    proc.kill()
    proc.wait()
    assert port == handed[1] != taken


def test_launch_gives_up_after_the_attempts(
    make_binary: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    with _silent_listener() as taken:
        handed = _ports(monkeypatch, taken, times=99)
        started = harness.launch(make_binary("giveup"), GOOD_PIN, timeout=15.0)
    assert started is None
    assert len(handed) == harness._LAUNCH_ATTEMPTS


def test_launch_stops_retrying_when_the_time_is_spent(
    make_binary: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    with _silent_listener() as taken:
        handed = _ports(monkeypatch, taken, times=99)
        started = harness.launch(make_binary("late"), GOOD_PIN, timeout=0.0)
    assert started is None
    assert len(handed) == 1
