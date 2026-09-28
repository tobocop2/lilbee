"""Tests for DaemonCall, a waitable call on a daemon thread."""

from __future__ import annotations

import threading

import pytest

from lilbee.runtime.daemon_call import DaemonCall


def test_result_returns_the_calls_value() -> None:
    assert DaemonCall(lambda: 42, name="answer").result() == 42


def test_result_reraises_the_calls_exception() -> None:
    def _boom() -> int:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        DaemonCall(_boom, name="boom").result()


def test_result_reraises_a_base_exception() -> None:
    def _exit() -> int:
        raise SystemExit(3)

    with pytest.raises(SystemExit):
        DaemonCall(_exit, name="exit").result()


def test_result_waits_for_a_call_still_running() -> None:
    release = threading.Event()
    call = DaemonCall(lambda: release.wait(5.0), name="slow")
    assert call.wait(timeout=0.01) is False
    release.set()
    assert call.result() is True


def test_wait_reports_whether_the_call_finished() -> None:
    release = threading.Event()
    call = DaemonCall(lambda: release.wait(5.0), name="gate")
    assert call.wait(timeout=0.01) is False
    release.set()
    assert call.wait(timeout=5.0) is True


def test_call_runs_on_a_named_daemon_thread() -> None:
    call = DaemonCall(threading.current_thread, name="probe-thread")
    thread = call.result()
    assert thread.name == "probe-thread"
    assert thread.daemon is True
