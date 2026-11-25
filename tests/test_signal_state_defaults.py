"""The starting state and the stop steps of the signal code, one assertion per mutated value."""

from __future__ import annotations

import logging
import signal
import threading
import time

import pytest

from lilbee.app import services as services_mod
from lilbee.app.services import _EngineLifecycle, _ServerExit, _StopStep

_STOP_LOG = "Received signal SIGTERM; stopping the engine fleet before exit"


@pytest.fixture
def restored_signals():
    """Snapshot the real signal table, and put it back after the test."""
    watched = (signal.SIGINT, *_EngineLifecycle._hard_exit_signals())
    saved = {sig: signal.getsignal(sig) for sig in watched}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


@pytest.fixture
def pool_exit_stubbed(monkeypatch):
    """Keep the thread pools' exit hook from running: it sets a flag for the whole process."""
    monkeypatch.setattr(services_mod, "_end_pool_workers", lambda status: status)


@pytest.fixture
def stop_steps(monkeypatch):
    """Replace the two teardown calls with recorders that note the stop step they run in."""
    lifecycle = _EngineLifecycle()
    seen: list[tuple[str, _StopStep]] = []
    monkeypatch.setattr(
        services_mod,
        "answer_provider_waiters",
        lambda: seen.append(("answer", lifecycle.stop_step)),
    )
    monkeypatch.setattr(
        services_mod, "reset_services", lambda: seen.append(("reset", lifecycle.stop_step))
    )
    return lifecycle, seen


def test_a_new_server_exit_holds_no_hook_and_is_not_stopping():
    server_exit = _ServerExit()

    assert server_exit.request(0) is False
    assert server_exit.stopping is False


def test_a_new_lifecycle_has_installed_nothing():
    assert _EngineLifecycle()._installed is False


def test_a_new_lifecycle_installs_the_hard_exit_handlers(restored_signals):
    lifecycle = _EngineLifecycle()

    lifecycle.install()

    assert lifecycle._installed is True
    assert signal.getsignal(signal.SIGTERM) == lifecycle._on_hard_exit


def test_a_forked_child_gets_no_stop_thread():
    lifecycle = _EngineLifecycle()
    lifecycle._ensure_stop_thread()
    assert lifecycle._thread is not None

    lifecycle._reset_in_forked_child()

    assert lifecycle._thread is None


def test_forgetting_a_stop_clears_every_request():
    lifecycle = _EngineLifecycle()
    lifecycle.stop_requested = True
    lifecycle._signalled = True
    lifecycle.owed_status = 143
    lifecycle.stop_step = _StopStep.RESET_SERVICES

    lifecycle._forget_stop()

    assert lifecycle.stop_requested is False
    assert lifecycle._signalled is False
    assert lifecycle.owed_status is None
    assert lifecycle.stop_step is _StopStep.STARTING
    assert not lifecycle._stopped.is_set()


def test_the_stop_thread_runs_the_stop_with_the_queued_signal(monkeypatch):
    lifecycle = _EngineLifecycle()
    lifecycle._ensure_stop_thread()
    signums: list[int] = []
    monkeypatch.setattr(lifecycle, "_stop", signums.append)

    lifecycle.request_stop(signal.SIGTERM)

    assert lifecycle.wait_stopped(5)
    assert signums == [signal.SIGTERM]


def test_a_stop_for_a_signal_logs_the_signal_name(stop_steps, caplog):
    lifecycle, _ = stop_steps

    with caplog.at_level(logging.INFO, logger="lilbee.app.services"):
        lifecycle._stop(signal.SIGTERM)

    assert [record.getMessage() for record in caplog.records] == [_STOP_LOG]


def test_a_stop_for_an_exit_logs_no_signal(stop_steps, caplog):
    lifecycle, _ = stop_steps

    with caplog.at_level(logging.INFO, logger="lilbee.app.services"):
        lifecycle._stop(0)

    assert caplog.records == []


def test_the_stop_answers_the_waiters_before_it_stops_the_services(stop_steps):
    lifecycle, seen = stop_steps

    lifecycle._stop(0)

    assert seen == [
        ("answer", _StopStep.ANSWER_WAITERS),
        ("reset", _StopStep.RESET_SERVICES),
    ]


def test_an_exit_with_no_stop_thread_runs_the_stop_in_place_and_is_not_waited_for(
    stop_steps, caplog
):
    lifecycle, seen = stop_steps

    with caplog.at_level(logging.INFO, logger="lilbee.app.services"):
        waited = lifecycle.stop_for_exit()

    assert waited is False
    assert lifecycle.stop_requested is True
    assert [name for name, _ in seen] == ["answer", "reset"]
    assert caplog.records == []
    lifecycle.request_stop(signal.SIGTERM)
    assert lifecycle._requests.empty()


def test_an_exit_with_a_stop_thread_queues_an_exit_stop_and_is_waited_for(stop_steps, caplog):
    lifecycle, _ = stop_steps
    lifecycle._ensure_stop_thread()

    with caplog.at_level(logging.INFO, logger="lilbee.app.services"):
        waited = lifecycle.stop_for_exit()
        assert lifecycle.wait_stopped(5)

    assert waited is True
    assert caplog.records == []


class _FlushFails:
    """A stream or handler whose flush raises *error*."""

    def __init__(self, error: type[BaseException]) -> None:
        self._error = error

    def flush(self) -> None:
        raise self._error("flush failed")


class _FlushFailsHandler(logging.Handler):
    def __init__(self, error: type[BaseException]) -> None:
        super().__init__()
        self._error = error

    def flush(self) -> None:
        raise self._error("flush failed")


@pytest.mark.parametrize("error", [OSError, ValueError])
def test_flushing_the_logs_survives_a_handler_that_cannot_flush(error):
    handler = _FlushFailsHandler(error)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        services_mod._flush_logs()
    finally:
        root.removeHandler(handler)


@pytest.mark.parametrize("error", [AttributeError, OSError, ValueError])
@pytest.mark.parametrize("name", ["stdout", "stderr"])
def test_flushing_the_logs_survives_a_stream_that_cannot_flush(monkeypatch, name, error):
    # logging.shutdown flushes the handlers first and reads sys.stderr for its last-resort
    # handler, so it is replaced here: this test is about the two stdio flushes after it.
    events: list[str] = []
    other = "stderr" if name == "stdout" else "stdout"

    class _Recorder:
        def flush(self) -> None:
            events.append(f"{other} flushed")

    monkeypatch.setattr(services_mod.logging, "shutdown", lambda: events.append("handlers"))
    monkeypatch.setattr(services_mod.sys, name, _FlushFails(error))
    monkeypatch.setattr(services_mod.sys, other, _Recorder())

    services_mod._flush_logs()

    assert events == ["handlers", f"{other} flushed"]


def test_the_stragglers_are_the_live_non_daemon_threads_besides_the_caller():
    release = threading.Event()
    daemon = threading.Thread(target=release.wait, daemon=True)
    straggler = threading.Thread(target=release.wait)
    daemon.start()
    straggler.start()
    try:
        found = services_mod.straggler_threads()
    finally:
        release.set()
        straggler.join()
        daemon.join()

    assert straggler in found
    assert daemon not in found
    assert threading.main_thread() not in found


def test_the_straggler_join_waits_the_time_left_before_the_deadline(monkeypatch):
    waited: list[float | None] = []

    class _Thread:
        def join(self, timeout: float | None = None) -> None:
            waited.append(timeout)

    pending = [[_Thread()], []]
    monkeypatch.setattr(services_mod, "straggler_threads", lambda: pending.pop(0))

    services_mod._exit_if_stragglers(0, time.monotonic() + 2.0)

    assert len(waited) == 1
    assert waited[0] is not None
    assert 0.0 < waited[0] <= 2.0


def test_the_straggler_join_waits_no_time_once_the_deadline_passed(monkeypatch):
    waited: list[float | None] = []

    class _Thread:
        def join(self, timeout: float | None = None) -> None:
            waited.append(timeout)

    pending = [[_Thread()], []]
    monkeypatch.setattr(services_mod, "straggler_threads", lambda: pending.pop(0))

    services_mod._exit_if_stragglers(0, time.monotonic() - 10.0)

    assert waited == [0.0]


def test_the_straggler_exit_gives_the_stragglers_the_whole_budget(monkeypatch, pool_exit_stubbed):
    deadlines: list[float] = []
    monkeypatch.setattr(services_mod._state, "interactive", True)
    monkeypatch.setattr(
        services_mod, "_exit_if_stragglers", lambda status, deadline: deadlines.append(deadline)
    )
    before = time.monotonic()

    services_mod.exit_when_stragglers_would_hang(0)

    assert len(deadlines) == 1
    assert before + services_mod._STRAGGLER_BUDGET_S <= deadlines[0]
    assert deadlines[0] <= time.monotonic() + services_mod._STRAGGLER_BUDGET_S


def test_the_exit_after_a_signal_gives_the_stragglers_the_whole_budget(
    monkeypatch, restored_signals, pool_exit_stubbed
):
    deadlines: list[float] = []
    monkeypatch.setattr(services_mod._lifecycle, "stop_for_exit", lambda: False)
    monkeypatch.setattr(
        services_mod, "_exit_if_stragglers", lambda status, deadline: deadlines.append(deadline)
    )
    before = time.monotonic()

    services_mod.exit_after_signal(0)

    assert len(deadlines) == 1
    assert before + services_mod._STRAGGLER_BUDGET_S <= deadlines[0]
    assert deadlines[0] <= time.monotonic() + services_mod._STRAGGLER_BUDGET_S


def test_an_exit_that_cuts_the_stop_logs_the_step_it_cut(monkeypatch, restored_signals, caplog):
    monkeypatch.setattr(services_mod._lifecycle, "stop_for_exit", lambda: True)
    monkeypatch.setattr(services_mod._lifecycle, "wait_stopped", lambda timeout: False)
    monkeypatch.setattr(services_mod._lifecycle, "stop_step", _StopStep.ANSWER_WAITERS)
    monkeypatch.setattr(services_mod, "_exit_now", lambda status, note: None)
    monkeypatch.setattr(services_mod, "_exit_if_stragglers", lambda status, deadline: None)

    with caplog.at_level(logging.WARNING, logger="lilbee.app.services"):
        services_mod.exit_after_signal(0)

    assert [record.getMessage() for record in caplog.records] == [
        "The exit cut the engine stop while answering the callers that wait on the provider"
    ]
