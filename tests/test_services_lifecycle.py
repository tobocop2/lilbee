"""Hard-exit signals must tear the engine fleet down, not orphan it."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import logging
import os
import selectors
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from conftest import clean_env
from lilbee.app import services as services_mod
from lilbee.app.services import (
    ServicesStoppingError,
    exit_status_after_stop,
    get_services,
    install_engine_lifecycle_hooks,
    loop_takes_signals,
    peek_services,
    reset_services,
    reset_services_on_exit,
    set_server_exit_hook,
    set_services,
    stop_on_interrupt,
    wait_for_hard_exit_teardown,
)
from lilbee.cli.helpers import _run_foreground_sync

# The exact signals production installs: (SIGTERM, SIGHUP) on POSIX, SIGTERM alone
# on Windows, which has no SIGHUP. Deriving from the helper keeps this file
# collectable on every platform and pins the test set to what install() really does.
_HARD_EXIT_SIGNALS = [
    pytest.param(sig, id=sig.name.lower())
    for sig in services_mod._EngineLifecycle._hard_exit_signals()
]


@pytest.fixture(autouse=True)
def _restore_handlers():
    """Snapshot and restore the real signal table around every test."""
    watched = (signal.SIGINT, *services_mod._EngineLifecycle._hard_exit_signals())
    saved = {sig: signal.getsignal(sig) for sig in watched}
    for sig in saved:
        # A nohup or background launch must not change what the lifecycle routes.
        signal.signal(sig, signal.default_int_handler if sig == signal.SIGINT else signal.SIG_DFL)
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)
    reset_services()


def test_windows_routes_sigbreak_as_the_launcher_stop(monkeypatch) -> None:
    """The launcher stops serve with CTRL_BREAK, which Python delivers as SIGBREAK."""
    sigbreak = 21
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(signal, "SIGBREAK", sigbreak, raising=False)
    assert services_mod._EngineLifecycle._hard_exit_signals() == (signal.SIGTERM, sigbreak)


def _join_teardown() -> None:
    """Wait out the stop thread's teardown so assertions don't race it."""
    assert services_mod._lifecycle.wait_stopped(5)


def _install_stub_services() -> MagicMock:
    """Bind a mock container whose provider records shutdown()."""
    provider = MagicMock()
    container = MagicMock()
    container.provider = provider
    set_services(container)
    return provider


@pytest.mark.parametrize("sig", _HARD_EXIT_SIGNALS)
def test_hard_exit_signal_shuts_down_the_provider(sig):
    """SIGTERM / SIGHUP must run teardown before the process dies."""
    provider = _install_stub_services()
    install_engine_lifecycle_hooks()

    handler = signal.getsignal(sig)
    assert callable(handler), f"{sig!r} still has its default disposition"

    with pytest.raises(SystemExit):
        handler(sig, None)
    _join_teardown()

    provider.shutdown.assert_called_once()
    assert peek_services() is None


def test_install_is_idempotent():
    """Installing twice must not stack handlers or double-shutdown."""
    provider = _install_stub_services()
    install_engine_lifecycle_hooks()
    first = signal.getsignal(signal.SIGTERM)
    install_engine_lifecycle_hooks()
    assert signal.getsignal(signal.SIGTERM) is first

    with pytest.raises(SystemExit):
        first(signal.SIGTERM, None)
    _join_teardown()
    provider.shutdown.assert_called_once()


def test_teardown_runs_once_when_signal_and_atexit_both_fire():
    """The atexit hook after a signal handler must not shut down twice."""
    provider = _install_stub_services()
    install_engine_lifecycle_hooks()

    with pytest.raises(SystemExit):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    _join_teardown()
    reset_services()

    provider.shutdown.assert_called_once()


def test_teardown_runs_off_the_main_thread():
    """Regression: a second signal (the kernel pairs SIGCONT with SIGHUP for an
    orphaned process group) interrupts the main thread mid-handler; a teardown
    running there was aborted half-done, orphaning a loaded fleet. The reap
    must run on its own thread, out of any signal handler's reach."""
    provider = _install_stub_services()
    teardown_threads: list[str] = []
    provider.shutdown.side_effect = lambda: teardown_threads.append(threading.current_thread().name)
    install_engine_lifecycle_hooks()

    with pytest.raises(SystemExit):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    _join_teardown()

    assert teardown_threads == ["engine-stop"]


def test_signal_without_services_still_exits():
    """A hard exit before any container was built must not raise."""
    install_engine_lifecycle_hooks()
    with pytest.raises(SystemExit):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)


@pytest.mark.parametrize("sig", _HARD_EXIT_SIGNALS)
def test_a_serving_loop_takes_the_hard_exit_and_its_status(sig):
    """The twin of test_signal_without_services_still_exits: nothing is raised into the loop."""
    provider = _install_stub_services()
    statuses: list[int] = []
    set_server_exit_hook(statuses.append)
    try:
        install_engine_lifecycle_hooks()
        signal.getsignal(sig)(sig, None)
        _join_teardown()
    finally:
        set_server_exit_hook(None)
    assert statuses == [128 + sig]
    provider.shutdown.assert_called_once()


def test_the_first_ctrl_c_stops_the_server_and_the_second_forces_the_exit(monkeypatch):
    """One press takes the hard-exit path; the next press ends the process with status 130.

    The lifecycle starts with no stop thread, so the stop runs only if the routing starts one.
    """
    _without_a_stop_thread(monkeypatch)
    provider = _install_stub_services()
    statuses: list[int] = []
    exits: list[int] = []
    monkeypatch.setattr(services_mod.os, "_exit", exits.append)
    set_server_exit_hook(statuses.append)
    previous = signal.getsignal(signal.SIGINT)
    try:
        stop_on_interrupt()
        handler = signal.getsignal(signal.SIGINT)
        handler(signal.SIGINT, None)
        _join_teardown()
        assert exits == []
        handler(signal.SIGINT, None)
    finally:
        signal.signal(signal.SIGINT, previous)
        set_server_exit_hook(None)
    assert exits == [services_mod._FORCE_QUIT_EXIT_CODE]
    assert statuses == [128 + signal.SIGINT]
    provider.shutdown.assert_called_once()


def test_an_inherited_ignore_of_sigint_survives_stop_on_interrupt():
    """A process started with SIGINT ignored keeps ignoring it, as Python does at start."""
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        stop_on_interrupt()
        assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
    finally:
        signal.signal(signal.SIGINT, previous)


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no SIGHUP")
def test_an_inherited_ignore_of_sighup_survives_install():
    """A process started under nohup keeps ignoring SIGHUP; SIGTERM is still routed."""
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    install_engine_lifecycle_hooks()
    assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
    assert signal.getsignal(signal.SIGTERM) == services_mod._lifecycle._on_hard_exit


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal disposition")
def test_an_inherited_ignore_of_sigterm_does_not_stop_install_from_routing_it():
    """SIGTERM is the stop request, so install routes it even when the launch ignored it."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    install_engine_lifecycle_hooks()
    assert signal.getsignal(signal.SIGTERM) == services_mod._lifecycle._on_hard_exit


_ORDER_CONFTEST = """
from tests.conftest import _forget_engine_stop, _restore_server_stop_flag  # noqa: F401
"""

_ORDER_TEST = """
from lilbee.app import services as services_mod


def test_a_test_that_leaves_the_server_stopping():
    life = services_mod._lifecycle
    reset = type(life).reset

    def reset_after_checking_the_flag():
        # The wait fixture ends here. The restore fixture must not have run yet,
        # or a stop that ends late sets the flag after it.
        assert services_mod._server_exit.stopping is True
        del life.__dict__["reset"]
        reset(life)

    life.reset = reset_after_checking_the_flag
    services_mod._server_exit.stopping = True
"""


def test_the_stop_wait_tears_down_before_the_server_stop_flag_is_restored(pytester):
    """The server-stopping flag is restored after the wait for the engine stop, not before.

    A stop sets the flag before it ends, so a restore that ran first would let a
    late stop set the flag again for the next test. The inner conftest sets the
    fixtures up in the order that is wrong unless the wait fixture asks for the restore
    fixture, so the check fails when that dependency goes.
    """
    pytester.makeconftest(_ORDER_CONFTEST)
    pytester.makepyfile(_ORDER_TEST)

    result = pytester.runpytest_inprocess("-p", "no:cacheprovider")

    result.assert_outcomes(passed=1)


def test_install_routes_sigterm_to_the_stop_handler():
    """The handler is the lifecycle's own, so a SIGTERM reaches the stop and not the default."""
    install_engine_lifecycle_hooks()
    assert signal.getsignal(signal.SIGTERM) == services_mod._lifecycle._on_hard_exit


def test_the_handler_force_quit_writes_to_fd_2_and_takes_no_lock(monkeypatch):
    """The second press runs in handler context: a bare fd write, then the exit; no log flush."""
    writes: list[tuple[int, bytes]] = []
    exits: list[int] = []

    def _no_lock_taking_call(*_args: object) -> None:
        raise AssertionError("the force quit took a lock in the signal handler")

    monkeypatch.setattr(services_mod.os, "write", lambda fd, data: writes.append((fd, data)))
    monkeypatch.setattr(services_mod.os, "_exit", exits.append)
    monkeypatch.setattr(services_mod, "_write_exit_note", _no_lock_taking_call)
    monkeypatch.setattr(services_mod, "_flush_logs", _no_lock_taking_call)
    services_mod._lifecycle._signalled = True

    services_mod._lifecycle._on_hard_exit(signal.SIGINT, None)

    assert writes == [(2, services_mod._FORCE_QUIT_NOTE.encode())]
    assert exits == [services_mod._FORCE_QUIT_EXIT_CODE]


def test_the_exit_flush_uses_the_standard_logging_shutdown_then_flushes_stdio(monkeypatch):
    """logging.shutdown reaches every live handler; a walk of the logger table misses some."""
    steps: list[str] = []
    monkeypatch.setattr(logging, "shutdown", lambda: steps.append("shutdown"))
    out, err = MagicMock(), MagicMock()
    out.flush.side_effect = lambda: steps.append("stdout")
    err.flush.side_effect = lambda: steps.append("stderr")
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    services_mod._flush_logs()

    assert steps == ["shutdown", "stdout", "stderr"]


def test_the_force_quit_kills_the_ingest_workers_then_ends_the_process(monkeypatch):
    """Only Linux ends a worker with its parent, and this exit skips the join of the workers."""
    steps: list[object] = []
    worker = MagicMock()
    worker.kill.side_effect = lambda: steps.append("kill")
    monkeypatch.setattr(services_mod.multiprocessing, "active_children", lambda: [worker, worker])
    monkeypatch.setattr(services_mod, "_write_exit_note", steps.append)
    monkeypatch.setattr(services_mod.os, "_exit", steps.append)

    services_mod.force_quit()

    exit_code = services_mod._FORCE_QUIT_EXIT_CODE
    assert steps == ["kill", "kill", services_mod._FORCE_QUIT_NOTE, exit_code]


def test_the_ctrl_c_force_quit_from_a_handler_kills_the_workers_first(monkeypatch):
    steps: list[object] = []
    worker = MagicMock()
    worker.kill.side_effect = lambda: steps.append("kill")
    monkeypatch.setattr(services_mod.multiprocessing, "active_children", lambda: [worker, worker])
    monkeypatch.setattr(services_mod.os, "write", lambda fd, data: steps.append((fd, data)))
    monkeypatch.setattr(services_mod.os, "_exit", steps.append)

    services_mod.force_quit_from_handler()

    note = (2, services_mod._FORCE_QUIT_NOTE.encode())
    assert steps == ["kill", "kill", note, services_mod._FORCE_QUIT_EXIT_CODE]


@pytest.mark.parametrize("error", [ValueError("process object is closed"), PermissionError()])
def test_a_worker_that_cannot_be_killed_does_not_stop_the_force_quit(monkeypatch, error):
    """Another thread can close or reap a worker between the listing and the kill."""
    killed: list[str] = []
    gone, live = MagicMock(), MagicMock()
    gone.kill.side_effect = error
    live.kill.side_effect = lambda: killed.append("live")
    monkeypatch.setattr(services_mod.multiprocessing, "active_children", lambda: [gone, live])

    services_mod._kill_worker_processes()

    assert killed == ["live"]


def test_the_handler_force_quit_exits_when_fd_2_is_gone(monkeypatch):
    exits: list[int] = []

    def _closed(_fd: int, _data: bytes) -> int:
        raise OSError("bad file descriptor")

    monkeypatch.setattr(services_mod.os, "write", _closed)
    monkeypatch.setattr(services_mod.os, "_exit", exits.append)

    services_mod._force_quit_from_handler(128 + signal.SIGTERM)

    assert exits == [128 + signal.SIGTERM]


def test_a_ctrl_c_after_the_server_hook_is_gone_joins_the_stop_without_raising():
    """A stuck lifespan leaves the hook cleared; the press must not raise into the loop."""
    provider = _install_stub_services()
    set_server_exit_hook(lambda _status: None)
    services_mod.answer_provider_waiters()
    set_server_exit_hook(None)
    previous = signal.getsignal(signal.SIGINT)
    try:
        stop_on_interrupt()
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        _join_teardown()
    finally:
        signal.signal(signal.SIGINT, previous)
        set_server_exit_hook(lambda _status: None)
        set_server_exit_hook(None)
    provider.shutdown.assert_called_once()


@pytest.mark.parametrize("sig", _HARD_EXIT_SIGNALS)
def test_a_terminate_signal_after_the_server_hook_is_gone_joins_the_stop_without_raising(sig):
    """The twin of the Ctrl-C case: SIGTERM and SIGHUP must not raise into a stuck lifespan."""
    provider = _install_stub_services()
    set_server_exit_hook(lambda _status: None)
    services_mod.answer_provider_waiters()
    set_server_exit_hook(None)
    try:
        install_engine_lifecycle_hooks()
        signal.getsignal(sig)(sig, None)
        _join_teardown()
    finally:
        set_server_exit_hook(lambda _status: None)
        set_server_exit_hook(None)
    provider.shutdown.assert_called_once()


def test_a_ctrl_c_before_any_server_runs_still_unwinds_the_main_thread():
    """With no loop and no stop under way, the press exits through SystemExit."""
    _install_stub_services()
    previous = signal.getsignal(signal.SIGINT)
    try:
        stop_on_interrupt()
        with pytest.raises(SystemExit) as exit_info:
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        _join_teardown()
    finally:
        signal.signal(signal.SIGINT, previous)
    assert exit_info.value.code == 128 + signal.SIGINT


@pytest.mark.parametrize("sig", _HARD_EXIT_SIGNALS)
def test_a_first_terminate_signal_with_no_server_raises_even_when_the_teardown_wins_the_race(
    sig, monkeypatch
):
    """The teardown marks the server stopping; the first signal must not read that as a stop."""
    _install_stub_services()
    started: list[int] = []

    def _teardown_marks_stopping_at_once(signum: int) -> None:
        started.append(signum)
        services_mod.answer_provider_waiters()

    install_engine_lifecycle_hooks()
    monkeypatch.setattr(services_mod._lifecycle, "request_stop", _teardown_marks_stopping_at_once)
    with pytest.raises(SystemExit) as exit_info:
        signal.getsignal(sig)(sig, None)
    assert exit_info.value.code == 128 + sig
    assert started == [sig]


@pytest.mark.parametrize("sig", _HARD_EXIT_SIGNALS)
@pytest.mark.parametrize("server", [False, True], ids=["no-server", "server-takes-it"])
def test_a_first_signal_that_joins_no_stop_owes_no_exit_status(sig, server):
    """Only a signal that joins a stop already under way owes its status to the exit."""
    _install_stub_services()
    statuses: list[int] = []
    set_server_exit_hook(statuses.append if server else None)
    try:
        install_engine_lifecycle_hooks()
        if server:
            signal.getsignal(sig)(sig, None)
        else:
            with pytest.raises(SystemExit):
                signal.getsignal(sig)(sig, None)
        _join_teardown()
    finally:
        set_server_exit_hook(None)
    assert statuses == ([128 + sig] if server else [])
    assert exit_status_after_stop(0) == 0


def test_cli_entry_point_installs_the_hooks():
    """Every surface enters through the Typer callback, so the hooks install there."""
    import inspect

    from lilbee.cli.app import _default

    assert "install_engine_lifecycle_hooks()" in inspect.getsource(_default)


def test_mounting_the_tui_does_not_touch_the_signal_table():
    """Installing from on_mount would leave real handlers behind in every TUI test."""
    import inspect

    from lilbee.cli.tui.app import LilbeeApp

    assert "install_engine_lifecycle_hooks" not in inspect.getsource(LilbeeApp.on_mount)


def test_install_off_main_thread_is_a_noop():
    """signal.signal raises off the main thread; installing there must not crash."""
    import threading

    errors: list[BaseException] = []

    def _worker() -> None:
        try:
            install_engine_lifecycle_hooks()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()
    assert errors == []


@pytest.mark.parametrize("sig", _HARD_EXIT_SIGNALS)
def test_hard_exit_logs_the_received_signal(sig, caplog):
    """server.log must record which signal ended the process, for diagnostics."""
    _install_stub_services()
    install_engine_lifecycle_hooks()
    handler = signal.getsignal(sig)
    assert callable(handler)

    with caplog.at_level(logging.INFO, logger="lilbee.app.services"):
        with pytest.raises(SystemExit):
            handler(sig, None)
        _join_teardown()

    assert any(signal.Signals(sig).name in rec.message for rec in caplog.records)


def test_wait_for_hard_exit_teardown_waits_for_the_signal_stop():
    """serve's lock release must not outrun a signal-driven fleet stop."""
    provider = _install_stub_services()
    release = threading.Event()
    provider.shutdown.side_effect = lambda: release.wait(timeout=5)
    install_engine_lifecycle_hooks()
    with pytest.raises(SystemExit):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

    timer = threading.Timer(0.2, release.set)
    timer.start()
    wait_for_hard_exit_teardown()
    timer.join()

    assert release.is_set()
    provider.shutdown.assert_called_once()


def test_wait_for_hard_exit_teardown_without_a_teardown_is_a_noop():
    wait_for_hard_exit_teardown()
    assert services_mod._lifecycle.stop_requested is False


def test_exit_teardown_runs_on_the_stop_thread():
    """With handlers installed, Ctrl-C lands on the main thread, so the teardown leaves it."""
    provider = _install_stub_services()
    runners: list[threading.Thread] = []
    provider.shutdown.side_effect = lambda: runners.append(threading.current_thread())
    install_engine_lifecycle_hooks()

    reset_services_on_exit()

    assert [runner.name for runner in runners] == ["engine-stop"]
    assert peek_services() is None


def _without_a_stop_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap in a lifecycle that never installed a handler, as library use has."""
    monkeypatch.setattr(services_mod, "_lifecycle", services_mod._EngineLifecycle())


def test_an_exit_without_handlers_stops_on_the_calling_thread_and_starts_no_thread(
    monkeypatch,
):
    """atexit refuses new threads on 3.12 and later, so the library teardown runs in place."""
    _without_a_stop_thread(monkeypatch)
    provider = _install_stub_services()
    runners: list[threading.Thread] = []
    provider.shutdown.side_effect = lambda: runners.append(threading.current_thread())

    def _refuse(self: threading.Thread) -> None:
        raise AssertionError(f"the exit teardown started thread {self.name}")

    monkeypatch.setattr(threading.Thread, "start", _refuse)
    reset_services_on_exit()
    monkeypatch.undo()

    assert runners == [threading.current_thread()]
    assert peek_services() is None


def test_a_ctrl_c_during_a_teardown_in_place_force_quits(monkeypatch):
    _without_a_stop_thread(monkeypatch)
    provider = _install_stub_services()
    provider.shutdown.side_effect = KeyboardInterrupt
    exits: list[int] = []
    monkeypatch.setattr(services_mod.os, "_exit", exits.append)

    reset_services_on_exit()

    assert exits == [services_mod._FORCE_QUIT_EXIT_CODE]


def test_a_second_interrupt_force_quits_instead_of_waiting(monkeypatch):
    """A second Ctrl-C force-quits: the child guard reaps the engine."""
    exits: list[int] = []
    monkeypatch.setattr(services_mod.os, "_exit", exits.append)

    def _interrupted(timeout: float | None) -> bool:
        raise KeyboardInterrupt

    _without_a_stop_thread(monkeypatch)
    services_mod._lifecycle.stop_requested = True
    monkeypatch.setattr(services_mod._lifecycle, "wait_stopped", _interrupted)

    services_mod.wait_for_hard_exit_teardown()

    assert exits == [services_mod._FORCE_QUIT_EXIT_CODE]


def test_the_signal_handler_starts_no_thread(monkeypatch):
    """A handler that starts a thread can deadlock on threading's own lock (3.11, 3.12)."""
    provider = _install_stub_services()
    install_engine_lifecycle_hooks()
    handler = signal.getsignal(signal.SIGTERM)

    def _refuse(self: threading.Thread) -> None:
        raise AssertionError(f"the signal handler started thread {self.name}")

    monkeypatch.setattr(threading.Thread, "start", _refuse)
    with pytest.raises(SystemExit) as raised:
        handler(signal.SIGTERM, None)
    monkeypatch.undo()
    _join_teardown()

    assert raised.value.code == 128 + signal.SIGTERM
    provider.shutdown.assert_called_once()


def test_every_exit_path_waits_for_a_stop_that_already_cleared_the_services():
    """The stop clears the singleton first; a later exit must still wait for the teardown."""
    provider = _install_stub_services()
    release = threading.Event()
    provider.shutdown.side_effect = lambda: release.wait(timeout=5)
    install_engine_lifecycle_hooks()
    with pytest.raises(SystemExit):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    deadline = time.monotonic() + 5
    while peek_services() is not None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert peek_services() is None

    waiter = threading.Thread(target=reset_services_on_exit)
    waiter.start()
    waiter.join(timeout=0.3)
    still_waiting = waiter.is_alive()
    release.set()
    waiter.join(timeout=5)

    assert still_waiting
    provider.shutdown.assert_called_once()


def test_after_the_stop_began_the_services_refuse_and_build_nothing(monkeypatch):
    """A container built after the stop would outlive it and leave its lock file behind."""
    provider = _install_stub_services()
    reset_services_on_exit()
    build = MagicMock()
    monkeypatch.setattr(services_mod, "build_services", build)

    with pytest.raises(ServicesStoppingError, match="lilbee is stopping"):
        get_services()

    build.assert_not_called()
    assert peek_services() is None
    provider.shutdown.assert_called_once()


def test_a_container_built_while_the_stop_begins_is_shut_down_not_kept(monkeypatch):
    """The stop swapped the singleton out during the build; the new one must not survive."""
    import lilbee.app.settings as settings_mod
    import lilbee.modelhub.registry as registry_mod

    _without_a_stop_thread(monkeypatch)
    reset_services()
    built = MagicMock()

    def _build_as_the_stop_begins(*_args: object, **_kwargs: object) -> MagicMock:
        services_mod._lifecycle.stop_requested = True
        return built

    monkeypatch.setattr(registry_mod, "ModelRegistry", MagicMock())
    monkeypatch.setattr(settings_mod, "reconcile_embedding_dim", MagicMock())
    monkeypatch.setattr(services_mod, "build_services", _build_as_the_stop_begins)

    with pytest.raises(ServicesStoppingError):
        get_services()

    built.provider.shutdown.assert_called_once_with()
    built.store.close.assert_called_once_with()
    assert peek_services() is None


def test_a_store_rebuilt_while_the_stop_begins_is_closed_not_kept(monkeypatch):
    """A /reset store rebuild must not bring back the container the stop cleared."""
    import dataclasses

    import lilbee.retrieval.canary_probe as canary_mod
    import lilbee.retrieval.clustering as clustering_mod
    import lilbee.retrieval.concepts as concepts_mod
    import lilbee.retrieval.query as query_mod

    _without_a_stop_thread(monkeypatch)
    _install_stub_services()
    new_store = MagicMock()

    def _rebuild_as_the_stop_begins(*_args: object) -> MagicMock:
        services_mod._lifecycle.stop_requested = True
        reset_services()
        return new_store

    monkeypatch.setattr(services_mod, "_new_store", _rebuild_as_the_stop_begins)
    monkeypatch.setattr(canary_mod, "canary_probe", MagicMock())
    monkeypatch.setattr(concepts_mod, "ConceptGraph", MagicMock())
    monkeypatch.setattr(clustering_mod, "Clusterer", MagicMock())
    monkeypatch.setattr(query_mod, "Searcher", MagicMock())
    monkeypatch.setattr(dataclasses, "replace", lambda svc, **_changes: MagicMock())

    with pytest.raises(ServicesStoppingError):
        services_mod.reset_store()

    new_store.close.assert_called_once_with()
    assert peek_services() is None
    with pytest.raises(ServicesStoppingError):
        get_services()


def _rebuild_with(monkeypatch, new_stores, during_build):
    """Build each store from *new_stores* in turn; run *during_build* in the first build.

    Returns the list of builds, so a test can count them. Everything else in
    ``reset_store`` runs for real.
    """
    builds = iter(new_stores)
    calls: list[object] = []

    def _build(*_args: object) -> MagicMock:
        calls.append(None)
        if len(calls) == 1:
            during_build()
        return next(builds)

    monkeypatch.setattr(services_mod, "_new_store", _build)
    return calls


_THREAD_TIMEOUT_S = 10.0


def _start_thread(fn, errors: list[BaseException]) -> threading.Thread:
    """Run *fn* on a daemon thread; an exception it raises goes to *errors*."""

    def _run() -> None:
        try:
            fn()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


def _run_within_timeout(fn) -> None:
    """Run *fn* on a thread and fail, not hang, when it does not return in time."""
    errors: list[BaseException] = []
    thread = _start_thread(fn, errors)
    thread.join(_THREAD_TIMEOUT_S)
    assert not thread.is_alive(), "the call did not return: it waits on a lock it should not take"
    if errors:
        raise errors[0]


def test_a_store_reset_does_not_bring_back_a_container_that_was_cleared_meanwhile(monkeypatch):
    """A reset_services that ran during the build is not undone, and the new store is closed."""
    from tests.conftest import make_mock_services

    _without_a_stop_thread(monkeypatch)
    set_services(make_mock_services())
    new_store = MagicMock()
    calls = _rebuild_with(monkeypatch, [new_store], reset_services)

    _run_within_timeout(services_mod.reset_store)

    new_store.close.assert_called_once_with()
    assert peek_services() is None
    assert len(calls) == 1


def test_a_store_reset_keeps_a_container_that_replaced_it_meanwhile(monkeypatch):
    """A container installed during the build stays; the new store is closed, not swapped in."""
    from tests.conftest import make_mock_services

    _without_a_stop_thread(monkeypatch)
    set_services(make_mock_services())
    replacement = make_mock_services()
    new_store = MagicMock()

    def _a_new_container_is_installed() -> None:
        services_mod._state.singleton = replacement

    calls = _rebuild_with(monkeypatch, [new_store], _a_new_container_is_installed)

    services_mod.reset_store()

    new_store.close.assert_called_once_with()
    assert peek_services() is replacement
    replacement.store.close.assert_not_called()
    assert len(calls) == 1


def test_a_store_reset_refuses_the_swap_once_the_stop_began_though_the_container_is_the_same(
    monkeypatch,
):
    """The stop flag alone refuses the swap, though the container is still the current one."""
    from tests.conftest import make_mock_services

    _without_a_stop_thread(monkeypatch)
    svc = make_mock_services()
    set_services(svc)
    new_store = MagicMock()

    def _the_stop_begins() -> None:
        services_mod._lifecycle.stop_requested = True

    _rebuild_with(monkeypatch, [new_store], _the_stop_begins)

    with pytest.raises(services_mod.ServicesStoppingError):
        services_mod.reset_store()

    new_store.close.assert_called_once_with()
    assert services_mod._state.singleton is svc


@pytest.mark.parametrize(
    ("target", "error"),
    [
        ("lilbee.retrieval.concepts.ConceptGraph", OSError),
        ("lilbee.retrieval.clustering.Clusterer", OSError),
        ("lilbee.retrieval.query.Searcher", OSError),
        ("lilbee.retrieval.query.Searcher", KeyboardInterrupt),
    ],
)
def test_a_store_reset_closes_the_new_store_when_a_dependent_raises(monkeypatch, target, error):
    """A dependent that fails to build leaves no open new store, and the old container stays."""
    from tests.conftest import make_mock_services

    _without_a_stop_thread(monkeypatch)
    current = make_mock_services()
    set_services(current)
    new_store = MagicMock()
    _rebuild_with(monkeypatch, [new_store], lambda: None)
    monkeypatch.setattr(target, MagicMock(side_effect=error("boom")))

    with pytest.raises(error, match="boom"):
        services_mod.reset_store()

    new_store.close.assert_called_once_with()
    assert peek_services() is current
    current.store.close.assert_not_called()
    assert not services_mod._store_rebuild_lock.locked()


def test_a_store_reset_holds_the_rebuild_lock_for_the_build_but_not_the_swap_lock(monkeypatch):
    """Two resets never build at once, and the build never blocks the stop's swap."""
    from tests.conftest import make_mock_services

    _without_a_stop_thread(monkeypatch)
    set_services(make_mock_services())
    seen: list[tuple[bool, bool]] = []

    def _probe_the_locks() -> None:
        seen.append(
            (services_mod._store_rebuild_lock.locked(), services_mod._reset_swap_lock.locked())
        )

    _rebuild_with(monkeypatch, [MagicMock()], _probe_the_locks)

    services_mod.reset_store()

    assert seen == [(True, False)]
    assert not services_mod._store_rebuild_lock.locked()


def test_a_store_reset_reads_the_stop_inside_the_swap_lock(monkeypatch):
    """A stop cannot land between the stop check and the swap: the check holds the swap lock."""
    from tests.conftest import make_mock_services

    class _WatchedStop(services_mod._EngineLifecycle):
        """A lifecycle whose stop flag records whether the swap lock was held when it was read."""

        def __init__(self) -> None:
            self.reads: list[bool] = []
            super().__init__()

        @property
        def stop_requested(self) -> bool:
            self.reads.append(services_mod._reset_swap_lock.locked())
            return False

        @stop_requested.setter
        def stop_requested(self, value: bool) -> None:
            """The base class assigns the flag; the watched one stays False."""

    watched = _WatchedStop()
    monkeypatch.setattr(services_mod, "_lifecycle", watched)
    set_services(make_mock_services())
    _rebuild_with(monkeypatch, [MagicMock()], lambda: None)

    services_mod.reset_store()

    assert watched.reads == [True]


class _TwoStoreResets:
    """Two reset_store calls on threads: the first holds its build open until released."""

    def __init__(self, monkeypatch) -> None:
        from tests.conftest import make_mock_services

        _without_a_stop_thread(monkeypatch)
        self.original = make_mock_services()
        set_services(self.original)
        self.first_store, self.second_store = MagicMock(), MagicMock()
        self.in_first_build = threading.Event()
        self.release_first = threading.Event()
        self.second_building = threading.Event()
        self.seen: list[object] = []
        self.errors: list[list[BaseException]] = [[], []]
        stores = iter([self.first_store, self.second_store])

        def _build(*_args: object) -> MagicMock:
            self.seen.append(services_mod._state.singleton)
            if len(self.seen) == 1:
                self.in_first_build.set()
                self.release_first.wait(_THREAD_TIMEOUT_S)
            else:
                self.second_building.set()
            return next(stores)

        monkeypatch.setattr(services_mod, "_new_store", _build)

    def start(self) -> list[threading.Thread]:
        """Start the first reset, wait until it builds, then start the second."""
        first = _start_thread(services_mod.reset_store, self.errors[0])
        assert self.in_first_build.wait(_THREAD_TIMEOUT_S)
        second = _start_thread(services_mod.reset_store, self.errors[1])
        return [first, second]

    def finish(self, threads: list[threading.Thread]) -> None:
        self.release_first.set()
        for thread in threads:
            thread.join(_THREAD_TIMEOUT_S)
            assert not thread.is_alive()


def test_two_store_resets_at_once_build_one_after_the_other(monkeypatch):
    """The second reset waits for the first, then builds from the container the first installed."""
    resets = _TwoStoreResets(monkeypatch)
    threads = resets.start()
    try:
        assert not resets.second_building.wait(0.3)
    finally:
        resets.finish(threads)

    assert resets.errors == [[], []]
    assert len(resets.seen) == 2
    assert resets.seen[0] is resets.original
    second_read = resets.seen[1]
    assert second_read is not resets.original
    assert second_read.store is resets.first_store
    kept = peek_services()
    assert kept is not None
    assert kept.store is resets.second_store
    resets.original.store.close.assert_called_once_with()
    resets.first_store.close.assert_called_once_with()
    resets.second_store.close.assert_not_called()


def test_a_stop_while_a_second_store_reset_waits_leaves_no_container_and_no_open_store(
    monkeypatch,
):
    """The waiting reset builds nothing and the first one closes its store."""
    resets = _TwoStoreResets(monkeypatch)
    threads = resets.start()
    try:
        services_mod._lifecycle.stop_requested = True
        reset_services()
    finally:
        resets.finish(threads)

    assert [type(errors[0]) for errors in resets.errors if errors] == [ServicesStoppingError]
    assert resets.errors[1] == []
    assert len(resets.seen) == 1
    resets.first_store.close.assert_called_once_with()
    resets.second_store.close.assert_not_called()
    assert peek_services() is None


@pytest.mark.parametrize("sig", _HARD_EXIT_SIGNALS)
def test_a_signal_during_an_exit_stop_joins_it_and_sets_the_exit_status(sig):
    """A signal that lands during a clean exit's stop gives its status once the stop ends."""
    provider = _install_stub_services()
    release = threading.Event()
    provider.shutdown.side_effect = lambda: release.wait(timeout=5)
    install_engine_lifecycle_hooks()
    assert services_mod._lifecycle.stop_for_exit()

    signal.getsignal(sig)(sig, None)
    release.set()
    _join_teardown()

    assert exit_status_after_stop(0) == 128 + sig
    provider.shutdown.assert_called_once()


@pytest.mark.parametrize("sig", _HARD_EXIT_SIGNALS)
def test_a_second_signal_force_quits_with_its_status(sig, monkeypatch):
    """The first signal raises for the clean stop; the next one ends the process at once."""
    exits: list[int] = []
    monkeypatch.setattr(services_mod.os, "_exit", exits.append)
    _install_stub_services()
    install_engine_lifecycle_hooks()
    with pytest.raises(SystemExit) as raised:
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    _join_teardown()
    assert exits == []

    signal.getsignal(sig)(sig, None)

    assert raised.value.code == 128 + signal.SIGTERM
    assert exits == [128 + sig]


_STUCK_WORKER_CHILD = """
import asyncio
import logging
import sys
import time

from lilbee.app.services import install_engine_lifecycle_hooks

# The stop thread logs a signal after the handler took it: the test waits for that line.
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")


def _worker() -> None:
    print("ready", flush=True)
    time.sleep(float(sys.argv[1]))  # a worker thread: no signal interrupts it


async def _main() -> None:
    await asyncio.to_thread(_worker)


install_engine_lifecycle_hooks()
asyncio.run(_main())
"""


def _start_worker_child(tmp_path, worker_s: float) -> subprocess.Popen[str]:
    """Start a process whose asyncio.run waits on a worker thread for *worker_s* seconds."""
    script = tmp_path / "stuck_worker.py"
    script.write_text(_STUCK_WORKER_CHILD, encoding="utf-8")
    child = subprocess.Popen(
        [sys.executable, str(script), str(worker_s)],
        env=clean_env(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    assert child.stdout is not None
    assert child.stdout.readline().strip() == "ready", child.stderr.read() if child.stderr else ""
    return child


def _wait_or_kill(child: subprocess.Popen[str], timeout: float) -> tuple[int | None, str]:
    """The child's status within *timeout* (None when it still ran, then killed) and stderr."""
    try:
        status: int | None = child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        status = None
        child.kill()
        child.wait()
    _, stderr = child.communicate()
    return status, stderr


@pytest.mark.skipif(sys.platform == "win32", reason="Windows delivers no SIGTERM to a handler")
def test_a_second_terminate_ends_a_process_whose_worker_thread_is_stuck(tmp_path):
    """The exit after a first SIGTERM joins the stuck worker; the second SIGTERM ends it."""
    child = _start_worker_child(tmp_path, worker_s=120.0)
    assert child.stdout is not None
    try:
        child.send_signal(signal.SIGTERM)
        # A second SIGTERM sent before the handler ran merges with the first one.
        took_first = child.stdout.readline()
        still_running = child.poll() is None
        child.send_signal(signal.SIGTERM)
        status = child.wait()  # bounded by the test timeout
    finally:
        child.kill()  # a no-op once the child ended
    _, stderr = child.communicate()

    assert took_first.startswith("Received signal SIGTERM"), stderr
    assert still_running, stderr
    assert status == 128 + signal.SIGTERM, stderr
    assert services_mod._FORCE_QUIT_NOTE in stderr


@pytest.mark.skipif(sys.platform == "win32", reason="Windows delivers no SIGTERM to a handler")
def test_a_first_terminate_waits_for_a_worker_that_finishes(tmp_path):
    """A worker that returns lets the first SIGTERM end the process cleanly, with no force quit."""
    child = _start_worker_child(tmp_path, worker_s=1.0)
    child.send_signal(signal.SIGTERM)

    status, stderr = _wait_or_kill(child, timeout=30.0)

    assert status == 128 + signal.SIGTERM, stderr
    assert services_mod._FORCE_QUIT_NOTE not in stderr


_EXIT_WITH_A_WORKER_CHILD = """
import multiprocessing
import signal
import sys
import time
from unittest.mock import MagicMock

from lilbee.app.services import (
    exit_when_stragglers_would_hang,
    install_engine_lifecycle_hooks,
    reset_services_on_exit,
    set_services,
)

if __name__ == "__main__":
    install_engine_lifecycle_hooks()
    # Started as the catalog download starts its child: spawn, daemonic.
    worker = multiprocessing.get_context("spawn").Process(
        target=time.sleep, args=(120.0,), daemon=True
    )
    worker.start()
    if sys.argv[1:] == ["stop-thread-takes-signals"]:
        # Threads keep the mask they started with, so only the stop thread takes them.
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    container = MagicMock()
    container.provider.shutdown.side_effect = lambda: time.sleep(120.0)
    set_services(container)
    print(worker.pid, flush=True)
    # The TUI's exit, in its order: the engine stop hangs, so the exit waits.
    reset_services_on_exit()
    exit_when_stragglers_would_hang(0)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="Windows sends no Ctrl-C to a child process")
@pytest.mark.parametrize(
    ("signals", "taker"),
    [
        ((signal.SIGINT,), "any-thread"),
        ((signal.SIGTERM, signal.SIGINT), "stop-thread-takes-signals"),
    ],
    ids=["ctrl-c", "terminate-then-ctrl-c-on-the-stop-thread"],
)
def test_a_ctrl_c_in_the_exit_wait_leaves_no_worker_process_running(tmp_path, signals, taker):
    """The force quit kills the workers; a daemonic child outlives a bare os._exit on every OS.

    A signal that the stop thread takes must still reach the main thread's exit wait.
    """
    import psutil

    script = tmp_path / "exit_with_a_worker.py"
    script.write_text(_EXIT_WITH_A_WORKER_CHILD, encoding="utf-8")
    child = subprocess.Popen(
        [sys.executable, str(script), taker],
        env=clean_env(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    assert child.stdout is not None and child.stderr is not None
    worker = psutil.Process(int(child.stdout.readline()))
    try:
        for line in child.stderr:
            if line == services_mod._STOPPING_NOTE:
                break
        for sig in signals:
            child.send_signal(sig)
        status, stderr = _wait_or_kill(child, timeout=10.0)
        _, alive = psutil.wait_procs([worker], timeout=5.0)
    finally:
        with contextlib.suppress(psutil.NoSuchProcess):
            worker.kill()

    assert status == services_mod._FORCE_QUIT_EXIT_CODE, stderr
    assert alive == []


def _record_worker_kills(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Record each worker kill and each exit, in order."""
    events: list[object] = []
    worker = MagicMock()
    worker.kill.side_effect = lambda: events.append("kill")
    monkeypatch.setattr(services_mod.multiprocessing, "active_children", lambda: [worker])
    monkeypatch.setattr(services_mod.os, "_exit", events.append)
    return events


def test_a_ctrl_c_in_the_exit_wait_kills_the_workers_before_the_exit(monkeypatch):
    events = _record_worker_kills(monkeypatch)

    def _interrupted(timeout: float | None) -> bool:
        raise KeyboardInterrupt

    _without_a_stop_thread(monkeypatch)
    services_mod._lifecycle.stop_requested = True
    monkeypatch.setattr(services_mod._lifecycle, "wait_stopped", _interrupted)

    services_mod.wait_for_hard_exit_teardown()

    assert events == ["kill", services_mod._FORCE_QUIT_EXIT_CODE]


def test_the_exit_while_an_open_runs_kills_the_workers_before_the_exit(monkeypatch):
    events = _record_worker_kills(monkeypatch)
    monkeypatch.setattr(services_mod, "wait_for_opens", lambda timeout: False)

    services_mod.exit_if_opens_remain(143)

    assert events == ["kill", 143]


def test_a_cut_engine_stop_kills_the_workers_before_the_exit(monkeypatch):
    provider = _install_stub_services()
    release = threading.Event()
    provider.shutdown.side_effect = lambda: release.wait(timeout=5)
    events = _record_worker_kills(monkeypatch)
    monkeypatch.setattr(services_mod, "_STRAGGLER_BUDGET_S", 0.2)
    monkeypatch.setattr(services_mod, "_all_threads", lambda: [threading.main_thread()])
    install_engine_lifecycle_hooks()

    try:
        services_mod.exit_after_signal(143)
    finally:
        release.set()

    assert events == ["kill", 143]


def test_a_forked_child_drops_the_parent_stop_thread():
    """A fork copies no thread; the child must start its own stop thread."""
    lifecycle = services_mod._EngineLifecycle()
    lifecycle._ensure_stop_thread()
    parent_thread = lifecycle._thread
    assert parent_thread is not None

    lifecycle._reset_in_forked_child()

    assert lifecycle._thread is None
    assert lifecycle.stop_requested is False
    lifecycle._ensure_stop_thread()
    assert lifecycle.stop_for_exit()
    assert lifecycle.wait_stopped(5)
    assert lifecycle._thread is not parent_thread


def test_a_slow_exit_names_the_wait_and_the_way_out(capsys, monkeypatch):
    """A teardown that outlives the grace says what is happening."""
    monkeypatch.setattr(services_mod, "_STOPPING_NOTE_GRACE_S", 0.0)
    provider = _install_stub_services()
    release = threading.Event()
    provider.shutdown.side_effect = lambda: release.wait(timeout=5)
    install_engine_lifecycle_hooks()

    try:
        waiter = threading.Thread(target=reset_services_on_exit)
        waiter.start()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if "stopping the engine" in capsys.readouterr().err:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("the slow exit never named the wait")
    finally:
        release.set()
        waiter.join(timeout=5)


def test_the_exit_wait_never_blocks_longer_than_a_slice(monkeypatch):
    """After the note the wait runs in slices until the stop ends; a signal lands between two."""
    timeouts: list[float] = []
    results = iter([False, False, True])

    def _wait(timeout: float) -> bool:
        timeouts.append(timeout)
        return next(results)

    monkeypatch.setattr(services_mod._lifecycle, "wait_stopped", _wait)
    monkeypatch.setattr(services_mod, "_write_exit_note", lambda note: None)

    services_mod._wait_for_stop()

    slice_s = services_mod._SIGNAL_CHECK_S
    assert timeouts == [services_mod._STOPPING_NOTE_GRACE_S, slice_s, slice_s]


def test_the_exit_wait_lasts_until_the_stop_ends(monkeypatch):
    """No count of slices ends the wait; only the end of the stop does."""
    slices_before_the_end = 500
    calls = 0

    def _wait(timeout: float) -> bool:
        nonlocal calls
        calls += 1
        return calls > slices_before_the_end

    monkeypatch.setattr(services_mod._lifecycle, "wait_stopped", _wait)
    monkeypatch.setattr(services_mod, "_write_exit_note", lambda note: None)

    services_mod._wait_for_stop()

    assert calls == slices_before_the_end + 1


def test_a_fast_exit_stays_silent(capsys):
    """A teardown inside the grace keeps the exit as quiet as it always was."""
    _install_stub_services()

    reset_services_on_exit()

    assert "stopping the engine" not in capsys.readouterr().err


def test_exit_note_survives_a_closed_stderr(monkeypatch):
    """atexit can run after stderr is gone; the note must not raise."""

    class _Closed:
        def write(self, _note: str) -> int:
            raise ValueError("I/O operation on closed file")

        def flush(self) -> None:  # pragma: no cover - never reached
            raise ValueError

    monkeypatch.setattr(services_mod.sys, "stderr", _Closed())
    services_mod._write_exit_note("note")


def test_exit_teardown_without_services_starts_no_thread(monkeypatch):
    """Every process exit runs this hook; an unused container costs no thread."""
    reset_services()
    spawned = MagicMock()
    monkeypatch.setattr(services_mod.threading, "Thread", spawned)

    reset_services_on_exit()

    spawned.assert_not_called()


class _Wedged:
    """A non-daemon thread facade that never finishes joining."""

    daemon = False
    name = "wedged-straggler"  # the leak-check fixture reads thread names

    def is_alive(self) -> bool:
        return True

    def join(self, timeout: float | None = None) -> None:
        return None


class TestStragglerExit:
    @pytest.fixture(autouse=True)
    def _interactive(self, monkeypatch):
        monkeypatch.setattr(services_mod._state, "interactive", True)
        # The real hook sets the pools' global shutdown flag in this process.
        monkeypatch.setattr(services_mod, "_end_pool_workers", lambda status: status)

    def test_idle_pool_workers_the_hook_ends_leave_the_normal_exit(self, monkeypatch) -> None:
        exits: list[int] = []
        monkeypatch.setattr(services_mod.os, "_exit", exits.append)
        threads = self._fake_enumerate(monkeypatch, [_Wedged()])

        def hook_ends_them(status: int) -> int:
            del threads[1:]
            return status

        monkeypatch.setattr(services_mod, "_end_pool_workers", hook_ends_them)

        services_mod.exit_when_stragglers_would_hang(0)

        assert exits == []

    def test_a_signal_in_the_pool_hook_gives_the_exit_its_status(self, monkeypatch) -> None:
        exits: list[int] = []
        monkeypatch.setattr(services_mod.os, "_exit", exits.append)
        monkeypatch.setattr(services_mod, "_STRAGGLER_BUDGET_S", 0.0)
        monkeypatch.setattr(services_mod, "_end_pool_workers", lambda status: 143)
        threads = self._fake_enumerate(monkeypatch, [_Wedged()])

        services_mod.exit_when_stragglers_would_hang(0)
        del threads[1:]

        assert exits == [143]

    def test_a_non_interactive_process_keeps_its_threads(self, monkeypatch) -> None:
        """A test host or library embedder owns its threads; never exit them."""
        monkeypatch.setattr(services_mod._state, "interactive", False)
        exits: list[int] = []
        monkeypatch.setattr(services_mod.os, "_exit", exits.append)
        threads = self._fake_enumerate(monkeypatch, [_Wedged()])

        services_mod.exit_when_stragglers_would_hang(0)
        del threads[1:]

        assert exits == []

    def _fake_enumerate(self, monkeypatch, extra):
        """Patch enumerate with a mutable list, cleared before teardown so the
        leak-check fixture never meets the fakes."""
        threads = [threading.main_thread(), *extra]
        monkeypatch.setattr(services_mod, "_all_threads", lambda: list(threads))
        return threads

    def test_a_wedged_thread_exits_instead_of_hanging(self, monkeypatch, capsys) -> None:
        exits: list[int] = []
        monkeypatch.setattr(services_mod.os, "_exit", exits.append)
        monkeypatch.setattr(services_mod, "_STRAGGLER_BUDGET_S", 0.0)
        threads = self._fake_enumerate(monkeypatch, [_Wedged()])

        services_mod.exit_when_stragglers_would_hang(0)
        del threads[1:]

        assert exits == [0]
        assert "would not stop" in capsys.readouterr().err

    def test_a_wedged_thread_exits_with_the_status_of_the_app(self, monkeypatch) -> None:
        exits: list[int] = []
        monkeypatch.setattr(services_mod.os, "_exit", exits.append)
        monkeypatch.setattr(services_mod, "_STRAGGLER_BUDGET_S", 0.0)
        threads = self._fake_enumerate(monkeypatch, [_Wedged()])

        services_mod.exit_when_stragglers_would_hang(1)
        del threads[1:]

        assert exits == [1]

    @pytest.mark.parametrize("join_raises", [False, True], ids=["wedged", "ctrl-c"])
    def test_a_forced_straggler_exit_kills_the_workers_first(
        self, monkeypatch, join_raises
    ) -> None:
        events = _record_worker_kills(monkeypatch)
        monkeypatch.setattr(services_mod, "_STRAGGLER_BUDGET_S", 0.0)

        class _Straggler(_Wedged):
            def join(self, timeout: float | None = None) -> None:
                if join_raises:
                    raise KeyboardInterrupt

        threads = self._fake_enumerate(monkeypatch, [_Straggler()])

        services_mod.exit_when_stragglers_would_hang(1)
        del threads[1:]

        status = services_mod._FORCE_QUIT_EXIT_CODE if join_raises else 1
        assert events == ["kill", status]

    def test_a_clean_shutdown_exits_normally(self, monkeypatch, capsys) -> None:
        exits: list[int] = []
        monkeypatch.setattr(services_mod.os, "_exit", exits.append)
        self._fake_enumerate(monkeypatch, [])

        services_mod.exit_when_stragglers_would_hang(0)

        assert exits == []
        assert capsys.readouterr().err == ""

    def test_an_interrupt_during_the_budget_force_quits(self, monkeypatch) -> None:
        exits: list[int] = []
        monkeypatch.setattr(services_mod.os, "_exit", exits.append)

        class _Interruptible(_Wedged):
            def join(self, timeout: float | None = None) -> None:
                raise KeyboardInterrupt

        threads = self._fake_enumerate(monkeypatch, [_Interruptible()])

        services_mod.exit_when_stragglers_would_hang(0)
        del threads[1:]

        assert exits == [services_mod._FORCE_QUIT_EXIT_CODE]


@pytest.mark.skipif(sys.platform == "win32", reason="Windows delivers no SIGTERM to a handler")
@pytest.mark.parametrize("sig", [signal.SIGINT, *_HARD_EXIT_SIGNALS], ids=lambda s: s.name.lower())
@pytest.mark.parametrize("loop_owns_signals", [False, True], ids=["plain", "loop-owned"])
def test_a_signal_in_a_running_loop_stops_the_fleet_and_exits_with_its_status(
    sig, loop_owns_signals
):
    """A loop blocked in select is interrupted by the signal and unwinds with its status."""
    provider = _install_stub_services()
    previous = signal.getsignal(signal.SIGINT)

    async def serve() -> None:
        with contextlib.ExitStack() as stack:
            if loop_owns_signals:
                stack.enter_context(loop_takes_signals(asyncio.get_running_loop()))
            os.kill(os.getpid(), sig)
            await asyncio.sleep(30)

    try:
        install_engine_lifecycle_hooks()
        stop_on_interrupt()
        with pytest.raises(SystemExit) as raised:
            asyncio.run(serve())
        _join_teardown()
    finally:
        signal.signal(signal.SIGINT, previous)

    assert raised.value.code == 128 + sig
    provider.shutdown.assert_called_once()


class _TerminateInTheNextSleep(selectors.DefaultSelector):
    """A selector whose next sleep runs the hard-exit handler, as a SIGTERM landing there does."""

    def __init__(self) -> None:
        super().__init__()
        self.armed = False

    def select(self, timeout: float | None = None):
        if self.armed:
            self.armed = False
            services_mod._lifecycle._on_hard_exit(signal.SIGTERM, None)
        return super().select(timeout)


def test_a_terminate_after_the_add_sync_finished_exits_with_its_status(monkeypatch):
    """The finished run's stop is still queued when the signal lands; the exit keeps 143.

    The sync leaves a task, so the runner's close runs the loop for more than one
    step, and a stale stop would end that run before the cancelled task finished.
    """
    provider = _install_stub_services()
    install_engine_lifecycle_hooks()
    selector = _TerminateInTheNextSleep()
    monkeypatch.setattr(
        asyncio.events, "new_event_loop", lambda: asyncio.SelectorEventLoop(selector)
    )
    cleaned: list[str] = []

    async def left() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.append("left")

    async def finishes(**_kwargs: object) -> str:
        asyncio.get_running_loop().create_task(left())
        await asyncio.sleep(0)
        selector.armed = True
        return "synced"

    monkeypatch.setattr("lilbee.data.ingest.sync", finishes)
    with pytest.raises(SystemExit) as raised:
        _run_foreground_sync()
    _join_teardown()

    assert raised.value.code == 128 + signal.SIGTERM
    assert cleaned == ["left"]
    provider.shutdown.assert_called_once()


def test_no_module_runs_a_loop_past_the_exit_hold():
    """Every loop run on the main thread goes through run_main, so a late exit keeps its status."""
    src = Path(services_mod.__file__).resolve().parents[1]
    calls = [
        f"{path.relative_to(src)}:{node.lineno}"
        for path in sorted(src.rglob("*.py"))
        if path.name != "asyncio_loop.py"
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Attribute)
        and node.attr == "run"
        and isinstance(node.value, ast.Name)
        and node.value.id == "asyncio"
    ]
    assert calls == []


def _signal_once_the_loop_runs(ready: threading.Event) -> None:
    # raise_signal from this thread delivers the signal to this thread, so the main
    # thread, asleep in the loop's select, is not interrupted by it.
    assert ready.wait(10)
    signal.raise_signal(signal.SIGINT)


class _NoSignalHandlerLoop(asyncio.SelectorEventLoop):
    """A loop shaped like the Windows proactor: no signal handlers, its own wakeup descriptor.

    The proactor points the signal wakeup descriptor at its self-pipe when it is made on
    the main thread, so a signal that lands while it waits wakes it.
    """

    def __init__(self) -> None:
        super().__init__()
        self._wakeup_before = signal.set_wakeup_fd(self._csock.fileno())

    def add_signal_handler(self, sig, callback, *args):
        raise NotImplementedError

    def close(self) -> None:
        signal.set_wakeup_fd(self._wakeup_before)
        super().close()


# On Windows the real proactor runs the no-handler cases; elsewhere a loop shaped like it.
_LOOPS = [
    pytest.param(
        asyncio.SelectorEventLoop,
        id="loop-handlers",
        marks=pytest.mark.skipif(sys.platform == "win32", reason="Windows has no loop handlers"),
    ),
    pytest.param(
        getattr(asyncio, "ProactorEventLoop", _NoSignalHandlerLoop), id="no-loop-handlers"
    ),
]


def _record_loop_signals(monkeypatch) -> list[int]:
    """Record every signal that reaches the stop as a loop callback.

    The plain SIGINT handler is the same function, so only this record shows that a
    loop callback, not the plain handler, took a press.
    """
    taken: list[int] = []
    on_loop_signal = services_mod._lifecycle.on_loop_signal

    def recording_loop_signal(signum: int) -> None:
        taken.append(signum)
        on_loop_signal(signum)

    monkeypatch.setattr(services_mod._lifecycle, "on_loop_signal", recording_loop_signal)
    return taken


@pytest.mark.parametrize("loop_factory", _LOOPS)
def test_a_ctrl_c_inside_a_catch_all_step_still_ends_the_loop(monkeypatch, loop_factory):
    """A press that lands in a step whose library swallows every exception still stops the loop."""
    provider = _install_stub_services()
    loop_signals = _record_loop_signals(monkeypatch)
    swallowed: list[BaseException] = []

    async def serve() -> None:
        with loop_takes_signals(asyncio.get_running_loop()):
            try:
                signal.raise_signal(signal.SIGINT)
            except BaseException as exc:  # a library's catch-all
                swallowed.append(exc)
            await asyncio.sleep(30)

    install_engine_lifecycle_hooks()
    stop_on_interrupt()
    started = time.monotonic()
    with pytest.raises(SystemExit) as raised, asyncio.Runner(loop_factory=loop_factory) as runner:
        runner.run(serve())
    waited = time.monotonic() - started
    _join_teardown()

    assert swallowed == []
    assert raised.value.code == 128 + signal.SIGINT
    assert loop_signals == [signal.SIGINT]
    assert waited < 10  # the stop ended the loop; it did not sleep out the 30 s
    provider.shutdown.assert_called_once()


@pytest.mark.parametrize("loop_factory", _LOOPS)
def test_a_signal_on_another_thread_ends_an_idle_loop_with_its_status(monkeypatch, loop_factory):
    """The loop takes the signal as a callback, so it ends the loop wherever the signal lands."""
    provider = _install_stub_services()
    loop_signals = _record_loop_signals(monkeypatch)
    previous = signal.getsignal(signal.SIGINT)
    ready = threading.Event()
    sender = threading.Thread(target=_signal_once_the_loop_runs, args=(ready,))

    async def idle() -> None:
        loop = asyncio.get_running_loop()
        with loop_takes_signals(loop):
            loop.call_soon(ready.set)
            sender.start()
            await asyncio.sleep(30)

    try:
        install_engine_lifecycle_hooks()
        stop_on_interrupt()
        interrupt_handler = signal.getsignal(signal.SIGINT)
        with (
            pytest.raises(SystemExit) as raised,
            asyncio.Runner(loop_factory=loop_factory) as runner,
        ):
            runner.run(idle())
        _join_teardown()
        after = signal.getsignal(signal.SIGINT)
    finally:
        signal.signal(signal.SIGINT, previous)
        sender.join(timeout=5)

    assert raised.value.code == 128 + signal.SIGINT
    assert loop_signals == [signal.SIGINT]
    assert after == interrupt_handler
    provider.shutdown.assert_called_once()


@pytest.mark.parametrize("loop_factory", _LOOPS)
def test_a_second_ctrl_c_in_a_loop_owned_loop_force_quits(monkeypatch, loop_factory):
    """The loop callback reaches the one handler: the second press ends the process."""
    provider = _install_stub_services()
    forced: list[int] = []
    monkeypatch.setattr(services_mod, "_force_quit_from_handler", forced.append)
    loop_signals = _record_loop_signals(monkeypatch)
    # A registered server hook takes the first press, so the callback does not raise.
    set_server_exit_hook(lambda _status: None)
    previous = signal.getsignal(signal.SIGINT)

    async def until(condition) -> None:
        async with asyncio.timeout(10):
            while not condition():
                await asyncio.sleep(0.01)

    async def serve() -> None:
        with loop_takes_signals(asyncio.get_running_loop()):
            # raise_signal, not os.kill: on Windows os.kill(SIGINT) ends the process outright.
            signal.raise_signal(signal.SIGINT)
            await until(lambda: services_mod._lifecycle._signalled)
            first_press_forced = list(forced)
            signal.raise_signal(signal.SIGINT)
            await until(lambda: forced)
            assert first_press_forced == []

    try:
        stop_on_interrupt()
        with asyncio.Runner(loop_factory=loop_factory) as runner:
            runner.run(serve())
        _join_teardown()
    finally:
        signal.signal(signal.SIGINT, previous)
        set_server_exit_hook(lambda _status: None)
        set_server_exit_hook(None)

    assert forced == [128 + signal.SIGINT]
    assert loop_signals == [signal.SIGINT, signal.SIGINT]
    provider.shutdown.assert_called_once()


@pytest.mark.parametrize("loop_factory", _LOOPS)
def test_an_ordinary_exception_in_the_loop_signals_block_still_restores_the_handlers(loop_factory):
    """The restore is a finally: a body that raises anything leaves the earlier handlers back."""
    install_engine_lifecycle_hooks()
    stop_on_interrupt()
    routed = services_mod._lifecycle.routed_in_loop()
    before = {sig: signal.getsignal(sig) for sig in routed}

    async def failing() -> None:
        with loop_takes_signals(asyncio.get_running_loop()):
            raise RuntimeError("the body failed")

    with (
        pytest.raises(RuntimeError, match="the body failed"),
        asyncio.Runner(loop_factory=loop_factory) as runner,
    ):
        runner.run(failing())

    assert routed
    assert {sig: signal.getsignal(sig) for sig in routed} == before
    assert all(callable(handler) for handler in before.values())


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no SIGHUP")
def test_an_ignored_sighup_stays_ignored_inside_the_loop_signals():
    """The loop takes no signal the launch ignored: an inherited ignore of SIGHUP holds."""
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    async def inside() -> signal.Handlers | object:
        with loop_takes_signals(asyncio.get_running_loop()):
            return signal.getsignal(signal.SIGHUP)

    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
        assert runner.run(inside()) == signal.SIG_IGN
    assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN


@pytest.mark.parametrize("loop_factory", _LOOPS)
def test_the_loop_takes_only_the_signals_the_process_routes(loop_factory):
    """With Ctrl-C not routed, as in the TUI, the loop leaves it a KeyboardInterrupt."""
    install_engine_lifecycle_hooks()
    routed = signal.getsignal(signal.SIGTERM)

    async def inside() -> tuple[object, object, object]:
        # The runner puts its own Ctrl-C handler in place for the run.
        interrupt_before = signal.getsignal(signal.SIGINT)
        with loop_takes_signals(asyncio.get_running_loop()):
            interrupt = signal.getsignal(signal.SIGINT)
            return interrupt_before, interrupt, signal.getsignal(signal.SIGTERM)

    with asyncio.Runner(loop_factory=loop_factory) as runner:
        interrupt_before, interrupt, terminate = runner.run(inside())

    assert interrupt == interrupt_before
    assert terminate != routed  # the loop's handler, or the one that queues for it, took it
    assert signal.getsignal(signal.SIGTERM) == routed


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no loop signal handlers")
def test_the_wakeup_descriptor_set_before_the_loop_signals_returns_after_them():
    """The TUI's descriptor wakes its loop again once the loop gives the signals back."""
    install_engine_lifecycle_hooks()
    reader, writer = socket.socketpair()
    writer.setblocking(False)
    wakeup = writer.fileno()

    async def inside() -> int:
        with loop_takes_signals(asyncio.get_running_loop()):
            current = signal.set_wakeup_fd(-1)
            signal.set_wakeup_fd(current)
            return current

    with reader, writer, asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
        previous = signal.set_wakeup_fd(wakeup)
        try:
            during = runner.run(inside())
            after = signal.set_wakeup_fd(previous)
        finally:
            signal.set_wakeup_fd(previous)

    assert during not in (wakeup, -1)  # the loop's own descriptor while it takes the signals
    assert after == wakeup


def test_the_thread_seam_lists_the_running_threads():
    assert threading.current_thread() in services_mod._all_threads()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no loop signal handlers")
def test_a_closed_loop_leaves_the_terminate_handlers_in_place():
    """A signal after the loop closed must still reach the stop, not the default disposition."""
    from lilbee import mcp_server

    install_engine_lifecycle_hooks()
    installed = {sig: signal.getsignal(sig) for sig in services_mod._lifecycle._hard_exit_signals()}

    async def serve() -> None:
        pass

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mcp_server, "get_services", lambda: None)
        patch.setattr(mcp_server, "build_mcp_server", MagicMock())
        mcp_server.build_mcp_server.return_value.run_stdio_async = serve
        mcp_server.main()

    after = {sig: signal.getsignal(sig) for sig in installed}
    assert after == installed
    assert all(callable(handler) for handler in after.values())


class TestExitAfterSignal:
    def test_a_stop_past_the_budget_is_cut_and_names_its_step(
        self, monkeypatch, caplog, capsys
    ) -> None:
        provider = _install_stub_services()
        release = threading.Event()
        provider.shutdown.side_effect = lambda: release.wait(timeout=5)
        exits: list[int] = []
        monkeypatch.setattr(services_mod.os, "_exit", exits.append)
        monkeypatch.setattr(services_mod, "_STRAGGLER_BUDGET_S", 0.2)
        monkeypatch.setattr(services_mod, "_all_threads", lambda: [threading.main_thread()])
        install_engine_lifecycle_hooks()

        try:
            with caplog.at_level(logging.WARNING, logger="lilbee.app.services"):
                services_mod.exit_after_signal(143)
        finally:
            release.set()

        assert exits == [143]
        assert "would not stop" in capsys.readouterr().err
        assert any("stopping the services" in rec.getMessage() for rec in caplog.records)

    def test_a_signal_that_joined_a_cut_stop_sets_the_exit_status(self, monkeypatch) -> None:
        provider = _install_stub_services()
        release = threading.Event()
        provider.shutdown.side_effect = lambda: release.wait(timeout=5)
        exits: list[int] = []
        monkeypatch.setattr(services_mod.os, "_exit", exits.append)
        monkeypatch.setattr(services_mod, "_STRAGGLER_BUDGET_S", 0.2)
        monkeypatch.setattr(services_mod._lifecycle, "owed_status", 143)
        monkeypatch.setattr(services_mod, "_all_threads", lambda: [threading.main_thread()])
        install_engine_lifecycle_hooks()

        try:
            services_mod.exit_after_signal(0)
        finally:
            release.set()

        assert exits == [143]

    def test_a_wedged_thread_cannot_hold_the_exit_status_back(self, monkeypatch) -> None:
        # The stop runs in place, so the zero budget cuts only the straggler wait.
        _without_a_stop_thread(monkeypatch)
        provider = _install_stub_services()
        exits: list[int] = []
        monkeypatch.setattr(services_mod.os, "_exit", exits.append)
        monkeypatch.setattr(services_mod, "_STRAGGLER_BUDGET_S", 0.0)
        threads = [threading.main_thread(), _Wedged()]
        monkeypatch.setattr(services_mod, "_all_threads", lambda: list(threads))

        services_mod.exit_after_signal(143)
        del threads[1:]

        assert exits == [143]
        provider.shutdown.assert_called_once()

    def test_a_clean_unwind_returns_to_the_caller(self, monkeypatch) -> None:
        exits: list[int] = []
        monkeypatch.setattr(services_mod.os, "_exit", exits.append)
        monkeypatch.setattr(services_mod, "_all_threads", lambda: [threading.main_thread()])

        services_mod.exit_after_signal(130)

        assert exits == []

    def test_a_signal_that_joined_the_stop_sets_the_exit_status(self, monkeypatch) -> None:
        _without_a_stop_thread(monkeypatch)
        _install_stub_services()
        exits: list[int] = []
        monkeypatch.setattr(services_mod.os, "_exit", exits.append)
        monkeypatch.setattr(services_mod, "_STRAGGLER_BUDGET_S", 0.0)
        monkeypatch.setattr(services_mod._lifecycle, "owed_status", 143)
        threads = [threading.main_thread(), _Wedged()]
        monkeypatch.setattr(services_mod, "_all_threads", lambda: list(threads))

        services_mod.exit_after_signal(0)
        del threads[1:]

        assert exits == [143]

    def test_a_ctrl_c_during_the_stop_wait_force_quits(self, monkeypatch) -> None:
        exits: list[int] = []
        monkeypatch.setattr(services_mod.os, "_exit", exits.append)
        monkeypatch.setattr(services_mod._lifecycle, "stop_for_exit", lambda: True)

        def interrupted(timeout: float) -> bool:
            raise KeyboardInterrupt

        monkeypatch.setattr(services_mod._lifecycle, "wait_stopped", interrupted)
        monkeypatch.setattr(services_mod, "_all_threads", lambda: [threading.main_thread()])

        services_mod.exit_after_signal(0)

        assert exits == [services_mod._FORCE_QUIT_EXIT_CODE]


@pytest.mark.parametrize(("code", "status"), [(None, 0), (3, 3), ("a message", 1)])
def test_exit_code_of_maps_a_code_as_the_interpreter_does(code: object, status: int) -> None:
    assert services_mod.exit_code_of(SystemExit(code)) == status


class TestEndCommand:
    def _threads(self, monkeypatch, extra):
        """Patch enumerate with a mutable list, cleared before teardown."""
        threads = [threading.main_thread(), *extra]
        monkeypatch.setattr(services_mod, "_all_threads", lambda: list(threads))
        # The real hook sets the pools' global shutdown flag in this process.
        monkeypatch.setattr(services_mod, "_end_pool_workers", lambda status: status)
        return threads

    def test_idle_pool_workers_the_hook_ends_leave_the_normal_exit(self, monkeypatch) -> None:
        bounded = MagicMock()
        monkeypatch.setattr(services_mod, "exit_after_signal", bounded)
        threads = self._threads(monkeypatch, [_Wedged()])

        def hook_ends_them(status: int) -> int:
            del threads[1:]
            return status

        monkeypatch.setattr(services_mod, "_end_pool_workers", hook_ends_them)
        stop = SystemExit(0)

        with pytest.raises(SystemExit) as raised:
            services_mod.end_command(stop)

        assert raised.value is stop
        bounded.assert_not_called()

    def test_a_signal_during_the_pool_hook_sets_the_status(self, monkeypatch) -> None:
        monkeypatch.setattr(services_mod, "exit_after_signal", MagicMock())
        threads = self._threads(monkeypatch, [_Wedged()])

        def signalled(status: int) -> int:
            del threads[1:]
            return 143

        monkeypatch.setattr(services_mod, "_end_pool_workers", signalled)

        with pytest.raises(SystemExit) as raised:
            services_mod.end_command(SystemExit(0))

        assert raised.value.code == 143

    def test_with_no_thread_left_it_raises_the_command_exit(self, monkeypatch) -> None:
        bounded = MagicMock()
        monkeypatch.setattr(services_mod, "exit_after_signal", bounded)
        self._threads(monkeypatch, [])
        stop = SystemExit(2)

        with pytest.raises(SystemExit) as raised:
            services_mod.end_command(stop)

        assert raised.value is stop
        bounded.assert_not_called()

    def test_a_thread_left_takes_the_bounded_exit_with_the_command_status(
        self, monkeypatch
    ) -> None:
        bounded = MagicMock()
        monkeypatch.setattr(services_mod, "exit_after_signal", bounded)
        threads = self._threads(monkeypatch, [_Wedged()])
        stop = SystemExit(None)

        with pytest.raises(SystemExit) as raised:
            services_mod.end_command(stop)
        del threads[1:]

        assert raised.value is stop
        bounded.assert_called_once_with(0)

    def test_a_signal_during_the_pool_hook_gives_the_bounded_exit_its_status(
        self, monkeypatch
    ) -> None:
        bounded = MagicMock()
        monkeypatch.setattr(services_mod, "exit_after_signal", bounded)
        self._threads(monkeypatch, [_Wedged()])
        monkeypatch.setattr(services_mod, "_end_pool_workers", lambda status: 143)

        with pytest.raises(SystemExit):
            services_mod.end_command(SystemExit(0))

        bounded.assert_called_once_with(143)

    def test_a_signal_during_the_bounded_exit_sets_the_status(self, monkeypatch) -> None:
        monkeypatch.setattr(services_mod, "exit_after_signal", MagicMock())
        monkeypatch.setattr(services_mod._lifecycle, "owed_status", 143)
        threads = self._threads(monkeypatch, [_Wedged()])

        with pytest.raises(SystemExit) as raised:
            services_mod.end_command(SystemExit(0))
        del threads[1:]

        assert raised.value.code == 143


class TestEndPoolWorkers:
    def test_the_pools_exit_hook_runs_and_the_status_stays(self, monkeypatch) -> None:
        ran = threading.Event()
        monkeypatch.setattr(services_mod.concurrent.futures.thread, "_python_exit", ran.set)

        assert services_mod._end_pool_workers(3) == 3
        assert ran.is_set()

    def test_the_hook_runs_on_a_daemon_thread_so_a_busy_worker_cannot_hold_the_exit(
        self, monkeypatch
    ) -> None:
        seen: list[tuple[str, bool]] = []

        def hook() -> None:
            current = threading.current_thread()
            seen.append((current.name, current.daemon))

        monkeypatch.setattr(services_mod.concurrent.futures.thread, "_python_exit", hook)

        services_mod._end_pool_workers(0)

        assert seen == [("pool-exit", True)]

    def test_the_wait_lasts_until_a_slow_hook_ends(self, monkeypatch) -> None:
        ended = threading.Event()

        def slow_hook() -> None:
            time.sleep(0.3)
            ended.set()

        monkeypatch.setattr(services_mod.concurrent.futures.thread, "_python_exit", slow_hook)

        assert services_mod._end_pool_workers(0) == 0
        assert ended.is_set()

    def test_a_hook_held_by_a_busy_worker_ends_the_wait_at_the_budget(self, monkeypatch) -> None:
        release = threading.Event()
        monkeypatch.setattr(services_mod.concurrent.futures.thread, "_python_exit", release.wait)
        monkeypatch.setattr(services_mod, "_STRAGGLER_BUDGET_S", 0.05)

        try:
            assert services_mod._end_pool_workers(0) == 0
        finally:
            release.set()

    def test_a_signal_during_the_wait_gives_its_status(self, monkeypatch) -> None:
        monkeypatch.setattr(services_mod.concurrent.futures.thread, "_python_exit", lambda: None)

        @contextlib.contextmanager
        def signalled():
            raise SystemExit(143)
            yield

        monkeypatch.setattr(services_mod, "_interrupt_force_quits", signalled)

        assert services_mod._end_pool_workers(0) == 143


class TestExitIfOpensRemain:
    @pytest.mark.parametrize(
        ("open_done", "expected_events", "expected_note"),
        [
            # The teardown atexit would run comes before the exit that skips atexit.
            pytest.param(False, ["teardown", "exit 143"], True, id="open-still-running"),
            pytest.param(True, [], False, id="no-open-running"),
        ],
    )
    def test_exits_with_the_status_only_while_an_open_runs(
        self, monkeypatch, capsys, open_done, expected_events, expected_note
    ) -> None:
        provider = _install_stub_services()
        events: list[str] = []
        provider.shutdown.side_effect = lambda: events.append("teardown")

        def record_exit(status: int) -> None:
            events.append(f"exit {status}")

        monkeypatch.setattr(services_mod.os, "_exit", record_exit)
        waits: list[float] = []

        def wait(timeout: float) -> bool:
            waits.append(timeout)
            return open_done

        monkeypatch.setattr(services_mod, "wait_for_opens", wait)

        services_mod.exit_if_opens_remain(143)

        assert waits == [0.0]
        assert events == expected_events
        stderr = capsys.readouterr().err
        assert ("the library is still opening" in stderr) is expected_note

    def test_the_exit_flushes_every_log_handler_first(self, monkeypatch) -> None:
        order: list[str] = []

        class _Recording(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                pass

            def flush(self) -> None:
                order.append("flush")

        handler = _Recording()
        named = logging.getLogger("lilbee.test_exit_flush")
        named.addHandler(handler)
        monkeypatch.setattr(services_mod.os, "_exit", lambda status: order.append(f"exit {status}"))
        monkeypatch.setattr(services_mod, "wait_for_opens", lambda timeout: False)
        try:
            services_mod.exit_if_opens_remain(143)
        finally:
            named.removeHandler(handler)

        assert order == ["flush", "exit 143"]


def _run_pending_handlers() -> None:
    """Give the main thread bytecode to run, so a pending signal handler runs."""
    for _ in range(20):
        time.sleep(0.01)


def test_a_parent_death_stops_the_process_as_one_terminate_does():
    """The handler runs on the main thread and raises the terminate status."""
    provider = _install_stub_services()
    install_engine_lifecycle_hooks()

    with pytest.raises(SystemExit) as raised:
        services_mod.stop_for_parent_death()
        _run_pending_handlers()
    _join_teardown()

    assert raised.value.code == 128 + signal.SIGTERM
    provider.shutdown.assert_called_once()


def test_a_parent_death_after_a_terminate_sends_no_second_stop(monkeypatch):
    """A second run of the handler is the force quit, so the stop under way gets no other."""
    forced: list[int] = []
    monkeypatch.setattr(services_mod, "_force_quit_from_handler", forced.append)
    _install_stub_services()
    install_engine_lifecycle_hooks()
    with pytest.raises(SystemExit):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    _join_teardown()

    services_mod.stop_for_parent_death()
    _run_pending_handlers()

    assert forced == []
