"""Hard-exit signals must tear the engine fleet down, not orphan it."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import sys
import threading
import time
from unittest.mock import MagicMock

import pytest

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
    """One press takes the hard-exit path; the next press ends the process with status 130."""
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


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal disposition")
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
    services_mod._lifecycle._interrupts = 1

    services_mod._lifecycle._on_interrupt(signal.SIGINT, None)

    assert writes == [(2, services_mod._FORCE_QUIT_NOTE.encode())]
    assert exits == [services_mod._FORCE_QUIT_EXIT_CODE]


def test_the_handler_force_quit_exits_when_fd_2_is_gone(monkeypatch):
    exits: list[int] = []

    def _closed(_fd: int, _data: bytes) -> int:
        raise OSError("bad file descriptor")

    monkeypatch.setattr(services_mod.os, "write", _closed)
    monkeypatch.setattr(services_mod.os, "_exit", exits.append)

    services_mod._force_quit_from_handler()

    assert exits == [services_mod._FORCE_QUIT_EXIT_CODE]


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


def test_the_first_signal_status_wins_over_a_joined_one():
    """A second signal joins the stop the first one raised for; it owes no status."""
    _install_stub_services()
    install_engine_lifecycle_hooks()
    with pytest.raises(SystemExit) as raised:
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    _join_teardown()

    assert raised.value.code == 128 + signal.SIGTERM
    assert exit_status_after_stop(raised.value.code) == 128 + signal.SIGTERM


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

    def test_a_non_interactive_process_keeps_its_threads(self, monkeypatch) -> None:
        """A test host or library embedder owns its threads; never exit them."""
        monkeypatch.setattr(services_mod._state, "interactive", False)
        exits: list[int] = []
        monkeypatch.setattr(services_mod.os, "_exit", exits.append)
        threads = self._fake_enumerate(monkeypatch, [_Wedged()])

        services_mod.exit_when_stragglers_would_hang()
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

        services_mod.exit_when_stragglers_would_hang()
        del threads[1:]

        assert exits == [0]
        assert "would not stop" in capsys.readouterr().err

    def test_a_clean_shutdown_exits_normally(self, monkeypatch, capsys) -> None:
        exits: list[int] = []
        monkeypatch.setattr(services_mod.os, "_exit", exits.append)
        self._fake_enumerate(monkeypatch, [])

        services_mod.exit_when_stragglers_would_hang()

        assert exits == []
        assert capsys.readouterr().err == ""

    def test_an_interrupt_during_the_budget_force_quits(self, monkeypatch) -> None:
        exits: list[int] = []
        monkeypatch.setattr(services_mod.os, "_exit", exits.append)

        class _Interruptible(_Wedged):
            def join(self, timeout: float | None = None) -> None:
                raise KeyboardInterrupt

        threads = self._fake_enumerate(monkeypatch, [_Interruptible()])

        services_mod.exit_when_stragglers_would_hang()
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


def _signal_once_the_loop_runs(ready: threading.Event) -> None:
    # raise_signal from this thread delivers the signal to this thread, so the main
    # thread, asleep in the loop's select, is not interrupted by it.
    assert ready.wait(10)
    signal.raise_signal(signal.SIGINT)


def test_a_signal_on_another_thread_ends_an_idle_loop_with_its_status():
    """The loop takes the signal as a callback, so it ends the loop wherever the signal lands."""
    provider = _install_stub_services()
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
            asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner,
        ):
            runner.run(idle())
        _join_teardown()
        after = signal.getsignal(signal.SIGINT)
    finally:
        signal.signal(signal.SIGINT, previous)
        sender.join(timeout=5)

    assert raised.value.code == 128 + signal.SIGINT
    assert after == interrupt_handler
    provider.shutdown.assert_called_once()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no loop signal handlers")
def test_a_second_ctrl_c_in_a_loop_owned_loop_force_quits(monkeypatch):
    """The loop callback counts the press: the first joins the stop, the second ends the process."""
    provider = _install_stub_services()
    forced: list[int] = []
    monkeypatch.setattr(services_mod, "_force_quit_from_handler", lambda: forced.append(1))
    # A registered server hook takes the first press, so the callback does not raise.
    set_server_exit_hook(lambda _status: None)
    previous = signal.getsignal(signal.SIGINT)

    async def until(condition) -> None:
        async with asyncio.timeout(10):
            while not condition():
                await asyncio.sleep(0.01)

    async def serve() -> None:
        with loop_takes_signals(asyncio.get_running_loop()):
            os.kill(os.getpid(), signal.SIGINT)
            await until(lambda: services_mod._lifecycle._interrupts == 1)
            first_press_forced = list(forced)
            os.kill(os.getpid(), signal.SIGINT)
            await until(lambda: forced)
            assert first_press_forced == []

    try:
        stop_on_interrupt()
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            runner.run(serve())
        _join_teardown()
    finally:
        signal.signal(signal.SIGINT, previous)
        set_server_exit_hook(lambda _status: None)
        set_server_exit_hook(None)

    assert forced == [1]
    provider.shutdown.assert_called_once()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no loop signal handlers")
def test_an_ordinary_exception_in_the_loop_signals_block_still_restores_the_handlers():
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
        asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner,
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
