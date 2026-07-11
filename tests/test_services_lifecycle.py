"""Hard-exit signals must tear the engine fleet down, not orphan it."""

from __future__ import annotations

import signal
import threading
from unittest.mock import MagicMock

import pytest

from lilbee.app import services as services_mod
from lilbee.app.services import (
    install_engine_lifecycle_hooks,
    peek_services,
    reset_services,
    set_services,
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
    saved = {
        sig: signal.getsignal(sig) for sig in services_mod._EngineLifecycle._hard_exit_signals()
    }
    services_mod._lifecycle.reset()
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)
    services_mod._lifecycle.reset()
    reset_services()


def _join_teardown() -> None:
    """Wait out the handler's teardown thread so assertions don't race it."""
    for thread in threading.enumerate():
        if thread.name == "hard-exit-teardown":
            thread.join(timeout=5)


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

    assert teardown_threads == ["hard-exit-teardown"]


def test_signal_without_services_still_exits():
    """A hard exit before any container was built must not raise."""
    install_engine_lifecycle_hooks()
    with pytest.raises(SystemExit):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)


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
