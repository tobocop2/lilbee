"""Hard-exit signals must tear the engine fleet down, not orphan it."""

from __future__ import annotations

import signal
from unittest.mock import MagicMock

import pytest

from lilbee.app import services as services_mod
from lilbee.app.services import (
    install_engine_lifecycle_hooks,
    peek_services,
    reset_services,
    set_services,
)

_HARD_EXIT_SIGNALS = [
    pytest.param(signal.SIGTERM, id="sigterm"),
    pytest.param(signal.SIGHUP, id="sighup-terminal-close"),
]


@pytest.fixture(autouse=True)
def _restore_handlers():
    """Snapshot and restore the real signal table around every test."""
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
    services_mod._lifecycle.reset()
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)
    services_mod._lifecycle.reset()
    reset_services()


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
    provider.shutdown.assert_called_once()


def test_teardown_runs_once_when_signal_and_atexit_both_fire():
    """The atexit hook after a signal handler must not shut down twice."""
    provider = _install_stub_services()
    install_engine_lifecycle_hooks()

    with pytest.raises(SystemExit):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    reset_services()

    provider.shutdown.assert_called_once()


def test_signal_without_services_still_exits():
    """A hard exit before any container was built must not raise."""
    install_engine_lifecycle_hooks()
    with pytest.raises(SystemExit):
        signal.getsignal(signal.SIGHUP)(signal.SIGHUP, None)


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
