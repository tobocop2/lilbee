"""Chat is not reachable while its models cannot resolve.

A fresh install has model refs configured but nothing on disk, so the chat
screen has no engine behind it. Every route into chat lands on the catalog
with the setup wizard over it, which makes Escape leave the user somewhere
models can be installed rather than on a prompt that cannot answer.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.screens.catalog import CatalogScreen
from lilbee.cli.tui.screens.chat import ChatScreen
from lilbee.cli.tui.screens.setup import SetupWizard
from tests._lilbee_app_test_host import LilbeeAppHost

_WAIT_TIMEOUT_S = 60.0
_POLL_S = 0.02


def _no_models():
    """Pin the app's readiness probe to "nothing resolves", as on a fresh install."""
    return patch("lilbee.cli.tui.app.models_ready", return_value=False)


def _patch_setup_scan():
    return patch(
        "lilbee.cli.tui.screens.setup._scan_installed_models",
        return_value=([], []),
    )


def _patch_setup_ram(ram_gb: float = 16.0):
    return patch("lilbee.cli.tui.screens.setup.get_system_ram_gb", return_value=ram_gb)


async def _push_a_screen(app) -> None:
    """Stand in for the startup gate: a view switch replaces a *pushed* screen."""
    from textual.screen import Screen

    await app.push_screen(Screen())


async def _wait_for(pilot, predicate, what: str):
    """Pump until *predicate* holds, or fail naming what never happened."""
    deadline = time.monotonic() + _WAIT_TIMEOUT_S
    while time.monotonic() < deadline:
        await pilot.pause()
        if predicate():
            return
        await asyncio.sleep(_POLL_S)
    raise AssertionError(f"{what} never happened")


@pytest.mark.asyncio
async def test_the_handover_waits_for_the_readiness_answer() -> None:
    """The gate must not hand over to chat while the answer is still outstanding.

    The probe reads disk and can call a local model server, so it takes real
    time. A handover that races it reads whatever the readiness flag happened
    to hold and flashes up the very chat screen this gate exists to prevent,
    before the wizard replaces it. Sample the stack the whole way through:
    chat must never appear on it, not merely be gone by the end.
    """
    seen: set[str] = set()

    def _slow_probe() -> bool:
        time.sleep(0.5)
        return False

    app = LilbeeApp()
    with (
        patch("lilbee.cli.tui.app.models_ready", side_effect=_slow_probe),
        _patch_setup_scan(),
        _patch_setup_ram(),
    ):
        async with app.run_test(size=(120, 40)) as pilot:

            def _sample_until_wizard() -> bool:
                seen.update(type(screen).__name__ for screen in app.screen_stack)
                return isinstance(app.screen, SetupWizard)

            await _wait_for(pilot, _sample_until_wizard, "the setup wizard")
    assert ChatScreen.__name__ not in seen, f"chat was on screen during boot: {sorted(seen)}"


@pytest.mark.asyncio
async def test_escaping_the_wizard_lands_on_the_catalog_not_a_dead_chat() -> None:
    """Escape must leave a surface that works, never a chat with no engine."""
    app = LilbeeApp()
    with _no_models(), _patch_setup_scan(), _patch_setup_ram():
        async with app.run_test(size=(120, 40)) as pilot:
            await _wait_for(
                pilot,
                lambda: isinstance(app.screen, SetupWizard),
                "the setup wizard",
            )
            # The wizard is layered over the catalog, so dismissing it reveals
            # the place models are installed.
            assert not any(isinstance(s, ChatScreen) for s in app.screen_stack)

            await pilot.press("escape")
            await _wait_for(
                pilot,
                lambda: not isinstance(app.screen, SetupWizard),
                "the wizard dismissal",
            )
            assert isinstance(app.screen, CatalogScreen)
            assert not any(isinstance(s, ChatScreen) for s in app.screen_stack)


@pytest.mark.asyncio
async def test_navigating_to_chat_without_models_re_presents_setup() -> None:
    """The Chat view is a route into setup while no model resolves."""
    app = LilbeeApp()
    with _no_models(), _patch_setup_scan(), _patch_setup_ram():
        async with app.run_test(size=(120, 40)) as pilot:
            await _wait_for(
                pilot,
                lambda: isinstance(app.screen, SetupWizard),
                "the setup wizard",
            )
            await pilot.press("escape")
            await _wait_for(
                pilot,
                lambda: isinstance(app.screen, CatalogScreen),
                "the catalog",
            )

            app.switch_view("Chat")
            await _wait_for(
                pilot,
                lambda: isinstance(app.screen, SetupWizard),
                "the setup wizard re-presenting",
            )
            assert app.active_view != "Chat"
            assert not any(isinstance(s, ChatScreen) for s in app.screen_stack)


@pytest.mark.asyncio
async def test_settling_runs_setup_before_it_returns_to_the_gate() -> None:
    """The answer is recorded and acted on before the gate is told what to do.

    Driven from a worker thread, as the gate drives it: the marshalled apply
    has to have landed by the time the call returns, or the gate could hand
    over against a flag nobody has written yet.
    """
    app = LilbeeAppHost()
    async with app.run_test(size=(120, 40)):
        with (
            patch("lilbee.cli.tui.app.models_ready", return_value=False),
            patch.object(LilbeeApp, "open_setup") as open_setup,
        ):
            took_the_screen = await asyncio.to_thread(app.settle_setup_state)
        assert took_the_screen is True
        assert app.models_are_ready is False
        open_setup.assert_called_once_with()


@pytest.mark.asyncio
async def test_setup_waits_for_an_in_flight_view_switch() -> None:
    """A wizard pushed mid-switch would sit on the screen the switch is leaving."""
    app = LilbeeAppHost()
    async with app.run_test(size=(120, 40)) as pilot:
        await _push_a_screen(app)
        app._switching = True
        app.open_setup()
        await pilot.pause()
        assert not isinstance(app.screen, SetupWizard)

        app._switching = False
        await _wait_for(
            pilot,
            lambda: isinstance(app.screen, SetupWizard),
            "the deferred wizard",
        )
        assert isinstance(app.screen_stack[-2], CatalogScreen)


@pytest.mark.asyncio
async def test_setup_is_not_re_opened_over_itself() -> None:
    """Two routes into setup landing at once must not stack two wizards."""
    app = LilbeeAppHost()
    async with app.run_test(size=(120, 40)) as pilot:
        await _push_a_screen(app)
        with _patch_setup_scan(), _patch_setup_ram():
            app.open_setup()
            await _wait_for(
                pilot,
                lambda: isinstance(app.screen, SetupWizard),
                "the setup wizard",
            )
            depth = len(app.screen_stack)
            app.open_setup()
            await pilot.pause()
            assert len(app.screen_stack) == depth


@pytest.mark.asyncio
async def test_a_model_reassignment_re_answers_readiness() -> None:
    """A model landing anywhere -- wizard, catalog, /set -- unblocks chat."""
    app = LilbeeAppHost()
    async with app.run_test(size=(120, 40)) as pilot:
        with patch.object(LilbeeApp, "refresh_models_ready") as refresh:
            app.settings_changed_signal.publish(("chat_model", "owner/repo/model.gguf"))
            await pilot.pause()
            refresh.assert_called_once_with()
            refresh.reset_mock()
            app.settings_changed_signal.publish(("theme", "rose-pine"))
            await pilot.pause()
            refresh.assert_not_called()


def test_settling_runs_setup_only_when_there_is_setup_to_do() -> None:
    """A brand new lilbee sees the wizard even when its models already resolve."""
    for ready, fresh, expected in (
        (True, False, (True, False)),  # nothing to do
        (False, False, (False, True)),  # no models: show the wizard
        (True, True, (True, True)),  # models, but this lilbee is brand new
    ):
        app = LilbeeApp()
        with (
            patch("lilbee.cli.tui.app.models_ready", return_value=ready),
            patch("lilbee.cli.tui.app.is_fresh_install", return_value=fresh),
            patch("lilbee.cli.tui.app.call_from_thread") as marshal,
        ):
            assert app.settle_setup_state() is expected[1]
        assert marshal.call_args.args[2:] == expected


def test_a_later_refresh_never_re_opens_the_wizard() -> None:
    """The refresh path runs when a model lands, which is how the wizard closes."""
    body = LilbeeApp.refresh_models_ready.__wrapped__
    app = LilbeeApp()
    with (
        patch("lilbee.cli.tui.app.models_ready", return_value=False),
        patch("lilbee.cli.tui.app.is_fresh_install", return_value=True),
        patch("lilbee.cli.tui.app.call_from_thread") as marshal,
    ):
        body(app)
    assert marshal.call_args.args[2:] == (False, False)
