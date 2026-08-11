"""The startup gate lands on what the machine can serve.

A resolvable chat model lands on Chat. Anything else lands on the Catalog,
where models are installed. Chat itself stays reachable either way; with no
model it shows an empty state instead of a prompt that cannot answer.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.screens.catalog import CatalogScreen
from lilbee.cli.tui.screens.chat import ChatScreen
from tests._lilbee_app_test_host import LilbeeAppHost

_WAIT_TIMEOUT_S = 60.0
_POLL_S = 0.02


def _readiness(chat: bool, embedding: bool):
    """Pin the app's per-role readiness probes."""
    return (
        patch("lilbee.cli.tui.app.chat_ready", return_value=chat),
        patch("lilbee.cli.tui.app.embedding_ready", return_value=embedding),
    )


def _no_container_build():
    """Keep the gate's container build out of a routing test."""
    return patch.object(LilbeeApp, "adopt_services")


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
async def test_no_models_lands_on_the_catalog_and_chat_never_flashes() -> None:
    """A fresh install boots into the catalog, and chat is never on screen first.

    The probe reads disk and can call a local model server, so it takes real
    time. A handover that races it would flash up a chat screen with no engine
    behind it. Sample the stack the whole way through: chat must never appear
    on it, not merely be gone by the end.
    """
    seen: set[str] = set()

    def _slow_probe() -> bool:
        time.sleep(0.5)
        return False

    _, embed_pin = _readiness(False, False)
    app = LilbeeApp()
    with (
        patch("lilbee.cli.tui.app.chat_ready", side_effect=_slow_probe),
        embed_pin,
        _no_container_build(),
    ):
        async with app.run_test(size=(120, 40)) as pilot:

            def _sample_until_catalog() -> bool:
                seen.update(type(screen).__name__ for screen in app.screen_stack)
                return isinstance(app.screen, CatalogScreen)

            await _wait_for(pilot, _sample_until_catalog, "the catalog landing")
            assert app.active_view == msg.CATALOG_VIEW
    assert ChatScreen.__name__ not in seen, f"chat was on screen during boot: {sorted(seen)}"


@pytest.mark.asyncio
async def test_a_resolvable_chat_model_lands_on_chat() -> None:
    """The landing rule's other arm: chat resolves, so the user starts there."""
    chat_pin, embed_pin = _readiness(True, True)
    app = LilbeeApp()
    with chat_pin, embed_pin, _no_container_build():
        async with app.run_test(size=(120, 40)) as pilot:
            await _wait_for(
                pilot,
                lambda: isinstance(app.screen, ChatScreen),
                "the chat landing",
            )
            assert app.active_view == msg.DEFAULT_VIEW


@pytest.mark.asyncio
async def test_chat_stays_reachable_without_models() -> None:
    """Chat is a view, not a gate: no models means an empty state, not a refusal."""
    chat_pin, embed_pin = _readiness(False, False)
    app = LilbeeApp()
    with chat_pin, embed_pin, _no_container_build():
        async with app.run_test(size=(120, 40)) as pilot:
            await _wait_for(
                pilot,
                lambda: isinstance(app.screen, CatalogScreen),
                "the catalog landing",
            )
            # The user's own route into chat: the c key, not the method behind it.
            await pilot.press("c")
            await _wait_for(
                pilot,
                lambda: isinstance(app.screen, ChatScreen),
                "the chat view",
            )
            assert app.active_view == msg.DEFAULT_VIEW


@pytest.mark.asyncio
async def test_settling_records_the_answer_before_it_returns() -> None:
    """The flags are written by the time settle_landing returns.

    Driven from a worker thread, as the gate drives it: the marshalled apply
    has to have landed by the time the call returns, or the gate could hand
    over against a flag nobody has written yet.
    """
    app = LilbeeAppHost()
    async with app.run_test(size=(120, 40)):
        chat_pin, embed_pin = _readiness(False, True)
        with chat_pin, embed_pin:
            await asyncio.to_thread(app.settle_landing)
        assert app.chat_is_ready is False
        assert app.embedding_is_ready is True


@pytest.mark.asyncio
async def test_a_model_reassignment_re_answers_readiness() -> None:
    """A model landing anywhere -- catalog, /set, a download -- updates the flags."""
    app = LilbeeAppHost()
    async with app.run_test(size=(120, 40)) as pilot:
        with patch.object(LilbeeApp, "refresh_readiness") as refresh:
            app.settings_changed_signal.publish(("chat_model", "owner/repo/model.gguf"))
            await pilot.pause()
            refresh.assert_called_once_with()
            refresh.reset_mock()
            app.settings_changed_signal.publish(("theme", "rose-pine"))
            await pilot.pause()
            refresh.assert_not_called()


def test_the_first_ready_role_builds_the_container_off_the_ui_thread() -> None:
    """First run reaches a ready role with no container; building one is not UI work."""
    body = LilbeeApp.refresh_readiness.__wrapped__
    app = LilbeeApp()
    chat_pin, embed_pin = _readiness(True, False)
    with (
        chat_pin,
        embed_pin,
        patch("lilbee.cli.tui.app.peek_services", return_value=None),
        patch.object(LilbeeApp, "adopt_services") as adopt,
        patch("lilbee.cli.tui.app.call_from_thread"),
    ):
        body(app)
    adopt.assert_called_once_with()


def test_a_container_that_already_exists_is_not_adopted_again() -> None:
    """Swapping a model on a running app must not stack another subscription."""
    body = LilbeeApp.refresh_readiness.__wrapped__
    app = LilbeeApp()
    chat_pin, embed_pin = _readiness(True, True)
    with (
        chat_pin,
        embed_pin,
        patch("lilbee.cli.tui.app.peek_services", return_value=object()),
        patch.object(LilbeeApp, "adopt_services") as adopt,
        patch("lilbee.cli.tui.app.call_from_thread"),
    ):
        body(app)
    adopt.assert_not_called()


def test_no_ready_role_builds_nothing() -> None:
    """Nothing resolving is not a reason to spawn role servers."""
    body = LilbeeApp.refresh_readiness.__wrapped__
    app = LilbeeApp()
    chat_pin, embed_pin = _readiness(False, False)
    with (
        chat_pin,
        embed_pin,
        patch("lilbee.cli.tui.app.peek_services", return_value=None),
        patch.object(LilbeeApp, "adopt_services") as adopt,
        patch("lilbee.cli.tui.app.call_from_thread"),
    ):
        body(app)
    adopt.assert_not_called()
