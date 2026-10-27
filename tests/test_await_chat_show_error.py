"""await_chat fails at once, with the handler's error, when the chat show handler raises."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from lilbee.app.services import set_services
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.screens.chat import ChatScreen
from tests._lilbee_app_test_host import _record_show, await_chat
from tests.conftest import make_mock_services

_FAST_S = 10.0


@pytest.fixture(autouse=True)
def _chat_setup():
    store = MagicMock()
    store.get_sources.return_value = []
    set_services(make_mock_services(store=store))
    with (
        patch("lilbee.cli.tui.app.chat_ready", return_value=True),
        patch("lilbee.cli.tui.app.embedding_ready", return_value=True),
        patch("lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready", return_value=False),
        patch("lilbee.cli.tui.widgets.model_bar.ModelBar.on_mount"),
    ):
        yield
    set_services(None)


async def test_raising_show_handler_fails_the_wait_at_once(monkeypatch):
    def boom(self: ChatScreen) -> None:
        raise RuntimeError("show handler boom")

    monkeypatch.setattr(ChatScreen, "on_show", _record_show(boom))
    started = time.monotonic()
    with pytest.raises(Exception, match="show handler boom"):
        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await await_chat(app, pilot)
            await pilot.pause()
    assert time.monotonic() - started < _FAST_S
