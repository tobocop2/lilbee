"""Chat session lifecycle: auto-save, new chat, and resume."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lilbee.app.services import get_services, set_services
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.widgets.message import AssistantMessage, UserMessage
from lilbee.core.config import cfg
from lilbee.sessions import MessageRole, SessionMessage, TitleSource
from tests._lilbee_app_test_host import await_chat
from tests.conftest import make_mock_services


@pytest.fixture(autouse=True)
def _services():
    store = MagicMock()
    store.get_sources.return_value = []
    set_services(make_mock_services(store=store))
    yield
    set_services(None)


@pytest.fixture(autouse=True)
def _patch_chat_setup():
    with (
        patch("lilbee.cli.tui.screens.chat.needs_setup", return_value=False),
        patch("lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready", return_value=False),
        patch("lilbee.cli.tui.widgets.model_bar.ModelBar.on_mount"),
    ):
        yield


@pytest.fixture
def sessions():
    return get_services().session_store


async def test_first_turn_creates_titled_session(sessions):
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen._persist_user_turn("Torque specs in the manual")
        metas = sessions.list()
        assert len(metas) == 1
        assert metas[0].title == "Torque specs in the manual"
        assert metas[0].message_count == 1
        assert screen._session_id == metas[0].id


async def test_turns_append_to_the_same_session(sessions):
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen._persist_user_turn("first question")
        session_id = screen._session_id
        screen._persist_assistant_turn("the answer", ["manual.pdf"])
        screen._persist_user_turn("second question")
        assert screen._session_id == session_id
        assert len(sessions.list()) == 1
        session = sessions.get(session_id)
        assert session.meta.message_count == 3
        assert session.messages[1].role == MessageRole.ASSISTANT
        assert session.messages[1].sources == ("manual.pdf",)


async def test_assistant_persist_is_noop_without_a_session(sessions):
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen._persist_assistant_turn("orphan answer", [])
        assert sessions.list() == []


async def test_reset_conversation_opens_a_fresh_session(sessions):
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen._persist_user_turn("first")
        first_id = screen._session_id
        screen._reset_conversation()
        assert screen._session_id is None
        screen._persist_user_turn("second")
        assert screen._session_id != first_id
        assert len(sessions.list()) == 2


async def test_resume_loads_history_and_renders(sessions):
    session_id = sessions.create(model_ref=cfg.chat_model, scope="both")
    sessions.set_title(session_id, "Seeded", TitleSource.AUTO)
    sessions.add_message(session_id, SessionMessage(role=MessageRole.USER, content="the question"))
    sessions.add_message(
        session_id,
        SessionMessage(role=MessageRole.ASSISTANT, content="the answer", sources=("s.pdf",)),
    )
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(session_id)
        await pilot.pause()
        assert screen._session_id == session_id
        assert screen._history == [
            {"role": "user", "content": "the question"},
            {"role": "assistant", "content": "the answer"},
        ]
        assert len(screen.query(UserMessage)) == 1
        assert len(screen.query(AssistantMessage)) == 1


async def test_resume_restores_a_different_model_when_installed(sessions):
    session_id = sessions.create(model_ref="other/model:latest", scope="both")
    sessions.add_message(session_id, SessionMessage(role=MessageRole.USER, content="q"))
    get_services().registry.is_installed.return_value = True
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        with patch.object(app, "set_active_model") as mock_set:
            screen.resume_session(session_id)
        mock_set.assert_called_once_with("chat_model", "other/model:latest")


async def test_resume_keeps_current_model_when_session_model_is_gone(sessions):
    session_id = sessions.create(model_ref="deleted/model:latest", scope="both")
    sessions.add_message(session_id, SessionMessage(role=MessageRole.USER, content="q"))
    get_services().registry.is_installed.return_value = False  # model was deleted
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        with patch.object(app, "set_active_model") as mock_set:
            screen.resume_session(session_id)
        mock_set.assert_not_called()  # never point chat at a missing model
        assert screen.session_id == session_id  # the conversation still loaded
        assert screen._history[0] == {"role": "user", "content": "q"}
