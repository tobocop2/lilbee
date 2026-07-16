"""TUI session surfaces: the left drawer and the full-screen Sessions tab."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lilbee.app.services import get_services, set_services
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.screens.sessions import SessionsScreen
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog
from lilbee.cli.tui.widgets.session_list import SessionListPanel, SessionRow
from lilbee.cli.tui.widgets.sessions_drawer import SessionsDrawer
from lilbee.sessions import MessageRole, SessionMessage, SessionNotFoundError, TitleSource
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


def _seed(store, title: str) -> str:
    session_id = store.create(model_ref="gpt-oss-20b", scope="both")
    store.set_title(session_id, title, TitleSource.AUTO)
    store.add_message(session_id, SessionMessage(role=MessageRole.USER, content="q"))
    return session_id


async def _open_drawer(app, pilot) -> SessionsDrawer:
    screen = await await_chat(app, pilot)
    app.action_toggle_sessions()
    await pilot.pause()
    return screen.query_one(SessionsDrawer)


# --- drawer ---------------------------------------------------------------
async def test_drawer_opens_lists_and_insets_bars(sessions):
    _seed(sessions, "Torque specs")
    _seed(sessions, "Board email")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        app.action_toggle_sessions()
        await pilot.pause()
        drawer = screen.query_one(SessionsDrawer)
        assert len(drawer.query(SessionRow)) == 2
        assert screen.has_class("sessions-open")


async def test_drawer_toggle_closes_and_restores_bars(sessions):
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _open_drawer(app, pilot)
        app.action_toggle_sessions()
        await pilot.pause()
        assert not screen.query(SessionsDrawer)
        assert not screen.has_class("sessions-open")


async def test_drawer_filters_by_title(sessions):
    _seed(sessions, "Torque specs")
    _seed(sessions, "Board email")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        drawer = await _open_drawer(app, pilot)
        for ch in "board":
            await pilot.press(ch)
        await pilot.pause()
        rows = drawer.query(SessionRow)
        assert len(rows) == 1


# --- the sessions_enabled toggle -----------------------------------------
async def test_sessions_on_by_default() -> None:
    from lilbee.core.config import cfg

    assert cfg.sessions_enabled is True


async def test_disabled_hides_the_footer_binding(sessions, monkeypatch) -> None:
    """check_action returns None so ctrl+o leaves the footer entirely."""
    from lilbee.core.config import cfg

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        monkeypatch.setattr(cfg, "sessions_enabled", True)
        assert app.check_action("toggle_sessions", ()) is not None
        monkeypatch.setattr(cfg, "sessions_enabled", False)
        assert app.check_action("toggle_sessions", ()) is None


async def test_disabled_shows_notice_on_toggle(sessions, monkeypatch) -> None:
    from lilbee.cli.tui.widgets.notice_dialog import NoticeDialog
    from lilbee.core.config import cfg

    monkeypatch.setattr(cfg, "sessions_enabled", False)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        app.action_toggle_sessions()
        await pilot.pause()
        assert isinstance(app.screen, NoticeDialog)
        assert not screen.query(SessionsDrawer)


async def test_disabled_shows_notice_on_the_sessions_tab(sessions, monkeypatch) -> None:
    from lilbee.cli.tui.widgets.notice_dialog import NoticeDialog
    from lilbee.core.config import cfg

    monkeypatch.setattr(cfg, "sessions_enabled", False)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        app.switch_view("Sessions")
        await pilot.pause()
        assert isinstance(app.screen, NoticeDialog)
        assert not app.query(SessionsScreen)


async def test_disabled_does_not_persist(sessions, monkeypatch) -> None:
    """A turn while off is never written to disk; _session_id stays None."""
    from lilbee.core.config import cfg

    monkeypatch.setattr(cfg, "sessions_enabled", False)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen._persist_user_turn("this must not be saved")
        assert screen._session_id is None
        assert sessions.list() == []
        assert rows.first().meta.title == "Board email"


async def test_filtering_does_not_re_read_the_store(sessions):
    """Each keystroke must filter the loaded list, not re-fold every session file.

    list() replays every event of every session, so re-listing per keystroke made
    typing cost O(vault bytes) on the UI thread: measured at 190ms per keystroke
    for 300 sessions x 200 messages, i.e. 1.3s to type a 7-character filter.
    """
    for i in range(3):
        _seed(sessions, f"session {i}")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await _open_drawer(app, pilot)
        calls = 0
        real_list = sessions.list

        def counting_list():
            nonlocal calls
            calls += 1
            return real_list()

        with patch.object(sessions, "list", counting_list):
            for ch in "sess":
                await pilot.press(ch)
            await pilot.pause()
        assert calls == 0, f"filter keystrokes re-read the store {calls} times"


async def test_resume_from_drawer_loads_and_closes(sessions):
    session_id = _seed(sessions, "Torque specs")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _open_drawer(app, pilot)
        await pilot.press("enter")
        await pilot.pause()
        assert app.chat_screen().session_id == session_id
        assert not screen.query(SessionsDrawer)


async def test_new_chat_from_drawer(sessions):
    session_id = _seed(sessions, "Torque specs")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(app, pilot)
        chat.resume_session(session_id)
        await pilot.pause()
        screen = await _open_drawer(app, pilot)
        await pilot.press("ctrl+n")
        await pilot.pause()
        assert chat.session_id is None
        assert not screen.query(SessionsDrawer)


async def test_rename_in_drawer(sessions):
    session_id = _seed(sessions, "Old name")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await _open_drawer(app, pilot)
        await pilot.press("ctrl+r")
        for ch in " new":
            await pilot.press(ch)
        await pilot.press("enter")
        await pilot.pause()
        assert sessions.get(session_id).meta.title == "Old name new"


async def test_rename_cancel_leaves_title(sessions):
    session_id = _seed(sessions, "Keep me")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        drawer = await _open_drawer(app, pilot)
        panel = drawer.query_one(SessionListPanel)
        await pilot.press("ctrl+r")
        await pilot.pause()
        panel.action_close()  # escape cancels the rename, does not close
        await pilot.pause()
        assert sessions.get(session_id).meta.title == "Keep me"
        assert drawer.is_mounted


async def test_delete_confirmed_removes_session(sessions):
    session_id = _seed(sessions, "Delete me")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        drawer = await _open_drawer(app, pilot)
        panel = drawer.query_one(SessionListPanel)
        meta = sessions.list()[0]
        await pilot.press("ctrl+d")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmDialog)
        panel._on_delete_confirmed(meta, confirmed=True)
        await pilot.pause()
        assert sessions.list() == []
        with pytest.raises(SessionNotFoundError):
            sessions.get(session_id)


async def test_delete_declined_keeps_session(sessions):
    _seed(sessions, "Keep me")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        drawer = await _open_drawer(app, pilot)
        panel = drawer.query_one(SessionListPanel)
        meta = sessions.list()[0]
        panel._on_delete_confirmed(meta, confirmed=False)
        await pilot.pause()
        assert len(sessions.list()) == 1


async def test_close_drawer_with_escape(sessions):
    _seed(sessions, "Torque specs")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _open_drawer(app, pilot)
        await pilot.press("escape")
        await pilot.pause()
        assert not screen.query(SessionsDrawer)


async def test_panel_close_request_removes_drawer(sessions):
    _seed(sessions, "Torque specs")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _open_drawer(app, pilot)
        screen.query_one(SessionListPanel).action_close()
        await pilot.pause()
        assert not screen.query(SessionsDrawer)


async def test_active_session_gets_the_filled_dot(sessions):
    session_id = _seed(sessions, "Active one")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(app, pilot)
        chat.resume_session(session_id)
        await pilot.pause()
        drawer = await _open_drawer(app, pilot)
        row = drawer.query_one(SessionRow)
        assert row._active is True


# --- cursor + empty -------------------------------------------------------
async def test_cursor_moves_and_empty_state(sessions):
    _seed(sessions, "one")
    _seed(sessions, "two")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        drawer = await _open_drawer(app, pilot)
        await pilot.press("down")
        await pilot.press("up")
        await pilot.pause()
        panel = drawer.query_one(SessionListPanel)
        assert panel._selected() is not None
        for ch in "zzz-nomatch":
            await pilot.press(ch)
        await pilot.pause()
        assert not drawer.query(SessionRow)


# --- full-screen tab ------------------------------------------------------
async def test_sessions_tab_shows_list(sessions):
    _seed(sessions, "Torque specs")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        app.switch_view("Sessions")
        await pilot.pause()
        assert isinstance(app.screen, SessionsScreen)
        assert len(app.screen.query(SessionRow)) == 1


async def test_sessions_tab_resume_switches_to_chat(sessions):
    session_id = _seed(sessions, "Torque specs")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        app.switch_view("Sessions")
        await pilot.pause()
        app.screen.query_one(SessionListPanel).post_message(SessionListPanel.Resumed(session_id))
        await pilot.pause()
        assert app.active_view == "Chat"
        assert app.chat_screen().session_id == session_id


async def test_sessions_tab_new_and_close_go_to_chat(sessions):
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        app.switch_view("Sessions")
        await pilot.pause()
        panel = app.screen.query_one(SessionListPanel)
        panel.post_message(SessionListPanel.NewChat())
        await pilot.pause()
        assert app.active_view == "Chat"
        app.switch_view("Sessions")
        await pilot.pause()
        app.screen.query_one(SessionListPanel).post_message(SessionListPanel.CloseRequested())
        await pilot.pause()
        assert app.active_view == "Chat"


async def test_sessions_tab_q_goes_back(sessions):
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        app.switch_view("Sessions")
        await pilot.pause()
        app.screen.action_go_back()
        await pilot.pause()
        assert app.active_view == "Chat"


async def test_toggle_sessions_is_noop_on_the_tab(sessions):
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        app.switch_view("Sessions")
        await pilot.pause()
        app.action_toggle_sessions()
        await pilot.pause()
        assert not app.screen.query(SessionsDrawer)


async def test_actions_noop_on_empty_list(sessions):
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        drawer = await _open_drawer(app, pilot)
        panel = drawer.query_one(SessionListPanel)
        await pilot.press("ctrl+r")  # rename with nothing selected
        await pilot.press("ctrl+d")  # delete with nothing selected
        await pilot.press("enter")  # resume with nothing selected
        await pilot.pause()
        assert panel._renaming_id is None
        assert drawer.is_mounted


async def test_slash_sessions_command_opens_drawer(sessions):
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(app, pilot)
        chat._cmd_sessions("")
        await pilot.pause()
        assert chat.query(SessionsDrawer)


# --- app helpers with no chat screen -------------------------------------
async def test_app_session_helpers_noop_without_chat(sessions):
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        with patch.object(app, "chat_screen", return_value=None):
            app.resume_session("x")
            app.new_chat()
            assert app.current_session_id() is None
        await pilot.pause()
