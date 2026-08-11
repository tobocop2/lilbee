"""TUI session surfaces: the left drawer and the full-screen Sessions tab."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from textual.widgets import ListView

from lilbee.app.services import get_services, set_services
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.screens.sessions import SessionsScreen
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog
from lilbee.cli.tui.widgets.session_list import SessionListPanel, SessionRow
from lilbee.cli.tui.widgets.sessions_drawer import SessionsDrawer
from lilbee.sessions import MessageRole, SessionMessage, SessionNotFoundError, TitleSource
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import await_chat, shown_footer_keys
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
        patch("lilbee.cli.tui.app.chat_ready", return_value=True),
        patch("lilbee.cli.tui.app.embedding_ready", return_value=True),
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


async def _laid_out(pilot, widget):
    """Return *widget* once it owns a hit area.

    The drawer mounts before it lays out, and one pause covers the mount but
    not the layout. Until a widget has a region, every point derived from it is
    the origin, so the event lands on whatever occupies the top-left corner --
    the drawer itself -- and the test asserts against the wrong widget. That is
    the whole of this file's windows-runner failure history, so the wait lives
    here once rather than inline at each hit-test.
    """
    await wait_until(pilot, lambda: bool(widget.region.size))
    assert widget.region.size, "the widget was never laid out, so it has no hit area"
    return widget


async def _hittable_row(pilot, drawer):
    """Return the drawer's first session row once it owns a hit area."""
    rows = list(drawer.query(".session-row-meta").results())
    assert rows, "the drawer rendered no session rows"
    return await _laid_out(pilot, rows[0])


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
        assert rows.first().meta.title == "Board email"


# --- the sessions_enabled toggle -----------------------------------------
async def test_sessions_on_by_default() -> None:
    from lilbee.core.config import cfg

    assert cfg.sessions_enabled is True


async def test_disabled_hides_the_footer_binding(sessions, monkeypatch) -> None:
    """With sessions off, ctrl+o leaves the footer row entirely.

    Asserted on the row rather than on check_action's return value: the guard
    used to return None, which reads as "hidden" but which Textual renders
    greyed-and-present, so a return-value check passed while the footer still
    advertised a toggle with nothing to toggle. Only False drops the cell.
    """
    from lilbee.core.config import cfg

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)

        monkeypatch.setattr(cfg, "sessions_enabled", True)
        app.screen.refresh_bindings()
        await pilot.pause()
        assert "ctrl+o" in shown_footer_keys(app)

        monkeypatch.setattr(cfg, "sessions_enabled", False)
        app.screen.refresh_bindings()
        await pilot.pause()
        assert "ctrl+o" not in shown_footer_keys(app)


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


async def test_the_notice_dismisses_and_never_stacks(sessions, monkeypatch) -> None:
    """The modal closes by key and by click, and a second ctrl+o never stacks a copy."""
    from lilbee.cli.tui.widgets.notice_dialog import NoticeDialog
    from lilbee.core.config import cfg

    monkeypatch.setattr(cfg, "sessions_enabled", False)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        app.action_toggle_sessions()
        await pilot.pause()
        notice = app.screen
        assert isinstance(notice, NoticeDialog)
        # A second press while the notice is up must not stack another copy.
        app.action_toggle_sessions()
        await pilot.pause()
        assert app.screen is notice
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, NoticeDialog)
        # Reopen and close by clicking the pill instead.
        app.action_toggle_sessions()
        await pilot.pause()
        await pilot.click("#notice-dismiss")
        await pilot.pause()
        assert not isinstance(app.screen, NoticeDialog)


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


@pytest.mark.parametrize("count", [1, 2])
async def test_clicking_a_row_resumes_it(sessions, count):
    """A mouse click on a row resumes it, whether or not the list is the only one."""
    for i in range(count):
        _seed(sessions, f"Session {i}")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _open_drawer(app, pilot)
        row = await _laid_out(pilot, screen.query(SessionRow)[0])
        clicked_id = row.meta.id
        landed = await pilot.click(row)
        await pilot.pause()
        # Without this the same miss reads as "resuming a session is broken",
        # which is what made this class take eight rounds to name.
        assert landed, "the click never reached the session row"
        assert app.chat_screen().session_id == clicked_id
        assert not screen.query(SessionsDrawer)


async def test_enter_resumes_while_chat_is_in_normal_mode(sessions):
    """The chat screen must not eat the drawer's enter while it sits in NORMAL mode.

    Clicking a row moves focus off the chat input, which drops chat out of INSERT;
    the screen-level vim handler then swallowed every later enter in the drawer.
    """
    session_id = _seed(sessions, "Torque specs")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(app, pilot)
        chat._insert_mode = False
        screen = await _open_drawer(app, pilot)
        await pilot.press("enter")
        await pilot.pause()
        assert app.chat_screen().session_id == session_id
        assert not screen.query(SessionsDrawer)


async def test_enter_resumes_after_filtering(sessions):
    """Enter must resume the row a filter narrowed to, not just an unfiltered list."""
    _seed(sessions, "Alpha")
    _seed(sessions, "Beta")
    wanted = _seed(sessions, "Gamma")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _open_drawer(app, pilot)
        for ch in "Gamma":
            await pilot.press(ch)
        await pilot.pause()
        assert len(screen.query(SessionRow)) == 1
        await pilot.press("enter")
        await pilot.pause()
        assert app.chat_screen().session_id == wanted


async def test_enter_on_the_focused_list_resumes(sessions):
    """Enter resumes when the list holds focus, not just when the filter does.

    Clicking a row moves focus off the filter box, so a resume path that only
    listens on the filter's Submitted leaves Enter dead for the rest of the visit.
    """
    session_id = _seed(sessions, "Torque specs")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _open_drawer(app, pilot)
        screen.query_one("#sessions-list", ListView).focus()
        await pilot.pause()
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
        # The filter and chat inputs both eat ctrl+d as delete-right; the list
        # leaves it to bubble to the panel binding, so press from there.
        drawer.query_one("#sessions-list", ListView).focus()
        await pilot.pause()
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


async def test_sessions_tab_enter_resumes(sessions):
    """The tab focuses the list rather than the filter, so Enter must work there."""
    session_id = _seed(sessions, "Torque specs")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        app.switch_view("Sessions")
        await pilot.pause()
        await pilot.press("enter")
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


async def test_agent_sessions_never_appear_in_the_drawer(sessions) -> None:
    """Agent (MCP) sessions are working state, not conversations: the TUI
    session list must not show them at all."""
    from lilbee.sessions import SessionOrigin

    mine = sessions.create(model_ref="gpt-oss-20b", scope="both")
    sessions.create(model_ref="gpt-oss-20b", scope="both", origin=SessionOrigin.MCP)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        drawer = await _open_drawer(app, pilot)
        panel = drawer.query_one(SessionListPanel)
        assert [meta.id for meta in panel._metas] == [mine]


async def test_resuming_an_obsidian_session_keeps_its_origin(sessions) -> None:
    """TUI, HTTP, and CLI are one conversation space: resuming a session the
    plugin started needs no ownership transfer, and turns still persist."""
    from lilbee.sessions import MessageRole, SessionMessage, SessionOrigin

    sid = sessions.create(model_ref="gpt-oss-20b", scope="both", origin=SessionOrigin.HTTP)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(sid)
        await pilot.pause()
        assert sessions.get(sid).meta.origin is SessionOrigin.HTTP
        sessions.add_message(
            sid,
            SessionMessage(role=MessageRole.USER, content="from the tui"),
            surface=SessionOrigin.TUI,
        )
        assert sessions.get(sid).meta.message_count == 1


async def test_sessions_tab_vocabulary_walks_the_list(sessions):
    """j/k/g/G drive the sessions list from the full-screen tab."""
    from textual.widgets import ListView

    for n in range(3):
        _seed(sessions, f"s{n}")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        app.switch_view("Sessions")
        await pilot.pause()
        assert isinstance(app.screen, SessionsScreen)
        lv = app.screen.query_one("#sessions-list", ListView)
        lv.focus()
        await pilot.pause()
        await pilot.press("G")
        await pilot.pause()
        assert lv.index == len(lv) - 1
        await pilot.press("g")
        await pilot.pause()
        assert lv.index == 0
        await pilot.press("j")
        await pilot.pause()
        assert lv.index == 1
        await pilot.press("k")
        await pilot.pause()
        assert lv.index == 0


async def test_sessions_escape_returns_to_previous_view(sessions):
    """Escape leaves Sessions with the same semantics as q: back, not Chat."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    _seed(sessions, "one")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        app.switch_view("Settings")
        await pilot.pause()
        app.switch_view("Sessions")
        await pilot.pause()
        assert isinstance(app.screen, SessionsScreen)
        await pilot.press("escape")
        for _ in range(10):
            await pilot.pause()
            if isinstance(app.screen, SettingsScreen):
                break
        assert isinstance(app.screen, SettingsScreen)


async def test_sessions_jump_on_empty_list_is_safe(sessions):
    """g/G on an empty sessions list must not raise."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        app.switch_view("Sessions")
        await pilot.pause()
        assert isinstance(app.screen, SessionsScreen)
        panel = app.screen.query_one(SessionListPanel)
        panel.jump_to(0)
        panel.jump_to(-1)


async def test_clicking_a_session_row_never_starts_a_text_selection(sessions, monkeypatch) -> None:
    """A click on a row must not enter Textual's selection path.

    That path takes ``content_widget.parent`` and dereferences
    ``container.region`` without a None check, so a click landing on a row that
    has just been unparented crashed the app: AttributeError on
    `_MessagePump__parent`, reported from a live session. Rows are unparented on
    every store mutation and every filter keystroke, because `_render_rows`
    clears the ListView without awaiting the removal, so the window is open
    often and cannot be closed by ordering alone.

    Asserted on the entry condition rather than by racing a detach against a
    click, which pilot cannot time: with selection off for row text the path is
    unreachable regardless of when the click lands. Flip ALLOW_SELECT back to
    True on _RowText and this fails.
    """
    from lilbee.core.config import cfg

    monkeypatch.setattr(cfg, "sessions_enabled", True)
    _seed(sessions, "first session")
    _seed(sessions, "second session")

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        drawer = await _open_drawer(app, pilot)
        row = await _hittable_row(pilot, drawer)
        assert app.screen._select_state is None

        # mouse_down, not click: a full click is down-then-up, and MouseUp calls
        # clear_selection(), so _select_state reads None afterwards either way.
        # MouseDown is also the event in the reported traceback.
        landed = await pilot.mouse_down(row)
        await pilot.pause()

        # Pilot reports whether the final event reached the widget it was
        # given. Asserting it keeps a miss reading as a miss instead of as a
        # clean row: an event that lands elsewhere leaves the row's selection
        # state untouched, which is indistinguishable from the fix working.
        assert landed, "the MouseDown never reached the session row"
        assert app.screen._select_state is None, (
            "MouseDown on a session row started a text selection, which is the "
            "path that crashes on an unparented row"
        )


async def test_mouse_down_on_a_detached_session_row_does_not_crash(sessions, monkeypatch) -> None:
    """End-to-end: the reported crash, driven through Textual's real event path.

    The report was a MouseDown on a row's meta line raising AttributeError:
    'NoneType' object has no attribute 'region', because Textual's selection
    path takes content_widget.parent and dereferences container.region without a
    None check. This builds that exact state: paint the rows so the compositor
    maps clicks to them, unparent one, confirm it is still hit-testable, then
    forward a real MouseDown at its screen coordinates.

    The detachment is injected rather than raced. `_render_rows` clears the
    ListView without awaiting the removal, so the real trigger is a click
    arriving between the detach and the next repaint, which the pilot cannot
    time; injecting it reproduces the same state the race produces. With
    ALLOW_SELECT left on for row text this raises.
    """
    from textual import events

    from lilbee.core.config import cfg

    monkeypatch.setattr(cfg, "sessions_enabled", True)
    _seed(sessions, "first session")
    _seed(sessions, "second session")

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        drawer = await _open_drawer(app, pilot)
        target = await _hittable_row(pilot, drawer)
        x = target.region.offset.x + 2
        y = target.region.offset.y

        # Unparent it while the painted frame still maps clicks to it, the way
        # Textual's own _detach does. Restored before teardown: the DOM prune
        # asserts on a live parent, so leaving it nulled fails the test for an
        # unrelated reason and hides the result of the click.
        original_parent = target._parent
        target._parent = None
        try:
            assert target.parent is None, "the row is still attached; nothing to test"
            hit, _offset = app.screen.get_widget_and_offset_at(x, y)
            assert hit is target, "the compositor no longer maps that point to the row"

            crash: Exception | None = None
            try:
                app.screen._forward_event(
                    events.MouseDown(
                        None,
                        x=x,
                        y=y,
                        delta_x=0,
                        delta_y=0,
                        button=1,
                        shift=False,
                        meta=False,
                        ctrl=False,
                    )
                )
            except Exception as exc:
                crash = exc
        finally:
            target._parent = original_parent

        assert crash is None, f"MouseDown on a detached row still crashes: {crash!r}"
        await pilot.pause()
