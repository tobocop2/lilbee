"""TUI session surfaces: the left drawer and the full-screen Sessions tab."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from textual.widgets import Input, ListView

from lilbee.app.services import get_services, set_services
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.screens.sessions import SessionsScreen
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog
from lilbee.cli.tui.widgets.session_list import SessionListPanel, SessionRow
from lilbee.cli.tui.widgets.sessions_drawer import SessionsDrawer
from lilbee.sessions import MessageRole, SessionMessage, SessionNotFoundError, TitleSource
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import await_chat, send_key_burst, shown_footer_keys
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
    await app.action_toggle_sessions()
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


async def _start_rename(pilot, drawer, title: str) -> Input:
    """Press ctrl+r once the drawer highlights a row, and return the filter box it took over.

    The rows mount after the drawer does, so a ctrl+r that lands first finds no
    selection and starts nothing. The rename assertions then pass without a rename.
    """
    rows = drawer.query_one("#sessions-list", ListView)
    await wait_until(pilot, lambda: rows.highlighted_child is not None)
    await pilot.press("ctrl+r")
    field = drawer.query_one("#sessions-filter", Input)
    assert field.placeholder == msg.SESSIONS_RENAME_PLACEHOLDER, "ctrl+r did not start a rename"
    assert field.value == title
    return field


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
        await app.action_toggle_sessions()
        await pilot.pause()
        drawer = screen.query_one(SessionsDrawer)
        assert len(drawer.query(SessionRow)) == 2
        assert screen.has_class("sessions-open")


async def test_drawer_toggle_closes_and_restores_bars(sessions):
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await _open_drawer(app, pilot)
        await app.action_toggle_sessions()
        await pilot.pause()
        assert not app.screen.query(SessionsDrawer)
        assert not app.screen.has_class("sessions-open")


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
        await app.action_toggle_sessions()
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
        await app.action_toggle_sessions()
        await pilot.pause()
        notice = app.screen
        assert isinstance(notice, NoticeDialog)
        # A second press while the notice is up must not stack another copy.
        await app.action_toggle_sessions()
        await pilot.pause()
        assert app.screen is notice
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, NoticeDialog)
        # Reopen and close by clicking the pill instead.
        await app.action_toggle_sessions()
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
        await _open_drawer(app, pilot)
        await pilot.press("enter")
        await pilot.pause()
        assert app.chat_screen().session_id == session_id
        assert not app.screen.query(SessionsDrawer)


@pytest.mark.parametrize("count", [1, 2])
async def test_clicking_a_row_resumes_it(sessions, count):
    """A mouse click on a row resumes it, whether or not the list is the only one."""
    for i in range(count):
        _seed(sessions, f"Session {i}")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        drawer = await _open_drawer(app, pilot)
        row = await _laid_out(pilot, drawer.query(SessionRow)[0])
        clicked_id = row.meta.id
        landed = await pilot.click(row)
        await pilot.pause()
        # Without this the same miss reads as "resuming a session is broken",
        # which is what made this class take eight rounds to name.
        assert landed, "the click never reached the session row"
        assert app.chat_screen().session_id == clicked_id
        assert not app.screen.query(SessionsDrawer)


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
        await _open_drawer(app, pilot)
        await pilot.press("enter")
        await pilot.pause()
        assert app.chat_screen().session_id == session_id
        assert not app.screen.query(SessionsDrawer)


async def _leave_drawer(pilot, chat, key: str) -> None:
    """Open the drawer, press *key* once a row is highlighted, and wait for the drawer to close."""
    await pilot.press("ctrl+o")
    rows = chat.query_one("#sessions-list", ListView)
    assert await wait_until(pilot, lambda: rows.highlighted_child is not None)
    await pilot.press(key)
    assert await wait_until(pilot, lambda: not chat.query(SessionsDrawer))


@pytest.mark.parametrize("key", ["enter", "ctrl+n"], ids=["resume", "new_chat"])
async def test_keys_typed_after_leaving_the_drawer_from_normal_mode_reach_the_prompt(sessions, key):
    """In NORMAL mode the letters before the first i / a / o would run as vim commands."""
    _seed(sessions, "Torque specs")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(app, pilot)
        await pilot.press("escape")
        assert await wait_until(pilot, lambda: not chat._insert_mode)
        await _leave_drawer(pilot, chat, key)
        await pilot.press(*"What is 5 plus 5?")
        await wait_until(pilot, lambda: chat._chat_input.value == "What is 5 plus 5?")
        assert chat._chat_input.value == "What is 5 plus 5?"


@pytest.mark.parametrize("key", ["enter", "ctrl+n"], ids=["resume", "new_chat"])
async def test_leaving_the_drawer_in_insert_mode_focuses_the_prompt(sessions, key):
    _seed(sessions, "Torque specs")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(app, pilot)
        await _leave_drawer(pilot, chat, key)
        assert await wait_until(pilot, lambda: chat._chat_input.has_focus)


@pytest.mark.parametrize("key", ["enter", "ctrl+n"], ids=["resume", "new_chat"])
async def test_keys_typed_in_the_same_burst_as_leaving_the_drawer_reach_the_prompt(sessions, key):
    """Typing ahead of the drawer key must not land in the drawer's filter box."""
    session_id = _seed(sessions, "Torque specs")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(app, pilot)
        await pilot.press("ctrl+o")
        rows = chat.query_one("#sessions-list", ListView)
        assert await wait_until(pilot, lambda: rows.highlighted_child is not None)
        send_key_burst(app, key, *"hello")
        await wait_until(pilot, lambda: chat._chat_input.value == "hello")
        assert chat._chat_input.value == "hello"
        assert chat._chat_input.has_focus
        assert not chat.query(SessionsDrawer)
        assert (chat.session_id == session_id) is (key == "enter")


async def test_enter_in_the_same_burst_as_opening_the_drawer_resumes(sessions):
    """Enter pressed before the drawer's rows have mounted still resumes the top row."""
    session_id = _seed(sessions, "Torque specs")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(app, pilot)
        send_key_burst(app, "ctrl+o", "enter")
        await wait_until(pilot, lambda: chat.session_id == session_id)
        assert chat.session_id == session_id
        assert not chat.query(SessionsDrawer)


@pytest.mark.parametrize("opening", [(), ("ctrl+o",)], ids=["drawer_open", "with_ctrl_o"])
async def test_a_filter_typed_in_the_same_burst_as_enter_picks_the_row(sessions, opening):
    """Enter resumes the row the keys typed before it filtered to, not the top of the full list."""
    wanted = _seed(sessions, "Gamma")
    _seed(sessions, "Alpha")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(app, pilot)
        if not opening:
            await pilot.press("ctrl+o")
            rows = chat.query_one("#sessions-list", ListView)
            assert await wait_until(pilot, lambda: rows.highlighted_child is not None)
            assert rows.highlighted_child.meta.title == "Alpha", "the filter must change the pick"
        send_key_burst(app, *opening, *"Gamma", "enter", *"hello")
        await wait_until(pilot, lambda: chat._chat_input.value == "hello")
        assert chat.session_id == wanted
        assert chat._chat_input.value == "hello"


@pytest.mark.parametrize(
    ("view", "cursor_key"),
    [("Chat", "down"), ("Sessions", "down"), ("Sessions", "j")],
    ids=["drawer_down", "tab_down", "tab_j"],
)
async def test_enter_in_the_same_burst_as_a_cursor_key_resumes_the_moved_to_row(
    sessions, view, cursor_key
):
    """Enter resumes the row the cursor key typed before it moved to, not the top row."""
    wanted = _seed(sessions, "Gamma")
    _seed(sessions, "Alpha")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(app, pilot)
        if view == "Chat":
            await pilot.press("ctrl+o")
        else:
            app.switch_view(view)
            assert await wait_until(pilot, lambda: bool(app.screen.query("#sessions-list")))
        rows = app.screen.query_one("#sessions-list", ListView)
        assert await wait_until(pilot, lambda: rows.highlighted_child is not None)
        assert rows.highlighted_child.meta.title == "Alpha", "the cursor key must change the pick"
        send_key_burst(app, cursor_key, "enter")
        await wait_until(pilot, lambda: chat.session_id == wanted)
        assert chat.session_id == wanted


@pytest.mark.parametrize("typed", ["a", "am"], ids=["one_letter", "two_letters"])
async def test_a_cursor_key_after_filter_text_in_one_burst_moves_within_the_filtered_rows(
    sessions, typed
):
    """Down typed after filter text moves the cursor in the filtered list, and it stays there."""
    wanted = _seed(sessions, "Gamma one")
    _seed(sessions, "Gamma two")
    _seed(sessions, "Alpha")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(app, pilot)
        await pilot.press("ctrl+o")
        rows = chat.query_one("#sessions-list", ListView)
        assert await wait_until(pilot, lambda: rows.highlighted_child is not None)
        assert rows.highlighted_child.meta.title == "Alpha"
        send_key_burst(app, *typed, "down", "down", "enter")
        await wait_until(pilot, lambda: chat.session_id == wanted)
        assert chat.session_id == wanted


@pytest.mark.parametrize("stale", ["a", "am"], ids=["older_text", "same_text"])
async def test_a_filter_change_that_arrives_after_a_cursor_key_keeps_the_cursor(sessions, stale):
    """A Changed event the list receives late must not re-render over the cursor the user moved.

    Built deterministically: the filter changes with its Changed event held back,
    Down renders for the new text and moves, and then the held event arrives.
    """
    wanted = _seed(sessions, "Gamma one")
    _seed(sessions, "Gamma two")
    _seed(sessions, "Alpha")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(app, pilot)
        drawer = await _open_drawer(app, pilot)
        field = drawer.query_one("#sessions-filter", Input)
        with field.prevent(Input.Changed):
            field.value = "am"
        await pilot.press("down")
        rows = drawer.query_one("#sessions-list", ListView)
        assert rows.index == 1
        field.post_message(Input.Changed(field, stale))
        await pilot.pause()
        await pilot.press("enter")
        assert await wait_until(pilot, lambda: chat.session_id == wanted)


def _seed_three(sessions) -> str:
    """Seed rows that list as Alpha, Gamma two, Gamma one; return Gamma one's id."""
    wanted = _seed(sessions, "Gamma one")
    _seed(sessions, "Gamma two")
    _seed(sessions, "Alpha")
    return wanted


@pytest.mark.parametrize("edit", ["backspace", "ctrl+w"], ids=["backspace", "delete_word"])
async def test_an_editing_key_in_the_same_burst_as_a_cursor_key_applies_first(sessions, edit):
    """An edit to the filter typed before Down and Enter changes the rows they act on."""
    _seed_three(sessions)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(app, pilot)
        await pilot.press("ctrl+o")
        rows = chat.query_one("#sessions-list", ListView)
        assert await wait_until(pilot, lambda: rows.highlighted_child is not None)
        await pilot.press(*"am")
        assert await wait_until(pilot, lambda: len(chat.query(SessionRow)) == 2)
        send_key_burst(app, edit, "down", "enter")
        await wait_until(pilot, lambda: chat.session_id is not None)
        assert sessions.get(chat.session_id).meta.title == "Gamma two"


async def test_a_key_the_filter_declines_still_reaches_the_chat(sessions):
    """PageUp in the filter box, which has nothing to scroll, still scrolls the chat behind it."""
    from lilbee.cli.tui.screens.chat import ChatScreen

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        drawer = await _open_drawer(app, pilot)
        assert drawer.query_one("#sessions-filter", Input).has_focus
        with patch.object(ChatScreen, "action_scroll_up") as scroll_up:
            await pilot.press("pageup")
            await pilot.pause()
        scroll_up.assert_called_once_with()


async def test_an_editing_key_in_the_filter_edits_it_once(sessions):
    """Backspace in the filter box deletes one character, not one per place that binds it."""
    _seed(sessions, "Gamma")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        drawer = await _open_drawer(app, pilot)
        field = drawer.query_one("#sessions-filter", Input)
        await pilot.press(*"amm", "backspace")
        assert field.value == "am"


@pytest.mark.parametrize(
    ("key", "title"),
    [("up", "Gamma two"), ("down", "Gamma one"), ("g", "Gamma two"), ("G", "Gamma one")],
)
async def test_a_list_key_applies_filter_text_the_list_has_not_shown_yet(sessions, key, title):
    """A list key first shows the rows for the filter box's text, then moves within them.

    Built deterministically: the filter changes with its Changed event held back.
    """
    _seed_three(sessions)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        drawer = await _open_drawer(app, pilot)
        rows = drawer.query_one("#sessions-list", ListView)
        assert await wait_until(pilot, lambda: rows.highlighted_child is not None)
        rows.index = 2
        field = drawer.query_one("#sessions-filter", Input)
        with field.prevent(Input.Changed):
            field.value = "am"
        rows.focus()
        await pilot.pause()
        assert len(drawer.query(SessionRow)) == 3
        await pilot.press(key)
        await pilot.pause()
        assert len(drawer.query(SessionRow)) == 2
        assert rows.highlighted_child.meta.title == title


async def test_rename_in_the_same_burst_as_filter_text_renames_the_filtered_row(sessions):
    """ctrl+r typed after filter text starts renaming the row the filter narrowed to."""
    _seed(sessions, "Gamma")
    _seed(sessions, "Alpha")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(app, pilot)
        await pilot.press("ctrl+o")
        rows = chat.query_one("#sessions-list", ListView)
        assert await wait_until(pilot, lambda: rows.highlighted_child is not None)
        send_key_burst(app, *"Gam", "ctrl+r")
        field = chat.query_one("#sessions-filter", Input)
        assert await wait_until(pilot, lambda: field.placeholder == msg.SESSIONS_RENAME_PLACEHOLDER)
        assert field.value == "Gamma"


async def test_delete_in_the_same_burst_as_filter_text_asks_about_the_filtered_row(sessions):
    """ctrl+d typed after filter text asks about the row the filter narrowed to, not the top row."""
    _seed(sessions, "Gamma")
    _seed(sessions, "Alpha")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(app, pilot)
        await pilot.press("ctrl+o")
        rows = chat.query_one("#sessions-list", ListView)
        assert await wait_until(pilot, lambda: rows.highlighted_child is not None)
        assert rows.highlighted_child.meta.title == "Alpha"
        with (
            patch.object(ConfirmDialog, "__init__", return_value=None) as dialog,
            patch.object(app, "push_screen"),
        ):
            send_key_burst(app, *"Gam", "ctrl+d")
            assert await wait_until(pilot, lambda: dialog.called)
        dialog.assert_called_once_with(
            msg.SESSIONS_DELETE_CONFIRM_TITLE, msg.SESSIONS_DELETE_CONFIRM.format(title="Gamma")
        )


@pytest.mark.parametrize(
    ("key", "start", "index"),
    [("j", 0, 1), ("G", 0, 2), ("k", 2, 1), ("g", 2, 0)],
    ids=["j_moves_down", "G_jumps_to_the_end", "k_moves_up", "g_jumps_to_the_top"],
)
async def test_sessions_tab_list_keys_work_with_focus_outside_the_list(sessions, key, start, index):
    """j / k / g / G drive the tab's list wherever focus is, like the other browse screens."""
    from lilbee.cli.tui.widgets.status_bar import ViewTab

    for n in range(3):
        _seed(sessions, f"s{n}")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        app.switch_view("Sessions")
        assert await wait_until(pilot, lambda: bool(app.screen.query("#sessions-list")))
        rows = app.screen.query_one("#sessions-list", ListView)
        assert await wait_until(pilot, lambda: rows.highlighted_child is not None)
        rows.index = start
        tab = app.screen.query(ViewTab).first()
        tab.focus()
        await pilot.pause()
        assert tab.has_focus
        assert rows.index == start
        await pilot.press(key)
        await pilot.pause()
        assert rows.index == index


async def test_a_readiness_change_just_before_the_drawer_opens_leaves_focus_in_the_drawer(
    sessions,
):
    """A focus restore that the chat screen asks for must not land after the drawer takes focus.

    The readiness answer arrives from a worker thread while the app is still busy
    opening the drawer, so both happen inside one step of the app's queue.
    """
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(app, pilot)

        async def readiness_lands_while_the_drawer_opens() -> None:
            app.chat_is_ready = False
            app.chat_is_ready = True
            await app.action_toggle_sessions()

        app.call_later(readiness_lands_while_the_drawer_opens)
        assert await wait_until(pilot, lambda: bool(chat.query(SessionsDrawer)))
        await pilot.pause()
        field = chat.query_one("#sessions-filter", Input)
        assert field.has_focus


async def test_enter_after_a_key_that_breaks_the_filter_lets_the_app_exit(sessions, monkeypatch):
    """Enter waits for the filter to catch up; a filter whose queue has stopped must not hang it."""
    _seed(sessions, "Torque specs")

    def broken_insert(self, text: str) -> None:
        raise RuntimeError("the filter broke")

    app = LilbeeApp()
    with pytest.raises(RuntimeError, match="the filter broke"):
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_drawer(app, pilot)
            monkeypatch.setattr(Input, "insert_text_at_cursor", broken_insert)
            send_key_burst(app, "x", "enter")
            await wait_until(pilot, lambda: not app.is_running)


async def test_enter_on_a_session_deleted_elsewhere_says_so_and_refreshes(sessions):
    """Another process deleting the highlighted session must not crash the TUI on Enter."""
    gone = _seed(sessions, "Gone")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(app, pilot)
        drawer = await _open_drawer(app, pilot)
        rows = drawer.query_one("#sessions-list", ListView)
        assert await wait_until(pilot, lambda: rows.highlighted_child is not None)
        sessions.delete(gone)
        with patch.object(app, "notify") as notify:
            await pilot.press("enter")
            await pilot.pause()
        assert app.is_running
        assert chat.session_id != gone
        notify.assert_called_once_with(msg.SESSIONS_GONE, severity="warning")
        assert await wait_until(pilot, lambda: not drawer.query(SessionRow))


async def test_enter_uses_the_filter_text_before_the_list_has_caught_up(sessions):
    """Enter resumes by the text in the filter box even when the list has not re-rendered for it.

    Built deterministically: the filter changes with its Changed message held back,
    which is the state a burst leaves when Enter overtakes the list's re-render.
    """
    wanted = _seed(sessions, "Gamma")
    _seed(sessions, "Alpha")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(app, pilot)
        drawer = await _open_drawer(app, pilot)
        field = drawer.query_one("#sessions-filter", Input)
        with field.prevent(Input.Changed):
            field.value = "Gamma"
        await pilot.press("enter")
        assert await wait_until(pilot, lambda: chat.session_id == wanted)


async def test_enter_resumes_after_filtering(sessions):
    """Enter must resume the row a filter narrowed to, not just an unfiltered list."""
    _seed(sessions, "Alpha")
    _seed(sessions, "Beta")
    wanted = _seed(sessions, "Gamma")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        drawer = await _open_drawer(app, pilot)
        for ch in "Gamma":
            await pilot.press(ch)
        await pilot.pause()
        assert len(drawer.query(SessionRow)) == 1
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
        drawer = await _open_drawer(app, pilot)
        drawer.query_one("#sessions-list", ListView).focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app.chat_screen().session_id == session_id
        assert not app.screen.query(SessionsDrawer)


async def test_new_chat_from_drawer(sessions):
    session_id = _seed(sessions, "Torque specs")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        chat = await await_chat(app, pilot)
        chat.resume_session(session_id)
        await pilot.pause()
        await _open_drawer(app, pilot)
        await pilot.press("ctrl+n")
        await pilot.pause()
        assert chat.session_id is None
        assert not app.screen.query(SessionsDrawer)


async def test_rename_in_drawer(sessions):
    session_id = _seed(sessions, "Old name")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        drawer = await _open_drawer(app, pilot)
        await _start_rename(pilot, drawer, "Old name")
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
        field = await _start_rename(pilot, drawer, "Keep me")
        await pilot.press("escape")  # cancels the rename, does not close
        await pilot.pause()
        assert field.placeholder == msg.SESSIONS_FILTER_PLACEHOLDER
        assert field.value == ""
        assert sessions.get(session_id).meta.title == "Keep me"
        assert app.screen.query(SessionsDrawer)


async def test_ctrl_d_in_a_rename_deletes_a_character(sessions):
    """While renaming, ctrl+d edits the name instead of asking to delete the session."""
    session_id = _seed(sessions, "Keep me")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        drawer = await _open_drawer(app, pilot)
        field = await _start_rename(pilot, drawer, "Keep me")
        await pilot.press("home", "ctrl+d")
        await pilot.pause()
        assert not isinstance(app.screen, ConfirmDialog)
        assert field.value == "eep me"
        await pilot.press("enter")
        await pilot.pause()
        assert sessions.get(session_id).meta.title == "eep me"


async def test_finishing_a_rename_of_a_deleted_session_refreshes_the_list(sessions):
    """A session deleted during its rename leaves the list on Enter; the app keeps running."""
    _seed(sessions, "Keep me")
    doomed = _seed(sessions, "Doomed")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        drawer = await _open_drawer(app, pilot)
        field = await _start_rename(pilot, drawer, "Doomed")
        sessions.delete(doomed)
        with patch.object(app, "notify") as mock_notify:
            await pilot.press("x", "enter")
            await pilot.pause()
        mock_notify.assert_called_once_with(msg.SESSIONS_RENAME_GONE, severity="warning")
        assert app.is_running
        assert field.placeholder == msg.SESSIONS_FILTER_PLACEHOLDER
        assert [row.meta.title for row in drawer.query(SessionRow)] == ["Keep me"]


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


async def test_delete_bracketed_title_notifies_literally(sessions):
    """A bracketed title must render as literal text, not be parsed as markup."""
    _seed(sessions, "[red]Delete")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        drawer = await _open_drawer(app, pilot)
        panel = drawer.query_one(SessionListPanel)
        meta = sessions.list()[0]
        with patch.object(app, "notify") as mock_notify:
            panel._on_delete_confirmed(meta, confirmed=True)
            await pilot.pause()
        mock_notify.assert_called_once()
        assert "[red]Delete" in mock_notify.call_args[0][0]


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
        await _open_drawer(app, pilot)
        await pilot.press("escape")
        await pilot.pause()
        assert not app.screen.query(SessionsDrawer)


async def test_panel_close_request_removes_drawer(sessions):
    _seed(sessions, "Torque specs")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        drawer = await _open_drawer(app, pilot)
        drawer.query_one(SessionListPanel).action_close()
        await pilot.pause()
        assert not app.screen.query(SessionsDrawer)


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


async def test_sessions_tab_click_resume_switches_to_chat(sessions):
    session_id = _seed(sessions, "Torque specs")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        app.switch_view("Sessions")
        await pilot.pause()
        row = await _laid_out(pilot, app.screen.query(SessionRow)[0])
        assert await pilot.click(row), "the click never reached the session row"
        assert await wait_until(pilot, lambda: app.active_view == "Chat")
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
        await pilot.press("ctrl+n")
        assert await wait_until(pilot, lambda: app.active_view == "Chat")
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
        await app.action_toggle_sessions()
        await pilot.pause()
        assert not app.screen.query(SessionsDrawer)


async def test_actions_noop_on_empty_list(sessions):
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        drawer = await _open_drawer(app, pilot)
        panel = drawer.query_one(SessionListPanel)
        # The keys below only reach the panel while focus is inside it.
        assert drawer.query_one("#sessions-filter", Input).has_focus
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
