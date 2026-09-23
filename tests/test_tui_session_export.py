"""The /export-chat slash command: the file it writes and every refusal."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from lilbee.app.services import get_services, set_services
from lilbee.app.session_export import session_markdown
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.widgets.notice_dialog import NoticeDialog
from lilbee.core.config import cfg
from lilbee.sessions import MessageRole, SessionMessage, TitleSource
from tests._lilbee_app_test_host import await_chat, pump_until
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


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """An empty working directory apart from the data dir under tmp_path."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return cwd


def _seed(store) -> str:
    session_id = store.create(model_ref=cfg.chat_model, scope="both")
    store.set_title(session_id, "Torque", TitleSource.AUTO)
    store.add_message(session_id, SessionMessage(role=MessageRole.USER, content="Q1"))
    store.add_message(
        session_id,
        SessionMessage(role=MessageRole.ASSISTANT, content="A1", sources=("manual.pdf",)),
    )
    return session_id


async def _submit(pilot, text: str) -> None:
    keys = ["slash" if ch == "/" else ch for ch in text]
    await pilot.press(*keys, "enter")
    await pilot.pause()


def _notified(mock_notify: MagicMock) -> list[str]:
    return [str(call.args[0]) for call in mock_notify.call_args_list]


async def test_export_writes_the_default_file_in_the_working_directory(sessions, workdir):
    source = _seed(sessions)
    expected = (workdir / f"torque-{source[:8]}.md").resolve()
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        with patch.object(screen, "notify") as notify:
            await _submit(pilot, "/export-chat")
        assert _notified(notify) == [msg.EXPORT_CHAT_DONE.format(path=expected)]
    assert expected.read_text(encoding="utf-8") == session_markdown(sessions.get(source))
    if sys.platform != "win32":
        assert os.stat(expected).st_mode & 0o777 == 0o600


async def test_export_to_a_named_file(sessions, workdir):
    source = _seed(sessions)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        with patch.object(screen, "notify") as notify:
            await _submit(pilot, "/export-chat notes.md")
        target = (workdir / "notes.md").resolve()
        assert _notified(notify) == [msg.EXPORT_CHAT_DONE.format(path=target)]
    assert target.is_file()


@pytest.mark.parametrize("folder", ["[draft]", "[/x]"])
async def test_the_notice_shows_a_bracketed_path_as_written(sessions, workdir, folder):
    source = _seed(sessions)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        with patch.object(screen, "notify", wraps=screen.notify) as notify:
            await _submit(pilot, f"/export-chat {folder}/")
        target = (workdir / folder / f"torque-{source[:8]}.md").resolve()
        assert _notified(notify) == [msg.EXPORT_CHAT_DONE.format(path=target)]
        assert notify.call_args.kwargs["markup"] is False
        assert app.is_running
    assert target.is_file()


async def test_the_failure_notice_is_not_markup(sessions, workdir):
    source = _seed(sessions)
    (workdir / "[draft]").write_text("x", encoding="utf-8")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        with patch.object(screen, "notify", wraps=screen.notify) as notify:
            await _submit(pilot, "/export-chat [draft]/out.md")
        assert notify.call_args.kwargs["markup"] is False
        assert notify.call_args.kwargs["severity"] == "error"


async def test_export_into_a_directory_uses_the_default_name(sessions, workdir):
    source = _seed(sessions)
    (workdir / "notes").mkdir()
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        await _submit(pilot, "/export-chat notes")
    assert (workdir / "notes" / f"torque-{source[:8]}.md").is_file()


async def test_export_that_cannot_write_says_why(sessions, workdir):
    source = _seed(sessions)
    (workdir / "file").write_text("x", encoding="utf-8")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        with patch.object(screen, "notify") as notify:
            await _submit(pilot, "/export-chat file/out.md")
        [notice] = _notified(notify)
        assert notice.startswith(msg.EXPORT_CHAT_FAILED.format(error=""))
        assert notify.call_args.kwargs["severity"] == "error"


async def test_export_before_the_first_message_says_there_is_nothing_to_export(workdir):
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        with patch.object(screen, "notify") as notify:
            await _submit(pilot, "/export-chat")
        assert _notified(notify) == [msg.EXPORT_CHAT_NO_SESSION]
    assert list(workdir.iterdir()) == []


async def test_export_of_a_deleted_session_says_it_is_gone(sessions, workdir):
    source = _seed(sessions)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        sessions.delete(source)
        with patch.object(screen, "notify") as notify:
            await _submit(pilot, "/export-chat")
        assert _notified(notify) == [msg.EXPORT_CHAT_SESSION_GONE]
        assert app.is_running
    assert list(workdir.iterdir()) == []


async def test_export_while_a_reply_is_still_saving_is_refused(sessions, workdir):
    source = _seed(sessions)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        with screen._live_streams, patch.object(screen, "notify") as notify:
            await _submit(pilot, "/export-chat")
        assert _notified(notify) == [msg.EXPORT_CHAT_WHILE_FINISHING]
    assert list(workdir.iterdir()) == []


async def test_export_with_sessions_off_shows_the_sessions_notice(sessions, workdir, monkeypatch):
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await await_chat(app, pilot)
        monkeypatch.setattr(cfg, "sessions_enabled", False)
        await _submit(pilot, "/export-chat")
        assert await pump_until(pilot, lambda: isinstance(app.screen, NoticeDialog))
    assert list(workdir.iterdir()) == []
