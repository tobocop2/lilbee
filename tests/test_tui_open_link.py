"""Answer citation links: the markdown parser admits file: URLs and clicks open
the source file with the OS opener instead of a browser."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from unittest import mock
from urllib.parse import urlparse
from urllib.request import url2pathname

import pytest
from textual.widgets import Markdown

from lilbee.app.services import set_services
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.screens import chat as chat_screen
from lilbee.cli.tui.screens.chat_helpers import (
    LinkOpenResult,
    _opener_argv,
    open_local_file,
)
from lilbee.cli.tui.widgets.message import _answer_markdown_parser
from lilbee.core.config import cfg
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import await_chat
from tests.conftest import make_mock_services


class TestAnswerMarkdownParser:
    def test_admits_file_links(self):
        assert _answer_markdown_parser().validateLink("file:///Users/t/doc.md")

    def test_keeps_default_scheme_rejections(self):
        # Only file: is re-admitted; the XSS-guard defaults stay in force.
        assert not _answer_markdown_parser().validateLink("javascript:alert(1)")

    def test_parses_a_sources_block_file_link(self):
        tokens = _answer_markdown_parser().parse("1. [label](file:///Users/t/My%20Docs/doc.md)")
        links = [t for tok in tokens if tok.children for t in tok.children if t.type == "link_open"]
        assert [link.attrs["href"] for link in links] == ["file:///Users/t/My%20Docs/doc.md"]


class TestOpenerArgv:
    @pytest.mark.parametrize(
        ("platform", "expected"),
        [
            ("darwin", ["open"]),
            ("linux", ["xdg-open"]),
            ("linux2", ["xdg-open"]),
            ("win32", None),
        ],
    )
    def test_platform_opener(self, platform, expected):
        assert _opener_argv(platform) == expected


class TestOpenLocalFile:
    def test_opens_decoded_path_with_platform_opener(self, monkeypatch):
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers.sys.platform", "darwin")
        run = mock.Mock()
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers.subprocess.run", run)
        doc = cfg.documents_dir / "My Docs" / "doc.md"
        open_local_file(doc.as_uri())
        opener, path = run.call_args.args[0]
        assert opener == "open"
        assert Path(path) == doc.resolve()
        assert "%20" not in path
        assert run.call_args.kwargs == {"check": False, "timeout": 10}

    def test_unknown_platform_falls_back_to_webbrowser(self, monkeypatch):
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers.sys.platform", "win32")
        opened = mock.Mock()
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers.webbrowser.open", opened)
        href = (cfg.documents_dir / "doc.md").as_uri()
        open_local_file(href)
        opened.assert_called_once_with((cfg.documents_dir / "doc.md").resolve().as_uri())

    def test_opener_failure_is_logged_not_raised(self, monkeypatch, caplog):
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers.sys.platform", "linux")
        monkeypatch.setattr(
            "lilbee.cli.tui.screens.chat_helpers.subprocess.run",
            mock.Mock(side_effect=OSError("no xdg-open")),
        )
        href = (cfg.documents_dir / "doc.md").as_uri()
        with caplog.at_level("WARNING"):
            open_local_file(href)  # must not raise
        assert caplog.messages == [
            f"Could not open source file: {url2pathname(urlparse(href).path)}"
        ]

    def test_path_outside_roots_is_refused_and_logged(self, monkeypatch, caplog):
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers.sys.platform", "linux")
        run = mock.Mock()
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers.subprocess.run", run)
        with caplog.at_level("WARNING"):
            open_local_file("file:///etc/passwd")
        run.assert_not_called()
        assert caplog.messages == [
            "Refused to open a file outside the document roots: "
            + url2pathname(urlparse("file:///etc/passwd").path)
        ]

    def test_nul_byte_path_is_refused(self, monkeypatch):
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers.sys.platform", "linux")
        run = mock.Mock()
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers.subprocess.run", run)
        open_local_file((cfg.documents_dir / "doc.md").as_uri() + "%00.md")
        run.assert_not_called()


class TestOpenAnswerLink:
    def _click(self, href: str) -> Markdown.LinkClicked:
        return Markdown.LinkClicked(mock.Mock(spec=Markdown), href)

    def test_file_link_opens_locally(self, monkeypatch):
        opened = mock.Mock()
        monkeypatch.setattr(chat_screen, "open_local_file", opened)
        screen = mock.Mock()
        chat_screen.ChatScreen._open_answer_link(screen, self._click("file:///tmp/doc.md"))
        screen._open_file_link.assert_called_once_with("file:///tmp/doc.md")
        opened.assert_not_called()
        screen.app.open_url.assert_not_called()

    @pytest.mark.parametrize(
        ("result", "notice"),
        [
            (
                LinkOpenResult.OUTSIDE_ROOTS,
                "Link not opened: the file is outside your document folders",
            ),
            (
                LinkOpenResult.NOT_A_DOCUMENT,
                "Link not opened: the target is a program or a folder, not a document",
            ),
        ],
    )
    def test_refused_file_link_tells_the_user(self, monkeypatch, result, notice):
        monkeypatch.setattr(chat_screen, "open_local_file", mock.Mock(return_value=result))
        told = mock.Mock()
        monkeypatch.setattr(chat_screen, "call_from_thread", told)
        screen = mock.Mock()
        chat_screen.ChatScreen._open_file_link.__wrapped__(screen, "file:///etc/passwd")
        told.assert_called_once_with(screen, screen.notify, notice, severity="warning")

    def test_opened_file_link_shows_no_notice(self, monkeypatch):
        monkeypatch.setattr(
            chat_screen, "open_local_file", mock.Mock(return_value=LinkOpenResult.OPENED)
        )
        told = mock.Mock()
        monkeypatch.setattr(chat_screen, "call_from_thread", told)
        chat_screen.ChatScreen._open_file_link.__wrapped__(mock.Mock(), "file:///tmp/doc.md")
        told.assert_not_called()

    def test_web_link_opens_in_browser(self, monkeypatch):
        opened = mock.Mock()
        monkeypatch.setattr(chat_screen, "open_local_file", opened)
        screen = mock.Mock()
        chat_screen.ChatScreen._open_answer_link(screen, self._click("https://example.com/page"))
        screen.app.open_url.assert_called_once_with("https://example.com/page")
        opened.assert_not_called()


class TestProgramsAndFoldersAreNotOpened:
    """The opener shows documents; it never starts a program or enters a folder."""

    @pytest.fixture()
    def run(self, monkeypatch):
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers.sys.platform", "darwin")
        run = mock.Mock()
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers.subprocess.run", run)
        return run

    @pytest.mark.parametrize("name", ["Tool.command", "Shell.terminal", "run.SH", "a.exe"])
    def test_a_program_type_is_refused(self, run, name):
        target = cfg.documents_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("echo hi", encoding="utf-8")
        assert open_local_file(target.as_uri()) is LinkOpenResult.NOT_A_DOCUMENT
        run.assert_not_called()

    def test_an_app_bundle_folder_is_refused(self, run):
        bundle = cfg.documents_dir / "Tool.app"
        bundle.mkdir(parents=True)
        assert open_local_file(bundle.as_uri()) is LinkOpenResult.NOT_A_DOCUMENT
        run.assert_not_called()

    def test_a_plain_folder_is_refused(self, run):
        folder = cfg.documents_dir / "sub"
        folder.mkdir(parents=True)
        assert open_local_file(folder.as_uri()) is LinkOpenResult.NOT_A_DOCUMENT
        run.assert_not_called()

    @pytest.mark.skipif(sys.platform == "win32", reason="Windows has no execute bit")
    @pytest.mark.parametrize("mode", [0o700, 0o610, 0o601])
    def test_a_file_with_any_execute_bit_is_refused(self, run, mode):
        script = cfg.documents_dir / "notes.txt"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("#!/bin/sh\n", encoding="utf-8")
        script.chmod(mode)
        assert open_local_file(script.as_uri()) is LinkOpenResult.NOT_A_DOCUMENT
        run.assert_not_called()

    @pytest.mark.skipif(sys.platform == "win32", reason="Windows has no execute bit")
    def test_a_file_that_is_writable_but_not_executable_opens(self, run):
        doc = cfg.documents_dir / "notes.txt"
        doc.parent.mkdir(parents=True, exist_ok=True)
        doc.write_text("hi", encoding="utf-8")
        doc.chmod(0o662)
        assert open_local_file(doc.as_uri()) is LinkOpenResult.OPENED
        run.assert_called_once()

    def test_a_regular_document_still_opens(self, run):
        doc = cfg.documents_dir / "notes.md"
        doc.parent.mkdir(parents=True, exist_ok=True)
        doc.write_text("hi", encoding="utf-8")
        assert open_local_file(doc.as_uri()) is LinkOpenResult.OPENED
        run.assert_called_once()


class TestOpenerKeepsTheScreenResponsive:
    @pytest.fixture(autouse=True)
    def _chat_setup(self):
        store = mock.MagicMock()
        store.get_sources.return_value = []
        set_services(make_mock_services(store=store))
        with (
            mock.patch("lilbee.cli.tui.app.chat_ready", return_value=True),
            mock.patch("lilbee.cli.tui.app.embedding_ready", return_value=True),
            mock.patch(
                "lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready", return_value=False
            ),
            mock.patch("lilbee.cli.tui.widgets.model_bar.ModelBar.on_mount"),
        ):
            yield
        set_services(None)

    async def test_a_slow_opener_does_not_block_the_event_loop(self, monkeypatch):
        started, release = threading.Event(), threading.Event()

        def slow_opener(href: str) -> LinkOpenResult:
            started.set()
            release.wait(timeout=15)
            return LinkOpenResult.NOT_A_DOCUMENT

        monkeypatch.setattr(chat_screen, "open_local_file", slow_opener)
        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            chat = await await_chat(app, pilot)
            chat.post_message(Markdown.LinkClicked(mock.Mock(spec=Markdown), "file:///tmp/a.md"))
            began = time.monotonic()
            assert await wait_until(pilot, started.is_set, timeout=5)
            await pilot.pause()
            assert time.monotonic() - began < 5, "the opener held the event loop"
            release.set()
            assert await wait_until(
                pilot,
                lambda: any("not a document" in n.message for n in app._notifications),
                timeout=5,
            )
