"""Answer citation links: the markdown parser admits file: URLs and clicks open
the source file with the OS opener instead of a browser."""

from __future__ import annotations

from pathlib import Path
from unittest import mock
from urllib.parse import urlparse
from urllib.request import url2pathname

import pytest
from textual.widgets import Markdown

from lilbee.cli.tui.screens import chat as chat_screen
from lilbee.cli.tui.screens.chat_helpers import _opener_argv, open_local_file
from lilbee.cli.tui.widgets.message import _answer_markdown_parser
from lilbee.core.config import cfg


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
        opened.assert_called_once_with(href)

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
        opened.assert_called_once_with("file:///tmp/doc.md")
        screen.app.open_url.assert_not_called()

    def test_web_link_opens_in_browser(self, monkeypatch):
        opened = mock.Mock()
        monkeypatch.setattr(chat_screen, "open_local_file", opened)
        screen = mock.Mock()
        chat_screen.ChatScreen._open_answer_link(screen, self._click("https://example.com/page"))
        screen.app.open_url.assert_called_once_with("https://example.com/page")
        opened.assert_not_called()
