"""Answer citation links: the markdown parser admits file: URLs and clicks open
the source file with the OS opener instead of a browser."""

from __future__ import annotations

from unittest import mock

import pytest
from textual.widgets import Markdown

from lilbee.cli.tui.screens import chat as chat_screen
from lilbee.cli.tui.screens.chat_helpers import _opener_argv, open_local_file
from lilbee.cli.tui.widgets.message import _answer_markdown_parser


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
        # url2pathname's separator style varies by host OS, so assert the two
        # behaviors that matter (opener choice, %20 decoded) not the literal path.
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers.sys.platform", "darwin")
        run = mock.Mock()
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers.subprocess.run", run)
        open_local_file("file:///Users/t/My%20Docs/doc.md")
        opener, path = run.call_args.args[0]
        assert opener == "open"
        assert "My Docs" in path
        assert path.endswith("doc.md")
        assert "%20" not in path

    def test_unknown_platform_falls_back_to_webbrowser(self, monkeypatch):
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers.sys.platform", "win32")
        opened = mock.Mock()
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers.webbrowser.open", opened)
        open_local_file("file:///C:/docs/doc.md")
        opened.assert_called_once_with("file:///C:/docs/doc.md")

    def test_opener_failure_is_logged_not_raised(self, monkeypatch):
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers.sys.platform", "linux")
        monkeypatch.setattr(
            "lilbee.cli.tui.screens.chat_helpers.subprocess.run",
            mock.Mock(side_effect=OSError("no xdg-open")),
        )
        open_local_file("file:///home/t/doc.md")  # must not raise


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
