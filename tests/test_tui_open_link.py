"""Answer citation links: the markdown parser admits file: URLs and clicks open
the source file with the OS opener instead of a browser."""

from __future__ import annotations

import ctypes
import os
import socket
import subprocess
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
    LINK_REFUSAL_NOTICES,
    LinkOpenResult,
    _has_alias_flag,
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


def _write_doc(name: str, content: bytes = b"hi") -> Path:
    path = cfg.documents_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


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
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers._has_alias_flag", lambda p: False)
        monkeypatch.setattr(
            "lilbee.cli.tui.screens.chat_helpers._has_resource_fork", lambda p: False
        )
        run = mock.Mock()
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers.subprocess.run", run)
        doc = _write_doc("My Docs/doc.md")
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
        doc = _write_doc("doc.md")
        open_local_file(doc.as_uri())
        opened.assert_called_once_with(doc.resolve().as_uri())

    def test_opener_failure_is_logged_not_raised(self, monkeypatch, caplog):
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers.sys.platform", "linux")
        monkeypatch.setattr(
            "lilbee.cli.tui.screens.chat_helpers.subprocess.run",
            mock.Mock(side_effect=OSError("no xdg-open")),
        )
        href = _write_doc("doc.md").as_uri()
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
        open_local_file(_write_doc("doc.md").as_uri() + "%00.md")
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
                "Link not opened: the target is not a document lilbee can show",
            ),
            (
                LinkOpenResult.RESOURCE_FORK,
                "Link not opened: the file carries Finder resource data, such as a custom icon",
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

    def test_every_refusal_has_a_notice(self):
        refusals = set(LinkOpenResult) - {LinkOpenResult.OPENED}
        assert len(refusals) == 3
        assert set(LINK_REFUSAL_NOTICES) == refusals

    def test_opened_file_link_shows_no_notice(self, monkeypatch):
        monkeypatch.setattr(
            chat_screen, "open_local_file", mock.Mock(return_value=LinkOpenResult.OPENED)
        )
        told = mock.Mock()
        monkeypatch.setattr(chat_screen, "call_from_thread", told)
        chat_screen.ChatScreen._open_file_link.__wrapped__(mock.Mock(), "file:///tmp/doc.md")
        told.assert_not_called()

    def test_an_unexpected_opener_error_reaches_the_user_and_the_log(self, monkeypatch, caplog):
        monkeypatch.setattr(
            chat_screen, "open_local_file", mock.Mock(side_effect=RuntimeError("boom"))
        )
        told = mock.Mock()
        monkeypatch.setattr(chat_screen, "call_from_thread", told)
        screen = mock.Mock()
        with caplog.at_level("ERROR"):
            chat_screen.ChatScreen._open_file_link.__wrapped__(screen, "file:///tmp/doc.md")
        told.assert_called_once_with(
            screen,
            screen.notify,
            "Link not opened: the file could not be opened. See the log for details",
            severity="error",
        )
        assert "Could not open link: file:///tmp/doc.md" in caplog.messages
        assert caplog.records[0].exc_info is not None

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
        # Linux has no Finder metadata; TestMacosAliases covers these two reads.
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers._has_alias_flag", lambda p: False)
        monkeypatch.setattr(
            "lilbee.cli.tui.screens.chat_helpers._has_resource_fork", lambda p: False
        )
        run = mock.Mock()
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers.subprocess.run", run)
        return run

    @pytest.mark.parametrize(
        "name",
        [
            "Tool.command",
            "Shell.terminal",
            "run.SH",
            "a.exe",
            "a.fileloc",
            "a.jar",
            "a.jnlp",
            "a.pkg",
            "a.dmg",
            "a.iso",
            "a.mobileconfig",
            "a.webloc",
            "a.url",
            "no-suffix",
            "run.sh",
            "run.py",
            "run.ps1",
        ],
    )
    def test_a_type_that_is_not_an_ingestible_document_is_refused(self, run, name):
        target = _write_doc(name, b"echo hi")
        assert open_local_file(target.as_uri()) is LinkOpenResult.NOT_A_DOCUMENT
        run.assert_not_called()

    @pytest.mark.parametrize("name", ["alias", "alias.md", "alias.pdf", "alias.command"])
    def test_a_finder_alias_is_refused_whatever_its_name(self, run, name):
        # The first bytes of a real alias file made by Finder on macOS.
        alias = _write_doc(name, b"book\0\0\0\0mark\0\0\0\0\x38\0\0\0")
        assert open_local_file(alias.as_uri()) is LinkOpenResult.NOT_A_DOCUMENT
        run.assert_not_called()

    def test_a_text_file_that_starts_with_the_word_book_opens(self, run):
        doc = _write_doc("books.txt", b"bookmark list")
        assert open_local_file(doc.as_uri()) is LinkOpenResult.OPENED
        run.assert_called_once()

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs a POSIX fifo")
    def test_a_fifo_named_like_a_document_is_refused_without_a_read(self, run):
        fifo = cfg.documents_dir / "pipe.txt"
        fifo.parent.mkdir(parents=True, exist_ok=True)
        os.mkfifo(fifo, 0o644)
        assert open_local_file(fifo.as_uri()) is LinkOpenResult.NOT_A_DOCUMENT
        run.assert_not_called()

    @pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="needs a unix socket")
    def test_a_socket_named_like_a_document_is_refused(self, run, monkeypatch):
        cfg.documents_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.chdir(
            cfg.documents_dir
        )  # a relative name keeps the path under the socket limit
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.bind("s.txt")
            sock_path = cfg.documents_dir / "s.txt"
            sock_path.chmod(0o644)
            assert open_local_file(sock_path.as_uri()) is LinkOpenResult.NOT_A_DOCUMENT
        run.assert_not_called()

    def test_a_file_that_cannot_be_read_is_refused(self, run, monkeypatch):
        doc = _write_doc("doc.md")
        real_open = Path.open

        def refuse(self, *args, **kwargs):
            if self == doc.resolve():
                raise PermissionError("no read")
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", refuse)
        assert open_local_file(doc.as_uri()) is LinkOpenResult.NOT_A_DOCUMENT
        run.assert_not_called()

    def test_a_missing_file_is_refused(self, run):
        missing = cfg.documents_dir / "gone.md"
        missing.parent.mkdir(parents=True, exist_ok=True)
        assert open_local_file(missing.as_uri()) is LinkOpenResult.NOT_A_DOCUMENT
        run.assert_not_called()

    def test_a_source_file_that_an_editor_shows_opens(self, run):
        doc = _write_doc("lib.rs", b"fn main() {}")
        assert open_local_file(doc.as_uri()) is LinkOpenResult.OPENED
        run.assert_called_once()

    def test_no_code_file_opens_on_windows(self, monkeypatch):
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers.sys.platform", "win32")
        browser = mock.Mock()
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers.webbrowser.open", browser)
        for name in ("lib.rs", "run.js", "tool.vbs", "a.jar", "a.url", "a.hta"):
            target = _write_doc(name, b"x")
            assert open_local_file(target.as_uri()) is LinkOpenResult.NOT_A_DOCUMENT, name
        browser.assert_not_called()

    def test_a_document_opens_through_the_browser_on_windows(self, monkeypatch):
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers.sys.platform", "win32")
        browser = mock.Mock()
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers.webbrowser.open", browser)
        doc = _write_doc("notes.md")
        assert open_local_file(doc.as_uri()) is LinkOpenResult.OPENED
        browser.assert_called_once()

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
    @pytest.mark.parametrize("mode", [0o755, 0o700, 0o610, 0o601])
    @pytest.mark.parametrize("name", ["Dockerfile", "Makefile", "go", "sh"])
    def test_an_executable_file_without_a_suffix_is_refused(self, run, mode, name):
        # macOS opens an executable file without a suffix in Terminal, which runs it.
        script = cfg.documents_dir / "sub" / name
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("#!/bin/sh\n", encoding="utf-8")
        script.chmod(mode)
        assert open_local_file(script.as_uri()) is LinkOpenResult.NOT_A_DOCUMENT
        run.assert_not_called()

    @pytest.mark.skipif(sys.platform == "win32", reason="Windows has no execute bit")
    @pytest.mark.parametrize("mode", [0o700, 0o755, 0o610, 0o601])
    @pytest.mark.parametrize("name", ["notes.txt", "notes.md", "paper.pdf", "page.html"])
    def test_a_document_with_an_execute_bit_opens(self, run, mode, name):
        # FAT and exFAT volumes report every file as executable.
        doc = cfg.documents_dir / name
        doc.parent.mkdir(parents=True, exist_ok=True)
        doc.write_text("#!/bin/sh\n", encoding="utf-8")
        doc.chmod(mode)
        assert open_local_file(doc.as_uri()) is LinkOpenResult.OPENED
        run.assert_called_once()

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


def _fake_libc(finder_info: bytes | None) -> mock.Mock:
    """A libc whose getxattr returns *finder_info*, or -1 when it is None."""
    libc = mock.Mock()

    def getxattr(path, name, buf, size, position, options):
        assert isinstance(buf, ctypes.Array), "getxattr needs a writable buffer"
        if finder_info is None:
            return -1
        ctypes.memmove(buf, finder_info, len(finder_info))
        return len(finder_info)

    libc.getxattr.side_effect = getxattr
    return libc


def _patch_libc(monkeypatch, finder_info: bytes | None, errno_after: int = 0):
    """Patch ctypes.CDLL to hand out a fake libc; return the CDLL mock and the libc."""
    libc = _fake_libc(finder_info)
    cdll = mock.Mock(return_value=libc)
    monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers.ctypes.CDLL", cdll)
    monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers.ctypes.get_errno", lambda: errno_after)
    return cdll, libc


def _finder_info(flags: int) -> bytes:
    return b"\0" * 8 + flags.to_bytes(2, "big") + b"\0" * 22


class TestMacosAliases:
    """A legacy Finder alias keeps its target in the resource fork and a flag in the Finder info."""

    @pytest.fixture()
    def run(self, monkeypatch):
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers.sys.platform", "darwin")
        run = mock.Mock()
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers.subprocess.run", run)
        return run

    @pytest.mark.skipif(sys.platform != "darwin", reason="needs macOS resource forks and xattr")
    @pytest.mark.parametrize("data", [b"", b"# plain notes\n"], ids=["empty-data", "text-data"])
    def test_a_resource_fork_alias_is_refused_on_macos(self, run, data):
        alias = _write_doc("legacy.md", data)
        with open(f"{alias}/..namedfork/rsrc", "wb") as fork:
            fork.write(b"alis resource bytes")
        flags = _finder_info(0x8000).hex()
        subprocess.check_call(["/usr/bin/xattr", "-wx", "com.apple.FinderInfo", flags, str(alias)])
        assert open_local_file(alias.as_uri()) is LinkOpenResult.NOT_A_DOCUMENT
        run.assert_not_called()

    @pytest.mark.skipif(sys.platform != "darwin", reason="needs macOS xattr")
    def test_the_alias_flag_alone_is_refused_on_macos(self, run):
        doc = _write_doc("flag-only.md", b"# plain notes\n")
        flags = _finder_info(0x8000).hex()
        subprocess.check_call(["/usr/bin/xattr", "-wx", "com.apple.FinderInfo", flags, str(doc)])
        assert open_local_file(doc.as_uri()) is LinkOpenResult.NOT_A_DOCUMENT
        run.assert_not_called()

    @pytest.mark.skipif(sys.platform != "darwin", reason="needs macOS resource forks")
    def test_a_document_with_any_resource_fork_is_refused_as_finder_data_on_macos(self, run):
        doc = _write_doc("forked.md", b"hi")
        with open(f"{doc}/..namedfork/rsrc", "wb") as fork:
            fork.write(b"x")
        assert open_local_file(doc.as_uri()) is LinkOpenResult.RESOURCE_FORK
        run.assert_not_called()

    @pytest.mark.skipif(sys.platform != "darwin", reason="needs macOS Finder metadata")
    def test_a_plain_document_opens_on_macos(self, run):
        doc = _write_doc("plain.md", b"hi")
        assert open_local_file(doc.as_uri()) is LinkOpenResult.OPENED
        run.assert_called_once()

    @pytest.fixture()
    def no_fork(self, monkeypatch):
        monkeypatch.setattr(
            "lilbee.cli.tui.screens.chat_helpers._has_resource_fork", lambda p: False
        )

    @pytest.mark.parametrize(
        ("flags", "expected"),
        [
            (0x8000, LinkOpenResult.NOT_A_DOCUMENT),
            (0xFFFF, LinkOpenResult.NOT_A_DOCUMENT),
            (0x4000, LinkOpenResult.OPENED),
            (0x7FFF, LinkOpenResult.OPENED),
            (0x0080, LinkOpenResult.OPENED),
            (0x0000, LinkOpenResult.OPENED),
        ],
    )
    def test_only_the_alias_bit_of_the_finder_flags_refuses(
        self, run, no_fork, monkeypatch, flags, expected
    ):
        cdll, libc = _patch_libc(monkeypatch, _finder_info(flags))
        doc = _write_doc("flagged.md")
        assert open_local_file(doc.as_uri()) is expected
        cdll.assert_called_once_with(None, use_errno=True)
        libc.getxattr.assert_called_once_with(
            os.fsencode(doc.resolve()), b"com.apple.FinderInfo", mock.ANY, 32, 0, 0
        )

    def test_the_getxattr_signature_is_declared(self, run, no_fork, monkeypatch):
        _, libc = _patch_libc(monkeypatch, _finder_info(0))
        open_local_file(_write_doc("sig.md").as_uri())
        assert libc.getxattr.argtypes == [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_int,
        ]
        assert libc.getxattr.restype is ctypes.c_ssize_t

    @pytest.mark.parametrize("offset", [0, 7, 10, 31])
    def test_a_set_bit_outside_the_flags_field_does_not_refuse(
        self, run, no_fork, monkeypatch, offset
    ):
        info = bytearray(32)
        info[offset] = 0x80
        _patch_libc(monkeypatch, bytes(info))
        assert open_local_file(_write_doc("elsewhere.md").as_uri()) is LinkOpenResult.OPENED

    def test_an_empty_finder_info_attribute_is_not_an_alias(self, run, no_fork, monkeypatch):
        _patch_libc(monkeypatch, b"", errno_after=13)
        assert open_local_file(_write_doc("empty.md").as_uri()) is LinkOpenResult.OPENED
        run.assert_called_once()

    def test_a_missing_finder_info_attribute_is_not_an_alias(self, run, no_fork, monkeypatch):
        _patch_libc(monkeypatch, None, errno_after=93)
        doc = _write_doc("bare.md")
        assert open_local_file(doc.as_uri()) is LinkOpenResult.OPENED
        run.assert_called_once()

    def test_a_finder_info_read_error_refuses_the_document(self, run, no_fork, monkeypatch):
        _patch_libc(monkeypatch, None, errno_after=13)
        doc = _write_doc("denied.md")
        assert open_local_file(doc.as_uri()) is LinkOpenResult.NOT_A_DOCUMENT
        run.assert_not_called()

    def test_the_finder_info_read_error_names_the_file_and_the_cause(self, monkeypatch):
        _patch_libc(monkeypatch, None, errno_after=13)
        with pytest.raises(OSError, match="Permission denied") as caught:
            _has_alias_flag(Path("/some/file.md"))
        assert caught.value.errno == 13
        assert caught.value.filename == "/some/file.md"

    @pytest.mark.parametrize(
        ("fork_size", "expected"),
        [
            (5, LinkOpenResult.RESOURCE_FORK),
            (1, LinkOpenResult.RESOURCE_FORK),
            (0, LinkOpenResult.OPENED),
        ],
        ids=["forked", "one-byte-fork", "empty-fork"],
    )
    def test_the_resource_fork_size_decides(self, run, monkeypatch, fork_size, expected):
        doc = _write_doc("fork.md")
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers._has_alias_flag", lambda p: False)
        monkeypatch.setattr(
            "lilbee.cli.tui.screens.chat_helpers.os.stat", _stat_with_fork(fork_size, doc)
        )
        assert open_local_file(doc.as_uri()) is expected

    def test_a_resource_fork_refusal_is_logged_with_its_reason(self, run, monkeypatch, caplog):
        doc = _write_doc("icon.md")
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers._has_alias_flag", lambda p: False)
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers.os.stat", _stat_with_fork(3, doc))
        with caplog.at_level("WARNING"):
            open_local_file(doc.as_uri())
        assert caplog.messages == [f"Refused to open a file (resource_fork): {doc.resolve()}"]

    def test_a_program_with_a_resource_fork_is_refused_as_not_a_document(self, run, monkeypatch):
        tool = _write_doc("tool.command")
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers._has_alias_flag", lambda p: False)
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers.os.stat", _stat_with_fork(5, tool))
        assert open_local_file(tool.as_uri()) is LinkOpenResult.NOT_A_DOCUMENT
        run.assert_not_called()

    def test_a_file_without_a_resource_fork_is_not_an_alias(self, run, monkeypatch):
        doc = _write_doc("nofork.md")
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers._has_alias_flag", lambda p: False)
        monkeypatch.setattr(
            "lilbee.cli.tui.screens.chat_helpers.os.stat",
            _stat_with_fork(FileNotFoundError(), doc),
        )
        assert open_local_file(doc.as_uri()) is LinkOpenResult.OPENED

    def test_a_resource_fork_read_error_refuses_the_document(self, run, monkeypatch):
        doc = _write_doc("locked.md")
        monkeypatch.setattr("lilbee.cli.tui.screens.chat_helpers._has_alias_flag", lambda p: False)
        monkeypatch.setattr(
            "lilbee.cli.tui.screens.chat_helpers.os.stat",
            _stat_with_fork(PermissionError("denied"), doc),
        )
        assert open_local_file(doc.as_uri()) is LinkOpenResult.NOT_A_DOCUMENT
        run.assert_not_called()


def _stat_with_fork(fork: int | Exception, doc: Path):
    """An os.stat that answers the resource-fork path of *doc* with *fork*, other paths for real."""
    real_stat = os.stat
    fork_path = f"{doc.resolve()}/..namedfork/rsrc"

    def fake(path, *args, **kwargs):
        if str(path) == fork_path:
            if isinstance(fork, Exception):
                raise fork
            return mock.Mock(st_size=fork)
        return real_stat(path, *args, **kwargs)

    return fake


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
