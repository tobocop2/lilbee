"""A posted source links and opens only when its file lies inside a document root.

Acceptance for #1249. The roots are the documents directory (which holds the
crawled-page folder ``_web``) and every registered linked root. The oracle is
``os.path.realpath`` plus ``os.path.commonpath``, not the code under test.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from unittest import mock

import pytest

from lilbee.app.session_sources import message_sources, posted_message
from lilbee.cli.tui.screens.chat_helpers import open_local_file
from lilbee.core import settings
from lilbee.core.config import cfg, config_scope
from lilbee.retrieval.query.formatting import (
    SourceRef,
    append_sources_block,
    markdown_source_line,
    source_ref_line,
    source_ref_link,
)
from lilbee.sessions import MessageRole, StoredSourceRef

_HELPERS = "lilbee.cli.tui.screens.chat_helpers"


def _inside(path: str, roots: list[str]) -> bool:
    """Whether the real path of *path* lies in a real root (the oracle)."""
    real = os.path.realpath(path)
    for root in roots:
        real_root = os.path.realpath(root)
        try:
            if os.path.commonpath([real, real_root]) == real_root:
                return True
        except ValueError:
            continue
    return False


@pytest.fixture()
def world(tmp_path):
    """Documents dir, a linked root, a single-file root, and files outside every root."""
    docs = tmp_path / "docs"
    notes = tmp_path / "notes"
    evil = tmp_path / "docs-evil"
    outside = tmp_path / "outside"
    for folder in (docs / "sub", docs / "_web" / "example.com", notes, evil, outside):
        folder.mkdir(parents=True)
    paper = tmp_path / "paper.pdf"
    for file in (
        docs / "a.md",
        docs / "sub" / "b.md",
        docs / "has space.md",
        docs / "_web" / "example.com" / "guide~abc.md",
        notes / "c.md",
        paper,
        evil / "x.md",
        outside / "secret.md",
        tmp_path / "outside.md",
    ):
        file.write_text("x", encoding="utf-8")
    links = {}
    try:
        os.symlink(outside / "secret.md", docs / "escape.md")
        os.symlink(outside, docs / "linkdir", target_is_directory=True)
        os.symlink(docs / "a.md", docs / "inside-link.md")
        links = {"escape.md": False, "linkdir/secret.md": False, "inside-link.md": True}
    except (OSError, NotImplementedError):
        pass
    roots = {"notes": str(notes), "paper.pdf": str(paper)}
    config = cfg.model_copy(
        update={
            "data_root": tmp_path / "data",
            "documents_dir": docs,
            "linked_roots": dict(roots),
        }
    )
    (tmp_path / "data").mkdir()
    with config_scope(config):
        settings.set_value(config.data_root, "linked_roots", dict(roots))
        yield {
            "tmp": tmp_path,
            "docs": docs,
            "roots": [str(docs), *roots.values()],
            "links": links,
        }


def _safe_sources(w) -> list[str]:
    return [
        "a.md",
        "sub/b.md",
        "has space.md",
        "_web/example.com/guide~abc.md",
        "notes/c.md",
        "paper.pdf",
        "sub/../a.md",
        "missing-file.md",
        *[name for name, ok in w["links"].items() if ok],
    ]


def _unsafe_sources(w) -> list[str]:
    tmp = w["tmp"]
    return [
        "../../../../../../etc/passwd",
        "../outside.md",
        "../docs-evil/x.md",
        "notes/../../outside.md",
        "notes/../notes-evil/x.md",
        "paper.pdf/../outside.md",
        "/etc/passwd",
        str(tmp / "outside.md"),
        str(tmp / "docs-evil" / "x.md"),
        *[name for name, ok in w["links"].items() if not ok],
    ]


def _generated(w) -> list[tuple[str, bool]]:
    """Seeded traversal keys under the documents dir, each classed by the oracle."""
    rng = random.Random(1249)
    pieces = ["..", "sub", "x", ".", "docs-evil"]
    out = []
    for _ in range(40):
        key = "/".join([*rng.choices(pieces, k=rng.randint(0, 5)), "f.md"])
        out.append((key, _inside(os.path.join(w["docs"], key), w["roots"])))
    assert any(inside for _, inside in out)
    assert any(not inside for _, inside in out)
    return out


def _ref(source: str) -> SourceRef:
    return SourceRef(source, source, 1)


def _renderers():
    return {
        "terminal_link": source_ref_link,
        "terminal_line": source_ref_line,
        "markdown_line": markdown_source_line,
    }


class TestLinkBuilder:
    """Every Sources renderer links a safe source and shows an unsafe one as plain text."""

    @pytest.mark.parametrize("renderer", sorted(_renderers()))
    def test_safe_sources_link_to_their_real_path(self, world, renderer):
        sources = _safe_sources(world)
        assert sources
        for source in sources:
            line = _renderers()[renderer](_ref(source))
            first, _, rest = source.partition("/")
            labelled = {"notes": world["roots"][1], "paper.pdf": world["roots"][2]}
            base = Path(labelled[first]) / rest if first in labelled else world["docs"] / source
            expected = Path(os.path.realpath(base)).as_uri()
            assert f"]({expected})" in line, (source, line)

    @pytest.mark.parametrize("renderer", sorted(_renderers()))
    def test_unsafe_sources_show_as_plain_text(self, world, renderer):
        sources = _unsafe_sources(world)
        assert sources
        for source in sources:
            line = _renderers()[renderer](_ref(source))
            assert "file:" not in line, (source, line)
            assert "](" not in line, (source, line)
            if renderer != "markdown_line":
                assert line == source, (source, line)

    def test_generated_keys_link_exactly_when_the_oracle_says_inside(self, world):
        for key, inside in _generated(world):
            line = source_ref_link(_ref(key))
            assert ("file:" in line) is inside, (key, inside, line)

    def test_a_sibling_directory_with_the_root_name_as_prefix_is_outside(self, world):
        line = source_ref_link(_ref(str(world["tmp"] / "docs-evil" / "x.md")))
        assert "file:" not in line


class TestPostedTurnEndToEnd:
    """A turn posted with source_refs, read back as the TUI renders a resumed turn."""

    def test_only_the_safe_posted_sources_link(self, world):
        safe, unsafe = _safe_sources(world), _unsafe_sources(world)
        message = posted_message(
            MessageRole.ASSISTANT,
            "answer",
            (),
            [StoredSourceRef(s, i + 1, "") for i, s in enumerate([*safe, *unsafe])],
        )
        text, refs = message_sources(message)
        for render in (source_ref_line, markdown_source_line):
            shown = append_sources_block(text, refs, render).split("\n")
            lines = [line for line in shown if line[:1].isdigit()]
            assert len(lines) == len(safe) + len(unsafe)
            linked = [("file:" in line) for line in lines]
            assert linked == [True] * len(safe) + [False] * len(unsafe)


@pytest.fixture()
def opener(monkeypatch):
    """The OS opener and the browser fallback, both recorded and neither run."""
    run = mock.Mock()
    browser = mock.Mock()
    monkeypatch.setattr(f"{_HELPERS}.subprocess.run", run)
    monkeypatch.setattr(f"{_HELPERS}.webbrowser.open", browser)
    return run, browser


def _unsafe_hrefs(w) -> list[str]:
    tmp, docs = w["tmp"], w["docs"]
    return [
        (tmp / "outside.md").as_uri(),
        (tmp / "docs-evil" / "x.md").as_uri(),
        "file:///etc/passwd",
        f"{(docs / 'sub').as_uri()}/%2e%2e/%2e%2e/outside.md",
        f"{docs.as_uri()}/../outside.md",
        f"{docs.as_uri()}/a.md%00.txt",
        *[(docs / name).as_uri() for name, ok in w["links"].items() if not ok],
    ]


def _safe_hrefs(w) -> list[str]:
    docs = w["docs"]
    return [
        (docs / "a.md").as_uri(),
        (docs / "sub" / "b.md").as_uri(),
        (docs / "has space.md").as_uri(),
        (Path(w["roots"][1]) / "c.md").as_uri(),
        Path(w["roots"][2]).as_uri(),
    ]


class TestOpenTime:
    """The opener re-checks the path: any file: link in an answer reaches it."""

    @pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
    def test_an_unsafe_href_opens_nothing(self, world, opener, monkeypatch, platform):
        monkeypatch.setattr(f"{_HELPERS}.sys.platform", platform)
        run, browser = opener
        hrefs = _unsafe_hrefs(world)
        assert hrefs
        for href in hrefs:
            open_local_file(href)
            assert run.call_count == 0, href
            assert browser.call_count == 0, href

    @pytest.mark.parametrize("platform", ["darwin", "linux"])
    def test_a_safe_href_opens_its_decoded_path(self, world, opener, monkeypatch, platform):
        monkeypatch.setattr(f"{_HELPERS}.sys.platform", platform)
        run, _ = opener
        hrefs = _safe_hrefs(world)
        assert hrefs
        for href in hrefs:
            run.reset_mock()
            open_local_file(href)
            (argv,), _kw = run.call_args
            assert _inside(argv[1], world["roots"]), href

    def test_a_safe_href_on_a_platform_without_an_opener_uses_the_browser(
        self, world, opener, monkeypatch
    ):
        monkeypatch.setattr(f"{_HELPERS}.sys.platform", "win32")
        _, browser = opener
        href = (world["docs"] / "a.md").as_uri()
        assert open_local_file(href) is True
        browser.assert_called_once_with((world["docs"] / "a.md").resolve().as_uri())

    def test_the_browser_gets_the_checked_path_without_the_host(self, world, opener, monkeypatch):
        monkeypatch.setattr(f"{_HELPERS}.sys.platform", "win32")
        _, browser = opener
        local = (world["docs"] / "a.md").resolve()
        # A Windows path has no leading slash ("C:/..."), so the slash is added to end the host.
        open_local_file(f"file://attacker.example/{local.as_posix().lstrip('/')}")
        browser.assert_called_once_with(local.as_uri())

    def test_a_nul_byte_path_is_refused_when_resolve_accepts_it(self, world, opener, monkeypatch):
        """Python 3.13 on Windows resolves a path with a NUL byte without an error."""
        monkeypatch.setattr(f"{_HELPERS}.sys.platform", "linux")
        run, browser = opener
        monkeypatch.setattr(f"{_HELPERS}.Path.resolve", lambda self, strict=False: self)
        assert open_local_file(f"{world['docs'].as_uri()}/a.md%00.txt") is False
        assert run.call_count == 0
        assert browser.call_count == 0

    def test_the_opener_gets_the_resolved_path(self, world, opener, monkeypatch):
        monkeypatch.setattr(f"{_HELPERS}.sys.platform", "linux")
        run, _ = opener
        open_local_file(f"{world['docs'].as_uri()}/sub/../a.md")
        (argv,), _kw = run.call_args
        assert argv == ["xdg-open", str((world["docs"] / "a.md").resolve())]

    def test_a_symlink_loop_raises_no_error_whatever_the_platform_does(
        self, world, opener, monkeypatch
    ):
        monkeypatch.setattr(f"{_HELPERS}.sys.platform", "linux")
        run, browser = opener
        loop_a, loop_b = world["docs"] / "loop-a", world["docs"] / "loop-b"
        os.symlink(loop_b, loop_a)
        os.symlink(loop_a, loop_b)
        # Resolving a loop raises RuntimeError on Python 3.11 and 3.12 on Linux; other
        # platforms return the path unchanged, which lies inside the root.
        opened = open_local_file(loop_a.as_uri())
        assert browser.call_count == 0
        assert run.call_count == int(opened)

    def test_a_resolve_that_raises_on_a_loop_is_refused(self, world, opener, monkeypatch):
        monkeypatch.setattr(f"{_HELPERS}.sys.platform", "linux")
        run, browser = opener
        monkeypatch.setattr(
            f"{_HELPERS}.Path.resolve", mock.Mock(side_effect=RuntimeError("Symlink loop"))
        )
        assert open_local_file((world["docs"] / "a.md").as_uri()) is False
        assert run.call_count == 0
        assert browser.call_count == 0

    def test_a_refusal_returns_false_and_an_open_returns_true(self, world, opener, monkeypatch):
        monkeypatch.setattr(f"{_HELPERS}.sys.platform", "linux")
        assert open_local_file((world["docs"] / "a.md").as_uri()) is True
        assert open_local_file("file:///etc/passwd") is False


class TestRootThatIsASymlink:
    """A root reached through a symlink still contains the files it holds."""

    def test_files_under_symlinked_roots_link_and_others_do_not(
        self, tmp_path, opener, monkeypatch
    ):
        monkeypatch.setattr(f"{_HELPERS}.sys.platform", "linux")
        real_docs, real_notes = tmp_path / "real-docs", tmp_path / "real-notes"
        for folder in (real_docs, real_notes):
            folder.mkdir()
            (folder / "a.md").write_text("x", encoding="utf-8")
        (tmp_path / "outside.md").write_text("x", encoding="utf-8")
        docs, notes = tmp_path / "docs-link", tmp_path / "notes-link"
        os.symlink(real_docs, docs, target_is_directory=True)
        os.symlink(real_notes, notes, target_is_directory=True)
        (tmp_path / "data").mkdir()
        config = cfg.model_copy(
            update={
                "data_root": tmp_path / "data",
                "documents_dir": docs,
                "linked_roots": {"notes": str(notes)},
            }
        )
        with config_scope(config):
            settings.set_value(config.data_root, "linked_roots", {"notes": str(notes)})
            assert f"]({(real_docs / 'a.md').resolve().as_uri()})" in source_ref_link(_ref("a.md"))
            assert f"]({(real_notes / 'a.md').resolve().as_uri()})" in source_ref_link(
                _ref("notes/a.md")
            )
            assert "file:" not in source_ref_link(_ref("../outside.md"))
            assert open_local_file((real_docs / "a.md").as_uri()) is True
            assert open_local_file((tmp_path / "outside.md").as_uri()) is False
