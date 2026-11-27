"""The wiki and drafts screens read files and the store off the event loop."""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from textual.app import ComposeResult
from textual.widgets import DataTable, Footer, Input, Markdown, Static, Tree

import lilbee.cli.tui.screens.wiki as wiki_mod
import lilbee.cli.tui.screens.wiki_drafts as drafts_mod
import lilbee.wiki.browse as browse
from lilbee.cli.tui.screens.wiki import WikiScreen
from lilbee.cli.tui.screens.wiki_drafts import WikiDraftsScreen
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog
from lilbee.core.config import cfg
from lilbee.wiki.drafts import DraftInfo
from lilbee.wiki.entity_extractor import EntityKind
from lilbee.wiki.stubs import WikiStub
from tests._async_wait import comes, wait_until
from tests._lilbee_app_test_host import LilbeeAppHost

SIZES = [(80, 24), (120, 40)]
WAIT = 5.0


class _Host(LilbeeAppHost):
    """A bare app: each test pushes the screen it studies once its spies are in place."""

    CSS = ""

    def compose(self) -> ComposeResult:
        yield Footer()


class _Reads:
    """Records the thread that runs each wrapped call, beside the loop thread."""

    def __init__(self) -> None:
        self.loop = threading.get_ident()
        self.threads: dict[str, list[int]] = {}

    def wrap(self, name: str, real: Callable[..., Any]) -> Callable[..., Any]:
        def spy(*args: Any, **kwargs: Any) -> Any:
            self.threads.setdefault(name, []).append(threading.get_ident())
            return real(*args, **kwargs)

        return spy

    def on_loop(self) -> list[str]:
        return sorted(name for name, idents in self.threads.items() if self.loop in idents)

    def count(self, name: str) -> int:
        return len(self.threads.get(name, []))


def _page(root: Path, subdir: str, slug: str, title: str) -> None:
    page = root / subdir / f"{slug}.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(
        f"---\ntitle: {title}\ngenerated_at: 2025-01-01\nsource_count: 3\n---\nBody text\n",
        encoding="utf-8",
    )


def _wiki(tmp_path: Path, *, pages: bool = True) -> Path:
    cfg.wiki = True
    cfg.data_root = tmp_path
    root = tmp_path / cfg.wiki_dir
    root.mkdir(parents=True, exist_ok=True)
    if pages:
        _page(root, "summaries", "test-doc", "Test Document")
        _page(root, "synthesis", "some-synthesis", "Some Synthesis")
        (root / "index.md").write_text("# Index\n", encoding="utf-8")
    return root


def _stub() -> WikiStub:
    return WikiStub(
        slug="torque",
        label="Torque",
        kind=EntityKind.CONCEPT,
        type_hint="",
        source_mentions=(("src-1", 4),),
        chunk_refs=(("src-1", 0),),
    )


def _labels(tree: Tree[Any]) -> list[str]:
    out: list[str] = []
    stack = list(tree.root.children)
    while stack:
        node = stack.pop(0)
        out.append(str(node.label))
        stack = [*node.children, *stack]
    return out


def _find(tree: Tree[Any], slug: str) -> Any:
    stack = list(tree.root.children)
    while stack:
        node = stack.pop()
        if node.data == slug:
            return node
        stack.extend(node.children)
    return None


def _shows(screen: WikiScreen, text: str) -> bool:
    return any(text in label for label in _labels(screen.query_one("#wiki-page-list", Tree)))


async def _open_wiki(app: _Host) -> WikiScreen:
    screen = WikiScreen()
    await app.push_screen(screen)
    return screen


@contextmanager
def _spied(reads: _Reads, *, stubs: list[WikiStub] | None = None) -> Iterator[None]:
    """Wrap every read the wiki screen's scan makes, so each call logs its thread."""
    real_ungenerated = wiki_mod.ungenerated_stubs
    patches = [
        patch.object(browse, "list_pages", reads.wrap("list_pages", browse.list_pages)),
        patch.object(
            wiki_mod, "load_stub_index", reads.wrap("load_stub_index", wiki_mod.load_stub_index)
        ),
        patch.object(
            wiki_mod,
            "ungenerated_stubs",
            reads.wrap(
                "ungenerated_stubs",
                (lambda *_: stubs) if stubs is not None else real_ungenerated,
            ),
        ),
        patch.object(
            wiki_mod, "_root_shortcuts", reads.wrap("root_shortcuts", wiki_mod._root_shortcuts)
        ),
    ]
    for p in patches:
        p.start()
    try:
        yield
    finally:
        for p in reversed(patches):
            p.stop()


@pytest.mark.parametrize("size", SIZES)
async def test_resume_reads_run_off_the_loop(tmp_path: Path, size: tuple[int, int]) -> None:
    _wiki(tmp_path)
    reads = _Reads()
    app = _Host()
    with _spied(reads):
        async with app.run_test(size=size) as pilot:
            screen = await _open_wiki(app)
            assert await wait_until(pilot, lambda: _shows(screen, "Test Document"), timeout=WAIT)
    assert set(reads.threads) == {
        "list_pages",
        "load_stub_index",
        "ungenerated_stubs",
        "root_shortcuts",
    }
    assert reads.on_loop() == []


async def test_reload_returns_before_the_read_ends(tmp_path: Path) -> None:
    root = _wiki(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    real = browse.list_pages

    def held(wiki_root: Path) -> list[Any]:
        entered.set()
        release.wait(timeout=3)
        return real(wiki_root)

    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _open_wiki(app)
        assert await wait_until(pilot, lambda: _shows(screen, "Test Document"), timeout=WAIT)
        _page(root, "summaries", "fresh-page", "Fresh Page")
        entered.clear()
        with patch.object(browse, "list_pages", held):
            screen.reload()
            assert await wait_until(pilot, entered.is_set, timeout=WAIT)
            assert not _shows(screen, "Fresh Page")
            release.set()
            assert await wait_until(pilot, lambda: _shows(screen, "Fresh Page"), timeout=WAIT)


async def test_filtering_does_no_io_on_the_loop(tmp_path: Path) -> None:
    _wiki(tmp_path)
    reads = _Reads()
    app = _Host()
    with _spied(reads):
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open_wiki(app)
            assert await wait_until(pilot, lambda: _shows(screen, "Test Document"), timeout=WAIT)
            scans = reads.count("root_shortcuts")
            assert scans == 1
            screen.query_one("#wiki-search", Input).value = "synth"
            assert await wait_until(
                pilot, lambda: not _shows(screen, "Test Document"), timeout=WAIT
            )
            assert _shows(screen, "Some Synthesis")
    assert reads.count("root_shortcuts") == scans


async def test_opening_a_page_reads_off_the_loop(tmp_path: Path) -> None:
    _wiki(tmp_path)
    reads = _Reads()
    app = _Host()
    with patch.object(browse, "read_page", reads.wrap("read_page", browse.read_page)):
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open_wiki(app)
            assert await wait_until(pilot, lambda: _shows(screen, "Test Document"), timeout=WAIT)
            tree = screen.query_one("#wiki-page-list", Tree)
            node = _find(tree, "summaries/test-doc")
            assert node is not None
            tree.select_node(node)
            header = screen.query_one("#wiki-page-header", Static)
            assert await wait_until(
                pilot, lambda: "Test Document" in str(header.render()), timeout=WAIT
            )
    assert reads.count("read_page") == 1
    assert reads.on_loop() == []


async def test_opening_a_stub_queries_the_store_off_the_loop(tmp_path: Path) -> None:
    _wiki(tmp_path)
    stub = _stub()
    reads = _Reads()
    store = SimpleNamespace(source_paths=lambda ids: {i: "docs/handbook.md" for i in ids})
    services = SimpleNamespace(store=store)
    app = _Host()
    with (
        _spied(reads, stubs=[stub]),
        patch.object(wiki_mod, "get_services", lambda: services),
        patch.object(
            wiki_mod, "shown_sources", reads.wrap("shown_sources", wiki_mod.shown_sources)
        ),
    ):
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open_wiki(app)
            assert await wait_until(pilot, lambda: _shows(screen, "Torque"), timeout=WAIT)
            tree = screen.query_one("#wiki-page-list", Tree)
            node = _find(tree, stub.wiki_slug)
            assert node is not None
            tree.select_node(node)
            assert await wait_until(
                pilot, lambda: isinstance(app.screen, ConfirmDialog), timeout=WAIT
            )
            dialog_message = app.screen._message
            detail = screen.query_one("#wiki-content", Markdown).source
    assert reads.count("shown_sources") == 1
    assert reads.on_loop() == []
    assert "docs/handbook.md" in detail
    assert "docs/handbook.md" in dialog_message


@pytest.mark.parametrize("route", ["screen_key", "app_offer"])
async def test_wipe_offer_checks_content_off_the_loop(tmp_path: Path, route: str) -> None:
    _wiki(tmp_path, pages=False)
    reads = _Reads()
    store = SimpleNamespace(
        wiki_chunk_sources=reads.wrap("wiki_chunk_sources", lambda: ["row"]),
        wiki_citation_sources=reads.wrap("wiki_citation_sources", lambda: []),
    )
    services = SimpleNamespace(store=store)
    app = _Host()
    with patch.object(wiki_mod, "get_services", lambda: services):
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open_wiki(app)
            assert await wait_until(
                pilot,
                lambda: bool(_labels(screen.query_one("#wiki-page-list", Tree))),
                timeout=WAIT,
            )
            if route == "screen_key":
                await pilot.press("W")
            else:
                app._offer_wiki_wipe()
            assert await wait_until(
                pilot, lambda: isinstance(app.screen, ConfirmDialog), timeout=WAIT
            )
    assert reads.count("wiki_chunk_sources") == 1
    assert reads.on_loop() == []


async def test_a_slow_older_reload_never_overwrites_a_newer_one(tmp_path: Path) -> None:
    root = _wiki(tmp_path, pages=False)
    _page(root, "summaries", "new-page", "New Page")
    new_pages = browse.list_pages(root)
    old_pages = [
        dataclasses.replace(page, title="Old Page", slug="summaries/old-page") for page in new_pages
    ]
    calls: list[int] = []
    first_entered = threading.Event()
    release = threading.Event()
    first_done = threading.Event()

    def scan(_root: Path) -> list[Any]:
        calls.append(len(calls))
        if len(calls) > 1:
            return new_pages
        first_entered.set()
        release.wait(timeout=5)
        first_done.set()
        return old_pages

    app = _Host()
    with patch.object(browse, "list_pages", scan):
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _open_wiki(app)
            assert await wait_until(pilot, first_entered.is_set, timeout=WAIT)
            screen.reload()
            assert await wait_until(pilot, lambda: _shows(screen, "New Page"), timeout=WAIT)
            release.set()
            assert await wait_until(pilot, first_done.is_set, timeout=WAIT)
            assert not await comes(pilot, lambda: _shows(screen, "Old Page"))
            assert _shows(screen, "New Page")


async def test_drafts_screen_reads_run_off_the_loop(tmp_path: Path) -> None:
    _wiki(tmp_path)
    reads = _Reads()
    draft = DraftInfo(
        slug="summaries/torque",
        path=tmp_path / "torque.md",
        drift_ratio=0.2,
        faithfulness_score=0.9,
        bad_title=False,
        published_path=None,
        mtime=0.0,
    )
    services = SimpleNamespace(store=object())
    app = _Host()
    with (
        patch.object(drafts_mod, "list_drafts", reads.wrap("list_drafts", lambda _root: [draft])),
        patch.object(drafts_mod, "diff_draft", reads.wrap("diff_draft", lambda *_: "+a line")),
        patch.object(drafts_mod, "get_services", lambda: services),
    ):
        async with app.run_test(size=(120, 40)) as pilot:
            screen = WikiDraftsScreen()
            await app.push_screen(screen)
            table = screen.query_one("#wiki-drafts-table", DataTable)
            assert await wait_until(pilot, lambda: table.row_count == 1, timeout=WAIT)
            table.move_cursor(row=0)
            diff = screen.query_one("#wiki-drafts-diff", Static)
            assert await wait_until(pilot, lambda: "+a line" in str(diff.render()), timeout=WAIT)
    assert set(reads.threads) == {"list_drafts", "diff_draft"}
    assert reads.on_loop() == []
