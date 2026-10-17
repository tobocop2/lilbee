"""The move of saved pages to the names their canonical addresses give them."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lilbee.core.config import Config, cfg
from lilbee.crawler import migrate
from lilbee.crawler.migrate import move_saved_pages
from lilbee.crawler.save import (
    CrawlMeta,
    content_hash,
    load_crawl_metadata,
    save_crawl_metadata,
    url_to_filename,
)
from lilbee.data.store.types import RemoveResult, SourceRelocation

_PAGE = "https://example.com/t/page"
_EARLIER = "example.com/t/page/index.md"


@pytest.fixture(autouse=True)
def store(tmp_path: Path) -> Iterator[MagicMock]:
    """A documents and data root under *tmp_path*, and a store that settles every re-key."""
    snapshot = cfg.model_copy()
    cfg.documents_dir = tmp_path / "documents"
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.documents_dir.mkdir()
    cfg.data_dir.mkdir()
    store = MagicMock()
    store.relocate_sources.side_effect = lambda moves: SourceRelocation({}, len(moves))
    store.remove_documents.return_value = RemoveResult(removed=[], not_found=[])
    yield store
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _web() -> Path:
    return cfg.documents_dir / "_web"


def _write(name: str, text: str) -> Path:
    path = _web() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _moved(store: MagicMock, meta: dict[str, CrawlMeta]) -> dict[str, CrawlMeta]:
    """Store *meta*, run the move, and check the metadata it wrote is what it returned."""
    save_crawl_metadata(meta)
    moved = move_saved_pages(store)
    assert load_crawl_metadata() == moved
    return moved


def _moved_sources(store: MagicMock) -> list[tuple[tuple[str, ...], str]]:
    return [
        (move.candidates, move.new)
        for call in store.relocate_sources.call_args_list
        for move in call.args[0]
    ]


def test_a_page_moves_to_its_name_with_its_index_source(store: MagicMock):
    _write(_EARLIER, "# Page")
    moved = _moved(store, {_PAGE: CrawlMeta(_EARLIER, content_hash("# Page"), "t")})
    name = url_to_filename(_PAGE)
    assert moved == {_PAGE: CrawlMeta(name, content_hash("# Page"), "t")}
    assert (_web() / name).read_text(encoding="utf-8") == "# Page"
    assert _moved_sources(store) == [((f"_web/{_EARLIER}",), f"_web/{name}")]
    assert sorted(p.relative_to(_web()).as_posix() for p in _web().rglob("*")) == [
        "example.com",
        "example.com/t",
        f"{name}",
    ]


def test_a_second_run_moves_nothing_and_holds_no_lock(store: MagicMock, monkeypatch):
    _write(_EARLIER, "# Page")
    meta = _moved(store, {_PAGE: CrawlMeta(_EARLIER, content_hash("# Page"), "t")})
    store.relocate_sources.reset_mock()
    held = MagicMock()
    monkeypatch.setattr("lilbee.crawler.migrate.library_held", held)
    monkeypatch.setattr("lilbee.crawler.save.save_crawl_metadata", held)
    assert move_saved_pages(store) == meta
    held.assert_not_called()
    store.relocate_sources.assert_not_called()


def test_a_crawl_that_moved_the_pages_first_leaves_this_one_nothing_to_move(
    store: MagicMock, monkeypatch
):
    _write(_EARLIER, "# Page")
    save_crawl_metadata({_PAGE: CrawlMeta(_EARLIER, content_hash("# Page"), "t")})
    held = migrate.library_held

    def held_after_another_crawl(
        config: Config, running: str, wait: float
    ) -> AbstractContextManager[object]:
        monkeypatch.setattr("lilbee.crawler.migrate.library_held", held)
        move_saved_pages(store)
        return held(config, running, wait)

    monkeypatch.setattr("lilbee.crawler.migrate.library_held", held_after_another_crawl)
    name = url_to_filename(_PAGE)
    assert move_saved_pages(store) == {_PAGE: CrawlMeta(name, content_hash("# Page"), "t")}
    assert (_web() / name).read_text(encoding="utf-8") == "# Page"
    assert len(_moved_sources(store)) == 1


def test_spellings_of_one_page_keep_the_entry_crawled_last(store: MagicMock):
    name = url_to_filename(_PAGE)
    _write(name, "# Page")
    meta = {
        "https://example.com/t/pag%65": CrawlMeta(name, "older", "2026-01-01"),
        "https://EXAMPLE.com/t/page#top": CrawlMeta(name, "newer", "2026-02-01"),
        _PAGE: CrawlMeta(name, "oldest", "2025-01-01"),
    }
    assert _moved(store, meta) == {_PAGE: CrawlMeta(name, "newer", "2026-02-01")}


@pytest.mark.parametrize(
    "earlier", ["example.com/t/gone/index.md", "../outside.md"], ids=["no-file", "outside"]
)
def test_an_entry_with_no_file_under_the_web_folder_is_dropped(store: MagicMock, earlier: str):
    outside = cfg.documents_dir / "outside.md"
    outside.write_text("# Mine", encoding="utf-8")
    assert _moved(store, {_PAGE: CrawlMeta(earlier, "h", "t")}) == {}
    assert outside.read_text(encoding="utf-8") == "# Mine"


_SHARED = "example.com/t/a_b/index.md"
_COLON = "https://example.com/t/a:b"
_UNDERSCORE = "https://example.com/t/a_b"


@pytest.mark.parametrize(
    ("meta", "owner"),
    [
        pytest.param(
            {
                _COLON: CrawlMeta(_SHARED, "old", "2"),
                _UNDERSCORE: CrawlMeta(_SHARED, content_hash("# Second"), "1"),
            },
            (_UNDERSCORE, "1"),
            id="the-page-whose-text-it-holds",
        ),
        pytest.param(
            {_COLON: CrawlMeta(_SHARED, "old", "2"), _UNDERSCORE: CrawlMeta(_SHARED, "old", "1")},
            (_COLON, "2"),
            id="else-the-latest",
        ),
    ],
)
def test_a_shared_file_goes_to_one_page_and_the_others_are_saved_afresh(
    store: MagicMock, meta: dict[str, CrawlMeta], owner: tuple[str, str]
):
    _write(_SHARED, "# Second")
    address, crawled_at = owner
    name = url_to_filename(address)
    assert _moved(store, meta) == {address: CrawlMeta(name, content_hash("# Second"), crawled_at)}
    assert (_web() / name).read_text(encoding="utf-8") == "# Second"
    assert _moved_sources(store) == [((f"_web/{_SHARED}",), f"_web/{name}")]


def test_a_file_that_is_not_utf_8_goes_to_the_latest_page(store: MagicMock):
    (_web() / "example.com/t/a_b").mkdir(parents=True)
    (_web() / _SHARED).write_bytes(b"\xff\xfe")
    meta = {_COLON: CrawlMeta(_SHARED, "old", "2"), _UNDERSCORE: CrawlMeta(_SHARED, "old", "1")}
    assert list(_moved(store, meta)) == [_COLON]


def test_pages_that_share_a_file_through_an_empty_segment_move_it_once(store: MagicMock):
    _write("example.com/t/x/index.md", "# X")
    meta = {
        "https://example.com/t//x": CrawlMeta("example.com/t//x/index.md", "old", "1"),
        "https://example.com/t/x": CrawlMeta("example.com/t/x/index.md", content_hash("# X"), "2"),
    }
    assert list(_moved(store, meta)) == ["https://example.com/t/x"]
    assert _moved_sources(store) == [
        (
            ("_web/example.com/t/x/index.md",),
            f"_web/{url_to_filename('https://example.com/t/x')}",
        )
    ]


def test_a_disk_with_no_inodes_groups_names_by_their_resolved_path(store: MagicMock, monkeypatch):
    _write("example.com/t/a/index.md", "# A")
    _write("example.com/t/b/index.md", "# B")
    real_stat = migrate._file_stat

    def stat_without_inode(name: str) -> os.stat_result | None:
        stat = real_stat(name)
        assert stat is not None
        return os.stat_result((stat.st_mode, 0, *list(stat)[2:]))

    monkeypatch.setattr(migrate, "_file_stat", stat_without_inode)
    meta = {
        "https://example.com/t/a": CrawlMeta("example.com/t/a/index.md", content_hash("# A"), "1"),
        "https://example.com/t/b": CrawlMeta("example.com/t/b/index.md", content_hash("# B"), "1"),
    }
    assert sorted(_moved(store, meta)) == ["https://example.com/t/a", "https://example.com/t/b"]


def test_an_earlier_copy_beside_a_page_already_at_its_name_is_removed(store: MagicMock):
    name = url_to_filename(_PAGE)
    _write(name, "# Newer")
    _write(_EARLIER, "# Older")
    _write("example.com/t/page/keep.md", "# Mine")
    assert _moved(store, {_PAGE: CrawlMeta(_EARLIER, "h", "t")}) == {}
    assert not (_web() / _EARLIER).exists()
    assert (_web() / name).read_text(encoding="utf-8") == "# Newer"
    assert (_web() / "example.com/t/page/keep.md").exists()
    store.remove_documents.assert_called_once_with([f"_web/{_EARLIER}"])
    assert _moved_sources(store) == []
