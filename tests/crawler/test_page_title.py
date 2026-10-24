"""A crawled page's own title: read from crawlberg, kept in the crawl page table, read by sync."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from lilbee.core.config import cfg
from lilbee.core.config.enums import CrawlRenderMode
from lilbee.crawler.backends.crawlberg import fetcher as fetcher_mod
from lilbee.crawler.backends.crawlberg.fetcher import CrawlbergFetcher
from lilbee.crawler.events import _fetched_to_result
from lilbee.crawler.models import ConcurrencySpec, FetchedPage, FilterSpec
from lilbee.data.ingest import pipeline
from lilbee.data.ingest.pipeline import _title_is_stale
from lilbee.data.store import SourceRecord, Store
from tests._crawl_meta import CrawlMeta, claim_pages, crawl_store
from tests._crawlberg_stub import StubCrawlberg, complete, page

SEED = "https://example.com/"
NO_PACING = ConcurrencySpec()
LONG_TITLE = " ".join(["Growing tomatoes indoors under lights in winter"] * 6)


@pytest.fixture(autouse=True)
def _admit_every_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetcher_mod.url_filter, "validate_crawl_url", lambda url: None)


@pytest.fixture
def data_dir(tmp_path: Path) -> Iterator[Path]:
    snapshot = cfg.model_copy()
    cfg.data_dir = tmp_path / "data"
    cfg.data_dir.mkdir()
    cfg.lancedb_dir = cfg.data_dir / "lancedb"
    yield cfg.data_dir
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


async def _single(title: str | None) -> FetchedPage:
    stub = StubCrawlberg([page(SEED, "# Hello", depth=0, title=title), complete(1)])
    with stub.installed():
        fetcher = CrawlbergFetcher(render_mode=CrawlRenderMode.HTTP)
        return await fetcher.fetch_single(SEED, timeout=3, concurrency=NO_PACING)


class TestFetchedTitle:
    async def test_a_long_title_is_kept_in_full(self):
        assert len(LONG_TITLE.encode()) > 99
        assert (await _single(LONG_TITLE)).title == LONG_TITLE

    async def test_a_non_ascii_title_is_kept(self):
        assert (await _single("東京都 - Wikipedia")).title == "東京都 - Wikipedia"

    async def test_a_page_with_no_title_has_an_empty_title(self):
        fetched = await _single(None)
        assert (fetched.title, fetched.markdown) == ("", "# Hello")

    async def test_the_backend_hands_over_the_title_as_the_page_has_it(self):
        raw = "\n  Bees \t and\n honey  " + "x" * 2000
        assert (await _single(raw)).title == raw

    async def test_every_page_of_a_crawl_keeps_its_title(self):
        other = SEED + "b"
        script = [
            page(SEED, "# A", depth=0, title="Page A"),
            page(other, "# B", depth=1, title="Page B"),
            complete(2),
        ]
        with StubCrawlberg(script).installed():
            fetcher = CrawlbergFetcher(render_mode=CrawlRenderMode.HTTP)
            stream = fetcher.fetch_recursive(
                SEED,
                depth=1,
                max_pages=5,
                timeout=3,
                concurrency=NO_PACING,
                filters=FilterSpec(),
            )
            titles = {fetched.url: fetched.title async for fetched in stream}
        assert titles == {SEED: "Page A", other: "Page B"}

    def test_the_crawl_result_carries_the_title(self):
        result = _fetched_to_result(FetchedPage(url=SEED, markdown="# A", title="Page A"))
        assert result.title == "Page A"

    def test_the_crawl_result_title_is_one_line(self):
        fetched = FetchedPage(url=SEED, markdown="# A", title="\n  Bees \t and\n honey  ")
        assert _fetched_to_result(fetched).title == "Bees and honey"

    def test_the_crawl_result_title_is_cut_at_the_bound(self):
        fetched = FetchedPage(url=SEED, markdown="# A", title="x" * 2000)
        assert _fetched_to_result(fetched).title == "x" * 512


def _titles() -> dict[str, str]:
    return crawl_store().crawl_pages.titles()


class TestPageTitles:
    def test_no_table_gives_no_titles(self, data_dir: Path):
        assert _titles() == {}

    def test_each_page_has_the_title_of_its_last_claim(self, data_dir: Path):
        claim_pages({SEED + "a": CrawlMeta("example.com/a~1.md", "h", "t", title="Old")})
        claim_pages(
            {
                SEED + "a": CrawlMeta("example.com/a~1.md", "h", "t", title="New"),
                SEED + "b": CrawlMeta("example.com/b~2.md", "h", "t"),
            }
        )
        assert _titles() == {"_web/example.com/a~1.md": "New"}

    def test_a_claim_made_later_is_read_by_the_same_store(self, data_dir: Path):
        store = crawl_store()
        claim_pages({SEED + "a": CrawlMeta("example.com/a~1.md", "h", "t", title="One")})
        assert store.crawl_pages.titles() == {"_web/example.com/a~1.md": "One"}
        claim_pages({SEED + "b": CrawlMeta("example.com/b~2.md", "h", "t", title="Two")})
        assert store.crawl_pages.titles() == {
            "_web/example.com/a~1.md": "One",
            "_web/example.com/b~2.md": "Two",
        }

    def test_a_page_an_older_lilbee_recorded_has_no_title(self, data_dir: Path):
        (data_dir / "crawl_meta.json").write_text(
            '{"https://example.com/a": {"file": "example.com/a~1.md",'
            ' "content_hash": "h", "crawled_at": "t"}}',
            encoding="utf-8",
        )
        assert [row.file for row in crawl_store().crawl_pages.rows()] == ["example.com/a~1.md"]
        assert _titles() == {}


def _row(title: str | None) -> SourceRecord:
    return SourceRecord(
        filename="_web/example.com/a~1.md",
        file_hash="h",
        ingested_at="t",
        chunk_count=1,
        source_type="document",
        title=title,
    )


class TestTitleIsStale:
    @pytest.fixture(autouse=True)
    def _record(self, data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        claim_pages({SEED + "a": CrawlMeta("example.com/a~1.md", "h", "t", title="Page A")})
        services = SimpleNamespace(store=Store(cfg))
        monkeypatch.setattr(pipeline, "get_services", lambda: services)

    def test_a_row_with_the_stem_title_is_stale(self):
        assert (
            _title_is_stale("_web/example.com/a~1.md", _row("a"), pipeline._page_titles()) is True
        )

    def test_a_row_with_the_record_title_is_not_stale(self):
        assert (
            _title_is_stale("_web/example.com/a~1.md", _row("Page A"), pipeline._page_titles())
            is False
        )

    def test_a_page_the_record_gives_no_title_is_not_stale(self):
        assert (
            _title_is_stale("_web/example.com/b~2.md", _row("b"), pipeline._page_titles()) is False
        )

    def test_a_local_file_is_never_stale(self):
        assert _title_is_stale("notes/a.md", _row("a"), pipeline._page_titles()) is False


def test_a_list_with_no_crawled_page_starts_no_services(monkeypatch: pytest.MonkeyPatch):
    from lilbee.app import titles

    started: list[bool] = []

    def services() -> SimpleNamespace:
        started.append(True)
        return SimpleNamespace(
            store=SimpleNamespace(web_titles=lambda pages: dict.fromkeys(pages, "T"))
        )

    monkeypatch.setattr(titles, "get_services", services)
    assert titles.stored_web_titles(["notes.md", "memory:m1"]) == {}
    assert started == []
    page = "_web/example.com/a.md"
    assert titles.stored_web_titles(["notes.md", page]) == {page: "T"}
    assert started == [True]
