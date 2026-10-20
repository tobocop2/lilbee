"""A crawled page's own title: read from crawlberg, kept in the page record, read by sync."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from lilbee.core.config import cfg
from lilbee.core.config.enums import CrawlRenderMode
from lilbee.crawler import crawlberg_fetcher as fetcher_mod
from lilbee.crawler.crawlberg_fetcher import CrawlbergFetcher
from lilbee.crawler.events import _fetched_to_result
from lilbee.crawler.models import ConcurrencySpec, FetchedPage, FilterSpec
from lilbee.data.ingest.pipeline import _title_is_stale
from lilbee.data.page_record import CrawlMeta, page_titles, pages_path, record_line
from lilbee.data.store import SourceRecord
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

    async def test_a_title_over_lines_is_one_line(self):
        assert (await _single("\n  Bees \t and\n honey  ")).title == "Bees and honey"

    async def test_a_title_past_the_bound_is_cut(self):
        assert (await _single("x" * 2000)).title == "x" * 512

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


def _append(*lines: tuple[str, CrawlMeta]) -> None:
    with pages_path().open("a", encoding="utf-8") as out:
        out.writelines(f"{record_line(address, entry)}\n" for address, entry in lines)


class TestPageTitles:
    def test_no_record_gives_no_titles(self, data_dir: Path):
        assert page_titles() == {}

    def test_each_page_has_the_title_of_its_last_line(self, data_dir: Path):
        _append(
            (SEED + "a", CrawlMeta("example.com/a~1.md", "h", "t", title="Old")),
            (SEED + "a", CrawlMeta("example.com/a~1.md", "h", "t", title="New")),
            (SEED + "b", CrawlMeta("example.com/b~2.md", "h", "t")),
        )
        assert page_titles() == {"_web/example.com/a~1.md": "New"}

    def test_a_line_written_later_is_read(self, data_dir: Path):
        _append((SEED + "a", CrawlMeta("example.com/a~1.md", "h", "t", title="One")))
        assert page_titles() == {"_web/example.com/a~1.md": "One"}
        _append((SEED + "b", CrawlMeta("example.com/b~2.md", "h", "t", title="Two")))
        assert page_titles() == {
            "_web/example.com/a~1.md": "One",
            "_web/example.com/b~2.md": "Two",
        }

    def test_a_line_from_an_older_lilbee_has_no_title(self, data_dir: Path):
        pages_path().write_text(
            '{"url": "https://example.com/a", "file": "example.com/a~1.md",'
            ' "content_hash": "h", "crawled_at": "t", "replaces": ""}\n',
            encoding="utf-8",
        )
        assert page_titles() == {}


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
    def _record(self, data_dir: Path) -> None:
        _append((SEED + "a", CrawlMeta("example.com/a~1.md", "h", "t", title="Page A")))

    def test_a_row_with_the_stem_title_is_stale(self):
        assert _title_is_stale("_web/example.com/a~1.md", _row("a")) is True

    def test_a_row_with_the_record_title_is_not_stale(self):
        assert _title_is_stale("_web/example.com/a~1.md", _row("Page A")) is False

    def test_a_page_the_record_gives_no_title_is_not_stale(self):
        assert _title_is_stale("_web/example.com/b~2.md", _row("b")) is False

    def test_a_local_file_is_never_stale(self):
        assert _title_is_stale("notes/a.md", _row("a")) is False
