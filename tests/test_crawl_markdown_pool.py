"""Crawled HTML is converted in helper processes, not on the daemon's core."""

from __future__ import annotations

from concurrent.futures import BrokenExecutor

import pytest

from lilbee.crawler.markdown_pool import (
    MarkdownConversionPool,
    PooledMarkdownGenerator,
    _resolve_workers,
    _worker_convert,
)


class _Future:
    def __init__(self, value=None, exc: BaseException | None = None) -> None:
        self._value, self._exc = value, exc

    def result(self):
        if self._exc is not None:
            raise self._exc
        return self._value


class _FakeExecutor:
    """Stands in for a ProcessPoolExecutor without starting real processes."""

    def __init__(self, value=None, exc: BaseException | None = None) -> None:
        self._value, self._exc = value, exc
        self.shutdowns = 0

    def submit(self, _fn, *args):
        return _Future(self._value, self._exc)

    def shutdown(self, wait=True, cancel_futures=False) -> None:
        self.shutdowns += 1


class TestWorkerCount:
    """Conversion is CPU-bound, so more helpers than cores buys nothing."""

    def test_the_default_is_used_when_unset(self) -> None:
        assert _resolve_workers(None) >= 1

    def test_a_non_positive_count_falls_back_to_the_default(self) -> None:
        assert _resolve_workers(0) == _resolve_workers(None)
        assert _resolve_workers(-5) == _resolve_workers(None)

    def test_the_count_is_capped(self, monkeypatch) -> None:
        monkeypatch.setattr("os.cpu_count", lambda: 64)
        assert _resolve_workers(1000) == 8

    def test_it_never_exceeds_the_machine(self, monkeypatch) -> None:
        monkeypatch.setattr("os.cpu_count", lambda: 2)
        assert _resolve_workers(8) == 2

    def test_a_machine_that_will_not_say_still_gets_one(self, monkeypatch) -> None:
        monkeypatch.setattr("os.cpu_count", lambda: None)
        assert _resolve_workers(4) == 1


class TestThePoolNeverFailsACrawl:
    """Converting in-process is slower for the daemon but still correct, so every
    way the pool can fail degrades to that rather than losing the page."""

    def test_a_pool_that_cannot_start_returns_nothing_to_convert_with(self, monkeypatch) -> None:
        def _no_processes(**_kw):
            raise OSError("sandbox forbids new processes")

        monkeypatch.setattr("lilbee.crawler.markdown_pool.ProcessPoolExecutor", _no_processes)
        pool = MarkdownConversionPool(2)

        assert pool.convert("<p>x</p>", "", True) is None

    def test_it_stops_trying_after_a_failed_start(self, monkeypatch) -> None:
        """Retrying a start that cannot work would pay the failure per page."""
        starts: list[int] = []

        def _no_processes(**_kw):
            starts.append(1)
            raise OSError("nope")

        monkeypatch.setattr("lilbee.crawler.markdown_pool.ProcessPoolExecutor", _no_processes)
        pool = MarkdownConversionPool(2)
        for _ in range(3):
            pool.convert("<p>x</p>", "", True)

        assert len(starts) == 1

    def test_a_helper_dying_mid_crawl_falls_back_for_the_rest(self, monkeypatch) -> None:
        executor = _FakeExecutor(exc=BrokenExecutor("helper died"))
        monkeypatch.setattr(
            "lilbee.crawler.markdown_pool.ProcessPoolExecutor", lambda **_kw: executor
        )
        pool = MarkdownConversionPool(2)

        assert pool.convert("<p>x</p>", "", True) is None
        assert pool.convert("<p>y</p>", "", True) is None
        assert executor.shutdowns == 1

    def test_a_successful_conversion_is_returned(self, monkeypatch) -> None:
        executor = _FakeExecutor(value=("# raw", "# cited"))
        monkeypatch.setattr(
            "lilbee.crawler.markdown_pool.ProcessPoolExecutor", lambda **_kw: executor
        )
        pool = MarkdownConversionPool(2)

        assert pool.convert("<h1>raw</h1>", "", True) == ("# raw", "# cited")

    def test_shutdown_is_safe_before_anything_started(self, monkeypatch) -> None:
        """A daemon that never crawls must not start helpers, nor fail to stop."""
        started: list[int] = []
        monkeypatch.setattr(
            "lilbee.crawler.markdown_pool.ProcessPoolExecutor",
            lambda **_kw: started.append(1),
        )
        pool = MarkdownConversionPool(2)

        pool.shutdown()

        assert started == []

    def test_shutdown_twice_stops_the_helpers_once(self, monkeypatch) -> None:
        executor = _FakeExecutor(value=("x", "x"))
        monkeypatch.setattr(
            "lilbee.crawler.markdown_pool.ProcessPoolExecutor", lambda **_kw: executor
        )
        pool = MarkdownConversionPool(2)
        pool.convert("<p>x</p>", "", True)
        pool.shutdown()
        pool.shutdown()

        assert executor.shutdowns == 1


class TestTheGeneratorMatchesTheStockOne:
    """crawl4ai calls one method; anything the pool cannot reproduce exactly
    stays in-process rather than silently dropping options."""

    _HTML = "<html><body><h1>Title</h1><p>hello <a href='/x'>link</a></p></body></html>"

    def test_the_pooled_result_is_used(self, monkeypatch) -> None:
        class _Pool:
            def convert(self, *_a):
                return ("# pooled", "# pooled cited")

        result = PooledMarkdownGenerator(_Pool()).generate_markdown(self._HTML)

        assert str(result.raw_markdown) == "# pooled"
        assert str(result.markdown_with_citations) == "# pooled cited"

    def test_a_pool_that_declines_falls_back_in_process(self) -> None:
        class _Pool:
            def convert(self, *_a):
                return None

        result = PooledMarkdownGenerator(_Pool()).generate_markdown(self._HTML)

        assert "Title" in str(result.raw_markdown)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"content_filter": object()},
            {"html2text_options": {"ignore_links": True}},
            {"options": {"body_width": 80}},
        ],
        ids=["content_filter", "html2text_options", "options"],
    )
    def test_requests_the_pool_cannot_reproduce_stay_in_process(self, kwargs) -> None:
        class _Pool:
            def convert(self, *_a):
                raise AssertionError("the pool was used despite extra options")

        result = PooledMarkdownGenerator(_Pool()).generate_markdown(self._HTML, **kwargs)

        assert result is not None

    def test_the_pooled_markdown_matches_what_the_stock_generator_makes(self) -> None:
        """The whole point is a faster daemon, not different output."""
        from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

        stock = DefaultMarkdownGenerator().generate_markdown(self._HTML)
        raw, cited = _worker_convert(self._HTML, "", True)

        assert raw.strip() == str(stock.raw_markdown).strip()
        assert cited.strip() == str(stock.markdown_with_citations).strip()


class TestTheCrawlChoosesWhetherToPool:
    def test_zero_workers_keeps_the_conversion_in_the_server(self, monkeypatch) -> None:
        from lilbee.core.config import cfg
        from lilbee.crawler import crawl4ai_fetcher

        monkeypatch.setattr(cfg, "crawl_markdown_workers", 0)

        assert crawl4ai_fetcher._markdown_pool() is None

    def test_a_positive_count_builds_a_pool(self, monkeypatch) -> None:
        from lilbee.core.config import cfg
        from lilbee.crawler import crawl4ai_fetcher

        monkeypatch.setattr(cfg, "crawl_markdown_workers", 3)
        pool = crawl4ai_fetcher._markdown_pool()

        assert isinstance(pool, MarkdownConversionPool)
