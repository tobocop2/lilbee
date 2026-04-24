"""Parametrized contract test for :class:`lilbee.crawler.fetcher.WebFetcher`.

Pins the behaviour any backend must exhibit: lifecycle hooks work,
``fetch_single`` returns a ``FetchedPage`` with the expected shape,
``fetch_recursive`` streams pages as they arrive, and cancel tokens
are honoured. When a future backend (e.g. ``KreuzcrawlFetcher``)
lands, it gets added to the ``FETCHER_FACTORIES`` parametrization.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lilbee.crawler.crawl4ai_fetcher import Crawl4aiFetcher
from lilbee.crawler.fetcher import WebFetcher
from lilbee.crawler.models import ConcurrencySpec, FetchedPage, FilterSpec


def _mock_crawl4ai_modules(instance: Any) -> dict[str, Any]:
    """Build the ``sys.modules`` override shape that ``Crawl4aiFetcher`` needs.

    Mirrors the fixture used in ``tests/test_crawler.py`` so the contract
    test exercises the crawl4ai adapter without requiring the ``crawler``
    extra to be installed in the unit-test env.
    """
    mock_crawl4ai = MagicMock()
    mock_crawl4ai.AsyncWebCrawler = MagicMock(return_value=instance)
    mock_crawl4ai.CrawlerRunConfig = MagicMock()

    mock_deep = MagicMock()
    mock_deep.BFSDeepCrawlStrategy = MagicMock()

    mock_filters = MagicMock()
    mock_filters.FilterChain = MagicMock()
    mock_filters.URLPatternFilter = MagicMock()
    mock_filters.URLFilter = MagicMock
    mock_filters.DomainFilter = MagicMock

    mock_dispatcher = MagicMock()
    mock_dispatcher.RateLimiter = MagicMock()
    mock_dispatcher.SemaphoreDispatcher = MagicMock()

    return {
        "crawl4ai": mock_crawl4ai,
        "crawl4ai.deep_crawling": mock_deep,
        "crawl4ai.deep_crawling.filters": mock_filters,
        "crawl4ai.async_dispatcher": mock_dispatcher,
    }


def _make_crawl4ai_result(
    url: str = "https://example.com", markdown: str = "# Hello", success: bool = True
) -> Any:
    result = MagicMock()
    result.url = url
    result.markdown = markdown
    result.success = success
    result.error_message = None
    return result


@pytest.fixture(autouse=True)
def _stub_chromium(monkeypatch):
    """Crawl4aiFetcher calls ``chromium_installed`` before opening AsyncWebCrawler."""
    monkeypatch.setattr("lilbee.crawler.bootstrap.chromium_installed", lambda: True)


FetcherFactory = Callable[[], WebFetcher]


def _crawl4ai_factory() -> WebFetcher:
    """Build a :class:`Crawl4aiFetcher` for the contract run."""
    return Crawl4aiFetcher(quiet=True)


# Every backend implementation lives in this list; each must pass the
# suite below. When a ``KreuzcrawlFetcher`` lands, add a second factory.
FETCHER_FACTORIES: list[tuple[str, FetcherFactory]] = [
    ("crawl4ai", _crawl4ai_factory),
]


@pytest.fixture(params=FETCHER_FACTORIES, ids=[name for name, _ in FETCHER_FACTORIES])
def fetcher_factory(request) -> FetcherFactory:
    return request.param[1]


class TestFetcherContract:
    """Every :class:`WebFetcher` implementation must satisfy these expectations."""

    async def test_fetcher_implements_protocol(self, fetcher_factory):
        instance = fetcher_factory()
        assert isinstance(instance, WebFetcher)

    async def test_context_manager_round_trip(self, fetcher_factory):
        """``async with fetcher`` must yield a ``WebFetcher`` and exit cleanly."""
        mock_instance = AsyncMock()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        with patch.dict("sys.modules", _mock_crawl4ai_modules(mock_instance)):
            async with fetcher_factory() as f:
                assert isinstance(f, WebFetcher)
                assert hasattr(f, "fetch_single")
                assert hasattr(f, "fetch_recursive")

    async def test_fetch_single_returns_markdown(self, fetcher_factory):
        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=_make_crawl4ai_result(markdown="# Test"))
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        with patch.dict("sys.modules", _mock_crawl4ai_modules(mock_instance)):
            async with fetcher_factory() as f:
                page = await f.fetch_single("https://example.com", timeout=30.0)
        assert isinstance(page, FetchedPage)
        assert page.success is True
        assert page.markdown == "# Test"

    async def test_fetch_single_reports_failure_cleanly(self, fetcher_factory):
        failed = _make_crawl4ai_result(markdown="", success=False)
        failed.error_message = "404"
        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=failed)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        with patch.dict("sys.modules", _mock_crawl4ai_modules(mock_instance)):
            async with fetcher_factory() as f:
                page = await f.fetch_single("https://example.com", timeout=30.0)
        assert page.success is False
        assert page.error == "404"

    async def test_fetch_recursive_yields_pages(self, fetcher_factory):
        results = [
            _make_crawl4ai_result(url="https://example.com/a", markdown="# A"),
            _make_crawl4ai_result(url="https://example.com/b", markdown="# B"),
        ]
        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=results)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        received: list[FetchedPage] = []
        with patch.dict("sys.modules", _mock_crawl4ai_modules(mock_instance)):
            async with fetcher_factory() as f:
                async for page in f.fetch_recursive(
                    "https://example.com",
                    depth=1,
                    max_pages=5,
                    timeout=30.0,
                    concurrency=ConcurrencySpec(semaphore_count=1),
                    filters=FilterSpec(),
                ):
                    received.append(page)
        assert [p.url for p in received] == [
            "https://example.com/a",
            "https://example.com/b",
        ]
        assert all(p.success for p in received)

    async def test_fetch_recursive_honours_cancel_token(self, fetcher_factory):
        cancel = threading.Event()

        async def _gen() -> AsyncIterator[Any]:
            for i in range(1, 6):
                await asyncio.sleep(0)
                yield _make_crawl4ai_result(url=f"https://example.com/p{i}")
                if i == 2:
                    cancel.set()

        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=_gen())
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        received: list[FetchedPage] = []
        with patch.dict("sys.modules", _mock_crawl4ai_modules(mock_instance)):
            async with fetcher_factory() as f:
                async for page in f.fetch_recursive(
                    "https://example.com",
                    depth=2,
                    max_pages=100,
                    timeout=30.0,
                    concurrency=ConcurrencySpec(semaphore_count=1),
                    filters=FilterSpec(),
                    cancel=cancel,
                ):
                    received.append(page)
        # Cancel stops the stream promptly; we never see all five pages.
        assert len(received) <= 2

    async def test_fetch_recursive_translates_failure_pages(self, fetcher_factory):
        ok = _make_crawl4ai_result(url="https://example.com/a", markdown="# A")
        fail = _make_crawl4ai_result(url="https://example.com/b", success=False)
        fail.error_message = "boom"
        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=[ok, fail])
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        received: list[FetchedPage] = []
        with patch.dict("sys.modules", _mock_crawl4ai_modules(mock_instance)):
            async with fetcher_factory() as f:
                async for page in f.fetch_recursive(
                    "https://example.com",
                    depth=1,
                    max_pages=5,
                    timeout=30.0,
                    concurrency=ConcurrencySpec(semaphore_count=1),
                    filters=FilterSpec(),
                ):
                    received.append(page)
        assert received[0].success is True
        assert received[1].success is False
        assert received[1].error == "boom"
