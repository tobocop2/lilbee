"""Parametrized contract test for :class:`lilbee.crawler.fetcher.WebFetcher`.

Pins the behaviour any backend must exhibit: lifecycle hooks work,
``fetch_single`` returns a ``FetchedPage`` with the expected shape,
``fetch_recursive`` streams pages as they arrive, and cancel tokens
are honoured. Each backend and render mode is one entry in ``BACKENDS``,
with a stub that replays the same scripted pages through its SDK.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from lilbee.core.config.enums import CrawlRenderMode
from lilbee.crawler.crawl4ai_fetcher import Crawl4aiFetcher
from lilbee.crawler.crawlberg_fetcher import CrawlbergFetcher
from lilbee.crawler.fetcher import WebFetcher
from lilbee.crawler.models import ConcurrencySpec, FetchedPage, FilterSpec
from tests._crawlberg_stub import StubCrawlberg, error, page
from tests._sys_modules import inject_modules


@dataclass(frozen=True)
class Spec:
    """One scripted page: a success with *markdown*, or a failure when *error* is set."""

    url: str
    markdown: str = "# Hello"
    error: str | None = None


SpecStream = Callable[[], AsyncIterator[Spec]]


def _specs(*specs: Spec) -> SpecStream:
    async def replay() -> AsyncIterator[Spec]:
        for spec in specs:
            yield spec

    return replay


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
    mock_dispatcher.MemoryAdaptiveDispatcher = MagicMock()

    mock_strategy = MagicMock()
    mock_strategy.AsyncHTTPCrawlerStrategy = MagicMock()

    return {
        "crawl4ai": mock_crawl4ai,
        "crawl4ai.deep_crawling": mock_deep,
        "crawl4ai.deep_crawling.filters": mock_filters,
        "crawl4ai.async_dispatcher": mock_dispatcher,
        "crawl4ai.async_crawler_strategy": mock_strategy,
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


def _crawl4ai_result(spec: Spec) -> Any:
    result = _make_crawl4ai_result(url=spec.url, markdown=spec.markdown, success=spec.error is None)
    result.error_message = spec.error
    return result


@contextmanager
def _crawl4ai_stub(stream: SpecStream, *, single: bool) -> Iterator[None]:
    """Serve *stream* through a mocked crawl4ai ``AsyncWebCrawler``."""

    async def results() -> AsyncIterator[Any]:
        async for spec in stream():
            yield _crawl4ai_result(spec)

    async def first_result() -> Any:
        return await anext(aiter(results()), None)

    async def arun(**_kw: Any) -> Any:
        return await first_result() if single else results()

    instance = AsyncMock()
    instance.arun = AsyncMock(side_effect=arun)
    instance.__aenter__ = AsyncMock(return_value=instance)
    instance.__aexit__ = AsyncMock(return_value=False)
    with inject_modules(_mock_crawl4ai_modules(instance)):
        yield


@contextmanager
def _crawlberg_stub(stream: SpecStream, *, single: bool) -> Iterator[None]:
    """Serve *stream* through the stand-in crawlberg module."""

    async def events() -> AsyncIterator[dict[str, Any]]:
        async for spec in stream():
            if spec.error is not None:
                yield error(spec.url, spec.error)
            else:
                yield page(spec.url, spec.markdown, depth=0 if single else 1)

    with StubCrawlberg(events).installed():
        yield


@dataclass(frozen=True)
class Backend:
    """A fetcher under contract and the stub that feeds it scripted pages."""

    make: Callable[[], WebFetcher]
    stub: Callable[..., AbstractContextManager[None]]


@pytest.fixture(autouse=True)
def _stub_chromium(monkeypatch, tmp_path: Path):
    """Browser mode finds a Chromium on disk without one being installed."""
    shell = tmp_path / "chrome-headless-shell"
    monkeypatch.setattr("lilbee.crawler.bootstrap.chromium_installed", lambda: True)
    monkeypatch.setattr("lilbee.crawler.bootstrap.headless_shell_executable", lambda: shell)
    monkeypatch.setattr("lilbee.crawler.url_filter.validate_crawl_url", lambda url: None)
    monkeypatch.delenv("CHROME", raising=False)


# Every backend / mode lives in this table; each must pass the suite below.
BACKENDS: dict[str, Backend] = {
    "crawl4ai-http": Backend(
        lambda: Crawl4aiFetcher(quiet=True, render_mode=CrawlRenderMode.HTTP), _crawl4ai_stub
    ),
    "crawl4ai-browser": Backend(
        lambda: Crawl4aiFetcher(quiet=True, render_mode=CrawlRenderMode.BROWSER), _crawl4ai_stub
    ),
    "crawlberg-http": Backend(
        lambda: CrawlbergFetcher(render_mode=CrawlRenderMode.HTTP), _crawlberg_stub
    ),
    "crawlberg-browser": Backend(
        lambda: CrawlbergFetcher(render_mode=CrawlRenderMode.BROWSER), _crawlberg_stub
    ),
}


@pytest.fixture(params=list(BACKENDS))
def backend(request) -> Backend:
    return BACKENDS[request.param]


async def _single(backend: Backend, *specs: Spec) -> FetchedPage:
    with backend.stub(_specs(*specs), single=True):
        async with backend.make() as fetcher:
            return await fetcher.fetch_single("https://example.com", timeout=30.0)


async def _recursive(
    backend: Backend, stream: SpecStream, cancel: threading.Event | None = None
) -> list[FetchedPage]:
    received: list[FetchedPage] = []
    with backend.stub(stream, single=False):
        async with backend.make() as fetcher:
            async for fetched in fetcher.fetch_recursive(
                "https://example.com",
                depth=2,
                max_pages=100,
                timeout=30.0,
                concurrency=ConcurrencySpec(semaphore_count=1),
                filters=FilterSpec(),
                cancel=cancel,
            ):
                received.append(fetched)
    return received


class TestFetcherContract:
    """Every :class:`WebFetcher` implementation must satisfy these expectations."""

    async def test_fetcher_implements_protocol(self, backend):
        assert isinstance(backend.make(), WebFetcher)

    async def test_context_manager_round_trip(self, backend):
        """``async with fetcher`` must yield a ``WebFetcher`` and exit cleanly."""
        with backend.stub(_specs(), single=True):
            async with backend.make() as fetcher:
                assert isinstance(fetcher, WebFetcher)

    async def test_fetch_single_returns_markdown(self, backend):
        fetched = await _single(backend, Spec("https://example.com", markdown="# Test"))
        assert isinstance(fetched, FetchedPage)
        assert fetched.success is True
        assert fetched.markdown == "# Test"

    async def test_fetch_single_reports_failure_cleanly(self, backend):
        fetched = await _single(backend, Spec("https://example.com", markdown="", error="404"))
        assert fetched.success is False
        assert fetched.error == "404"

    async def test_fetch_recursive_yields_pages(self, backend):
        stream = _specs(
            Spec("https://example.com/a", markdown="# A"),
            Spec("https://example.com/b", markdown="# B"),
        )
        received = await _recursive(backend, stream)
        assert [p.url for p in received] == ["https://example.com/a", "https://example.com/b"]
        assert all(p.success for p in received)

    async def test_fetch_recursive_honours_cancel_token(self, backend):
        cancel = threading.Event()

        async def stream() -> AsyncIterator[Spec]:
            for i in range(1, 6):
                await asyncio.sleep(0)
                yield Spec(f"https://example.com/p{i}")
                if i == 2:
                    cancel.set()

        received = await _recursive(backend, stream, cancel)
        # Cancel stops the stream promptly; we never see all five pages.
        assert len(received) <= 2

    async def test_fetch_recursive_translates_failure_pages(self, backend):
        stream = _specs(
            Spec("https://example.com/a", markdown="# A"),
            Spec("https://example.com/b", markdown="", error="boom"),
        )
        received = await _recursive(backend, stream)
        assert received[0].success is True
        assert received[1].success is False
        assert received[1].error == "boom"
