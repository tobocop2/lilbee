"""Crawl with crawl4ai alone, with the settings lilbee's adapter passed; no lilbee is imported."""

from __future__ import annotations

import asyncio
import importlib.metadata
import inspect
import re
import sys
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlparse

from _driver_io import PageOut, RunOut, crawl_arguments, now, thread_count, write_pages, write_run
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.async_crawler_strategy import AsyncHTTPCrawlerStrategy
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

STARTED = now()
BASE_HREF = re.compile(r"<base\s[^>]*href\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
MAX_CONCURRENT = 3
MS_PER_SECOND = 1000


class SameHost(URLFilter):  # type: ignore[misc]
    """Keeps a crawl on the host of its seed."""

    def __init__(self, seed: str) -> None:
        super().__init__()
        self._host = (urlparse(seed).hostname or "").lower()

    def apply(self, url: str) -> bool:
        in_scope = (urlparse(url).hostname or "").lower() == self._host
        self._update_stats(in_scope)
        return in_scope


def new_crawler(mode: str) -> AsyncWebCrawler:
    """The crawler lilbee built for *mode*."""
    if mode == "http":
        return AsyncWebCrawler(crawler_strategy=AsyncHTTPCrawlerStrategy(), verbose=False)
    config = BrowserConfig(light_mode=True, text_mode=True, memory_saving_mode=True, verbose=False)
    return AsyncWebCrawler(config=config, verbose=False)


def convert(html: str, base_url: str) -> str:
    """crawl4ai's markdown for cleaned HTML."""
    result = DefaultMarkdownGenerator().generate_markdown(html, base_url=base_url)
    return str(result.raw_markdown or "")


def page_of(result: Any) -> PageOut:
    """The page of one crawl4ai result, converted the way lilbee converted it."""
    if not result.success:
        return PageOut(result.url, None, result.error_message or "Unknown error")
    base = BASE_HREF.search(result.html or "")
    base_url = base.group(1) if base else (result.redirected_url or result.url)
    markdown = convert(result.cleaned_html or "", base_url) if result.cleaned_html else ""
    if markdown.strip():
        return PageOut(result.url, markdown, saved_at=now())
    return PageOut(result.url, None, result.error_message or "No content extracted")


async def results_of(stream: Any) -> AsyncIterator[Any]:
    """crawl4ai returns an async generator, a list or one result; yield each result."""
    if inspect.isasyncgen(stream):
        async for item in stream:
            yield item
    elif isinstance(stream, list):  # the batch shape of an untyped return
        for item in stream:
            yield item
    else:
        yield stream


async def crawl(mode: str, seed: str, depth: int, timeout: float) -> list[PageOut]:
    """Every page crawl4ai reaches from *seed*."""
    settings: dict[str, Any] = {"page_timeout": int(timeout * MS_PER_SECOND)}
    if depth > 0:
        settings.update(
            deep_crawl_strategy=BFSDeepCrawlStrategy(
                max_depth=depth, filter_chain=FilterChain([SameHost(seed)])
            ),
            mean_delay=0.0,
            max_range=0.0,
            semaphore_count=MAX_CONCURRENT,
            stream=True,
        )
    pages: list[PageOut] = []
    async with new_crawler(mode) as crawler:
        stream = await crawler.arun(url=seed, config=CrawlerRunConfig(**settings))
        async for result in results_of(stream):
            pages.append(page_of(result))
    return pages


def run() -> int:
    """Crawl and write the result files."""
    args = crawl_arguments(__doc__ or "")
    versions = {"crawl4ai": importlib.metadata.version("crawl4ai")}
    record = RunOut(started=STARTED, versions=versions, threads_before=thread_count())
    record.crawl_started = now()
    pages = asyncio.run(crawl(args.mode, args.seed, args.depth, args.timeout))
    record.crawl_ended = now()
    record.threads_after = thread_count()
    write_pages(args.out, pages)
    write_run(args.out, record)
    return 0


if __name__ == "__main__":
    sys.exit(run())
