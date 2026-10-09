"""Convert HTML with crawl4ai alone: one JSON line in, one out; nothing is fetched."""

from __future__ import annotations

import asyncio

from _driver_io import serve_conversions
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.async_crawler_strategy import AsyncHTTPCrawlerStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

RAW_PREFIX = "raw:"


async def _cleaned_html(html: str) -> str:
    """The HTML crawl4ai keeps of a page, read through its ``raw:`` address form."""
    async with AsyncWebCrawler(
        crawler_strategy=AsyncHTTPCrawlerStrategy(), verbose=False
    ) as crawler:
        result = await crawler.arun(url=RAW_PREFIX + html, config=CrawlerRunConfig())
    if not result.success:
        raise RuntimeError(result.error_message or "crawl4ai gave no result")
    return str(result.cleaned_html or "")


def convert(html: str, base_url: str) -> str:
    """crawl4ai's markdown for *html*, cleaned and generated as in a crawl."""
    cleaned = asyncio.run(_cleaned_html(html))
    result = DefaultMarkdownGenerator().generate_markdown(cleaned, base_url=base_url)
    return str(result.raw_markdown or "")


if __name__ == "__main__":
    serve_conversions(convert)
