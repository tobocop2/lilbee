"""Crawl with crawlberg alone, with the settings lilbee's adapter passes; no lilbee is imported."""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import aclosing

import crawlberg
from _driver_io import PageOut, RunOut, crawl_arguments, now, thread_count, write_pages, write_run

STARTED = now()
BROWSER_MODES = {"http": "never", "browser": "always"}
USER_AGENTS = {
    "http": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "browser": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/116.0.0.0 Safari/537.36",
}
TRACKING_PARAMS = ["utm_*", "fbclid", "gclid", "ref"]
MAX_REDIRECTS = 10
MAX_CONCURRENT = 3
MS_PER_SECOND = 1000
SETTLE_SECONDS = 0.5


def crawl_config(
    mode: str, depth: int, timeout: float, chrome: str | None
) -> crawlberg.CrawlConfig:
    """The crawl settings of lilbee's adapter, with loopback allowed and no delay."""
    timeout_ms = round(timeout * MS_PER_SECOND)
    browser = mode == "browser"
    return crawlberg.CrawlConfig(
        max_depth=depth,
        max_concurrent=MAX_CONCURRENT,
        stay_on_domain=True,
        allow_subdomains=False,
        path_patterns_match_url=True,
        dedup_include_query=True,
        strip_tracking_params=True,
        tracking_params=TRACKING_PARAMS,
        request_timeout=timeout_ms,
        user_agent=USER_AGENTS[mode],
        rate_limit_ms=0,
        max_redirects=MAX_REDIRECTS,
        retry_count=0,
        respect_robots_txt=False,
        soft_http_errors=False,
        download_documents=False,
        content=crawlberg.ContentConfig(
            remove_navigation=False,
            remove_forms=False,
            exclude_selectors=[],
            preprocessing_preset="minimal",
            extract_metadata=False,
        ),
        browser=crawlberg.BrowserConfig(
            mode=BROWSER_MODES[mode],
            timeout=timeout_ms,
            overall_timeout=timeout_ms * 2,
            shutdown_timeout=5000,
            chrome_path=chrome if browser else None,
        ),
        ssrf=crawlberg.SsrfPolicy(
            deny_private=True,
            allowlist=[
                crawlberg.HostMatcher.cidr("127.0.0.0/8"),
                crawlberg.HostMatcher.cidr("::1/128"),
            ],
            max_redirects=MAX_REDIRECTS,
        ),
    )


def page_of(raw: object) -> PageOut | None:
    """The page of one crawl event, or None for an event that carries no page."""
    payload = json.loads(str(raw))
    if payload["type"] == "error":
        return PageOut(payload["url"], None, payload["error"])
    if payload["type"] != "page":
        return None
    result = payload["result"]
    markdown = (result.get("markdown") or {}).get("content") or ""
    if markdown.strip():
        return PageOut(result["url"], markdown, saved_at=now())
    return PageOut(result["url"], None, "No content extracted")


async def crawl(config: crawlberg.CrawlConfig, seed: str) -> list[PageOut]:
    """Every page crawlberg reaches from *seed*."""
    pages: list[PageOut] = []
    engine = crawlberg.create_engine(config)
    async with aclosing(crawlberg.crawl_stream(engine, seed)) as stream:
        async for raw in stream:
            page = page_of(raw)
            if page is not None:
                pages.append(page)
    del engine
    await asyncio.sleep(SETTLE_SECONDS)
    return pages


def run() -> int:
    """Crawl and write the result files."""
    args = crawl_arguments(__doc__ or "")
    record = RunOut(started=STARTED, versions={"crawlberg": crawlberg.__version__})
    record.threads_before = thread_count()
    record.crawl_started = now()
    config = crawl_config(args.mode, args.depth, args.timeout, args.chrome)
    pages = asyncio.run(crawl(config, args.seed))
    record.crawl_ended = now()
    record.threads_after = thread_count()
    write_pages(args.out, pages)
    write_run(args.out, record)
    return 0


if __name__ == "__main__":
    sys.exit(run())
