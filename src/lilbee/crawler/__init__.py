"""Web crawling: fetch pages as markdown and save them to the documents directory."""

from __future__ import annotations

import os

from lilbee.crawler.bootstrap import (
    CrawlerBackendError,
    CrawlerBrowserError,
    bootstrap_chromium,
    chromium_installed,
    crawler_browsers_path,
)
from lilbee.crawler.crawl4ai_fetcher import crawler_available
from lilbee.crawler.fetcher import WebFetcher
from lilbee.crawler.models import (
    CancelToken,
    ConcurrencySpec,
    CrawlResult,
    FetchedPage,
    FilterSpec,
)
from lilbee.crawler.runner import (
    crawl_and_save,
    crawl_recursive,
    crawl_single,
)
from lilbee.crawler.save import (
    METADATA_FLUSH_INTERVAL,
    CrawlMeta,
    content_hash,
    load_crawl_metadata,
    save_crawl_metadata,
    url_to_filename,
)
from lilbee.crawler.url_filter import (
    get_blocked_networks,
    is_url,
    require_valid_crawl_url,
    validate_crawl_url,
)

__all__ = [
    "METADATA_FLUSH_INTERVAL",
    "CancelToken",
    "ConcurrencySpec",
    "CrawlMeta",
    "CrawlResult",
    "CrawlerBackendError",
    "CrawlerBrowserError",
    "FetchedPage",
    "FilterSpec",
    "WebFetcher",
    "bootstrap_chromium",
    "chromium_installed",
    "content_hash",
    "crawl_and_save",
    "crawl_recursive",
    "crawl_single",
    "crawler_available",
    "crawler_browsers_path",
    "get_blocked_networks",
    "is_url",
    "load_crawl_metadata",
    "require_valid_crawl_url",
    "save_crawl_metadata",
    "url_to_filename",
    "validate_crawl_url",
]

# Pin Playwright's browser cache so install and launch agree on Chromium's
# location, regardless of wheel vs frozen-binary layout.
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(crawler_browsers_path()))
