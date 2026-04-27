"""Web crawling. Fetch pages as markdown and save to the documents directory.

This package is the public face of lilbee's crawling subsystem. All
callers (``cli/commands.py``, ``mcp.py``, ``server/handlers.py``,
``cli/tui/screens/chat.py``, ``crawl_task.py``, ``server/routes/setup.py``)
import symbols from here.

Layout:

- :mod:`lilbee.crawler.models`: value types (``CrawlResult``, ``FetchedPage``,
  specs)
- :mod:`lilbee.crawler.fetcher`: ``WebFetcher`` Protocol
- :mod:`lilbee.crawler.url_filter`: URL validation + host scope
- :mod:`lilbee.crawler.sitemap`: best-effort sitemap progress hint
- :mod:`lilbee.crawler.bootstrap`: Playwright Chromium install + detection
- :mod:`lilbee.crawler.save`: URL-to-filename, metadata I/O, per-page save
- :mod:`lilbee.crawler.discovery`: ``cfg`` -> backend-neutral concurrency /
  filter spec builders
- :mod:`lilbee.crawler.events`: per-page event emission, result translation,
  cancel-teardown classification
- :mod:`lilbee.crawler.runner`: orchestration (``crawl_single``,
  ``crawl_recursive``, ``crawl_and_save``)
- :mod:`lilbee.crawler.crawl4ai_fetcher`: crawl4ai-backed ``WebFetcher``.
  ONLY file importing ``crawl4ai``; the swap point for a future backend.
"""

from __future__ import annotations

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
