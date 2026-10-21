"""The crawlberg backend: builds the fetcher a crawl runs on."""

from __future__ import annotations

from lilbee.core.config import cfg
from lilbee.core.config.enums import CrawlRenderMode
from lilbee.crawler.backends.crawlberg.fetcher import CrawlbergFetcher
from lilbee.crawler.fetcher import WebFetcher
from lilbee.crawler.models import CrawlLogin

__all__ = ["fetcher_from_config"]


def fetcher_from_config(render_mode: CrawlRenderMode, login: CrawlLogin) -> WebFetcher:
    """The fetcher for one crawl, with its browser, user agent, robots.txt and login settings."""
    return CrawlbergFetcher(
        render_mode=render_mode,
        chrome_args=cfg.crawl_browser_extra_args,
        user_agent=cfg.crawl_user_agent,
        respect_robots_txt=cfg.crawl_respect_robots_txt,
        extra_wait_ms=cfg.crawl_browser_extra_wait_ms,
        login=login,
    )
