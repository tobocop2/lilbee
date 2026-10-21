"""The words a crawl shows a user when it cannot run, or when a site or crawlberg refuses it."""

from __future__ import annotations

CHROMIUM_MISSING_MESSAGE = (
    "The headless Chromium browser is not installed. "
    "Run 'lilbee setup crawler' to enable browser-mode crawling."
)
# ``{reason}`` is crawlberg's own error text.
CRAWL_REFUSED_MESSAGE = "crawlberg refuses to start this crawl: {reason}"
BROWSER_FLAGS_REFUSED_MESSAGE = (
    "The crawl_browser_extra_args setting holds a launch flag that crawlberg refuses: {reason}"
)
EXCLUDE_PATTERN_REFUSED_MESSAGE = (
    "The crawl_exclude_patterns setting holds a pattern that crawlberg refuses: {reason}"
)
ROBOTS_REFUSED_MESSAGE = (
    "The site's robots.txt does not allow crawlers to fetch this page. "
    "Set crawl_respect_robots_txt to false to crawl it."
)
ROBOTS_UNREACHABLE_MESSAGE = (
    "The site's robots.txt could not be read ({reason}), so no page of the site is crawled. "
    "Set crawl_respect_robots_txt to false to crawl the site without reading its robots.txt."
)
