"""Best-effort sitemap.xml lookups used as a progress-hint denominator.

Pure HTTP + regex: fetches ``/sitemap.xml`` at the root of the starting
host, counts ``<loc>`` entries matching the crawl scope, and returns
the count. Returns ``CRAWL_TOTAL_UNKNOWN`` on any failure so the
orchestrator can render ``[n/?]`` instead of a hard-coded ceiling.

Not load-bearing: correctness is best-effort and every branch falls
back cleanly on error.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from lilbee.crawler.url_filter import host_in_scope
from lilbee.runtime.progress import CRAWL_TOTAL_UNKNOWN

# Sitemap lookups are best-effort progress hints; never block the actual crawl.
_SITEMAP_FETCH_TIMEOUT_SECONDS = 5.0
_SITEMAP_MAX_URLS = 10_000
_SITEMAP_URL_TAG_RE = re.compile(r"<loc>\s*([^<]+?)\s*</loc>", re.IGNORECASE)


def _fetch_sitemap_text(start_url: str) -> str | None:
    """Return sitemap.xml body or None on any fetch/status failure."""
    import httpx

    parsed = urlparse(start_url)
    sitemap_url = f"{parsed.scheme}://{parsed.netloc}/sitemap.xml"
    try:
        resp = httpx.get(sitemap_url, timeout=_SITEMAP_FETCH_TIMEOUT_SECONDS, follow_redirects=True)
    except (httpx.HTTPError, OSError):
        return None
    if resp.status_code >= 400:
        return None
    return resp.text


def _count_sitemap_urls(start_url: str, *, include_subdomains: bool) -> int:
    """Best-effort count of URLs in the host's /sitemap.xml that match the crawl scope.

    Returns ``CRAWL_TOTAL_UNKNOWN`` on any failure (missing sitemap, timeout,
    parse error, redirect away from the starting host). This is purely a
    progress-hint denominator, so correctness is not load-bearing.

    Only fetches sitemap.xml directly at the root of the starting host; does
    not follow robots.txt references or nested sitemap indexes.
    """
    host = (urlparse(start_url).hostname or "").lower()
    if not host:
        return CRAWL_TOTAL_UNKNOWN
    text = _fetch_sitemap_text(start_url)
    if text is None:
        return CRAWL_TOTAL_UNKNOWN

    count = 0
    for match in _SITEMAP_URL_TAG_RE.finditer(text):
        link_host = (urlparse(match.group(1).strip()).hostname or "").lower()
        if host_in_scope(link_host, host, include_subdomains=include_subdomains):
            count += 1
        if count >= _SITEMAP_MAX_URLS:
            break
    return count if count > 0 else CRAWL_TOTAL_UNKNOWN
