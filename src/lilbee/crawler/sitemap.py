"""Best-effort ``/sitemap.xml`` lookup used as a progress-hint denominator."""

from __future__ import annotations

import re
from http import HTTPStatus
from urllib.parse import urlparse

from lilbee.crawler.url_filter import host_in_scope, require_valid_crawl_url
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
    # Validate the seed before any connection, and do not follow redirects: a
    # 3xx could otherwise steer this best-effort fetch to a private/metadata
    # host (SSRF) before the body is inspected. This is only a progress hint,
    # so a redirecting or unvalidated sitemap simply yields an unknown total.
    try:
        require_valid_crawl_url(sitemap_url)
    except ValueError:
        return None
    try:
        resp = httpx.get(
            sitemap_url, timeout=_SITEMAP_FETCH_TIMEOUT_SECONDS, follow_redirects=False
        )
    except (httpx.HTTPError, OSError):
        return None
    # Accept only a direct 2xx; an unfollowed 3xx (or any error status) yields
    # no usable sitemap and is treated as a miss.
    if not (HTTPStatus.OK <= resp.status_code < HTTPStatus.MULTIPLE_CHOICES):
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
