"""Best-effort ``/sitemap.xml`` lookup used as a progress-hint denominator."""

from __future__ import annotations

import re
from http import HTTPStatus
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse
from urllib.robotparser import RobotFileParser

from lilbee.crawler.models import NO_LOGIN, CrawlLogin
from lilbee.crawler.url_filter import host_in_scope, parsed_url, require_valid_crawl_url
from lilbee.runtime.progress import CRAWL_TOTAL_UNKNOWN

# ada-url is in the optional ``crawler`` extra, so it is imported on call.
if TYPE_CHECKING:
    import ada_url
    import httpx

# Sitemap lookups are best-effort progress hints; never block the actual crawl.
_SITEMAP_FETCH_TIMEOUT_SECONDS = 5.0
_SITEMAP_MAX_URLS = 10_000
_SITEMAP_URL_TAG_RE = re.compile(r"<loc>\s*([^<]+?)\s*</loc>", re.IGNORECASE)
_USER_AGENT_HEADER = "User-Agent"
_SITEMAP_PATH = "/sitemap.xml"
_ROBOTS_PATH = "/robots.txt"
_ROBOTS_ABSENT_STATUSES = frozenset({HTTPStatus.NOT_FOUND, HTTPStatus.GONE})
_ANY_USER_AGENT = "*"


def _basic_credentials(seed: ada_url.URL) -> tuple[str, str] | None:
    """The decoded username and password of *seed*, or None when it has neither."""
    if not seed.username and not seed.password:
        return None
    return unquote(seed.username), unquote(seed.password)


def _get(url: str, seed: ada_url.URL, user_agent: str, login: CrawlLogin) -> httpx.Response | None:
    """One GET of *url* with the seed's credentials and *login*, or None on a transport failure.

    The seed's own username and password win over *login*'s, as in crawlberg.

    Redirects are not followed: a 3xx could steer this best-effort request to a
    private or metadata host (SSRF) before the body is inspected.
    """
    import httpx

    try:
        return httpx.get(
            url,
            auth=_basic_credentials(seed) or login.basic_credentials,
            headers=_headers(user_agent, login),
            timeout=_SITEMAP_FETCH_TIMEOUT_SECONDS,
            follow_redirects=False,
        )
    except (httpx.HTTPError, OSError):
        return None


def _headers(user_agent: str, login: CrawlLogin) -> dict[str, str]:
    """The user agent, when set, and the authorization and cookie headers of *login*."""
    agent = {_USER_AGENT_HEADER: user_agent} if user_agent else {}
    return {**agent, **login.authorization_header, **login.cookie_header}


def _is_success(resp: httpx.Response) -> bool:
    return HTTPStatus.OK <= resp.status_code < HTTPStatus.MULTIPLE_CHOICES


def _robots_allow_sitemap(
    seed: ada_url.URL, sitemap_url: str, user_agent: str, login: CrawlLogin
) -> bool:
    """True when the site's robots.txt lets *user_agent* fetch *sitemap_url*.

    A missing robots.txt (404 or 410) allows everything. A robots.txt that cannot
    be read, or that answers with any other status, allows nothing.
    """
    resp = _get(f"{seed.origin}{_ROBOTS_PATH}", seed, user_agent, login)
    if resp is None:
        return False
    if resp.status_code in _ROBOTS_ABSENT_STATUSES:
        return True
    if not _is_success(resp):
        return False
    rules = RobotFileParser()
    rules.parse(resp.text.splitlines())
    return rules.can_fetch(user_agent or _ANY_USER_AGENT, sitemap_url)


def _fetch_sitemap_text(
    start_url: str, *, user_agent: str = "", respect_robots_txt: bool, login: CrawlLogin = NO_LOGIN
) -> str | None:
    """Return sitemap.xml body or None on any fetch/status failure.

    A blank *user_agent* leaves the user agent to the HTTP client. With
    *respect_robots_txt* the sitemap is requested only when robots.txt allows it.
    """
    parsed = parsed_url(start_url)
    if parsed is None:
        return None
    sitemap_url = f"{parsed.origin}{_SITEMAP_PATH}"
    # Validate the seed before any connection. This is only a progress hint, so
    # an unvalidated or redirecting sitemap simply yields an unknown total.
    try:
        require_valid_crawl_url(sitemap_url)
    except ValueError:
        return None
    if respect_robots_txt and not _robots_allow_sitemap(parsed, sitemap_url, user_agent, login):
        return None
    resp = _get(sitemap_url, parsed, user_agent, login)
    if resp is None or not _is_success(resp):
        return None
    return resp.text


def _count_sitemap_urls(
    start_url: str,
    *,
    include_subdomains: bool,
    respect_robots_txt: bool,
    user_agent: str = "",
    login: CrawlLogin = NO_LOGIN,
) -> int:
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
    text = _fetch_sitemap_text(
        start_url, user_agent=user_agent, respect_robots_txt=respect_robots_txt, login=login
    )
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
