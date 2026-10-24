"""The crawlberg configuration lilbee builds for one crawl."""

from __future__ import annotations

import ipaddress
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

from lilbee.core.config.enums import CrawlRenderMode
from lilbee.crawler import bootstrap, url_filter
from lilbee.crawler.bootstrap import ChromiumMissingError
from lilbee.crawler.messages import CHROMIUM_MISSING_MESSAGE
from lilbee.crawler.models import (
    AUTHORIZATION_HEADER,
    ConcurrencySpec,
    CrawlLogin,
    FilterSpec,
)

# crawlberg is the optional ``crawler`` extra and loads a native library, so each
# function that needs it imports it on call.
if TYPE_CHECKING:
    import crawlberg

log = logging.getLogger(__name__)

_Network = TypeVar("_Network", ipaddress.IPv4Network, ipaddress.IPv6Network)
# The networks crawlberg's ``deny_private`` refuses (crawlberg 1.12.0, DEFAULT_DENY_NET_CIDRS).
CRAWLBERG_DENIED_NETWORKS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = tuple(
    ipaddress.ip_network(cidr)
    for cidr in (
        "127.0.0.0/8",
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "169.254.0.0/16",
        "0.0.0.0/8",
        "224.0.0.0/4",
        "240.0.0.0/4",
        "100.64.0.0/10",
        "::1/128",
        "::/128",
        "fe80::/10",
        "fc00::/7",
        "ff00::/8",
    )
)
_MAX_REDIRECTS = 10
# Tracking query parameters stripped from a discovered link before it is queued, so the page
# is fetched once at its plain address. A trailing ``*`` matches by prefix. ``ref`` is not one:
# it names content on docs and forum sites, such as a branch, though crawlberg's default strips it.
_TRACKING_PARAMS = (
    "utm_*",
    "fbclid",
    "gclid",
    "msclkid",
    "yclid",
    "mc_cid",
    "mc_eid",
    "_hsenc",
    "_hsmi",
    "hsCtaTracking",
    "mkt_*",
    "trk",
    "trkInfo",
    "dm_i",
    "vero_id",
    "vero_conv",
    "oly_anon_id",
    "oly_enc_id",
    "igshid",
    "pk_*",
    "_ga",
    "affiliate",
    "aff_id",
    "aff_ref",
    "aff",
    "partner",
    "srsltid",
    "replytocom",
)
# A browser fetch as a whole (launch, navigation, render, close) may take this many page timeouts.
_BROWSER_OVERALL_TIMEOUTS = 2
_BROWSER_SHUTDOWN_TIMEOUT_MS = 5000
_RATE_LIMIT_STATUSES = (429, 503)
# The most retries crawlberg 1.12.0 accepts; it refuses a config with more.
_MAX_RETRIES = 20
_MS_PER_SECOND = 1000
_BROWSER_MODES = {CrawlRenderMode.HTTP: "never", CrawlRenderMode.BROWSER: "always"}
# The headers a browser sends with a page request, which crawlberg's HTTP client does not send.
# Browser mode sends none: crawlberg adds custom headers to Chrome's requests too.
_BROWSER_ACCEPT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Upgrade-Insecure-Requests": "1",
}
_CUSTOM_HEADERS = {CrawlRenderMode.HTTP: _BROWSER_ACCEPT_HEADERS, CrawlRenderMode.BROWSER: {}}


@dataclass(frozen=True)
class _CrawlSpec:
    """The per-call inputs of one crawl."""

    depth: int | None
    max_pages: int | None
    timeout: float
    concurrency: ConcurrencySpec = field(default_factory=ConcurrencySpec)
    filters: FilterSpec = field(default_factory=FilterSpec)


def _subtract(network: _Network, removed: _Network) -> list[_Network]:
    """The parts of *network* outside *removed*."""
    if not network.overlaps(removed):
        return [network]
    if network.subnet_of(removed):
        return []
    return list(network.address_exclude(removed))


def _subtract_any(
    network: ipaddress.IPv4Network | ipaddress.IPv6Network,
    removed: ipaddress.IPv4Network | ipaddress.IPv6Network,
) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    """:func:`_subtract` across address families; different families never overlap."""
    # isinstance narrows the union so each call gets one address family.
    if isinstance(network, ipaddress.IPv4Network) and isinstance(removed, ipaddress.IPv4Network):
        return list(_subtract(network, removed))
    if isinstance(network, ipaddress.IPv6Network) and isinstance(removed, ipaddress.IPv6Network):
        return list(_subtract(network, removed))
    return [network]


def admitted_networks(
    blocked: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...],
) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    """The networks crawlberg refuses by default that *blocked* does not block."""
    admitted: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for denied in CRAWLBERG_DENIED_NETWORKS:
        pieces = [denied]
        for network in blocked:
            pieces = [part for piece in pieces for part in _subtract_any(piece, network)]
        admitted.extend(pieces)
    return admitted


def _ssrf_policy() -> crawlberg.SsrfPolicy:
    """crawlberg's SSRF policy, read from lilbee's blocklist at crawl time."""
    import crawlberg

    # crawlberg 1.12.0's type stubs declare ``cidr`` as an attribute and as a constructor.
    allowlist = [
        crawlberg.HostMatcher.cidr(str(network))  # type: ignore[misc]
        for network in admitted_networks(url_filter.get_blocked_networks())
    ]
    return crawlberg.SsrfPolicy(
        deny_private=True, allowlist=allowlist, max_redirects=_MAX_REDIRECTS
    )


def _content_config() -> crawlberg.ContentConfig:
    """Markdown output with no frontmatter that keeps navigation, forms and all page text.

    All page text includes the text of hidden elements.
    """
    import crawlberg

    return crawlberg.ContentConfig(
        remove_navigation=False,
        remove_forms=False,
        exclude_selectors=[],
        preprocessing_preset="minimal",
        extract_metadata=False,
        keep_hidden=True,
    )


def _ms(seconds: float) -> int:
    """*seconds* in whole milliseconds."""
    return round(seconds * _MS_PER_SECOND)


@dataclass(frozen=True)
class _Pacing:
    """crawlberg's per-domain delay and jitter ratio for waits in ``[mean, mean + range]``.

    crawlberg scales the delay by a factor in ``[1 - ratio, 1 + ratio]``, so the delay is
    the middle of that span and the ratio is half the range over it.
    """

    rate_limit_ms: int
    jitter_ratio: float

    @classmethod
    def of(cls, spec: ConcurrencySpec) -> _Pacing:
        half_range = spec.max_delay_range / 2
        middle = spec.mean_delay + half_range
        return cls(_ms(middle), half_range / middle if middle > 0 else 0.0)


def _retry_count(spec: ConcurrencySpec) -> int:
    """The retries crawlberg makes for one page: the configured count, up to crawlberg's limit."""
    if not spec.retry_on_rate_limit:
        return 0
    if spec.retry_max_attempts > _MAX_RETRIES:
        log.warning(
            "crawl_retry_max_attempts is %d; a crawl retries a page at most %d times",
            spec.retry_max_attempts,
            _MAX_RETRIES,
        )
    return min(spec.retry_max_attempts, _MAX_RETRIES)


def _headless_shell() -> str:
    """The path of Playwright's headless shell, the only Chrome a crawl launches."""
    executable = bootstrap.headless_shell_executable()
    if executable is None:
        raise ChromiumMissingError(CHROMIUM_MISSING_MESSAGE)
    return str(executable)


def _browser_config(
    render_mode: CrawlRenderMode, timeout_ms: int, chrome_args: Sequence[str], extra_wait_ms: int
) -> crawlberg.BrowserConfig:
    """The browser settings, with a deadline on each whole browser fetch.

    Only browser mode names a Chrome binary, launch flags and the extra wait after a page loads.
    The deadline adds the extra wait, which crawlberg counts inside it.
    """
    import crawlberg

    browser = render_mode is CrawlRenderMode.BROWSER
    extra_wait = extra_wait_ms if browser else None
    return crawlberg.BrowserConfig(
        mode=_BROWSER_MODES[render_mode],
        timeout=timeout_ms,
        overall_timeout=timeout_ms * _BROWSER_OVERALL_TIMEOUTS + (extra_wait or 0),
        shutdown_timeout=_BROWSER_SHUTDOWN_TIMEOUT_MS,
        chrome_path=_headless_shell() if browser else None,
        chrome_args=list(chrome_args) if browser else [],
        extra_wait=extra_wait,
    )


def _crawl_config(
    render_mode: CrawlRenderMode,
    spec: _CrawlSpec,
    chrome_args: Sequence[str],
    user_agent: str,
    respect_robots_txt: bool,
    extra_wait_ms: int,
    login: CrawlLogin,
) -> crawlberg.CrawlConfig:
    """The crawlberg config for one crawl, with every default lilbee depends on set.

    A blank *user_agent* leaves the user agent to crawlberg: Chrome's own in browser mode.
    Browser mode accepts an invalid TLS certificate, as Chrome did before crawlberg 1.12 checked
    it; HTTP mode refuses one.
    crawlberg sends the *login* only to the seed host.
    """
    import crawlberg

    concurrency = spec.concurrency
    retries = concurrency.retry_on_rate_limit
    pacing = _Pacing.of(concurrency)
    timeout_ms = _ms(spec.timeout)
    return crawlberg.CrawlConfig(
        max_depth=spec.depth,
        max_pages=spec.max_pages,
        max_concurrent=concurrency.semaphore_count,
        stay_on_domain=True,
        allow_subdomains=spec.filters.include_subdomains,
        exclude_paths=list(spec.filters.exclude_patterns),
        path_patterns_match_url=True,
        dedup_include_query=True,
        strip_tracking_params=True,
        tracking_params=list(_TRACKING_PARAMS),
        request_timeout=timeout_ms,
        user_agent=user_agent or None,
        custom_headers={**_CUSTOM_HEADERS[render_mode], **login.cookie_header},
        auth=_auth(login),
        rate_limit_ms=pacing.rate_limit_ms,
        rate_limit_jitter_ratio=pacing.jitter_ratio,
        max_redirects=_MAX_REDIRECTS,
        retry_count=_retry_count(concurrency),
        retry_codes=list(_RATE_LIMIT_STATUSES) if retries else [],
        retry_initial_delay_ms=_ms(
            (concurrency.retry_base_delay_min + concurrency.retry_base_delay_max) / 2
        ),
        retry_max_delay_ms=_ms(concurrency.retry_max_backoff),
        respect_robots_txt=respect_robots_txt,
        danger_accept_invalid_certs=render_mode is CrawlRenderMode.BROWSER,
        cookies_enabled=True,
        soft_http_errors=False,
        download_documents=False,
        content=_content_config(),
        browser=_browser_config(render_mode, timeout_ms, chrome_args, extra_wait_ms),
        ssrf=_ssrf_policy(),
    )


def _auth(login: CrawlLogin) -> crawlberg.AuthConfig | None:
    """crawlberg's form of the basic credentials or authorization header of *login*."""
    import crawlberg

    # crawlberg 1.12.0's type stubs declare ``basic`` and ``header`` as attributes and as
    # constructors, so mypy reads each call as a call of None.

    if login.basic_credentials is not None:
        return crawlberg.AuthConfig.basic(*login.basic_credentials)  # type: ignore[no-any-return,misc]
    if login.authorization:
        return crawlberg.AuthConfig.header(  # type: ignore[no-any-return,misc]
            AUTHORIZATION_HEADER, login.authorization
        )
    return None
