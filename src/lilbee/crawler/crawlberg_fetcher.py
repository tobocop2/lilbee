"""crawlberg-backed implementation of :class:`lilbee.crawler.fetcher.WebFetcher`."""

from __future__ import annotations

import asyncio
import functools
import importlib.util
import ipaddress
import json
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypeVar, cast

from lilbee.core.config.enums import CrawlRenderMode
from lilbee.crawler import bootstrap, url_filter
from lilbee.crawler.bootstrap import CHROMIUM_MISSING_MESSAGE, ChromiumMissingError
from lilbee.crawler.models import CancelToken, ConcurrencySpec, FetchedPage, FilterSpec

# crawlberg is the optional ``crawler`` extra and loads a native library, so each
# function that needs it imports it on call.
if TYPE_CHECKING:
    import crawlberg

    from lilbee.crawler.fetcher import WebFetcher

log = logging.getLogger(__name__)

_Network = TypeVar("_Network", ipaddress.IPv4Network, ipaddress.IPv6Network)

# The networks crawlberg's ``deny_private`` refuses (crawlberg 1.7.1, DEFAULT_DENY_NET_CIDRS).
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
        "100.64.0.0/10",
        "::1/128",
        "::/128",
        "fe80::/10",
        "fc00::/7",
        "ff00::/8",
    )
)
_MAX_REDIRECTS = 10
_RATE_LIMIT_STATUSES = (429, 503)
_MS_PER_SECOND = 1000
_SEED_DEPTH = 0
_CHROME_ENV = "CHROME"
_SSRF_ERROR_CODE = "ssrf_policy_violation"
_NO_CONTENT = "No content extracted"
_BROWSER_MODES = {CrawlRenderMode.HTTP: "never", CrawlRenderMode.BROWSER: "always"}


class _EventKind(StrEnum):
    """The ``type`` of a crawlberg crawl event lilbee reads; the end-of-crawl event is not one."""

    PAGE = "page"
    ERROR = "error"


_EVENT_KINDS = {kind.value: kind for kind in _EventKind}


@dataclass(frozen=True)
class _Event:
    """The fields lilbee reads from one crawlberg crawl event."""

    kind: _EventKind
    url: str
    depth: int = _SEED_DEPTH
    markdown: str = ""
    error: str = ""

    @property
    def refused_by_ssrf(self) -> bool:
        """True for an error event crawlberg raised because its SSRF policy refused the URL."""
        return self.kind is _EventKind.ERROR and self.error.startswith(_SSRF_ERROR_CODE)


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

    allowlist = [
        crawlberg.HostMatcher.cidr(str(network))
        for network in admitted_networks(url_filter.get_blocked_networks())
    ]
    return crawlberg.SsrfPolicy(
        deny_private=True, allowlist=allowlist, max_redirects=_MAX_REDIRECTS
    )


def _content_config() -> crawlberg.ContentConfig:
    """Markdown output that keeps navigation, forms and all page text."""
    import crawlberg

    return crawlberg.ContentConfig(
        remove_navigation=False,
        remove_forms=False,
        exclude_selectors=[],
        preprocessing_preset="minimal",
    )


def _crawl_config(render_mode: CrawlRenderMode, spec: _CrawlSpec) -> crawlberg.CrawlConfig:
    """The crawlberg config for one crawl, with every default lilbee depends on set."""
    import crawlberg

    pacing = spec.concurrency
    retries = pacing.retry_on_rate_limit
    timeout_ms = int(spec.timeout * _MS_PER_SECOND)
    # crawlberg has no delay jitter or backoff range, so max_delay_range and the
    # retry delay settings do not apply.
    return crawlberg.CrawlConfig(
        max_depth=spec.depth,
        max_pages=spec.max_pages,
        max_concurrent=pacing.semaphore_count,
        stay_on_domain=True,
        allow_subdomains=spec.filters.include_subdomains,
        exclude_paths=list(spec.filters.exclude_patterns),
        request_timeout=timeout_ms,
        rate_limit_ms=int(pacing.mean_delay * _MS_PER_SECOND),
        max_redirects=_MAX_REDIRECTS,
        retry_count=pacing.retry_max_attempts if retries else 0,
        retry_codes=list(_RATE_LIMIT_STATUSES) if retries else [],
        respect_robots_txt=False,
        soft_http_errors=False,
        download_documents=False,
        content=_content_config(),
        browser=crawlberg.BrowserConfig(mode=_BROWSER_MODES[render_mode], timeout=timeout_ms),
        ssrf=_ssrf_policy(),
    )


def _point_at_headless_shell() -> None:
    """Make crawlberg launch Playwright's headless shell, never a system Chrome."""
    executable = bootstrap.headless_shell_executable()
    if executable is None:
        raise ChromiumMissingError(CHROMIUM_MISSING_MESSAGE)
    os.environ[_CHROME_ENV] = str(executable)


def _parse_event(raw: object) -> _Event | None:
    """Read a crawlberg event, whose payload is only reachable as JSON; None for other kinds."""
    payload: dict[str, Any] = json.loads(str(raw))
    kind = _EVENT_KINDS.get(payload["type"])
    if kind is None:
        return None
    if kind is _EventKind.ERROR:
        return _Event(kind, payload["url"], error=payload["error"])
    result: dict[str, Any] = payload["result"]
    markdown: dict[str, Any] = result.get("markdown") or {}
    return _Event(
        kind,
        result["url"],
        depth=result.get("depth", _SEED_DEPTH),
        markdown=markdown.get("content") or "",
    )


async def _events(
    config: crawlberg.CrawlConfig, seed_url: str, cancel: CancelToken | None
) -> AsyncGenerator[_Event, None]:
    """Stream one crawl's page and error events until it completes or *cancel* is set.

    The engine is created and dropped inside this generator, so it never leaves the
    thread whose event loop runs the crawl.
    """
    import crawlberg

    engine = crawlberg.create_engine(config)
    # crawl_stream is an async generator; its stub types it as an AsyncIterator.
    stream = cast(AsyncGenerator[object, None], crawlberg.crawl_stream(engine, seed_url))
    async with aclosing(stream):
        async for raw in stream:
            if cancel is not None and cancel.is_set():
                return
            event = _parse_event(raw)
            if event is not None:
                yield event


def _url_allowed(url: str) -> bool:
    """True when lilbee's own URL policy admits *url*."""
    try:
        url_filter.validate_crawl_url(url)
    except ValueError:
        return False
    return True


async def _to_page(event: _Event, seed_url: str) -> FetchedPage | None:
    """The fetched page for *event*, or None when lilbee's policy refuses its final URL."""
    if event.kind is _EventKind.ERROR:
        return FetchedPage(url=event.url, success=False, error=event.error)
    if not await asyncio.to_thread(_url_allowed, event.url):
        log.warning("Dropped %s: lilbee does not crawl the address it resolves to", event.url)
        return None
    url = seed_url if event.depth == _SEED_DEPTH else event.url
    return FetchedPage(url=url, markdown=event.markdown)


def _single_result(page: FetchedPage | None, url: str) -> FetchedPage:
    """A single-URL fetch result under the requested URL, failed when it has no text."""
    if page is not None and not page.success:
        return FetchedPage(url=url, success=False, error=page.error)
    markdown = page.markdown.strip() if page is not None else ""
    if markdown:
        return FetchedPage(url=url, markdown=markdown)
    return FetchedPage(url=url, success=False, error=_NO_CONTENT)


class CrawlbergFetcher:
    """:class:`WebFetcher` implementation backed by crawlberg."""

    def __init__(self, *, render_mode: CrawlRenderMode) -> None:
        self._render_mode = render_mode

    async def __aenter__(self) -> CrawlbergFetcher:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        return None

    def _config(self, spec: _CrawlSpec) -> crawlberg.CrawlConfig:
        if self._render_mode is CrawlRenderMode.BROWSER:
            _point_at_headless_shell()
        return _crawl_config(self._render_mode, spec)

    async def fetch_single(self, url: str, *, timeout: float) -> FetchedPage:
        """Fetch *url* alone; a redirected page keeps the requested URL."""
        config = self._config(_CrawlSpec(depth=_SEED_DEPTH, max_pages=1, timeout=timeout))
        async with aclosing(_events(config, url, None)) as events:
            async for event in events:
                return _single_result(await _to_page(event, url), url)
        return _single_result(None, url)

    async def fetch_recursive(
        self,
        seed_url: str,
        *,
        depth: int | None,
        max_pages: int | None,
        timeout: float,
        concurrency: ConcurrencySpec,
        filters: FilterSpec,
        cancel: CancelToken | None = None,
    ) -> AsyncGenerator[FetchedPage, None]:
        """Stream pages crawlberg reaches breadth-first from *seed_url*.

        A discovered link crawlberg's SSRF policy refuses is dropped without a page,
        like a link filtered out before it is fetched; a refused seed is a failed page.
        """
        spec = _CrawlSpec(depth, max_pages, timeout, concurrency, filters)
        config = self._config(spec)
        async with aclosing(_events(config, seed_url, cancel)) as events:
            async for event in events:
                if event.refused_by_ssrf and event.url != seed_url:
                    log.debug("crawlberg refused %s: %s", event.url, event.error)
                    continue
                page = await _to_page(event, seed_url)
                if page is not None:
                    yield page


# Protocol conformance check: CrawlbergFetcher is structurally a WebFetcher.
if TYPE_CHECKING:
    _: WebFetcher = CrawlbergFetcher(render_mode=CrawlRenderMode.HTTP)


@functools.cache
def crawler_available() -> bool:
    """Whether the crawlberg backend is installed, checked without importing it."""
    return importlib.util.find_spec("crawlberg") is not None
