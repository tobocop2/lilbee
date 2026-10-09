"""crawlberg-backed implementation of :class:`lilbee.crawler.fetcher.WebFetcher`."""

from __future__ import annotations

import asyncio
import functools
import importlib.util
import ipaddress
import json
import logging
import re
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypeVar, cast
from urllib.parse import parse_qsl, quote_plus, urlsplit, urlunsplit

from lilbee.core.config.enums import CrawlRenderMode
from lilbee.crawler import bootstrap, url_filter
from lilbee.crawler.bootstrap import (
    BROWSER_FLAGS_REFUSED_MESSAGE,
    CHROMIUM_MISSING_MESSAGE,
    CRAWL_REFUSED_MESSAGE,
    EXCLUDE_PATTERN_REFUSED_MESSAGE,
    ChromiumMissingError,
    CrawlEngineRefusedError,
)
from lilbee.crawler.models import CancelToken, ConcurrencySpec, FetchedPage, FilterSpec

# crawlberg is the optional ``crawler`` extra and loads a native library, so each
# function that needs it imports it on call.
if TYPE_CHECKING:
    import crawlberg

    from lilbee.crawler.fetcher import WebFetcher

log = logging.getLogger(__name__)

_Network = TypeVar("_Network", ipaddress.IPv4Network, ipaddress.IPv6Network)

# The networks crawlberg's ``deny_private`` refuses (crawlberg 1.9.0, DEFAULT_DENY_NET_CIDRS).
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
# is fetched once at its plain address. A trailing ``*`` matches by prefix.
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
    "ref",
)
_PREFIX_MARK = "*"
# What crawlberg's query encoder leaves as is besides letters, digits and ``-._``, and the one
# character Python's encoder leaves as is that crawlberg's does not.
_FORM_SAFE = "*"
_TILDE = "~"
_TILDE_ENCODED = "%7E"
# A browser fetch as a whole (launch, navigation, render, close) may take this many page timeouts.
_BROWSER_OVERALL_TIMEOUTS = 2
_BROWSER_SHUTDOWN_TIMEOUT_MS = 5000
_RATE_LIMIT_STATUSES = (429, 503)
# The most retries crawlberg 1.10.2 accepts; it refuses a config with more.
_MAX_RETRIES = 20
_MS_PER_SECOND = 1000
_SEED_DEPTH = 0
_SSRF_ERROR_CODE = "ssrf_policy_violation"
# How crawlberg 1.9.0 starts its refusal of a setting, and the message that names that setting.
_REFUSED_SETTING_MESSAGES = {
    "invalid_config: browser.chrome_args": BROWSER_FLAGS_REFUSED_MESSAGE,
    "invalid_config: invalid exclude_path regex": EXCLUDE_PATTERN_REFUSED_MESSAGE,
}
_NO_CONTENT = "No content extracted"
_BROWSER_MODES = {CrawlRenderMode.HTTP: "never", CrawlRenderMode.BROWSER: "always"}
# The Chrome release and the platform a browser crawl names, on every platform it runs on.
_CHROME_MAJOR = 116
_CHROME_PLATFORM = "Linux"
# The user agent of every request in each render mode.
_USER_AGENTS = {
    CrawlRenderMode.HTTP: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    CrawlRenderMode.BROWSER: (
        f"Mozilla/5.0 (X11; {_CHROME_PLATFORM} x86_64) AppleWebKit/537.36 "
        f"Chrome/{_CHROME_MAJOR}.0.0.0 Safari/537.36"
    ),
}
# The client hint a browser crawl adds to each request to the seed's host, and to no other host.
_BRAND_HINT = {
    "sec-ch-ua": (
        f'"Chromium";v="{_CHROME_MAJOR}", "Not_A Brand";v="8", "Google Chrome";v="{_CHROME_MAJOR}"'
    )
}
# The client hints Chrome sends only to a secure origin.
_SECURE_ORIGIN_HINTS = {"sec-ch-ua-mobile": "?0", "sec-ch-ua-platform": f'"{_CHROME_PLATFORM}"'}
_SECURE_SCHEME = "https"
_LOCALHOST = "localhost"


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
    links: tuple[str, ...] = ()

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
    """Markdown output with no frontmatter that keeps navigation, forms and all page text."""
    import crawlberg

    return crawlberg.ContentConfig(
        remove_navigation=False,
        remove_forms=False,
        exclude_selectors=[],
        preprocessing_preset="minimal",
        extract_metadata=False,
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
    render_mode: CrawlRenderMode, timeout_ms: int, chrome_args: Sequence[str]
) -> crawlberg.BrowserConfig:
    """The browser settings, with a deadline on each whole browser fetch.

    Only browser mode names a Chrome binary and launch flags.
    """
    import crawlberg

    browser = render_mode is CrawlRenderMode.BROWSER
    return crawlberg.BrowserConfig(
        mode=_BROWSER_MODES[render_mode],
        timeout=timeout_ms,
        overall_timeout=timeout_ms * _BROWSER_OVERALL_TIMEOUTS,
        shutdown_timeout=_BROWSER_SHUTDOWN_TIMEOUT_MS,
        chrome_path=_headless_shell() if browser else None,
        chrome_args=list(chrome_args) if browser else [],
    )


def _is_loopback(host: str) -> bool:
    """True when *host* is ``localhost``, a name under it, or a loopback address."""
    if host == _LOCALHOST or host.endswith(f".{_LOCALHOST}"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _client_hints(render_mode: CrawlRenderMode, seed_url: str) -> dict[str, str]:
    """The client hint headers of a crawl from *seed_url*; an HTTP crawl sends none.

    Chrome treats an https origin and a loopback host as secure.
    """
    if render_mode is not CrawlRenderMode.BROWSER:
        return {}
    seed = urlsplit(seed_url)
    if seed.scheme == _SECURE_SCHEME or _is_loopback(seed.hostname or ""):
        return _BRAND_HINT | _SECURE_ORIGIN_HINTS
    return dict(_BRAND_HINT)


def _crawl_config(
    render_mode: CrawlRenderMode, spec: _CrawlSpec, chrome_args: Sequence[str], seed_url: str
) -> crawlberg.CrawlConfig:
    """The crawlberg config for one crawl, with every default lilbee depends on set."""
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
        user_agent=_USER_AGENTS[render_mode],
        custom_headers=_client_hints(render_mode, seed_url),
        rate_limit_ms=pacing.rate_limit_ms,
        rate_limit_jitter_ratio=pacing.jitter_ratio,
        max_redirects=_MAX_REDIRECTS,
        retry_count=_retry_count(concurrency),
        retry_codes=list(_RATE_LIMIT_STATUSES) if retries else [],
        retry_initial_delay_ms=_ms(
            (concurrency.retry_base_delay_min + concurrency.retry_base_delay_max) / 2
        ),
        retry_max_delay_ms=_ms(concurrency.retry_max_backoff),
        respect_robots_txt=False,
        soft_http_errors=False,
        download_documents=False,
        content=_content_config(),
        browser=_browser_config(render_mode, timeout_ms, chrome_args),
        ssrf=_ssrf_policy(),
    )


def _refusal_message(reason: str) -> str:
    """crawlberg's *reason* for a refusal, with lilbee's setting named when the text tells it."""
    # crawlberg's error has no type or attribute for the field; only the start of its text names it.
    template = next(
        (
            message
            for start, message in _REFUSED_SETTING_MESSAGES.items()
            if reason.startswith(start)
        ),
        CRAWL_REFUSED_MESSAGE,
    )
    return template.format(reason=reason)


def _create_engine(config: crawlberg.CrawlConfig) -> crawlberg.CrawlEngineHandle:
    """The engine crawlberg builds for *config*.

    An exception of any type from ``crawlberg.create_engine`` is a
    :class:`CrawlEngineRefusedError` that carries crawlberg's reason. No other call is guarded.
    """
    import crawlberg

    try:
        return crawlberg.create_engine(config)
    except Exception as exc:
        raise CrawlEngineRefusedError(_refusal_message(str(exc))) from exc


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
        links=tuple(link["url"] for link in result.get("links") or ()),
    )


async def _events(
    config: crawlberg.CrawlConfig,
    seed_url: str,
    cancel: CancelToken | None,
) -> AsyncGenerator[_Event, None]:
    """Stream one crawl's page and error events until it completes or *cancel* is set.

    The engine is created and dropped inside this generator, so it never leaves the
    thread whose event loop runs the crawl.
    """
    import crawlberg

    engine = _create_engine(config)
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
    """The fetched page for *event*, or None when lilbee's URL policy refuses its URL."""
    if event.kind is _EventKind.ERROR:
        return FetchedPage(url=event.url, success=False, error=event.error)
    if not await asyncio.to_thread(_url_allowed, event.url):
        log.warning("Dropped %s: lilbee's URL policy refuses it", event.url)
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


def _is_tracking_param(name: str) -> bool:
    """True when crawlberg strips the query parameter *name* from a discovered link."""
    return any(
        name.startswith(param.removesuffix(_PREFIX_MARK))
        if param.endswith(_PREFIX_MARK)
        else name == param
        for param in _TRACKING_PARAMS
    )


def _form_encoded(text: str) -> str:
    """*text* as crawlberg writes it in a query: only letters, digits and ``*-._`` stay as is."""
    return quote_plus(text, safe=_FORM_SAFE).replace(_TILDE, _TILDE_ENCODED)


def _queued_url(link: str) -> str:
    """*link* as crawlberg queues it: no fragment, no tracking parameter, a form-encoded query."""
    parts = urlsplit(link)
    query = "&".join(
        f"{_form_encoded(name)}={_form_encoded(value)}"
        for name, value in parse_qsl(parts.query, keep_blank_values=True)
        if not _is_tracking_param(name)
    )
    return urlunsplit(parts._replace(query=query, fragment=""))


class _ExcludedLinks:
    """The addresses one crawl leaves out because an exclude pattern matches them.

    crawlberg counts the addresses it excludes and names none. This makes crawlberg's
    decision again for each link of a page, so the log can name the address and the pattern.
    """

    def __init__(self, spec: _CrawlSpec, seed_url: str) -> None:
        self._seed_host = urlsplit(seed_url).hostname or ""
        self._depth = spec.depth
        self._include_subdomains = spec.filters.include_subdomains
        self._patterns = tuple(re.compile(pattern) for pattern in spec.filters.exclude_patterns)
        self._seen_links: set[str] = set()
        self._checked: set[str] = set()

    def record(self, event: _Event) -> None:
        """Log each link of *event* that the crawl leaves out, the first time a page holds it."""
        if self._depth is not None and event.depth >= self._depth:
            return
        new_links = [link for link in event.links if link not in self._seen_links]
        self._seen_links.update(new_links)
        for url in map(_queued_url, new_links):
            if url not in self._checked and self._in_scope(url):
                self._checked.add(url)
                self._check(url)

    def _in_scope(self, url: str) -> bool:
        host = urlsplit(url).hostname or ""
        return url_filter.host_in_scope(
            host, self._seed_host, include_subdomains=self._include_subdomains
        )

    def _check(self, url: str) -> None:
        matched = next((p.pattern for p in self._patterns if p.search(url)), None)
        if matched is not None:
            log.warning(
                "Left out %s: it matches %s, a pattern of the crawl_exclude_patterns setting",
                url,
                matched,
            )


class CrawlbergFetcher:
    """:class:`WebFetcher` implementation backed by crawlberg."""

    def __init__(self, *, render_mode: CrawlRenderMode, chrome_args: Sequence[str] = ()) -> None:
        self._render_mode = render_mode
        self._chrome_args = tuple(chrome_args)

    async def __aenter__(self) -> CrawlbergFetcher:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        return None

    def _stream(
        self, spec: _CrawlSpec, seed_url: str, cancel: CancelToken | None
    ) -> AsyncGenerator[_Event, None]:
        """This crawl's events; browser mode finds the headless shell before any engine starts."""
        config = _crawl_config(self._render_mode, spec, self._chrome_args, seed_url)
        return _events(config, seed_url, cancel)

    async def fetch_single(self, url: str, *, timeout: float) -> FetchedPage:
        """Fetch *url* alone; a redirected page keeps the requested URL."""
        spec = _CrawlSpec(depth=_SEED_DEPTH, max_pages=1, timeout=timeout)
        async with aclosing(self._stream(spec, url, None)) as events:
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
        excluded = _ExcludedLinks(spec, seed_url)
        async with aclosing(self._stream(spec, seed_url, cancel)) as events:
            async for event in events:
                excluded.record(event)
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
