"""crawlberg-backed implementation of :class:`lilbee.crawler.fetcher.WebFetcher`."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, quote_plus, urlsplit, urlunsplit

from lilbee.core.config.enums import CrawlRenderMode
from lilbee.crawler import url_filter
from lilbee.crawler.backends.crawlberg.config import _TRACKING_PARAMS, _crawl_config, _CrawlSpec
from lilbee.crawler.backends.crawlberg.events import (
    _SEED_DEPTH,
    _Event,
    _EventKind,
    _events,
)
from lilbee.crawler.models import (
    NO_LOGIN,
    CancelToken,
    ConcurrencySpec,
    CrawlLogin,
    FetchedPage,
    FilterSpec,
)

if TYPE_CHECKING:
    from lilbee.crawler.fetcher import WebFetcher

log = logging.getLogger(__name__)


_PREFIX_MARK = "*"
# What crawlberg's query encoder leaves as is besides letters, digits and ``-._``, and the one
# character Python's encoder leaves as is that crawlberg's does not.
_FORM_SAFE = "*"
_TILDE = "~"
_TILDE_ENCODED = "%7E"
_NO_CONTENT = "No content extracted"


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
    return FetchedPage(url=url, markdown=event.markdown, title=event.title)


def _single_result(page: FetchedPage | None, url: str) -> FetchedPage:
    """A single-URL fetch result under the requested URL, failed when it has no text."""
    if page is not None and not page.success:
        return FetchedPage(url=url, success=False, error=page.error)
    if page is not None and page.markdown.strip():
        return FetchedPage(url=url, markdown=page.markdown.strip(), title=page.title)
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


def _python_patterns(patterns: Sequence[str]) -> tuple[re.Pattern[str], ...]:
    """The *patterns* Python compiles; crawlberg also accepts some that Python refuses.

    The log names an excluded address through Python. A pattern only crawlberg compiles
    still excludes its addresses and is left out of the log.
    """
    compiled = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern))
        except re.error:
            log.debug(
                "The exclude pattern %s is not a Python pattern; the log cannot name it", pattern
            )
    return tuple(compiled)


class _ExcludedLinks:
    """The addresses one crawl leaves out because an exclude pattern matches them.

    crawlberg counts the addresses it excludes and names none. This makes crawlberg's
    decision again for each link of a page, so the log can name the address and the pattern.
    """

    def __init__(self, spec: _CrawlSpec, seed_url: str) -> None:
        self._seed_host = urlsplit(seed_url).hostname or ""
        self._depth = spec.depth
        self._include_subdomains = spec.filters.include_subdomains
        self._patterns = _python_patterns(spec.filters.exclude_patterns)
        self._seen_links: set[str] = set()
        self._checked: set[str] = set()

    def record_seed(self, seed_url: str) -> None:
        """Log the seed when a pattern matches it: crawlberg then crawls nothing.

        The seed is matched as crawlberg's URL parser writes it; a seed that does not parse is
        one crawlberg refuses, so no pattern applies to it.
        """
        address = url_filter.parsed_address(seed_url)
        if address is None:
            return
        url = _queued_url(address)
        self._checked.add(url)
        self._check(url)

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

    def __init__(
        self,
        *,
        render_mode: CrawlRenderMode,
        chrome_args: Sequence[str] = (),
        user_agent: str = "",
        respect_robots_txt: bool = True,
        extra_wait_ms: int = 0,
        login: CrawlLogin = NO_LOGIN,
    ) -> None:
        self._render_mode = render_mode
        self._chrome_args = tuple(chrome_args)
        self._user_agent = user_agent
        self._respect_robots_txt = respect_robots_txt
        self._extra_wait_ms = extra_wait_ms
        self._login = login

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
        config = _crawl_config(
            self._render_mode,
            spec,
            self._chrome_args,
            self._user_agent,
            self._respect_robots_txt,
            self._extra_wait_ms,
            self._login,
        )
        return _events(config, seed_url, cancel)

    async def fetch_single(
        self, url: str, *, timeout: float, concurrency: ConcurrencySpec
    ) -> FetchedPage:
        """Fetch *url* alone; a redirected page keeps the requested URL, without its userinfo."""
        spec = _CrawlSpec(depth=_SEED_DEPTH, max_pages=1, timeout=timeout, concurrency=concurrency)
        shown = url_filter.without_userinfo(url)
        async with aclosing(self._stream(spec, url, None)) as events:
            async for event in events:
                return _single_result(await _to_page(event, shown), shown)
        return _single_result(None, shown)

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
        shown = url_filter.without_userinfo(seed_url)
        excluded = _ExcludedLinks(spec, shown)
        excluded.record_seed(shown)
        async with aclosing(self._stream(spec, seed_url, cancel)) as events:
            async for event in events:
                excluded.record(event)
                if event.refused_by_ssrf and event.url != shown:
                    log.debug("crawlberg refused %s: %s", event.url, event.error)
                    continue
                page = await _to_page(event, shown)
                if page is not None:
                    yield page


# Protocol conformance check: CrawlbergFetcher is structurally a WebFetcher.
if TYPE_CHECKING:
    _: WebFetcher = CrawlbergFetcher(render_mode=CrawlRenderMode.HTTP)
