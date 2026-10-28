"""crawlberg-backed implementation of :class:`lilbee.crawler.fetcher.WebFetcher`."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing
from typing import TYPE_CHECKING, Any

from lilbee.core.config.enums import CrawlRenderMode
from lilbee.crawler import url_filter
from lilbee.crawler.backends.crawlberg.config import _crawl_config, _CrawlSpec
from lilbee.crawler.backends.crawlberg.events import (
    _SEED_DEPTH,
    _Event,
    _EventKind,
    _events,
)
from lilbee.crawler.backends.crawlberg.links import _ExcludedLinks
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
        if excluded.record_seed(shown):
            return
        try:
            async with aclosing(self._stream(spec, seed_url, cancel)) as events:
                async for event in events:
                    excluded.record(event)
                    if event.refused_by_ssrf and event.url != shown:
                        log.debug("crawlberg refused %s: %s", event.url, event.error)
                        continue
                    page = await _to_page(event, shown)
                    if page is not None:
                        yield page
        finally:
            excluded.summarize()


# Protocol conformance check: CrawlbergFetcher is structurally a WebFetcher.
if TYPE_CHECKING:
    _: WebFetcher = CrawlbergFetcher(render_mode=CrawlRenderMode.HTTP)
