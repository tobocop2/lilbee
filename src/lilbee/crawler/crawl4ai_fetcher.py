"""crawl4ai-backed implementation of :class:`lilbee.crawler.fetcher.WebFetcher`.

THIS IS THE ONLY FILE IN THE PROJECT THAT IMPORTS ``crawl4ai``.

Swapping to a different web-fetching SDK is a one-file change:
delete this module, add a replacement that implements
:class:`lilbee.crawler.fetcher.WebFetcher`, and update the one
import in :mod:`lilbee.crawler.runner`.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import io
import logging
import math
from collections.abc import AsyncGenerator, AsyncIterator
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from lilbee.crawler import bootstrap
from lilbee.crawler.bootstrap import CrawlerBrowserError
from lilbee.crawler.models import (
    CancelToken,
    ConcurrencySpec,
    FetchedPage,
    FilterSpec,
)

if TYPE_CHECKING:
    from lilbee.crawler.fetcher import WebFetcher

log = logging.getLogger(__name__)


def _build_rate_limited_dispatcher(concurrency: ConcurrencySpec) -> Any:
    """Build a SemaphoreDispatcher + RateLimiter from a ConcurrencySpec, or None.

    BFSDeepCrawlStrategy calls ``crawler.arun_many()`` without a dispatcher
    kwarg, so per-domain rate limiting is only reachable by threading a
    dispatcher through AsyncWebCrawler itself. This helper centralizes the
    spec read so the TUI / CLI / server all get identical behavior.
    """
    if not concurrency.retry_on_rate_limit:
        return None
    from crawl4ai.async_dispatcher import RateLimiter, SemaphoreDispatcher

    rate_limiter = RateLimiter(
        base_delay=(concurrency.retry_base_delay_min, concurrency.retry_base_delay_max),
        max_delay=concurrency.retry_max_backoff,
        max_retries=concurrency.retry_max_attempts,
    )
    return SemaphoreDispatcher(
        semaphore_count=concurrency.semaphore_count,
        rate_limiter=rate_limiter,
    )


class _LilbeeAsyncCrawler:
    """AsyncWebCrawler wrapper that injects a default dispatcher on arun_many.

    crawl4ai's BFSDeepCrawlStrategy hard-codes crawler.arun_many(urls, config)
    without a dispatcher kwarg, so per-domain rate limiting and 429/503 retries
    can't be wired via CrawlerRunConfig. By giving the crawler a default
    dispatcher, every strategy-originated arun_many picks it up. An explicit
    dispatcher= on the call still wins.
    """

    def __init__(self, *, verbose: bool, dispatcher: Any) -> None:
        from crawl4ai import AsyncWebCrawler

        self._inner = AsyncWebCrawler(verbose=verbose)
        self._dispatcher = dispatcher

    async def __aenter__(self) -> _LilbeeAsyncCrawler:
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return await self._inner.__aexit__(exc_type, exc, tb)

    async def arun(self, *args: Any, **kwargs: Any) -> Any:
        return await self._inner.arun(*args, **kwargs)

    async def arun_many(
        self, urls: Any, config: Any = None, dispatcher: Any = None, **kwargs: Any
    ) -> Any:
        return await self._inner.arun_many(
            urls,
            config=config,
            dispatcher=dispatcher if dispatcher is not None else self._dispatcher,
            **kwargs,
        )


@contextlib.asynccontextmanager
async def _open_crawler(*, quiet: bool = False, dispatcher: Any = None) -> AsyncIterator[Any]:
    """Open a crawler.

    Raises :class:`CrawlerBrowserError` early if the Chromium binary
    hasn't been downloaded. Without this guard Playwright prints a full
    ASCII install banner that leaks into the TUI.

    When *dispatcher* is provided, wrap AsyncWebCrawler in _LilbeeAsyncCrawler
    so every strategy-originated arun_many call picks it up. The single-URL
    path (crawl_single) doesn't need a dispatcher because arun() doesn't accept
    one, so it passes None and gets a bare AsyncWebCrawler.
    """
    if not bootstrap.chromium_installed():
        raise CrawlerBrowserError(
            "Playwright Chromium browser not installed. "
            "Run 'uv run playwright install chromium' to enable /crawl."
        )

    from crawl4ai import AsyncWebCrawler

    stdout_ctx = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    stderr_ctx = contextlib.redirect_stderr(io.StringIO()) if quiet else contextlib.nullcontext()
    with stdout_ctx, stderr_ctx:
        if dispatcher is not None:
            async with _LilbeeAsyncCrawler(verbose=not quiet, dispatcher=dispatcher) as crawler:
                yield crawler
        else:
            async with AsyncWebCrawler(verbose=not quiet) as crawler:
                yield crawler


def _safe_strategy_cancel(strategy: Any) -> None:
    """Call ``strategy.cancel()`` if available, swallowing the known shapes.

    BFSDeepCrawlStrategy has ``.cancel()`` in crawl4ai 0.8.6. Older versions or
    third-party strategies may not. Belt-and-suspenders: should_cancel already
    gates between BFS levels, but ``cancel()`` also short-circuits arun_many.

    Narrow catch: ``AttributeError`` covers the rare case where ``cancel``
    exists but accesses a missing attribute mid-call; ``RuntimeError`` covers
    cancel-on-closed-strategy. Anything else propagates.
    """
    cancel_method = getattr(strategy, "cancel", None)
    if callable(cancel_method):
        try:
            cancel_method()
        except (AttributeError, RuntimeError) as exc:
            log.debug("strategy.cancel() raised: %s", exc)


async def _safe_aclose(stream: Any) -> None:
    """Close an async generator stream if that is what it is.

    ``_iter_crawl_stream`` normalizes over async-generator / list / single-result
    shapes; only the generator shape has an ``aclose()`` to call. A list or
    single object is a no-op.
    """
    if stream is None:
        return
    if inspect.isasyncgen(stream):
        with contextlib.suppress(Exception):
            await stream.aclose()


async def _iter_crawl_stream(stream: Any) -> AsyncIterator[Any]:
    """Normalize crawl4ai's ``arun()`` return to an async iterator.

    With ``stream=True`` on CrawlerRunConfig, crawl4ai 0.8 returns an async
    generator. Older call sites and some crawl4ai code paths return a list
    (batch mode) or a single CrawlResult. Accept all three shapes so tests
    that mock ``arun()`` with a plain list keep working.
    """
    # Three possible shapes from crawl4ai's arun(): async generator (stream=True),
    # plain list (batch), or a single CrawlResult. Tests mock arun() with any of
    # the three, so normalize here rather than in each caller.
    if inspect.isasyncgen(stream):
        async for item in stream:
            yield item
        return
    # A list is the batch-mode shape; iterate and yield each item.
    if isinstance(stream, list):
        for item in stream:
            yield item
        return
    yield stream


def _host_scope_filter(start_url: str, *, include_subdomains: bool) -> Any:
    """Build a URLFilter that scopes a crawl to the starting URL's host.

    Default behavior (``include_subdomains=False``) restricts link-following to
    the exact host of *start_url*. For ``https://en.wikipedia.org/...`` this
    excludes ``af.wikipedia.org`` and every other language subdomain.

    When ``include_subdomains=True``, crawl4ai's DomainFilter matches the host
    plus any of its subdomains (``foo.example.com`` matches ``example.com``),
    which is the loose "whole registrable domain" behavior users may want.
    """
    from crawl4ai.deep_crawling.filters import DomainFilter, URLFilter

    host = (urlparse(start_url).hostname or "").lower()
    if include_subdomains:
        return DomainFilter(allowed_domains=host) if host else None

    class _ExactHostFilter(URLFilter):  # type: ignore[misc]
        def __init__(self, allowed_host: str) -> None:
            super().__init__()
            self._host = allowed_host

        def apply(self, url: str) -> bool:
            link_host = (urlparse(url).hostname or "").lower()
            ok = link_host == self._host
            self._update_stats(ok)
            return ok

    return _ExactHostFilter(host) if host else None


class Crawl4aiFetcher:
    """:class:`WebFetcher` implementation backed by crawl4ai.

    Migrating off crawl4ai means replacing this class with another
    :class:`WebFetcher` implementor (e.g. a ``KreuzcrawlFetcher``) and
    updating the one construction site in :mod:`lilbee.crawler.runner`.
    """

    def __init__(self, *, quiet: bool = False) -> None:
        self._quiet = quiet

    async def __aenter__(self) -> Crawl4aiFetcher:
        # Crawl4ai opens a fresh ``AsyncWebCrawler`` per operation because
        # ``fetch_recursive`` needs a per-call dispatcher (which depends on
        # the :class:`ConcurrencySpec` for that call). Nothing to set up here.
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        return None

    async def fetch_single(self, url: str, *, timeout: float) -> FetchedPage:
        """Fetch a single URL via crawl4ai's ``arun``."""
        from crawl4ai import CrawlerRunConfig

        config = CrawlerRunConfig(page_timeout=int(timeout * 1000))
        async with _open_crawler(quiet=self._quiet) as crawler:
            result = await crawler.arun(url=url, config=config)
        markdown = (result.markdown or "").strip()
        if markdown:
            return FetchedPage(url=url, markdown=markdown, success=True)
        return FetchedPage(
            url=url,
            success=False,
            error=result.error_message or "No content extracted",
        )

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
        """Stream pages discovered by crawl4ai's native BFS.

        ``depth`` / ``max_pages`` of ``None`` mean unbounded; the adapter
        translates to ``math.inf`` for crawl4ai's BFSDeepCrawlStrategy, which
        is the sentinel it understands.
        """

        def _should_cancel() -> bool:
            return cancel is not None and cancel.is_set()

        from crawl4ai import CrawlerRunConfig
        from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
        from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter

        filter_chain_items: list[Any] = []
        host_filter = _host_scope_filter(seed_url, include_subdomains=filters.include_subdomains)
        if host_filter is not None:
            filter_chain_items.append(host_filter)
        if filters.exclude_patterns:
            filter_chain_items.append(
                URLPatternFilter(filters.exclude_patterns, use_glob=False, reverse=True)
            )
        filter_chain = FilterChain(filter_chain_items) if filter_chain_items else FilterChain()

        strategy = BFSDeepCrawlStrategy(
            max_depth=math.inf if depth is None else depth,
            max_pages=math.inf if max_pages is None else max_pages,
            should_cancel=_should_cancel,
            filter_chain=filter_chain,
        )
        config = CrawlerRunConfig(
            deep_crawl_strategy=strategy,
            page_timeout=int(timeout * 1000),
            mean_delay=concurrency.mean_delay,
            max_range=concurrency.max_delay_range,
            semaphore_count=concurrency.semaphore_count,
            stream=True,
        )

        dispatcher = _build_rate_limited_dispatcher(concurrency)
        stream: Any = None
        strategy_cancelled = False
        # Exceptions propagate to the orchestration layer, which decides
        # whether to log cancel-teardown noise at debug vs surface a real
        # failure. The adapter's only housekeeping is stream close + BFS
        # strategy cancel so Playwright tears down in order.
        async with _open_crawler(quiet=self._quiet, dispatcher=dispatcher) as crawler:
            stream = await crawler.arun(url=seed_url, config=config)
            try:
                async for cr in _iter_crawl_stream(stream):
                    if _should_cancel():
                        _safe_strategy_cancel(strategy)
                        strategy_cancelled = True
                        break
                    if cr.success:
                        yield FetchedPage(url=cr.url, markdown=cr.markdown or "")
                    else:
                        yield FetchedPage(
                            url=cr.url,
                            success=False,
                            error=cr.error_message or "Unknown error",
                        )
            finally:
                # If the consumer breaks out before we saw a cancel, still
                # short-circuit the BFS strategy so any in-flight arun_many
                # batch stops dispatching. Mirrors the orchestrator's
                # previous "hard cap on visible counter" behavior now that
                # the strategy object lives inside the adapter.
                if not strategy_cancelled:
                    _safe_strategy_cancel(strategy)
                # Close the async generator (if it is one) before the
                # crawler context exits, so Playwright tears down
                # in-flight URLs in order. Skipping this is what produced
                # the "BrowserContext.new_page: Connection closed" spam
                # on cancel.
                await _safe_aclose(stream)


# Protocol conformance check: Crawl4aiFetcher is structurally a WebFetcher.
# We don't instantiate at import time so the check stays purely structural.
if TYPE_CHECKING:
    _: WebFetcher = Crawl4aiFetcher()


@functools.cache
def crawler_available() -> bool:
    """Check if the crawl4ai backend is importable (i.e. the extra is installed)."""
    try:
        import crawl4ai  # noqa: F401

        return True
    except ImportError:
        return False
