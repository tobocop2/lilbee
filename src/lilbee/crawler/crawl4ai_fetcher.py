"""crawl4ai-backed implementation of :class:`lilbee.crawler.fetcher.WebFetcher`."""

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

from lilbee.core.config import cfg
from lilbee.core.config.enums import CrawlRenderMode
from lilbee.crawler import bootstrap
from lilbee.crawler.bootstrap import CrawlerBrowserError
from lilbee.crawler.models import (
    CancelToken,
    ConcurrencySpec,
    FetchedPage,
    FilterSpec,
)
from lilbee.crawler.url_filter import host_in_scope, validate_crawl_url

if TYPE_CHECKING:
    from lilbee.crawler.fetcher import WebFetcher

log = logging.getLogger(__name__)


def _build_inner_crawler(*, verbose: bool, render_mode: CrawlRenderMode) -> Any:
    """Construct a crawl4ai ``AsyncWebCrawler`` for the requested render mode.

    HTTP mode swaps in the browserless HTTP strategy; browser mode tunes
    Chromium for memory (light/text/memory-saving + periodic process recycle),
    reading the recycle threshold and launch flags from config.
    """
    from crawl4ai import AsyncWebCrawler

    if render_mode is CrawlRenderMode.HTTP:
        from crawl4ai.async_crawler_strategy import AsyncHTTPCrawlerStrategy

        return AsyncWebCrawler(crawler_strategy=AsyncHTTPCrawlerStrategy(), verbose=verbose)

    from crawl4ai import BrowserConfig

    config = BrowserConfig(
        light_mode=True,
        text_mode=True,
        memory_saving_mode=True,
        max_pages_before_recycle=cfg.crawl_browser_recycle_pages,
        extra_args=list(cfg.crawl_browser_extra_args),
        verbose=verbose,
    )
    return AsyncWebCrawler(config=config, verbose=verbose)


def _build_rate_limited_dispatcher(
    concurrency: ConcurrencySpec, render_mode: CrawlRenderMode
) -> Any:
    """Build the recursive-crawl dispatcher from a ConcurrencySpec, or None.

    BFSDeepCrawlStrategy calls ``crawler.arun_many()`` without a dispatcher
    kwarg, so per-domain rate limiting is only reachable by threading a
    dispatcher through AsyncWebCrawler itself. Browser mode uses a
    MemoryAdaptiveDispatcher so a crawl backs off when system memory is tight
    rather than steamrolling the machine; HTTP mode is light enough to stay on
    the plain semaphore path.
    """
    if not concurrency.retry_on_rate_limit:
        return None
    from crawl4ai.async_dispatcher import RateLimiter

    rate_limiter = RateLimiter(
        base_delay=(concurrency.retry_base_delay_min, concurrency.retry_base_delay_max),
        max_delay=concurrency.retry_max_backoff,
        max_retries=concurrency.retry_max_attempts,
    )
    if render_mode is CrawlRenderMode.BROWSER:
        from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher

        return MemoryAdaptiveDispatcher(
            max_session_permit=concurrency.semaphore_count,
            rate_limiter=rate_limiter,
        )
    from crawl4ai.async_dispatcher import SemaphoreDispatcher

    return SemaphoreDispatcher(
        semaphore_count=concurrency.semaphore_count,
        rate_limiter=rate_limiter,
    )


class _LilbeeAsyncCrawler:
    """AsyncWebCrawler wrapper that injects a default dispatcher on ``arun_many``.

    BFSDeepCrawlStrategy calls ``arun_many`` without a dispatcher kwarg, so the
    wrapper supplies one to make rate limiting and 429/503 retries reachable.
    An explicit ``dispatcher=`` on the call still wins.
    """

    def __init__(self, inner: Any, *, dispatcher: Any) -> None:
        self._inner = inner
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
async def _open_crawler(
    *, quiet: bool = False, render_mode: CrawlRenderMode, dispatcher: Any = None
) -> AsyncIterator[Any]:
    """Open an AsyncWebCrawler for ``render_mode``, wrapping with the dispatcher.

    Browser mode requires the Chromium binary and raises
    :class:`CrawlerBrowserError` if it is missing, so Playwright's ASCII install
    banner does not leak into the TUI. HTTP mode needs no browser at all.
    """
    if render_mode is CrawlRenderMode.BROWSER and not bootstrap.chromium_installed():
        raise CrawlerBrowserError(
            "Playwright Chromium browser not installed. "
            "Run 'uv run playwright install chromium' to enable browser-mode crawling."
        )

    inner = _build_inner_crawler(verbose=not quiet, render_mode=render_mode)

    stdout_ctx = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    stderr_ctx = contextlib.redirect_stderr(io.StringIO()) if quiet else contextlib.nullcontext()
    with stdout_ctx, stderr_ctx:
        if dispatcher is not None:
            async with _LilbeeAsyncCrawler(inner, dispatcher=dispatcher) as crawler:
                yield crawler
        else:
            async with inner as crawler:
                yield crawler


def _safe_strategy_cancel(strategy: Any) -> None:
    """Call ``strategy.cancel()`` if available; swallow only the known SDK shapes.

    Narrow catch: ``AttributeError`` covers a missing nested attribute mid-call;
    ``RuntimeError`` covers cancel-on-closed-strategy. Anything else propagates.
    """
    cancel_method = getattr(strategy, "cancel", None)
    if callable(cancel_method):
        try:
            cancel_method()
        except (AttributeError, RuntimeError) as exc:
            log.debug("strategy.cancel() raised: %s", exc)


async def _safe_aclose(stream: Any) -> None:
    """Close an async generator stream; no-op for list / single-result shapes.

    aclose() runs the generator's own cleanup, which can surface arbitrary
    downstream errors. A teardown failure must not mask the crawl's real result,
    so it is logged at debug rather than propagated or silently dropped.
    """
    if stream is None:
        return
    if inspect.isasyncgen(stream):
        try:
            await stream.aclose()
        except Exception as exc:
            log.debug("crawl stream aclose() raised during teardown: %s", exc)


async def _iter_crawl_stream(stream: Any) -> AsyncIterator[Any]:
    """Normalize crawl4ai's ``arun()`` return (async generator, list, or single result)."""
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


def _link_passes_ssrf(url: str) -> bool:
    """Return True when a discovered link resolves to a public, http(s) target.

    Re-validates every followed link against the IP blocklist so a discovered
    link to a private/metadata host is dropped before fetch. This is a
    best-effort check at filter time, not DNS-rebinding protection: the fetcher
    resolves the host again when it connects, so a record that rebinds between
    this check and the fetch is a TOCTOU window this does not close.
    """
    try:
        validate_crawl_url(url)
    except ValueError:
        return False
    return True


def _host_scope_filter(start_url: str, *, include_subdomains: bool) -> Any:
    """Build a URLFilter that scopes a crawl to the starting URL's host.

    Default behavior (``include_subdomains=False``) restricts link-following to
    the exact host of *start_url*. When ``include_subdomains=True`` the host
    plus any subdomain is in scope. Either way every followed link is also
    re-validated against the SSRF blocklist, since the host scope check alone
    would let a same-host link that resolves to a private IP through.
    """
    from crawl4ai.deep_crawling.filters import URLFilter

    host = (urlparse(start_url).hostname or "").lower()
    if not host:
        return None

    class _ScopedSsrfFilter(URLFilter):  # type: ignore[misc]
        def apply(self, url: str) -> bool:
            link_host = (urlparse(url).hostname or "").lower()
            ok = host_in_scope(
                link_host, host, include_subdomains=include_subdomains
            ) and _link_passes_ssrf(url)
            self._update_stats(ok)
            return ok

    return _ScopedSsrfFilter()


class Crawl4aiFetcher:
    """:class:`WebFetcher` implementation backed by crawl4ai."""

    def __init__(self, *, quiet: bool = False, render_mode: CrawlRenderMode) -> None:
        self._quiet = quiet
        self._render_mode = render_mode

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
        async with _open_crawler(quiet=self._quiet, render_mode=self._render_mode) as crawler:
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

        dispatcher = _build_rate_limited_dispatcher(concurrency, self._render_mode)
        stream: Any = None
        strategy_cancelled = False
        # Exceptions propagate to the orchestration layer, which decides
        # whether to log cancel-teardown noise at debug vs surface a real
        # failure. The adapter's only housekeeping is stream close + BFS
        # strategy cancel so Playwright tears down in order.
        async with _open_crawler(
            quiet=self._quiet, render_mode=self._render_mode, dispatcher=dispatcher
        ) as crawler:
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
    _: WebFetcher = Crawl4aiFetcher(render_mode=CrawlRenderMode.HTTP)


@functools.cache
def crawler_available() -> bool:
    """Check if the crawl4ai backend is importable (i.e. the extra is installed).

    Uses ``importlib.util.find_spec`` rather than ``import crawl4ai`` so the
    check stays fast on the UI thread. ``crawl4ai`` is in AGENTS.md's
    known-heavy-imports list; executing it on Windows with Defender
    real-time scanning takes seconds, and the Settings screen's feature-
    gate call (``_FEATURE_GATED_GROUPS``) hits it synchronously during
    ``compose``. ``find_spec`` just walks ``sys.path`` to locate the
    package; the actual import runs later from the crawler bootstrap
    where the cost is expected.
    """
    import importlib.util

    return importlib.util.find_spec("crawl4ai") is not None
