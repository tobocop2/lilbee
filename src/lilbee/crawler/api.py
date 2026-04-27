"""Thin orchestration layer: builds specs from ``cfg``, drives a :class:`WebFetcher`.

No crawl4ai imports. All backend-specific knowledge lives in
:mod:`lilbee.crawler.crawl4ai_fetcher`; this module only decides
*what* to crawl (depth/pages/filters/concurrency) and *where* to
put the bytes (per-page flush + metadata). Callers (CLI, MCP,
HTTP, TUI) import these functions via the package façade in
``lilbee.crawler.__init__``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import threading
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lilbee.core.config import cfg
from lilbee.crawler import bootstrap, save, sitemap
from lilbee.crawler.bootstrap import CrawlerBrowserError
from lilbee.crawler.crawl4ai_fetcher import Crawl4aiFetcher
from lilbee.crawler.models import (
    ConcurrencySpec,
    CrawlResult,
    FetchedPage,
    FilterSpec,
)
from lilbee.crawler.save import METADATA_FLUSH_INTERVAL, CrawlMeta
from lilbee.crawler.url_filter import validate_crawl_url
from lilbee.runtime.progress import (
    CrawlDoneEvent,
    CrawlPageEvent,
    CrawlStartEvent,
    DetailedProgressCallback,
    EventType,
)

log = logging.getLogger(__name__)


class CrawlerState:
    """Per-process mutable state for the crawler (semaphore, periodic sync tracking).

    Encapsulates state that would otherwise live as bare module-level globals.
    A single module-level instance (``_state``) is used because this state is
    inherently per-process (threading primitives, asyncio tasks tied to the
    running loop). Test isolation is via :meth:`reset`.
    """

    def __init__(self) -> None:
        self.semaphore: asyncio.Semaphore | None = None
        self.semaphore_limit: int = 0
        self.last_sync_time: float = 0.0
        self.sync_running: threading.Lock = threading.Lock()
        self.background_tasks: set[asyncio.Task[None]] = set()

    def reset(self) -> None:
        """Reset all state (useful for testing)."""
        self.semaphore = None
        self.semaphore_limit = 0
        self.last_sync_time = 0.0
        self.sync_running = threading.Lock()
        self.background_tasks = set()


_state = CrawlerState()


def _get_crawl_semaphore() -> asyncio.Semaphore | None:
    """Return an asyncio semaphore for crawl concurrency, or None if unlimited (0)."""
    limit = cfg.crawl_max_concurrent
    if limit <= 0:
        return None
    if _state.semaphore is None or _state.semaphore_limit != limit:
        _state.semaphore = asyncio.Semaphore(limit)
        _state.semaphore_limit = limit
    return _state.semaphore


def _resolve_limit(value: int | None, cfg_ceiling: int | None) -> int | None:
    """Resolve a caller-provided crawl limit to the number the fetcher consumes.

    None    -> cfg_ceiling (itself may be None; ``None`` means unbounded)
    n > 0   -> n (explicit caller intent; cfg is not a ceiling here)
    n <= 0  -> ValueError (use None for unbounded, not 0)
    """
    effective = value if value is not None else cfg_ceiling
    if effective is None:
        return None
    if effective <= 0:
        raise ValueError("crawl limit must be a positive int or None")
    return effective


def _build_concurrency_spec() -> ConcurrencySpec:
    """Snapshot the crawl-concurrency settings from ``cfg`` into a spec."""
    return ConcurrencySpec(
        semaphore_count=cfg.crawl_concurrent_requests,
        mean_delay=cfg.crawl_mean_delay,
        max_delay_range=cfg.crawl_max_delay_range,
        retry_on_rate_limit=cfg.crawl_retry_on_rate_limit,
        retry_base_delay_min=cfg.crawl_retry_base_delay_min,
        retry_base_delay_max=cfg.crawl_retry_base_delay_max,
        retry_max_backoff=cfg.crawl_retry_max_backoff,
        retry_max_attempts=cfg.crawl_retry_max_attempts,
    )


def _build_filter_spec(*, include_subdomains: bool) -> FilterSpec:
    """Snapshot the filter settings from ``cfg`` + caller flags."""
    return FilterSpec(
        exclude_patterns=list(cfg.crawl_exclude_patterns),
        include_subdomains=include_subdomains,
    )


def _fetched_to_result(page: FetchedPage) -> CrawlResult:
    """Translate the fetcher's value type to the public ``CrawlResult`` shape."""
    return CrawlResult(
        url=page.url,
        markdown=page.markdown,
        success=page.success,
        error=page.error,
    )


async def crawl_single(url: str, *, quiet: bool = False) -> CrawlResult:
    """Fetch a single URL.

    Raises :class:`CrawlerBackendError` if the crawler extra isn't installed.
    """
    validate_crawl_url(url)
    from lilbee.crawler import crawler_available

    if not crawler_available():
        raise bootstrap.CrawlerBackendError(
            "Web crawling is not available. Run 'uv sync --extra crawler' to enable it."
        )
    try:
        async with Crawl4aiFetcher(quiet=quiet) as fetcher:
            page = await fetcher.fetch_single(url, timeout=cfg.crawl_timeout)
        return _fetched_to_result(page)
    except CrawlerBrowserError:
        raise
    except Exception as exc:
        log.warning("Failed to crawl %s: %s", url, exc)
        return CrawlResult(url=url, success=False, error=str(exc))


def _pages_cap(pages: int | None) -> float:
    """Return the per-result counter ceiling for visible progress.

    ``None`` (unbounded) maps to ``math.inf`` so the streaming loop's hard
    cap check is a pure numeric compare with no branching.
    """
    return math.inf if pages is None else pages


async def _drain_page_stream(
    page_stream: Any,
    *,
    on_progress: DetailedProgressCallback | None,
    on_result: Callable[[CrawlResult], Any] | None,
    sitemap_total: int,
    pages_cap: float,
    cancel: threading.Event | None,
) -> list[CrawlResult]:
    """Consume a fetcher's page stream, emitting events and flushing per page.

    Returns the accumulated ``CrawlResult`` list. The stream is closed
    deterministically by the caller; this helper only iterates.
    """
    results: list[CrawlResult] = []
    counter = 0

    def _should_cancel() -> bool:
        return cancel is not None and cancel.is_set()

    async for page in page_stream:
        if _should_cancel():
            break
        counter += 1
        if on_progress:
            on_progress(
                EventType.CRAWL_PAGE,
                CrawlPageEvent(url=page.url, current=counter, total=sitemap_total),
            )
        new_result = _fetched_to_result(page)
        results.append(new_result)
        if on_result is not None:
            try:
                rv = on_result(new_result)
                if inspect.isawaitable(rv):
                    await rv
            except OSError:
                # A disk-side flush failure must not masquerade as a crawl
                # failure. Log and keep streaming; the caller still sees the
                # result in its returned list.
                log.exception("Flush callback failed for %s", new_result.url)
        # Hard cap on visible progress. The BFS may emit failed / redirected
        # pages that push the per-result counter past the cap even after the
        # strategy has stopped dispatching. Break explicitly so the
        # user-visible count never exceeds the number the caller asked for.
        if counter >= pages_cap:
            break
    return results


def _handle_crawl_teardown_error(
    url: str,
    exc: Exception,
    *,
    cancel: threading.Event | None,
    results: list[CrawlResult],
) -> None:
    """Classify a recursive-crawl exception: cancel-teardown vs real failure.

    After cancel, crawl4ai may raise BrowserContext teardown errors as
    in-flight URLs bail. That's expected noise, not a failure worth
    surfacing. Otherwise, log and append a synthetic error result (only
    when nothing was produced so callers always see at least one entry).
    """
    cancelled = cancel is not None and cancel.is_set()
    if cancelled:
        log.debug("Recursive crawl of %s ended during cancel teardown: %s", url, exc)
        return
    log.warning("Recursive crawl of %s failed: %s", url, exc)
    if not results:
        results.append(CrawlResult(url=url, success=False, error=str(exc)))


async def crawl_recursive(
    url: str,
    max_depth: int | None = None,
    max_pages: int | None = None,
    on_progress: DetailedProgressCallback | None = None,
    cancel: threading.Event | None = None,
    *,
    quiet: bool = False,
    include_subdomains: bool = False,
    on_result: Callable[[CrawlResult], Any] | None = None,
) -> list[CrawlResult]:
    """Crawl a URL recursively using BFS, streaming per-page progress.

    None values for ``max_depth`` / ``max_pages`` mean unbounded (constrained
    only by whatever ceiling the user has set in ``cfg.crawl_max_{depth,pages}``,
    if any). Positive ints are explicit caps. ``CRAWL_PAGE`` events fire as
    each page completes; total is ``CRAWL_TOTAL_UNKNOWN`` by default and
    promoted to the sitemap count when available.

    By default the crawl is scoped to the exact starting host so a Wikipedia
    article doesn't wander into other language editions. Pass
    ``include_subdomains=True`` to broaden scope to the starting host plus any
    subdomains (e.g. ``en.wikipedia.org`` plus ``af.wikipedia.org``).

    If ``on_result`` is provided, it's called for each streamed ``CrawlResult``
    the moment it arrives (before the next page yields). Callers use this to
    flush pages to disk incrementally so a cancelled crawl keeps its partial
    output.
    """
    validate_crawl_url(url)
    depth = _resolve_limit(max_depth, cfg.crawl_max_depth)
    pages = _resolve_limit(max_pages, cfg.crawl_max_pages)

    # Fail fast when the ``crawler`` extra wasn't installed so SSE
    # callers see ``event: error`` instead of a silent zero-results run.
    from lilbee.crawler import crawler_available

    if not crawler_available():
        raise bootstrap.CrawlerBackendError(
            "Web crawling is not available. Run 'uv sync --extra crawler' to enable it."
        )

    # Fail fast before pulling in backend submodules so callers get a clean
    # CrawlerBrowserError instead of a Playwright install banner.
    if not bootstrap.chromium_installed():
        raise CrawlerBrowserError(
            "Playwright Chromium browser not installed. "
            "Run 'uv run playwright install chromium' to enable /crawl."
        )

    # Best-effort sitemap lookup so the TUI / CLI can render a real page-count
    # denominator instead of [n/-1]. Falls back to CRAWL_TOTAL_UNKNOWN on any
    # failure; off the hot path so a slow/missing sitemap never blocks the crawl.
    sitemap_total = await asyncio.to_thread(
        sitemap._count_sitemap_urls, url, include_subdomains=include_subdomains
    )

    concurrency = _build_concurrency_spec()
    filters = _build_filter_spec(include_subdomains=include_subdomains)

    results: list[CrawlResult] = []
    try:
        async with Crawl4aiFetcher(quiet=quiet) as fetcher:
            # Hold an explicit reference to the generator so we can aclose
            # it deterministically on break. Without this, the generator's
            # finally block (which also short-circuits the BFS strategy) only
            # runs at gc time, which is too late for callers that expect the
            # strategy to stop the moment we hit ``max_pages``.
            page_stream = fetcher.fetch_recursive(
                url,
                depth=depth,
                max_pages=pages,
                timeout=cfg.crawl_timeout,
                concurrency=concurrency,
                filters=filters,
                cancel=cancel,
            )
            try:
                results = await _drain_page_stream(
                    page_stream,
                    on_progress=on_progress,
                    on_result=on_result,
                    sitemap_total=sitemap_total,
                    pages_cap=_pages_cap(pages),
                    cancel=cancel,
                )
            finally:
                await page_stream.aclose()
    except CrawlerBrowserError:
        raise
    except Exception as exc:
        _handle_crawl_teardown_error(url, exc, cancel=cancel, results=results)

    return results


async def _maybe_periodic_sync() -> None:
    """Fire off a background sync if the ``crawl_sync_interval`` has elapsed.

    Skips if a sync is already running or periodic sync is disabled
    (``interval=0``). Uses a ``threading.Lock`` to avoid asyncio
    event-loop binding issues when called from different loops.
    """
    interval = cfg.crawl_sync_interval
    if interval <= 0 or not _state.sync_running.acquire(blocking=False):
        return

    now = time.monotonic()
    if now - _state.last_sync_time < interval:
        _state.sync_running.release()
        return

    _state.last_sync_time = now

    async def _run_sync() -> None:
        try:
            from lilbee.data.ingest import sync

            await sync(quiet=True)
        except Exception as exc:
            log.warning("Periodic sync during crawl failed: %s", exc)
        finally:
            _state.sync_running.release()

    task = asyncio.create_task(_run_sync())
    _state.background_tasks.add(task)
    task.add_done_callback(_state.background_tasks.discard)


def _make_flush_page(
    meta: dict[str, CrawlMeta],
    written_paths: list[Path],
    counter: dict[str, int],
) -> Callable[[CrawlResult], Any]:
    """Build a per-result flush closure that batches metadata writes.

    Filesystem work runs through ``asyncio.to_thread`` so the streaming
    event loop isn't blocked by per-page writes on slow filesystems.

    ``counter`` is a single-entry dict used as a mutable int so the closure
    can share counter state with the caller without nonlocal gymnastics.
    """

    def _sync_flush(result: CrawlResult) -> Path | None:
        outcome = save._save_single_result(result, meta)
        if outcome is None:
            return None
        save._update_single_metadata(meta, result.url, outcome, datetime.now(UTC).isoformat())
        counter["pending"] += 1
        if counter["pending"] >= METADATA_FLUSH_INTERVAL:
            save.save_crawl_metadata(meta)
            counter["pending"] = 0
        return outcome.path

    async def flush_page(result: CrawlResult) -> Path | None:
        path = await asyncio.to_thread(_sync_flush, result)
        if path is not None:
            written_paths.append(path)
        return path

    return flush_page


async def _ensure_crawler_ready(
    on_progress: DetailedProgressCallback | None,
) -> None:
    """Reject early when the extra is missing; bootstrap Chromium on first use.

    Runs before the Chromium bootstrap so a user without [crawler] doesn't pay
    the ~160 MB download just to hit the same error afterward. The bootstrap
    short-circuits when Chromium is already installed; any progress is forwarded
    through ``on_progress`` so downstream UIs surface a 'setup' stage.
    """
    from lilbee.crawler import crawler_available

    if not crawler_available():
        raise bootstrap.CrawlerBackendError(
            "Web crawling is not available. Run 'uv sync --extra crawler' to enable it."
        )

    if not bootstrap.chromium_installed():
        await bootstrap.bootstrap_chromium(on_progress=on_progress)


async def _run_crawl(
    url: str,
    *,
    depth: int | None,
    max_pages: int | None,
    on_progress: DetailedProgressCallback | None,
    cancel: threading.Event | None,
    quiet: bool,
    include_subdomains: bool,
    flush_page: Callable[[Any], Awaitable[Path | None]],
) -> int:
    """Run the single-URL or recursive crawl. Returns ``pages_seen``."""
    if depth == 0:
        result = await crawl_single(url, quiet=quiet)
        try:
            await flush_page(result)
        except OSError:
            log.exception("Flush failed for %s", result.url)
        if on_progress:
            on_progress(EventType.CRAWL_PAGE, CrawlPageEvent(url=url, current=1, total=1))
        return 1
    results = await crawl_recursive(
        url,
        max_depth=depth,
        max_pages=max_pages,
        on_progress=on_progress,
        cancel=cancel,
        quiet=quiet,
        include_subdomains=include_subdomains,
        on_result=flush_page,
    )
    return len(results)


async def crawl_and_save(
    url: str,
    *,
    depth: int | None = None,
    max_pages: int | None = None,
    on_progress: DetailedProgressCallback | None = None,
    cancel: threading.Event | None = None,
    quiet: bool = False,
    include_subdomains: bool = False,
) -> list[Path]:
    """Crawl URL(s), save as markdown, update metadata. Returns paths written.

    ``depth``: ``None`` = whole-site unbounded recursion (default). ``0`` =
    single URL, no recursion. ``N > 0`` = max link-follow depth.
    ``max_pages``: ``None`` = no limit. Positive int = cap.
    ``cfg.crawl_max_{depth,pages}`` act as user-opted-in ceilings applied only
    when ``depth``/``max_pages`` are ``None``.

    When recursing, the crawl is scoped to the exact starting host by default.
    Set ``include_subdomains=True`` to also follow links into sibling
    subdomains of the starting host.

    Uses hash-based change detection: always fetches, but only saves files
    whose content has changed (or is new). Pages are flushed to disk as they
    stream so a cancelled crawl preserves the pages already fetched instead
    of discarding them.
    """
    await _ensure_crawler_ready(on_progress)

    sem = _get_crawl_semaphore()
    if sem is not None:
        await sem.acquire()
    try:
        if on_progress:
            start_depth = depth if depth is not None else 0
            on_progress(EventType.CRAWL_START, CrawlStartEvent(url=url, depth=start_depth))

        meta = save.load_crawl_metadata()
        written_paths: list[Path] = []
        counter = {"pending": 0}
        flush_page = _make_flush_page(meta, written_paths, counter)

        pages_seen = await _run_crawl(
            url,
            depth=depth,
            max_pages=max_pages,
            on_progress=on_progress,
            cancel=cancel,
            quiet=quiet,
            include_subdomains=include_subdomains,
            flush_page=flush_page,
        )

        if counter["pending"] > 0:
            try:
                save.save_crawl_metadata(meta)
            except OSError:
                log.exception("Final metadata flush failed")

        cancelled = cancel is not None and cancel.is_set()
        if not cancelled:
            await _maybe_periodic_sync()

        if on_progress:
            on_progress(
                EventType.CRAWL_DONE,
                CrawlDoneEvent(pages_crawled=pages_seen, files_written=len(written_paths)),
            )

        return written_paths
    finally:
        if sem is not None:
            sem.release()
