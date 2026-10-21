"""Crawl orchestration: build specs from ``cfg``, drive a :class:`WebFetcher`.

By default a recursive crawl is scoped to the exact starting host so a
Wikipedia article does not wander into other language editions. Callers
opt into subdomain scope via ``include_subdomains=True``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from lilbee.app.ingest import remove_sources_and_forget_in_wiki
from lilbee.app.services import get_services
from lilbee.core.config import cfg
from lilbee.core.config.enums import CrawlRenderMode
from lilbee.crawler import availability, bootstrap, save, sitemap
from lilbee.crawler.availability import CRAWLER_MISSING_MESSAGE
from lilbee.crawler.bootstrap import (
    ChromiumMissingError,
    CrawlEngineRefusedError,
    CrawlerBrowserError,
)
from lilbee.crawler.crawlberg_fetcher import CrawlbergFetcher
from lilbee.crawler.discovery import build_concurrency_spec, build_filter_spec
from lilbee.crawler.events import (
    _browser_warm_up,
    _drain_page_stream,
    _emit_page_failure,
    _fetched_to_result,
    _handle_crawl_teardown_error,
    _pages_cap,
)
from lilbee.crawler.messages import CHROMIUM_MISSING_MESSAGE
from lilbee.crawler.models import (
    CRAWL_PAGES_UNLIMITED,
    NO_LOGIN,
    CancelToken,
    CrawlLogin,
    CrawlResult,
)
from lilbee.crawler.save import CrawlRecord
from lilbee.crawler.url_filter import validate_crawl_url, without_userinfo
from lilbee.runtime.progress import (
    CrawlDoneEvent,
    CrawlPageEvent,
    CrawlStartEvent,
    DetailedProgressCallback,
    EventType,
)

log = logging.getLogger(__name__)

# A crawl limited to this many pages fetches only its seed.
_ONE_PAGE = 1


def _get_crawl_semaphore() -> asyncio.Semaphore | None:
    """Return the process-wide crawl semaphore, or None when unlimited."""
    return get_services().crawler_semaphore


def _resolve_depth(value: int | None, cfg_ceiling: int | None) -> int | None:
    """Resolve a crawl depth to the value the dispatcher consumes.

    Depth has its own contract, distinct from the page-count limit: ``0`` is a
    valid "seed only / single page" depth, not "unbounded". (Page counts use
    :func:`_resolve_page_limit`, where ``0`` means "no limit".)

    None    -> cfg_ceiling (itself may be None; ``None`` means unbounded)
    n >= 0  -> n (0 = seed only; explicit caller intent overrides cfg)
    n < 0   -> ValueError (use None for unbounded)
    """
    effective = value if value is not None else cfg_ceiling
    if effective is None:
        return None
    if effective < 0:
        raise ValueError("crawl depth must be 0 (seed only) or a positive int")
    return effective


def _resolve_page_limit(max_pages: int | None) -> int | None:
    """Resolve the page bound the fetcher consumes (None means unbounded).

    ``CRAWL_PAGES_UNLIMITED`` (0) is an explicit "no limit" and returns None.
    ``None`` is unspecified: it falls back to ``cfg.crawl_max_pages`` if set,
    else the protective default ``cfg.crawl_safety_max_pages`` so a hostile site
    can't exhaust the disk on a crawl nobody bounded. A positive int is honored
    as-is, even above the default.
    """
    if max_pages == CRAWL_PAGES_UNLIMITED:
        return None
    if max_pages is not None:
        return max_pages
    if cfg.crawl_max_pages is not None:
        return cfg.crawl_max_pages
    return cfg.crawl_safety_max_pages


def _require_crawler_extra() -> None:
    """Raise :class:`CrawlerBackendError` when the crawler extra is not installed."""
    if not availability.crawler_available():
        raise bootstrap.CrawlerBackendError(CRAWLER_MISSING_MESSAGE)


def _fetcher(render_mode: CrawlRenderMode, login: CrawlLogin) -> CrawlbergFetcher:
    """The fetcher for one crawl, with its browser, user agent, robots.txt and login settings."""
    return CrawlbergFetcher(
        render_mode=render_mode,
        chrome_args=cfg.crawl_browser_extra_args,
        user_agent=cfg.crawl_user_agent,
        respect_robots_txt=cfg.crawl_respect_robots_txt,
        extra_wait_ms=cfg.crawl_browser_extra_wait_ms,
        login=login,
    )


async def _fetch_single_page(
    url: str,
    render_mode: CrawlRenderMode,
    on_progress: DetailedProgressCallback | None,
    login: CrawlLogin,
) -> CrawlResult:
    """One single-URL fetch; a failure to fetch the page becomes a failed result."""
    try:
        async with _fetcher(render_mode, login) as fetcher:
            warm_up_done = _browser_warm_up(on_progress, render_mode)
            page = await fetcher.fetch_single(
                url, timeout=cfg.crawl_timeout, concurrency=build_concurrency_spec()
            )
            if warm_up_done is not None:
                warm_up_done()
        return _fetched_to_result(page)
    except (CrawlerBrowserError, CrawlEngineRefusedError):
        raise
    except Exception as exc:
        shown = without_userinfo(url)
        log.warning("Failed to crawl %s: %s", shown, exc)
        return CrawlResult(url=shown, success=False, error=str(exc))


async def crawl_single(
    url: str,
    *,
    render_mode: CrawlRenderMode = CrawlRenderMode.BROWSER,
    on_progress: DetailedProgressCallback | None = None,
    login: CrawlLogin = NO_LOGIN,
) -> CrawlResult:
    """Fetch a single URL, sending *login* to its host.

    ``render_mode`` defaults to ``BROWSER`` for direct callers; the public
    entry point :func:`crawl_and_save` resolves it from ``cfg.crawl_render_mode``
    and passes the canonical value down. ``on_progress`` receives the setup
    bracket of a browser crawl's warm-up.

    Raises :class:`CrawlerBackendError` if the crawler extra isn't installed.
    When the headless Chromium shell is missing at launch, runs the bootstrap
    once and retries.
    """
    validate_crawl_url(url)
    _require_crawler_extra()
    try:
        return await _fetch_single_page(url, render_mode, on_progress, login)
    except ChromiumMissingError:
        log.warning("Chromium missing for %s; bootstrapping then retrying", without_userinfo(url))
        await bootstrap.bootstrap_chromium(on_progress=None)
        return await _fetch_single_page(url, render_mode, on_progress, login)


async def crawl_recursive(
    url: str,
    max_depth: int | None = None,
    max_pages: int | None = None,
    on_progress: DetailedProgressCallback | None = None,
    cancel: CancelToken | None = None,
    *,
    include_subdomains: bool = False,
    on_result: Callable[[CrawlResult], Any] | None = None,
    render_mode: CrawlRenderMode = CrawlRenderMode.BROWSER,
    login: CrawlLogin = NO_LOGIN,
) -> list[CrawlResult]:
    """Crawl a URL recursively using BFS, streaming per-page progress.

    The *login* goes to the seed host only, in the crawl and in the sitemap lookup.

    ``render_mode`` defaults to ``BROWSER`` for direct callers; the public
    entry point :func:`crawl_and_save` resolves it from ``cfg.crawl_render_mode``
    and passes the canonical value down.

    ``max_depth`` of None means unbounded depth. ``max_pages`` of
    ``CRAWL_PAGES_UNLIMITED`` (0) means no page limit; a positive int is that
    cap; None is unspecified and falls back to ``cfg.crawl_safety_max_pages`` so
    a hostile site can't exhaust the disk on a crawl nobody bounded.
    ``CRAWL_PAGE`` events fire as each page completes; total is
    ``CRAWL_TOTAL_UNKNOWN`` by default and promoted to the sitemap count
    when available.

    Pass ``include_subdomains=True`` to broaden scope from the exact host to the
    host plus any subdomains. If ``on_result`` is provided, it's called for each
    streamed ``CrawlResult`` the moment it arrives so callers can flush pages to
    disk incrementally and keep partial output across cancellation.
    """
    validate_crawl_url(url)
    # ``_run_crawl`` already resolved the depth ceiling (and routed a seed-only
    # 0 to the single-page path), so the recursive path takes ``max_depth`` as
    # given: None = unbounded, a positive int = the cap.
    depth = max_depth
    pages = _resolve_page_limit(max_pages)

    # Fail fast when the ``crawler`` extra wasn't installed so SSE
    # callers see ``event: error`` instead of a silent zero-results run.
    _require_crawler_extra()

    # HTTP mode needs no browser, so the guard only applies to browser-mode crawls.
    if render_mode is CrawlRenderMode.BROWSER and not bootstrap.chromium_installed():
        raise ChromiumMissingError(CHROMIUM_MISSING_MESSAGE)

    # Best-effort sitemap lookup so the TUI / CLI can render a real page-count
    # denominator instead of [n/-1]. Falls back to CRAWL_TOTAL_UNKNOWN on any
    # failure; off the hot path so a slow/missing sitemap never blocks the crawl.
    sitemap_total = await asyncio.to_thread(
        sitemap._count_sitemap_urls,
        url,
        include_subdomains=include_subdomains,
        respect_robots_txt=cfg.crawl_respect_robots_txt,
        user_agent=cfg.crawl_user_agent,
        login=login,
    )

    concurrency = build_concurrency_spec()
    filters = build_filter_spec(include_subdomains=include_subdomains)

    results: list[CrawlResult] = []
    try:
        async with _fetcher(render_mode, login) as fetcher:
            # The explicit reference lets the stream close, and the crawl stop,
            # the moment the drain breaks on max_pages or cancel.
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
                    render_mode=render_mode,
                )
            finally:
                await page_stream.aclose()
    except (CrawlerBrowserError, CrawlEngineRefusedError):
        raise
    except Exception as exc:
        _handle_crawl_teardown_error(without_userinfo(url), exc, cancel=cancel, results=results)

    return results


async def _maybe_periodic_sync(tasks: set[asyncio.Task[None]], cancel: CancelToken | None) -> None:
    """Fire off a background sync if the ``crawl_sync_interval`` has elapsed.

    Skips when periodic sync is disabled (``interval=0``) or another sync
    is already running. The spawned task is added to ``tasks`` so the
    caller can drain it before returning. The sync stops on the crawl's *cancel*.
    """
    interval = cfg.crawl_sync_interval
    sync_state = get_services().crawler_sync_state
    if interval <= 0 or not sync_state.lock.acquire(blocking=False):
        return

    now = time.monotonic()
    if now - sync_state.last_run < interval:
        sync_state.lock.release()
        return

    sync_state.last_run = now

    async def _run_sync() -> None:
        try:
            from lilbee.data.ingest import sync

            await sync(quiet=True, cancel=cancel)
        except Exception as exc:
            log.warning("Periodic sync during crawl failed: %s", exc)
        finally:
            sync_state.lock.release()

    task = asyncio.create_task(_run_sync())
    tasks.add(task)
    task.add_done_callback(tasks.discard)


def _make_flush_page(
    record: CrawlRecord, written_paths: list[Path]
) -> Callable[[CrawlResult], Any]:
    """A flush closure that saves a page at its name and records it in *record*."""

    def _sync_flush(result: CrawlResult) -> list[Path]:
        path = save.save_page(result, record, remove_sources_and_forget_in_wiki)
        return [] if path is None else [path]

    async def flush_page(result: CrawlResult) -> list[Path]:
        paths = await asyncio.to_thread(_sync_flush, result)
        written_paths.extend(paths)
        return paths

    return flush_page


async def _ensure_crawler_ready(
    on_progress: DetailedProgressCallback | None,
    render_mode: CrawlRenderMode,
) -> None:
    """Reject early when the extra is missing; bootstrap Chromium on first use.

    Runs before the Chromium bootstrap so a user without [crawler] doesn't pay
    the ~160 MB download just to hit the same error afterward. Only browser mode
    needs Chromium; HTTP mode skips the bootstrap entirely. The bootstrap
    short-circuits when Chromium is already installed; any progress is forwarded
    through ``on_progress`` so downstream UIs surface a 'setup' stage.
    """
    _require_crawler_extra()

    if render_mode is CrawlRenderMode.BROWSER and not bootstrap.chromium_installed():
        await bootstrap.bootstrap_chromium(on_progress=on_progress)


async def _run_crawl(
    url: str,
    *,
    depth: int | None,
    max_pages: int | None,
    on_progress: DetailedProgressCallback | None,
    cancel: CancelToken | None,
    include_subdomains: bool,
    flush_page: Callable[[Any], Awaitable[list[Path]]],
    render_mode: CrawlRenderMode,
    login: CrawlLogin,
) -> int:
    """Run the single-URL or recursive crawl. Returns ``pages_seen``.

    Resolves the depth ceiling here (permitting the seed-only ``0``) so a
    ``cfg.crawl_max_depth`` of 0 routes to the single-page path instead of
    blowing up inside the recursive resolver. A page limit of one also takes
    the single-page path, so the crawl reads no sitemap. A crawl whose
    ``cancel`` is already set fetches nothing and builds no engine, at any depth.
    """
    depth = _resolve_depth(depth, cfg.crawl_max_depth)
    if cancel is not None and cancel.is_set():
        return 0
    if depth == 0 or _resolve_page_limit(max_pages) == _ONE_PAGE:
        result = await crawl_single(
            url, render_mode=render_mode, on_progress=on_progress, login=login
        )
        try:
            await flush_page(result)
        except OSError:
            log.exception("Flush failed for %s", result.url)
        if on_progress:
            shown = without_userinfo(url)
            on_progress(EventType.CRAWL_PAGE, CrawlPageEvent(url=shown, current=1, total=1))
        _emit_page_failure(on_progress, result)
        return 1
    results = await crawl_recursive(
        url,
        max_depth=depth,
        max_pages=max_pages,
        on_progress=on_progress,
        cancel=cancel,
        include_subdomains=include_subdomains,
        on_result=flush_page,
        render_mode=render_mode,
        login=login,
    )
    return len(results)


async def crawl_and_save(
    url: str,
    *,
    depth: int | None = None,
    max_pages: int | None = None,
    on_progress: DetailedProgressCallback | None = None,
    cancel: CancelToken | None = None,
    include_subdomains: bool = False,
    render_mode: CrawlRenderMode | None = None,
    login: CrawlLogin = NO_LOGIN,
) -> list[Path]:
    """Crawl URL(s), save as markdown, record each saved page. Returns paths written.

    ``login`` goes to the seed host only and is never saved or logged.

    ``depth``: ``None`` = whole-site unbounded recursion (default). ``0`` =
    single URL, no recursion. ``N > 0`` = max link-follow depth. ``max_pages``:
    ``None`` (unspecified) defers to ``cfg.crawl_max_pages``, else the protective
    ``cfg.crawl_safety_max_pages`` cap. ``0`` (``CRAWL_PAGES_UNLIMITED``) is the
    only truly unbounded value; a positive int is honored as-is.
    ``cfg.crawl_max_depth`` acts as a ceiling applied only when ``depth`` is
    ``None``.

    ``render_mode``: ``None`` resolves to ``cfg.crawl_render_mode`` (the single
    write-boundary for the default). ``http`` fetches without a browser;
    ``browser`` runs a tuned Chromium with JavaScript enabled.

    Hash-based change detection: always fetches but only saves changed or new
    files. Pages flush to disk as they stream so a cancelled crawl preserves
    the pages already fetched.
    """
    mode = render_mode if render_mode is not None else cfg.crawl_render_mode
    await _ensure_crawler_ready(on_progress, mode)

    sem = _get_crawl_semaphore()
    if sem is not None:
        await sem.acquire()
    tasks: set[asyncio.Task[None]] = set()
    try:
        if on_progress:
            start_depth = depth if depth is not None else 0
            start = CrawlStartEvent(url=without_userinfo(url), depth=start_depth)
            on_progress(EventType.CRAWL_START, start)

        record = await asyncio.to_thread(save.load_crawl_record)
        written_paths: list[Path] = []
        flush_page = _make_flush_page(record, written_paths)

        pages_seen = await _run_crawl(
            url,
            depth=depth,
            max_pages=max_pages,
            on_progress=on_progress,
            cancel=cancel,
            include_subdomains=include_subdomains,
            flush_page=flush_page,
            render_mode=mode,
            login=login,
        )

        cancelled = cancel is not None and cancel.is_set()
        if not cancelled:
            await _maybe_periodic_sync(tasks, cancel)

        if on_progress:
            on_progress(
                EventType.CRAWL_DONE,
                CrawlDoneEvent(pages_crawled=pages_seen, files_written=len(written_paths)),
            )

        return written_paths
    finally:
        # Drain this call's periodic-sync tasks before returning so
        # asyncio.run() doesn't close the loop with a pending sync.
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if sem is not None:
            sem.release()
