"""Crawl orchestration: build specs from ``cfg``, drive a :class:`WebFetcher`.

By default a recursive crawl is scoped to the exact starting host so a
Wikipedia article does not wander into other language editions. Callers
opt into subdomain scope via ``include_subdomains=True``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lilbee.app.services import get_services
from lilbee.core.config import cfg
from lilbee.core.config.enums import CrawlRenderMode
from lilbee.crawler import bootstrap, save, sitemap
from lilbee.crawler.bootstrap import CrawlerBrowserError
from lilbee.crawler.crawl4ai_fetcher import Crawl4aiFetcher
from lilbee.crawler.discovery import build_concurrency_spec, build_filter_spec
from lilbee.crawler.events import (
    _drain_page_stream,
    _fetched_to_result,
    _handle_crawl_teardown_error,
    _pages_cap,
)
from lilbee.crawler.models import CRAWL_PAGES_UNLIMITED, CrawlResult
from lilbee.crawler.save import METADATA_FLUSH_INTERVAL, CrawlMeta
from lilbee.crawler.url_filter import validate_crawl_url
from lilbee.runtime.progress import (
    CrawlDoneEvent,
    CrawlPageEvent,
    CrawlStartEvent,
    DetailedProgressCallback,
    EventType,
    SetupDoneEvent,
    SetupStartEvent,
)

# Component name for the browser-warmup setup phase (distinct from the
# Chromium download, whose component is "chromium"). The crawl emits a
# start/done bracket around opening the crawler so the Task Center shows a
# "preparing crawler" stage instead of a silent stall on first use.
_BROWSER_SETUP_COMPONENT = "browser"

log = logging.getLogger(__name__)


def _get_crawl_semaphore() -> asyncio.Semaphore | None:
    """Return the process-wide crawl semaphore, or None when unlimited."""
    return get_services().crawler_semaphore


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


def _looks_like_missing_chromium(exc: BaseException) -> bool:
    """Heuristic for the Playwright "Executable doesn't exist" launch failure."""
    return "Executable doesn't exist" in str(exc)


async def crawl_single(
    url: str,
    *,
    quiet: bool = False,
    on_progress: DetailedProgressCallback | None = None,
    render_mode: CrawlRenderMode = CrawlRenderMode.BROWSER,
) -> CrawlResult:
    """Fetch a single URL.

    ``render_mode`` defaults to ``BROWSER`` for direct callers; the public
    entry point :func:`crawl_and_save` resolves it from ``cfg.crawl_render_mode``
    and passes the canonical value down.

    Raises :class:`CrawlerBackendError` if the crawler extra isn't installed.
    On a "Chromium executable missing" launch failure, re-runs the
    bootstrap once and retries -- ``chromium_installed()`` can return True
    when the wrong revision lives in the cache root, in which case the
    launch fails the first attempt.

    ``on_progress`` receives a setup_start/setup_done bracket around opening
    the crawler so the first crawl's browser warmup is visible rather than a
    silent stall.
    """
    validate_crawl_url(url)
    from lilbee.crawler import crawler_available

    if not crawler_available():
        raise bootstrap.CrawlerBackendError(
            "Web crawling is not available. Run 'uv sync --extra crawler' to enable it."
        )
    # The setup bracket exists to surface the Chromium warmup, which only
    # happens in browser mode; HTTP mode opens a browserless client with no
    # warmup, so emitting a "browser" setup stage there would be misleading.
    emit_setup = render_mode is CrawlRenderMode.BROWSER
    if on_progress is not None and emit_setup:
        on_progress(EventType.SETUP_START, SetupStartEvent(component=_BROWSER_SETUP_COMPONENT))
    try:
        async with Crawl4aiFetcher(quiet=quiet, render_mode=render_mode) as fetcher:
            if on_progress is not None and emit_setup:
                on_progress(
                    EventType.SETUP_DONE,
                    SetupDoneEvent(component=_BROWSER_SETUP_COMPONENT, success=True),
                )
            page = await fetcher.fetch_single(url, timeout=cfg.crawl_timeout)
        return _fetched_to_result(page)
    except CrawlerBrowserError:
        raise
    except Exception as exc:
        if _looks_like_missing_chromium(exc):
            log.warning("Chromium missing for %s; bootstrapping then retrying", url)
            await bootstrap.bootstrap_chromium(on_progress=None)
            try:
                async with Crawl4aiFetcher(quiet=quiet, render_mode=render_mode) as fetcher:
                    page = await fetcher.fetch_single(url, timeout=cfg.crawl_timeout)
                return _fetched_to_result(page)
            except Exception as retry_exc:
                log.warning("Crawl retry failed for %s: %s", url, retry_exc)
                return CrawlResult(url=url, success=False, error=str(retry_exc))
        log.warning("Failed to crawl %s: %s", url, exc)
        return CrawlResult(url=url, success=False, error=str(exc))


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
    render_mode: CrawlRenderMode = CrawlRenderMode.BROWSER,
) -> list[CrawlResult]:
    """Crawl a URL recursively using BFS, streaming per-page progress.

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
    depth = _resolve_limit(max_depth, cfg.crawl_max_depth)
    pages = _resolve_page_limit(max_pages)

    # Fail fast when the ``crawler`` extra wasn't installed so SSE
    # callers see ``event: error`` instead of a silent zero-results run.
    from lilbee.crawler import crawler_available

    if not crawler_available():
        raise bootstrap.CrawlerBackendError(
            "Web crawling is not available. Run 'uv sync --extra crawler' to enable it."
        )

    # Fail fast before pulling in backend submodules so callers get a clean
    # CrawlerBrowserError instead of a Playwright install banner. HTTP mode
    # needs no browser, so the guard only applies to browser-mode crawls.
    if render_mode is CrawlRenderMode.BROWSER and not bootstrap.chromium_installed():
        raise CrawlerBrowserError(
            "Playwright Chromium browser not installed. "
            "Run 'uv run playwright install chromium' to enable browser-mode crawling."
        )

    # Best-effort sitemap lookup so the TUI / CLI can render a real page-count
    # denominator instead of [n/-1]. Falls back to CRAWL_TOTAL_UNKNOWN on any
    # failure; off the hot path so a slow/missing sitemap never blocks the crawl.
    sitemap_total = await asyncio.to_thread(
        sitemap._count_sitemap_urls, url, include_subdomains=include_subdomains
    )

    concurrency = build_concurrency_spec()
    filters = build_filter_spec(include_subdomains=include_subdomains)

    results: list[CrawlResult] = []
    # Browser mode launches Chromium, whose one-time warmup can take many
    # seconds; bracket it with setup events so the Task Center shows a
    # "preparing crawler" stage instead of a silent stall. HTTP mode has no
    # browser warmup, so the bracket is skipped to avoid a misleading stage.
    emit_setup = render_mode is CrawlRenderMode.BROWSER
    if on_progress is not None and emit_setup:
        on_progress(EventType.SETUP_START, SetupStartEvent(component=_BROWSER_SETUP_COMPONENT))
    try:
        async with Crawl4aiFetcher(quiet=quiet, render_mode=render_mode) as fetcher:
            if on_progress is not None and emit_setup:
                on_progress(
                    EventType.SETUP_DONE,
                    SetupDoneEvent(component=_BROWSER_SETUP_COMPONENT, success=True),
                )
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


async def _maybe_periodic_sync(tasks: set[asyncio.Task[None]]) -> None:
    """Fire off a background sync if the ``crawl_sync_interval`` has elapsed.

    Skips when periodic sync is disabled (``interval=0``) or another sync
    is already running. The spawned task is added to ``tasks`` so the
    caller can drain it before returning.
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

            await sync(quiet=True)
        except Exception as exc:
            log.warning("Periodic sync during crawl failed: %s", exc)
        finally:
            sync_state.lock.release()

    task = asyncio.create_task(_run_sync())
    tasks.add(task)
    task.add_done_callback(tasks.discard)


def _make_flush_page(
    meta: dict[str, CrawlMeta],
    written_paths: list[Path],
    counter: dict[str, int],
) -> Callable[[CrawlResult], Any]:
    """Build a per-result flush closure that batches metadata writes via ``to_thread``."""

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
    render_mode: CrawlRenderMode,
) -> None:
    """Reject early when the extra is missing; bootstrap Chromium on first use.

    Runs before the Chromium bootstrap so a user without [crawler] doesn't pay
    the ~160 MB download just to hit the same error afterward. Only browser mode
    needs Chromium; HTTP mode skips the bootstrap entirely. The bootstrap
    short-circuits when Chromium is already installed; any progress is forwarded
    through ``on_progress`` so downstream UIs surface a 'setup' stage.
    """
    from lilbee.crawler import crawler_available

    if not crawler_available():
        raise bootstrap.CrawlerBackendError(
            "Web crawling is not available. Run 'uv sync --extra crawler' to enable it."
        )

    if render_mode is CrawlRenderMode.BROWSER and not bootstrap.chromium_installed():
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
    render_mode: CrawlRenderMode,
) -> int:
    """Run the single-URL or recursive crawl. Returns ``pages_seen``."""
    if depth == 0:
        result = await crawl_single(
            url, quiet=quiet, on_progress=on_progress, render_mode=render_mode
        )
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
        render_mode=render_mode,
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
    render_mode: CrawlRenderMode | None = None,
) -> list[Path]:
    """Crawl URL(s), save as markdown, update metadata. Returns paths written.

    ``depth``: ``None`` = whole-site unbounded recursion (default). ``0`` =
    single URL, no recursion. ``N > 0`` = max link-follow depth. ``max_pages``:
    ``None`` = no limit, positive int = cap. ``cfg.crawl_max_{depth,pages}`` act
    as ceilings applied only when ``depth``/``max_pages`` are ``None``.

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
            render_mode=mode,
        )

        if counter["pending"] > 0:
            try:
                save.save_crawl_metadata(meta)
            except OSError:
                log.exception("Final metadata flush failed")

        cancelled = cancel is not None and cancel.is_set()
        if not cancelled:
            await _maybe_periodic_sync(tasks)

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
