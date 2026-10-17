"""Per-page event emission, result translation, and cancel-teardown classification."""

from __future__ import annotations

import inspect
import logging
import math
from collections.abc import Callable
from typing import Any

from lilbee.core.config.enums import CrawlRenderMode
from lilbee.crawler.models import CancelToken, CrawlResult, FetchedPage
from lilbee.runtime.progress import (
    CrawlPageEvent,
    CrawlPageFailedEvent,
    DetailedProgressCallback,
    EventType,
    SetupDoneEvent,
    SetupStartEvent,
)

log = logging.getLogger(__name__)

# The setup component of a browser crawl's warm-up: the wait for the browser's first page.
BROWSER_SETUP_COMPONENT = "browser"


def _browser_warm_up(
    on_progress: DetailedProgressCallback | None, render_mode: CrawlRenderMode
) -> Callable[[], None] | None:
    """Report the start of a browser crawl's warm-up; the call that reports its end, or None."""
    if on_progress is None or render_mode is not CrawlRenderMode.BROWSER:
        return None
    on_progress(EventType.SETUP_START, SetupStartEvent(component=BROWSER_SETUP_COMPONENT))
    done = SetupDoneEvent(component=BROWSER_SETUP_COMPONENT, success=True)
    return lambda: on_progress(EventType.SETUP_DONE, done)


def _emit_page_failure(on_progress: DetailedProgressCallback | None, result: CrawlResult) -> None:
    """Log and emit ``CRAWL_PAGE_FAILED`` for a page that yields nothing to save."""
    reason = result.failure_reason()
    if reason is None:
        return
    log.warning("Crawled page yields no content: %s: %s", result.url, reason)
    if on_progress:
        on_progress(
            EventType.CRAWL_PAGE_FAILED,
            CrawlPageFailedEvent(url=result.url, reason=reason),
        )


def _fetched_to_result(page: FetchedPage) -> CrawlResult:
    """Translate the fetcher's value type to the public ``CrawlResult`` shape."""
    return CrawlResult(
        url=page.url,
        markdown=page.markdown,
        success=page.success,
        error=page.error,
    )


def _pages_cap(pages: int | None) -> float:
    """Return the per-result counter ceiling for visible progress.

    ``None`` (unbounded) maps to ``math.inf`` so the streaming loop's hard
    cap check is a pure numeric compare with no branching.
    """
    return math.inf if pages is None else pages


async def _flush(on_result: Callable[[CrawlResult], Any] | None, result: CrawlResult) -> None:
    """Hand *result* to *on_result*; a disk-side flush failure is logged, not a crawl failure."""
    if on_result is None:
        return
    try:
        rv = on_result(result)
        if inspect.isawaitable(rv):
            await rv
    except OSError:
        log.exception("Flush callback failed for %s", result.url)


async def _drain_page_stream(
    page_stream: Any,
    *,
    on_progress: DetailedProgressCallback | None,
    on_result: Callable[[CrawlResult], Any] | None,
    sitemap_total: int,
    pages_cap: float,
    cancel: CancelToken | None,
    render_mode: CrawlRenderMode,
) -> list[CrawlResult]:
    """Consume a fetcher's page stream, emitting events and flushing per page.

    Returns the accumulated ``CrawlResult`` list. The stream is closed
    deterministically by the caller; this helper only iterates. A browser
    crawl's wait for its first page is bracketed by setup events.
    """
    results: list[CrawlResult] = []
    counter = 0
    warm_up_done = _browser_warm_up(on_progress, render_mode)

    def _should_cancel() -> bool:
        return cancel is not None and cancel.is_set()

    async for page in page_stream:
        if warm_up_done is not None:
            warm_up_done()
            warm_up_done = None
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
        _emit_page_failure(on_progress, new_result)
        await _flush(on_result, new_result)
        # The user-visible count never exceeds the number the caller asked for.
        if counter >= pages_cap:
            break
    if warm_up_done is not None:
        warm_up_done()
    return results


def _handle_crawl_teardown_error(
    url: str,
    exc: Exception,
    *,
    cancel: CancelToken | None,
    results: list[CrawlResult],
) -> None:
    """Classify a recursive-crawl exception: cancel-teardown vs real failure.

    An error raised while a cancelled crawl tears down is expected, not a
    failure worth surfacing. Otherwise, log and append a synthetic error result
    (only when nothing was produced so callers always see at least one entry).
    """
    cancelled = cancel is not None and cancel.is_set()
    if cancelled:
        log.debug("Recursive crawl of %s ended during cancel teardown: %s", url, exc)
        return
    log.warning("Recursive crawl of %s failed: %s", url, exc)
    if not results:
        results.append(CrawlResult(url=url, success=False, error=str(exc)))
