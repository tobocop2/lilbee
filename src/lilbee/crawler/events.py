"""Per-page event emission, result translation, and cancel-teardown classification."""

from __future__ import annotations

import inspect
import logging
import math
import threading
from collections.abc import Callable
from typing import Any

from lilbee.crawler.models import CrawlResult, FetchedPage
from lilbee.runtime.progress import (
    CrawlPageEvent,
    DetailedProgressCallback,
    EventType,
)

log = logging.getLogger(__name__)


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
