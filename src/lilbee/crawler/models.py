"""Backend-agnostic value types for the crawler package.

These dataclasses cross the seam between the orchestration layer
(``runner.py``) and the web-fetcher backend (``crawl4ai_fetcher.py``).
No third-party types leak through them, so a future adapter can
satisfy the ``WebFetcher`` Protocol without pulling in crawl4ai.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TypeAlias


@dataclass
class CrawlResult:
    """Outcome of crawling a single URL.

    This is the high-level result surfaced to lilbee callers
    (CLI, MCP, HTTP, TUI). The adapter produces ``FetchedPage``
    and the orchestration layer converts it to ``CrawlResult``
    when returning up to the caller.
    """

    url: str
    markdown: str = ""
    success: bool = True
    error: str | None = None


@dataclass
class FetchedPage:
    """Single page produced by a ``WebFetcher`` backend.

    Distinct from :class:`CrawlResult` so the adapter surface
    stays narrow and neutral: just the bytes we needed out of
    the underlying SDK's response object.
    """

    url: str
    markdown: str = ""
    success: bool = True
    error: str | None = None
    links: list[str] = field(default_factory=list)


@dataclass
class ConcurrencySpec:
    """Backend-agnostic concurrency + rate-limit knobs.

    The crawl4ai adapter translates these into ``RateLimiter`` and
    ``SemaphoreDispatcher`` calls; a future adapter with its own
    BFS loop maps them onto ``asyncio.Semaphore`` + retry logic.
    """

    semaphore_count: int = 1
    mean_delay: float = 0.0
    max_delay_range: float = 0.0
    retry_on_rate_limit: bool = False
    retry_base_delay_min: float = 0.0
    retry_base_delay_max: float = 0.0
    retry_max_backoff: float = 0.0
    retry_max_attempts: int = 0


@dataclass
class FilterSpec:
    """Backend-agnostic filter settings applied to discovered links.

    Pure Python data; each adapter decides how to plug the settings
    into its own filter pipeline.
    """

    exclude_patterns: list[str] = field(default_factory=list)
    include_subdomains: bool = False


CancelToken: TypeAlias = threading.Event
"""Cancellation handle the orchestration layer passes to a fetcher.

An already-``set()`` event means "stop as soon as you can". The
crawl4ai adapter polls it in both its streaming loop and its BFS
strategy's ``should_cancel`` hook; a future adapter can poll it
in whatever granularity it supports.
"""
