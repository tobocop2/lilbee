"""Protocol for web-fetcher backends.

The orchestration layer (``api.py``) calls into a ``WebFetcher``
instance; the adapter (``crawl4ai_fetcher.py``) implements this
Protocol. Migrating to a different SDK is a one-file swap: delete
the adapter, add a new one, change the import in ``api.py``.

Lifecycle:

    async with fetcher:
        page = await fetcher.fetch_single(url, timeout=...)
        async for page in fetcher.fetch_recursive(...):
            ...

``__aenter__`` must be called before any fetch method; ``__aexit__``
tears the backend down (browser close, session cleanup, etc.).
``fetch_recursive`` is the streaming entry point: it yields
``FetchedPage`` objects as they arrive so callers can flush per-page.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, Protocol, runtime_checkable

from lilbee.crawler.models import (
    CancelToken,
    ConcurrencySpec,
    FetchedPage,
    FilterSpec,
)


@runtime_checkable
class WebFetcher(Protocol):
    """Backend contract for fetching web pages as markdown.

    Implementations must honour ``CancelToken`` promptly inside
    ``fetch_recursive`` so the streaming loop in ``api.py`` can
    abort without waiting for an in-flight batch to drain.

    Lifecycle ordering:

    1. ``__aenter__`` is called before any fetch method. Adapters with
       per-operation setup (e.g. crawl4ai opens a fresh
       ``AsyncWebCrawler`` inside each fetch method) may no-op here.
    2. ``fetch_single`` and ``fetch_recursive`` may be called multiple
       times during the same context; they must not assume fresh state.
    3. ``fetch_recursive`` returns an async generator; callers are
       expected to ``.aclose()`` it deterministically on early break.
    4. ``__aexit__`` tears the backend down and must succeed even if
       a prior fetch raised.
    """

    async def __aenter__(self) -> WebFetcher: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None: ...

    async def fetch_single(self, url: str, *, timeout: float) -> FetchedPage:
        """Fetch one URL and return its markdown + link set."""
        ...

    def fetch_recursive(
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
        """Stream pages discovered by BFS from ``seed_url``.

        ``depth`` / ``max_pages``: positive int caps, or ``None`` for
        unbounded. Adapters translate ``None`` into whatever sentinel the
        underlying SDK wants (crawl4ai uses ``math.inf``).

        Returns an async generator so the orchestration layer can
        react per page (progress events, save-to-disk, cancel) and
        deterministically ``.aclose()`` the stream when it breaks
        out early (e.g. on ``max_pages`` hard cap).
        """
        ...
