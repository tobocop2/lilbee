"""Convert crawled HTML to markdown in helper processes, not on the daemon's core.

Fetching a page costs almost no CPU: it waits on the network, and a rendered
page already runs in Playwright's own browser subprocess. Turning the HTML into
markdown is the opposite, a few hundred milliseconds of pure Python on a large
page, and it holds the GIL for all of it. Measured on documentation-shaped HTML:
roughly 0.33 ms per KiB, so 14 ms for a 42 KiB page and 282 ms for an 834 KiB
one. A crawl converting a handful of pages a second therefore spends a large
fraction of the one core Python can use, inside the same process that answers
MCP and HTTP requests, which is what makes searches feel sluggish mid-crawl.

Only the conversion moves. The pool is stateless (HTML in, markdown out), so it
needs none of the shared state a multi-process server would; the daemon stays a
single process and keeps owning the crawl, its cancellation and its output.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from crawl4ai.models import MarkdownGenerationResult

log = logging.getLogger(__name__)

# Conversion is CPU-bound, so more helpers than cores buys nothing and costs
# memory. Two is enough to keep a crawl moving while leaving the daemon a core
# to answer on; the point is to get the work off the daemon's core, not to
# convert as fast as the machine possibly can.
_DEFAULT_WORKERS = 2
_MAX_WORKERS = 8


def _worker_convert(html: str, base_url: str, citations: bool) -> tuple[str, str]:
    """Run one conversion in a helper process; returns (raw, with-citations).

    Imports inside the function because this runs in a fresh interpreter under
    the spawn start method, where nothing of the parent's state is inherited.
    """
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

    result = DefaultMarkdownGenerator().generate_markdown(
        html, base_url=base_url, citations=citations
    )
    return str(result.raw_markdown or ""), str(getattr(result, "markdown_with_citations", "") or "")


class MarkdownConversionPool:
    """A small pool of helper processes that turn HTML into markdown.

    Started on first use and shut down with the crawl, so a daemon that never
    crawls never pays for it. A pool that cannot start, or that dies mid-crawl,
    is not fatal: the caller falls back to converting in-process, which is
    exactly the behaviour that existed before this module.
    """

    def __init__(self, workers: int | None = None) -> None:
        self._workers = _resolve_workers(workers)
        self._pool: ProcessPoolExecutor | None = None
        self._broken = False

    def _executor(self) -> ProcessPoolExecutor | None:
        if self._broken:
            return None
        if self._pool is None:
            try:
                self._pool = ProcessPoolExecutor(max_workers=self._workers)
            except (OSError, ValueError):
                # A sandbox that forbids new processes, or a platform that
                # cannot spawn them. Converting in-process is slower for the
                # daemon but still correct.
                log.info(
                    "Could not start markdown helper processes; converting in the "
                    "server process instead. A large crawl may slow other requests.",
                    exc_info=log.isEnabledFor(logging.DEBUG),
                )
                self._broken = True
                return None
        return self._pool

    def convert(self, html: str, base_url: str, citations: bool) -> tuple[str, str] | None:
        """Convert off-process, or ``None`` when the caller should do it itself."""
        pool = self._executor()
        if pool is None:
            return None
        try:
            return pool.submit(_worker_convert, html, base_url, citations).result()
        except (BrokenExecutor, OSError) as exc:
            # A helper died (OOM-killed on a huge page, say). Stop using the
            # pool for the rest of this crawl rather than failing every page.
            log.warning(
                "A markdown helper process died (%s); converting in the server "
                "process for the rest of this crawl.",
                exc,
            )
            self._broken = True
            self.shutdown()
            return None

    def shutdown(self) -> None:
        """Stop the helpers. Safe to call twice, and when none were started."""
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)


def _resolve_workers(workers: int | None) -> int:
    """Helper count: the setting, else two, bounded by the machine and a cap."""
    if workers is None or workers < 1:
        workers = _DEFAULT_WORKERS
    return max(1, min(workers, _MAX_WORKERS, os.cpu_count() or 1))


class PooledMarkdownGenerator:
    """crawl4ai markdown generator that converts in helper processes.

    Implements the one method crawl4ai calls, and falls back to the stock
    generator whenever the pool cannot answer, so a crawl never fails because
    the offload is unavailable.
    """

    def __init__(self, pool: MarkdownConversionPool) -> None:
        self._pool = pool

    def generate_markdown(
        self,
        input_html: str,
        base_url: str = "",
        html2text_options: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        content_filter: Any = None,
        citations: bool = True,
        **kwargs: Any,
    ) -> MarkdownGenerationResult:
        from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

        stock = DefaultMarkdownGenerator()
        # Anything asking for filtering or custom html2text options is doing more
        # than the plain conversion this pool knows how to reproduce, so it stays
        # in-process rather than silently losing those options.
        if content_filter is not None or html2text_options or options:
            return stock.generate_markdown(
                input_html,
                base_url=base_url,
                html2text_options=html2text_options,
                options=options,
                content_filter=content_filter,
                citations=citations,
                **kwargs,
            )
        converted = self._pool.convert(input_html, base_url, citations)
        if converted is None:
            return stock.generate_markdown(input_html, base_url=base_url, citations=citations)
        raw, with_citations = converted
        from crawl4ai.models import MarkdownGenerationResult

        return MarkdownGenerationResult(
            raw_markdown=raw,
            markdown_with_citations=with_citations,
            references_markdown="",
            fit_markdown="",
            fit_html="",
        )
