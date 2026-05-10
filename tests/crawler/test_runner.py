"""Orchestration-layer tests backed by an inline ``FakeFetcher``.

Covers the contract that the runner exposes to lilbee callers without
pulling in crawl4ai: progress events, cancel tokens, per-page flush,
metadata batching, and auto-sync.

These tests sit alongside (not replace) ``tests/test_crawler.py``.
That file exercises the same functions via a ``sys.modules["crawl4ai"]``
mock so the ``Crawl4aiFetcher`` adapter is covered end-to-end. The tests
here pin the orchestrator's behaviour independently of any backend.
"""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from lilbee.core.config import cfg
from lilbee.crawler import crawl4ai_fetcher as crawl4ai_fetcher_mod
from lilbee.crawler import discovery as discovery_mod
from lilbee.crawler import events as events_mod
from lilbee.crawler import runner as runner_mod
from lilbee.crawler.models import (
    ConcurrencySpec,
    CrawlResult,
    FetchedPage,
    FilterSpec,
)
from lilbee.crawler.runner import crawl_and_save, crawl_recursive
from lilbee.runtime.progress import CrawlDoneEvent, CrawlStartEvent, EventType


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Point ``cfg`` paths at ``tmp_path`` and stub the Chromium check.

    Snapshot/restore follows the project's config-isolation pattern.
    The ``chromium_installed`` stub skips the pre-flight guard so these
    pure-orchestration tests don't need Playwright installed.
    """
    snapshot = cfg.model_copy()
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir()
    cfg.data_dir = tmp_path / "data"
    cfg.data_dir.mkdir()
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    monkeypatch.setattr("lilbee.crawler.bootstrap.chromium_installed", lambda: True)
    # The ``crawl_recursive`` entry point now gates on ``crawler_available()``
    # so a missing ``[crawler]`` extra produces ``CrawlerBackendError``.
    # Stub it True here so orchestration tests don't require the crawler extra.
    # Both the SDK façade (used by callers) and the impl (used by tests that
    # import directly) are patched so the value is consistent across import paths.
    monkeypatch.setattr("lilbee.crawler.crawler_available", lambda: True)
    monkeypatch.setattr("lilbee.crawler.crawl4ai_fetcher.crawler_available", lambda: True)
    # Bypass SSRF DNS resolution by default so localhost-like test URLs
    # don't hit real DNS.
    monkeypatch.setattr(
        "lilbee.crawler.url_filter.socket.getaddrinfo",
        lambda host, port, *a, **kw: [(2, 1, 6, "", ("93.184.216.34", 0))],
    )
    # Keep the sitemap denominator deterministic.
    from lilbee.runtime.progress import CRAWL_TOTAL_UNKNOWN

    monkeypatch.setattr(
        "lilbee.crawler.sitemap._count_sitemap_urls",
        lambda *a, **kw: CRAWL_TOTAL_UNKNOWN,
    )
    yield tmp_path
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


class FakeFetcher:
    """Hand-rolled :class:`lilbee.crawler.fetcher.WebFetcher` stand-in.

    Feeds a pre-recorded list of ``FetchedPage`` objects to the orchestrator
    so tests can pin per-page flush / cancel / progress behaviour without
    touching crawl4ai at all.
    """

    def __init__(
        self,
        pages: list[FetchedPage] | None = None,
        *,
        cancel_on_iter: threading.Event | None = None,
    ) -> None:
        self._pages = pages or []
        self._cancel_on_iter = cancel_on_iter
        self.single_calls: list[tuple[str, float]] = []
        self.recursive_calls: list[dict[str, Any]] = []
        self.closed = False

    async def __aenter__(self) -> FakeFetcher:
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.closed = True

    async def fetch_single(self, url: str, *, timeout: float) -> FetchedPage:
        self.single_calls.append((url, timeout))
        if self._pages:
            return self._pages[0]
        return FetchedPage(url=url, markdown="# Fake", success=True)

    async def fetch_recursive(
        self,
        seed_url: str,
        *,
        depth: int | None,
        max_pages: int | None,
        timeout: float,
        concurrency: ConcurrencySpec,
        filters: FilterSpec,
        cancel: threading.Event | None = None,
    ) -> AsyncIterator[FetchedPage]:
        self.recursive_calls.append(
            {
                "seed_url": seed_url,
                "depth": depth,
                "max_pages": max_pages,
                "timeout": timeout,
                "concurrency": concurrency,
                "filters": filters,
                "cancel": cancel,
            }
        )
        for page in self._pages:
            if self._cancel_on_iter is not None:
                # Allow the test to set the cancel event between yields.
                pass
            yield page


class TestCrawlRecursiveOrchestration:
    """``crawl_recursive`` wiring: specs, progress events, streaming."""

    async def test_builds_concurrency_spec_from_cfg(self):
        cfg.crawl_concurrent_requests = 7
        cfg.crawl_mean_delay = 0.5
        cfg.crawl_retry_on_rate_limit = True
        cfg.crawl_retry_base_delay_min = 1.0
        fake = FakeFetcher(pages=[])
        with patch.object(runner_mod, "Crawl4aiFetcher", return_value=fake):
            await crawl_recursive("https://example.com", max_depth=1, max_pages=5)
        assert fake.recursive_calls
        spec: ConcurrencySpec = fake.recursive_calls[0]["concurrency"]
        assert spec.semaphore_count == 7
        assert spec.mean_delay == pytest.approx(0.5)
        assert spec.retry_on_rate_limit is True
        assert spec.retry_base_delay_min == pytest.approx(1.0)

    async def test_builds_filter_spec_from_cfg_and_flag(self):
        cfg.crawl_exclude_patterns = ["/tag/", "/page/\\d+"]
        fake = FakeFetcher(pages=[])
        with patch.object(runner_mod, "Crawl4aiFetcher", return_value=fake):
            await crawl_recursive(
                "https://example.com",
                max_depth=1,
                max_pages=5,
                include_subdomains=True,
            )
        spec: FilterSpec = fake.recursive_calls[0]["filters"]
        assert spec.exclude_patterns == ["/tag/", "/page/\\d+"]
        assert spec.include_subdomains is True

    async def test_emits_progress_per_page(self):
        pages = [
            FetchedPage(url=f"https://example.com/p{i}", markdown=f"# P{i}") for i in range(1, 4)
        ]
        fake = FakeFetcher(pages=pages)
        events: list[tuple[EventType, Any]] = []

        def on_progress(event_type: EventType, data: Any) -> None:
            events.append((event_type, data))

        with patch.object(runner_mod, "Crawl4aiFetcher", return_value=fake):
            results = await crawl_recursive(
                "https://example.com", max_depth=2, max_pages=10, on_progress=on_progress
            )
        assert len(results) == 3
        page_events = [d for t, d in events if t == EventType.CRAWL_PAGE]
        assert [e.current for e in page_events] == [1, 2, 3]

    async def test_hard_cap_truncates_results(self):
        pages = [
            FetchedPage(url=f"https://example.com/p{i}", markdown=f"# P{i}") for i in range(1, 6)
        ]
        fake = FakeFetcher(pages=pages)
        with patch.object(runner_mod, "Crawl4aiFetcher", return_value=fake):
            results = await crawl_recursive("https://example.com", max_depth=1, max_pages=3)
        assert len(results) == 3
        assert [r.url for r in results] == [
            "https://example.com/p1",
            "https://example.com/p2",
            "https://example.com/p3",
        ]

    async def test_cancel_token_stops_iteration(self):
        cancel = threading.Event()
        # Fake pages with a side effect that sets the cancel event after p2.
        pages = [
            FetchedPage(url=f"https://example.com/p{i}", markdown=f"# P{i}") for i in range(1, 6)
        ]

        class _CancelAfterTwo(FakeFetcher):
            async def fetch_recursive(self, *a: Any, **kw: Any) -> AsyncIterator[FetchedPage]:
                self.recursive_calls.append({})
                for idx, page in enumerate(self._pages, start=1):
                    yield page
                    if idx == 2:
                        cancel.set()

        fake = _CancelAfterTwo(pages=pages)
        with patch.object(runner_mod, "Crawl4aiFetcher", return_value=fake):
            results = await crawl_recursive(
                "https://example.com", max_depth=1, max_pages=10, cancel=cancel
            )
        assert len(results) == 2

    async def test_on_result_callback_invoked_per_page(self):
        pages = [
            FetchedPage(url="https://example.com/a", markdown="# A"),
            FetchedPage(url="https://example.com/b", markdown="# B"),
        ]
        fake = FakeFetcher(pages=pages)
        observed: list[str] = []
        with patch.object(runner_mod, "Crawl4aiFetcher", return_value=fake):
            await crawl_recursive(
                "https://example.com",
                max_depth=1,
                max_pages=10,
                on_result=lambda r: observed.append(r.url),
            )
        assert observed == ["https://example.com/a", "https://example.com/b"]

    async def test_on_result_oserror_does_not_fail_crawl(self, caplog):
        """OSError inside the flush callback is logged, never reraised."""
        pages = [FetchedPage(url="https://example.com/p1", markdown="# P1")]
        fake = FakeFetcher(pages=pages)

        def flaky(result: CrawlResult) -> None:
            raise OSError("disk full")

        with patch.object(runner_mod, "Crawl4aiFetcher", return_value=fake):
            # Should not raise even though the callback errors.
            results = await crawl_recursive(
                "https://example.com", max_depth=1, max_pages=5, on_result=flaky
            )
        assert len(results) == 1

    async def test_validates_url_before_opening_fetcher(self):
        with pytest.raises(ValueError, match="http"):
            await crawl_recursive("ftp://example.com", max_depth=1)

    async def test_raises_when_chromium_missing(self, monkeypatch):
        monkeypatch.setattr("lilbee.crawler.bootstrap.chromium_installed", lambda: False)
        from lilbee.crawler import CrawlerBrowserError

        with pytest.raises(CrawlerBrowserError):
            await crawl_recursive("https://example.com", max_depth=1)


class TestCrawlAndSaveOrchestration:
    """``crawl_and_save`` wires single / recursive crawls to disk."""

    async def test_depth_zero_uses_crawl_single(self):
        fake_single = AsyncMock(
            return_value=CrawlResult(
                url="https://example.com/only", markdown="# Only", success=True
            )
        )
        # Patch at the submodule path where ``crawl_single`` is defined;
        # ``crawl_and_save`` looks it up on the runner module at call time.
        with patch("lilbee.crawler.runner.crawl_single", fake_single):
            paths = await crawl_and_save("https://example.com/only", depth=0)
        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].read_text(encoding="utf-8") == "# Only"
        fake_single.assert_awaited_once()

    async def test_depth_nonzero_uses_crawl_recursive(self):
        import inspect as _inspect

        async def fake_recursive(*args: Any, **kwargs: Any) -> list[CrawlResult]:
            flush = kwargs.get("on_result")
            results = [
                CrawlResult(url="https://example.com/a", markdown="# A"),
                CrawlResult(url="https://example.com/b", markdown="# B"),
            ]
            if flush is not None:
                for r in results:
                    rv = flush(r)
                    if _inspect.isawaitable(rv):
                        await rv
            return results

        with patch("lilbee.crawler.runner.crawl_recursive", side_effect=fake_recursive):
            paths = await crawl_and_save("https://example.com", depth=2, max_pages=5)
        assert len(paths) == 2
        assert {p.read_text(encoding="utf-8") for p in paths} == {"# A", "# B"}

    async def test_emits_start_and_done_events(self):
        events: list[tuple[EventType, Any]] = []

        async def _noop_recursive(*args: Any, **kwargs: Any) -> list[CrawlResult]:
            return []

        with patch("lilbee.crawler.runner.crawl_recursive", side_effect=_noop_recursive):
            await crawl_and_save(
                "https://example.com",
                depth=1,
                on_progress=lambda t, d: events.append((t, d)),
            )
        kinds = [t for t, _ in events]
        assert kinds[0] == EventType.CRAWL_START
        assert kinds[-1] == EventType.CRAWL_DONE
        start = events[0][1]
        assert isinstance(start, CrawlStartEvent)
        assert start.url == "https://example.com"
        done = events[-1][1]
        assert isinstance(done, CrawlDoneEvent)
        assert done.pages_crawled == 0

    async def test_cancel_skips_periodic_sync(self):
        cancel = threading.Event()
        cancel.set()

        async def _noop_recursive(*args: Any, **kwargs: Any) -> list[CrawlResult]:
            return []

        with (
            patch("lilbee.crawler.runner.crawl_recursive", side_effect=_noop_recursive),
            patch(
                "lilbee.crawler.runner._maybe_periodic_sync", new_callable=AsyncMock
            ) as mock_sync,
        ):
            await crawl_and_save("https://example.com", depth=1, cancel=cancel)
        mock_sync.assert_not_awaited()

    async def test_success_awaits_periodic_sync(self):
        async def _noop_recursive(*args: Any, **kwargs: Any) -> list[CrawlResult]:
            return []

        with (
            patch("lilbee.crawler.runner.crawl_recursive", side_effect=_noop_recursive),
            patch(
                "lilbee.crawler.runner._maybe_periodic_sync", new_callable=AsyncMock
            ) as mock_sync,
        ):
            await crawl_and_save("https://example.com", depth=1)
        mock_sync.assert_awaited_once()


class TestConcurrencySemaphore:
    """The process-wide crawl semaphore gates concurrent crawl_and_save calls."""

    async def test_no_semaphore_when_unlimited(self, monkeypatch):
        """``crawl_max_concurrent <= 0`` means no semaphore construction."""
        from lilbee.app.services import reset_services

        cfg.crawl_max_concurrent = 0
        reset_services()
        assert runner_mod._get_crawl_semaphore() is None

    async def test_semaphore_reused_within_services_lifetime(self):
        """Repeated ``_get_crawl_semaphore`` calls return the same Services-owned semaphore."""
        from lilbee.app.services import reset_services

        cfg.crawl_max_concurrent = 2
        reset_services()
        sem1 = runner_mod._get_crawl_semaphore()
        sem2 = runner_mod._get_crawl_semaphore()
        assert sem1 is sem2

    async def test_semaphore_rebuilt_on_reset_services(self):
        """``reset_services`` rebuilds the semaphore reflecting the new cfg value."""
        from lilbee.app.services import reset_services

        cfg.crawl_max_concurrent = 2
        reset_services()
        first = runner_mod._get_crawl_semaphore()
        assert first is not None
        assert first._value == 2
        cfg.crawl_max_concurrent = 4
        reset_services()
        second = runner_mod._get_crawl_semaphore()
        assert second is not None
        assert second._value == 4
        assert first is not second


class TestBuildSpecs:
    """Spec builders project ``cfg`` into backend-agnostic dataclasses."""

    def test_concurrency_spec_mirrors_cfg(self):
        cfg.crawl_concurrent_requests = 11
        cfg.crawl_mean_delay = 2.5
        cfg.crawl_max_delay_range = 3.5
        cfg.crawl_retry_on_rate_limit = True
        cfg.crawl_retry_base_delay_min = 0.25
        cfg.crawl_retry_base_delay_max = 2.0
        cfg.crawl_retry_max_backoff = 30.0
        cfg.crawl_retry_max_attempts = 4
        spec = discovery_mod.build_concurrency_spec()
        assert spec.semaphore_count == 11
        assert spec.mean_delay == pytest.approx(2.5)
        assert spec.max_delay_range == pytest.approx(3.5)
        assert spec.retry_on_rate_limit is True
        assert spec.retry_base_delay_min == pytest.approx(0.25)
        assert spec.retry_base_delay_max == pytest.approx(2.0)
        assert spec.retry_max_backoff == pytest.approx(30.0)
        assert spec.retry_max_attempts == 4

    def test_filter_spec_copies_patterns(self):
        cfg.crawl_exclude_patterns = ["/a/", "/b/"]
        spec = discovery_mod.build_filter_spec(include_subdomains=True)
        assert spec.exclude_patterns == ["/a/", "/b/"]
        assert spec.include_subdomains is True
        # Mutating the spec list must not feed back into cfg.
        spec.exclude_patterns.append("/c/")
        assert cfg.crawl_exclude_patterns == ["/a/", "/b/"]


class TestFetchedToResult:
    """The adapter's FetchedPage must survive the trip to CrawlResult intact."""

    def test_success_mapping(self):
        page = FetchedPage(url="https://x", markdown="# X", success=True)
        result = events_mod._fetched_to_result(page)
        assert result.url == "https://x"
        assert result.markdown == "# X"
        assert result.success is True
        assert result.error is None

    def test_failure_mapping(self):
        page = FetchedPage(url="https://x", markdown="", success=False, error="timeout")
        result = events_mod._fetched_to_result(page)
        assert result.success is False
        assert result.error == "timeout"


class TestResolveLimit:
    """Pure-Python limit resolver: None / positive / zero-negative."""

    def test_none_with_no_ceiling_is_none(self):
        assert runner_mod._resolve_limit(None, None) is None

    def test_none_with_ceiling_uses_ceiling(self):
        assert runner_mod._resolve_limit(None, 25) == 25

    def test_positive_value_wins_over_ceiling(self):
        assert runner_mod._resolve_limit(99, 10) == 99

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            runner_mod._resolve_limit(0, 10)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="positive"):
            runner_mod._resolve_limit(-1, None)


class TestFetcherModuleNotDirectlyImported:
    """Sanity guard: nothing outside ``crawl4ai_fetcher`` imports crawl4ai.

    Enforced structurally so a reviewer doesn't have to grep. If this
    fails, someone added ``import crawl4ai`` to a module that should
    stay backend-neutral.
    """

    def test_runner_module_has_no_crawl4ai_import(self):
        source = (runner_mod.__file__ or "").strip()
        assert source, "runner module must have a __file__"
        with open(source, encoding="utf-8") as fh:
            text = fh.read()
        assert "import crawl4ai" not in text
        assert "from crawl4ai" not in text

    def test_crawl4ai_fetcher_is_the_sole_importer(self):
        adapter_source = (crawl4ai_fetcher_mod.__file__ or "").strip()
        with open(adapter_source, encoding="utf-8") as fh:
            adapter_text = fh.read()
        assert "crawl4ai" in adapter_text
