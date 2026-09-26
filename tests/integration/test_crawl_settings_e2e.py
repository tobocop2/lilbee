"""End-to-end crawls through ``lilbee add --crawl`` with each crawl setting changed in config."""

from __future__ import annotations

import asyncio
import time
from itertools import pairwise
from pathlib import Path

import pytest

from lilbee.app.services import reset_services
from lilbee.core.config import cfg
from lilbee.core.config.defaults import DEFAULT_CRAWL_EXCLUDE_PATTERNS
from lilbee.crawler import crawl_and_save
from tests.integration import _crawl_site as site_mod
from tests.integration._crawl_site import (
    CrawlSite,
    all_saved_text,
    crawl_sandbox,
    needs_crawler,
    run_add,
)

pytestmark = [pytest.mark.slow, needs_crawler]

ONE_LEVEL = ("--crawl", "--depth", "1")
CONFIG_MAX_DEPTH = 1
CONFIG_PAGE_CAP = 5
MIN_PAGES_BEYOND_SEED = 2
TIMEOUT_S = 2
PARALLEL_REQUESTS = 4
PACED_DELAY_S = 1.0
# A delay wait may start a little before the mean on a loaded runner.
DELAY_MARGIN_S = 0.3
JITTER_RANGE_S = 2.0
# Four uniform(0, 2 s) waits sum below this with a probability of about 1e-4.
JITTER_MIN_TOTAL_S = 0.5
CONCURRENT_PREFIXES = ("/pool/", "/paced/")
USER_EXCLUDES = [r"drop-me\.html$", r"\?lang=\d+"]


class TestExcludePatterns:
    def test_user_patterns_drop_matching_pages_and_keep_the_rest(
        self, tmp_path: Path, crawl_site: CrawlSite
    ) -> None:
        with crawl_sandbox(tmp_path) as root:
            cfg.crawl_exclude_patterns = [*DEFAULT_CRAWL_EXCLUDE_PATTERNS, *USER_EXCLUDES]
            run_add(root, crawl_site.url("/filters/"), *ONE_LEVEL)
            saved = all_saved_text(root)
        assert site_mod.FILTER_TEXT.format(name="keep") in saved
        assert site_mod.QUERY_TEXT.format(query="page=ok") in saved
        assert site_mod.FILTER_TEXT.format(name="drop") not in saved
        assert site_mod.QUERY_TEXT.format(query="lang=7") not in saved

    @pytest.mark.parametrize(
        ("patterns", "expect_saved"),
        [(list(DEFAULT_CRAWL_EXCLUDE_PATTERNS), False), ([], True)],
        ids=["default-list", "empty-list"],
    )
    def test_default_list_drops_the_query_page(
        self,
        tmp_path: Path,
        crawl_site: CrawlSite,
        patterns: list[str],
        expect_saved: bool,
    ) -> None:
        with crawl_sandbox(tmp_path) as root:
            cfg.crawl_exclude_patterns = patterns
            run_add(root, crawl_site.url("/"), *ONE_LEVEL)
            saved = all_saved_text(root)
        assert (site_mod.EXCLUDED_TEXT in saved) is expect_saved


class TestPageAndDepthLimits:
    def test_config_max_depth_applies_without_a_flag(
        self, tmp_path: Path, crawl_site: CrawlSite
    ) -> None:
        with crawl_sandbox(tmp_path) as root:
            cfg.crawl_max_depth = CONFIG_MAX_DEPTH
            run = run_add(root, crawl_site.url("/a/"), "--crawl")
        assert set(run.pages) == {"a/index.md", "a/b/index.md"}

    @pytest.mark.parametrize("setting", ["crawl_max_pages", "crawl_safety_max_pages"])
    def test_config_page_cap_applies_without_a_flag(
        self, tmp_path: Path, crawl_site: CrawlSite, setting: str
    ) -> None:
        with crawl_sandbox(tmp_path) as root:
            setattr(cfg, setting, CONFIG_PAGE_CAP)
            run = run_add(root, crawl_site.url("/wide/"), "--crawl")
        assert MIN_PAGES_BEYOND_SEED <= len(run.pages) <= CONFIG_PAGE_CAP, sorted(run.pages)


class TestRenderMode:
    def test_http_mode_saves_the_page_without_running_its_scripts(
        self, tmp_path: Path, crawl_site: CrawlSite
    ) -> None:
        with crawl_sandbox(tmp_path) as root:
            run = run_add(root, crawl_site.url("/js/"))
        assert site_mod.SCRIPT_TEXT not in run.pages["js/index.md"]


class TestTimeout:
    def test_slow_page_fails_and_the_crawl_saves_the_rest(
        self, tmp_path: Path, crawl_site: CrawlSite
    ) -> None:
        with crawl_sandbox(tmp_path) as root:
            cfg.crawl_timeout = TIMEOUT_S
            run = run_add(root, crawl_site.url("/timeout/"), *ONE_LEVEL)
        assert set(run.pages) == {
            "timeout/index.md",
            "timeout/fast-a/index.md",
            "timeout/fast-b/index.md",
        }
        assert run.report["crawled"] == len(run.pages)


class TestPacing:
    @pytest.mark.parametrize("requests", [1, PARALLEL_REQUESTS])
    def test_concurrent_requests_bounds_the_pages_in_flight(
        self, tmp_path: Path, crawl_site: CrawlSite, requests: int
    ) -> None:
        with crawl_sandbox(tmp_path) as root:
            cfg.crawl_concurrent_requests = requests
            started = time.monotonic()
            run_add(root, crawl_site.url("/pool/"), *ONE_LEVEL)
        peak = crawl_site.peak_in_flight("/pool/p", since=started)
        if requests == 1:
            assert peak == 1
        else:
            assert 1 < peak <= requests

    @pytest.mark.parametrize(
        ("delay_range", "expect_spread"),
        [(JITTER_RANGE_S, True), (0.0, False)],
        ids=["jitter", "no-jitter"],
    )
    def test_delay_range_adds_jitter_between_requests(
        self, tmp_path: Path, crawl_site: CrawlSite, delay_range: float, expect_spread: bool
    ) -> None:
        with crawl_sandbox(tmp_path) as root:
            cfg.crawl_concurrent_requests = 1
            cfg.crawl_mean_delay = 0.0
            cfg.crawl_max_delay_range = delay_range
            started = time.monotonic()
            run_add(root, crawl_site.url("/paced/"), *ONE_LEVEL)
        starts = [r.at for r in crawl_site.requests_under("/paced/p", since=started)]
        assert len(starts) == site_mod.PACED_PAGE_COUNT
        total = sum(later - earlier for earlier, later in pairwise(starts))
        assert (total >= JITTER_MIN_TOTAL_S) is expect_spread, total

    def test_mean_delay_spaces_the_requests(self, tmp_path: Path, crawl_site: CrawlSite) -> None:
        with crawl_sandbox(tmp_path) as root:
            cfg.crawl_concurrent_requests = 1
            cfg.crawl_mean_delay = PACED_DELAY_S
            cfg.crawl_max_delay_range = 0.0
            started = time.monotonic()
            run_add(root, crawl_site.url("/paced/"), *ONE_LEVEL)
        starts = [r.at for r in crawl_site.requests_under("/paced/p", since=started)]
        assert len(starts) == site_mod.PACED_PAGE_COUNT
        gaps = [later - earlier for earlier, later in pairwise(starts)]
        assert min(gaps) >= PACED_DELAY_S - DELAY_MARGIN_S, gaps


class TestRateLimitRetries:
    def test_with_retries_off_a_rate_limited_page_is_requested_once_and_fails(
        self, tmp_path: Path, crawl_site: CrawlSite
    ) -> None:
        with crawl_sandbox(tmp_path) as root:
            cfg.crawl_retry_on_rate_limit = False
            run_add(root, crawl_site.url("/flaky/off/"), *ONE_LEVEL)
            saved = all_saved_text(root)
        assert site_mod.RATE_LIMITED_TEXT not in saved
        assert len(crawl_site.requests_under("/flaky/off/p")) == 1


class TestConcurrentCrawls:
    @pytest.mark.parametrize(("limit", "expect_overlap"), [(1, False), (0, True)])
    async def test_max_concurrent_crawls_queues_the_second_crawl(
        self, tmp_path: Path, crawl_site: CrawlSite, limit: int, expect_overlap: bool
    ) -> None:
        with crawl_sandbox(tmp_path):
            cfg.crawl_max_concurrent = limit
            reset_services()
            started = time.monotonic()
            await asyncio.gather(
                *(
                    crawl_and_save(crawl_site.url(prefix), depth=CONFIG_MAX_DEPTH)
                    for prefix in CONCURRENT_PREFIXES
                )
            )
        first, second = (crawl_site.requests_under(p, since=started) for p in CONCURRENT_PREFIXES)
        assert first and second, (first, second)
        overlap = min(r.at for r in second) < max(r.end for r in first) and min(
            r.at for r in first
        ) < max(r.end for r in second)
        assert overlap is expect_overlap
