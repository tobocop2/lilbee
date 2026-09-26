"""Unit tests for the crawlberg fetcher, run against the stand-in crawlberg module."""

from __future__ import annotations

import ipaddress
import os
import threading
from collections.abc import AsyncIterator, Iterator, MutableMapping

import pytest

from lilbee.core.config.enums import CrawlRenderMode
from lilbee.crawler import crawlberg_fetcher as fetcher_mod
from lilbee.crawler.bootstrap import ChromiumMissingError
from lilbee.crawler.crawlberg_fetcher import (
    _CHROME_ENV,
    _SSRF_ERROR_CODE,
    CRAWLBERG_DENIED_NETWORKS,
    CrawlbergFetcher,
    admitted_networks,
    crawler_available,
)
from lilbee.crawler.models import ConcurrencySpec, FetchedPage, FilterSpec
from lilbee.crawler.url_filter import _BLOCKED_NETWORKS
from tests._crawlberg_stub import Payload, StubCrawlberg, complete, error, page

SEED = "https://example.com/"
LOOPBACK = (ipaddress.ip_network("127.0.0.0/8"), ipaddress.ip_network("::1/128"))


@pytest.fixture(autouse=True)
def _admit_every_url(monkeypatch):
    """lilbee's URL policy admits every page unless a test says otherwise."""
    monkeypatch.setattr(fetcher_mod.url_filter, "validate_crawl_url", lambda url: None)


def _http() -> CrawlbergFetcher:
    return CrawlbergFetcher(render_mode=CrawlRenderMode.HTTP)


async def _recursive(
    fetcher: CrawlbergFetcher,
    *,
    cancel: threading.Event | None = None,
    concurrency: ConcurrencySpec | None = None,
    filters: FilterSpec | None = None,
) -> list[FetchedPage]:
    stream = fetcher.fetch_recursive(
        SEED,
        depth=2,
        max_pages=7,
        timeout=12.5,
        concurrency=concurrency or ConcurrencySpec(semaphore_count=4),
        filters=filters or FilterSpec(),
        cancel=cancel,
    )
    return [fetched async for fetched in stream]


class TestAdmittedNetworks:
    def test_lilbee_blocklist_leaves_only_multicast_to_admit(self):
        assert admitted_networks(_BLOCKED_NETWORKS) == [ipaddress.ip_network("224.0.0.0/4")]

    def test_empty_blocklist_admits_every_network_crawlberg_refuses(self):
        assert admitted_networks(()) == list(CRAWLBERG_DENIED_NETWORKS)

    def test_loopback_carve_out_is_admitted(self):
        blocked = tuple(n for n in _BLOCKED_NETWORKS if n not in LOOPBACK)
        admitted = admitted_networks(blocked)
        assert set(LOOPBACK) <= set(admitted)

    def test_partly_blocked_network_admits_the_rest(self):
        admitted = admitted_networks((ipaddress.ip_network("10.0.0.0/9"),))
        assert ipaddress.ip_network("10.128.0.0/9") in admitted
        assert ipaddress.ip_network("10.0.0.0/8") not in admitted

    def test_wider_blocked_network_removes_the_whole_range(self):
        admitted = admitted_networks((ipaddress.ip_network("192.0.0.0/2"),))
        assert not any(net.overlaps(ipaddress.ip_network("192.168.0.0/16")) for net in admitted)


class TestCrawlConfig:
    async def test_every_crawlberg_default_lilbee_relies_on_is_set_explicitly(self):
        stub = StubCrawlberg([page(SEED, depth=0)])
        with stub.installed():
            await _recursive(_http(), filters=FilterSpec(["/wp-admin"], include_subdomains=True))
        config = stub.config
        assert config["stay_on_domain"] is True
        assert config["allow_subdomains"] is True
        assert config["exclude_paths"] == ["/wp-admin"]
        assert config["max_depth"] == 2
        assert config["max_pages"] == 7
        assert config["max_concurrent"] == 4
        assert config["max_redirects"] == 10
        assert config["respect_robots_txt"] is False
        assert config["soft_http_errors"] is False
        assert config["download_documents"] is False
        assert config["request_timeout"] == 12500
        assert config["content"].kwargs == {
            "remove_navigation": False,
            "remove_forms": False,
            "exclude_selectors": [],
            "preprocessing_preset": "minimal",
            "extract_metadata": False,
        }
        assert config["browser"].kwargs == {
            "mode": "never",
            "timeout": 12500,
            "overall_timeout": 25000,
            "shutdown_timeout": 5000,
        }

    async def test_query_urls_are_matched_kept_apart_and_stripped_of_tracking(self):
        stub = StubCrawlberg()
        with stub.installed():
            await _recursive(_http())
        config = stub.config
        assert config["path_patterns_match_query"] is True
        assert config["dedup_include_query"] is True
        assert config["strip_tracking_params"] is True
        assert config["tracking_params"] == ["utm_*", "fbclid", "gclid", "ref"]

    async def test_pacing_and_retries_map_to_crawlberg(self):
        pacing = ConcurrencySpec(
            semaphore_count=2, mean_delay=0.75, retry_on_rate_limit=True, retry_max_attempts=5
        )
        stub = StubCrawlberg()
        with stub.installed():
            await _recursive(_http(), concurrency=pacing)
        assert stub.config["rate_limit_ms"] == 750
        assert stub.config["rate_limit_jitter_ratio"] == 0.0
        assert stub.config["retry_count"] == 5
        assert stub.config["retry_codes"] == [429, 503]

    @pytest.mark.parametrize(
        ("mean_delay", "delay_range", "rate_limit_ms", "jitter_ratio"),
        [(0.5, 0.5, 750, 1 / 3), (0.0, 2.0, 1000, 1.0), (0.0, 0.0, 0, 0.0)],
        ids=["mean-and-range", "range-only", "no-delay"],
    )
    async def test_waits_span_the_mean_delay_plus_the_delay_range(
        self, mean_delay: float, delay_range: float, rate_limit_ms: int, jitter_ratio: float
    ):
        pacing = ConcurrencySpec(mean_delay=mean_delay, max_delay_range=delay_range)
        stub = StubCrawlberg()
        with stub.installed():
            await _recursive(_http(), concurrency=pacing)
        delay = stub.config["rate_limit_ms"]
        ratio = stub.config["rate_limit_jitter_ratio"]
        assert delay == rate_limit_ms
        assert ratio == pytest.approx(jitter_ratio)
        assert delay * (1 - ratio) == pytest.approx(mean_delay * 1000)
        assert delay * (1 + ratio) == pytest.approx((mean_delay + delay_range) * 1000)

    async def test_retry_backoff_starts_mid_range_and_stops_at_the_max_backoff(self):
        pacing = ConcurrencySpec(
            retry_on_rate_limit=True,
            retry_base_delay_min=1.0,
            retry_base_delay_max=3.0,
            retry_max_backoff=30.0,
        )
        stub = StubCrawlberg()
        with stub.installed():
            await _recursive(_http(), concurrency=pacing)
        assert stub.config["retry_initial_delay_ms"] == 2000
        assert stub.config["retry_max_delay_ms"] == 30000

    async def test_rate_limit_retries_off_disables_retries(self):
        pacing = ConcurrencySpec(retry_on_rate_limit=False, retry_max_attempts=5)
        stub = StubCrawlberg()
        with stub.installed():
            await _recursive(_http(), concurrency=pacing)
        assert stub.config["retry_count"] == 0
        assert stub.config["retry_codes"] == []

    async def test_ssrf_policy_reads_the_blocklist_at_crawl_time(self, monkeypatch):
        blocked = tuple(n for n in _BLOCKED_NETWORKS if n not in LOOPBACK)
        monkeypatch.setattr(fetcher_mod.url_filter, "get_blocked_networks", lambda: blocked)
        stub = StubCrawlberg()
        with stub.installed():
            await _recursive(_http())
        policy = stub.config["ssrf"].kwargs
        assert policy["deny_private"] is True
        assert policy["max_redirects"] == 10
        assert ("cidr", "127.0.0.0/8") in policy["allowlist"]
        assert ("cidr", "::1/128") in policy["allowlist"]


SYSTEM_CHROME = "/usr/bin/system-chrome"


def _record_chrome_at_engine_build(stub: StubCrawlberg) -> list[str | None]:
    """The value of ``CHROME`` each time the stub builds an engine."""
    seen: list[str | None] = []
    build = stub.module.create_engine

    def create_engine(config: object) -> object:
        seen.append(os.environ.get(_CHROME_ENV))
        return build(config)

    stub.module.create_engine = create_engine
    return seen


class _RecordingEnviron(MutableMapping[str, str]):
    """``os.environ`` that records each value written to ``CHROME``, the same on every OS."""

    def __init__(self, environ: MutableMapping[str, str]) -> None:
        self._environ = environ
        self.chrome_writes: list[str] = []

    def __getitem__(self, key: str) -> str:
        return self._environ[key]

    def __setitem__(self, key: str, value: str) -> None:
        if key == _CHROME_ENV:
            self.chrome_writes.append(value)
        self._environ[key] = value

    def __delitem__(self, key: str) -> None:
        del self._environ[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._environ)

    def __len__(self) -> int:
        return len(self._environ)


class TestBrowserMode:
    @pytest.mark.parametrize("before", [None, SYSTEM_CHROME], ids=["unset", "set"])
    async def test_chrome_is_the_shell_before_any_engine_is_built(
        self, monkeypatch, tmp_path, before: str | None
    ):
        shell = tmp_path / "chrome-headless-shell"
        monkeypatch.setattr(fetcher_mod.bootstrap, "headless_shell_executable", lambda: shell)
        if before is not None:
            monkeypatch.setenv(_CHROME_ENV, before)
        stub = StubCrawlberg([page(SEED, depth=0)])
        at_build = _record_chrome_at_engine_build(stub)
        with stub.installed():
            await CrawlbergFetcher(render_mode=CrawlRenderMode.BROWSER).fetch_single(
                SEED, timeout=5
            )
        assert at_build == [str(shell)]
        assert stub.config["browser"].kwargs == {
            "mode": "always",
            "timeout": 5000,
            "overall_timeout": 10000,
            "shutdown_timeout": 5000,
        }

    @pytest.mark.parametrize("before", [None, SYSTEM_CHROME], ids=["unset", "set"])
    async def test_chrome_stays_on_the_shell_after_a_cancelled_crawl_closes(
        self, monkeypatch, tmp_path, before: str | None
    ):
        """crawlberg can launch Chrome after the stream closes, so CHROME is never taken back."""
        shell = tmp_path / "chrome-headless-shell"
        monkeypatch.setattr(fetcher_mod.bootstrap, "headless_shell_executable", lambda: shell)
        if before is not None:
            monkeypatch.setenv(_CHROME_ENV, before)
        cancel = threading.Event()
        cancel.set()
        stub = StubCrawlberg([page(SEED, depth=0), page(f"{SEED}a", depth=1), complete(2)])
        with stub.installed():
            pages = await _recursive(
                CrawlbergFetcher(render_mode=CrawlRenderMode.BROWSER), cancel=cancel
            )
        assert pages == []
        assert stub.closed == 1
        assert os.environ.get(_CHROME_ENV) == str(shell)

    async def test_chrome_stays_on_the_shell_after_a_crawl_runs_to_its_end(
        self, monkeypatch, tmp_path
    ):
        shell = tmp_path / "chrome-headless-shell"
        monkeypatch.setattr(fetcher_mod.bootstrap, "headless_shell_executable", lambda: shell)
        stub = StubCrawlberg(
            [page(SEED, depth=0), page(f"{SEED}a", depth=1), page(f"{SEED}b", depth=1), complete(3)]
        )
        with stub.installed():
            pages = await _recursive(CrawlbergFetcher(render_mode=CrawlRenderMode.BROWSER))
        assert [fetched.url for fetched in pages] == [SEED, f"{SEED}a", f"{SEED}b"]
        assert stub.closed == 1
        assert os.environ.get(_CHROME_ENV) == str(shell)

    async def test_later_crawls_do_not_write_chrome_again(self, monkeypatch, tmp_path):
        """crawlberg may read CHROME from another thread, so it is written once, not per crawl."""
        shell = tmp_path / "chrome-headless-shell"
        monkeypatch.setattr(fetcher_mod.bootstrap, "headless_shell_executable", lambda: shell)
        environ = _RecordingEnviron(os.environ)
        monkeypatch.setattr(os, "environ", environ)
        stub = StubCrawlberg([page(SEED, depth=0)])
        with stub.installed():
            for _ in range(3):
                await CrawlbergFetcher(render_mode=CrawlRenderMode.BROWSER).fetch_single(
                    SEED, timeout=5
                )
        assert environ.chrome_writes == [str(shell)]

    @pytest.mark.parametrize("before", [None, SYSTEM_CHROME], ids=["unset", "set"])
    async def test_http_mode_leaves_chrome_alone(self, monkeypatch, tmp_path, before: str | None):
        shell = tmp_path / "chrome-headless-shell"
        monkeypatch.setattr(fetcher_mod.bootstrap, "headless_shell_executable", lambda: shell)
        if before is not None:
            monkeypatch.setenv(_CHROME_ENV, before)
        stub = StubCrawlberg([page(SEED, depth=0)])
        with stub.installed():
            await CrawlbergFetcher(render_mode=CrawlRenderMode.HTTP).fetch_single(SEED, timeout=5)
        assert os.environ.get(_CHROME_ENV) == before

    async def test_missing_shell_raises_before_any_engine_starts(self, monkeypatch):
        monkeypatch.setattr(fetcher_mod.bootstrap, "headless_shell_executable", lambda: None)
        stub = StubCrawlberg([page(SEED, depth=0)])
        with stub.installed(), pytest.raises(ChromiumMissingError, match="lilbee setup crawler"):
            await CrawlbergFetcher(render_mode=CrawlRenderMode.BROWSER).fetch_single(
                SEED, timeout=5
            )
        assert stub.configs == []
        assert _CHROME_ENV not in os.environ


class TestFetchSingle:
    async def test_single_page_crawl_is_the_seed_alone(self):
        stub = StubCrawlberg([page(SEED, "  # Hello  \n", depth=0), complete(1)])
        with stub.installed():
            fetched = await _http().fetch_single(SEED, timeout=3)
        assert fetched == FetchedPage(url=SEED, markdown="# Hello")
        assert stub.config["max_depth"] == 0
        assert stub.config["max_pages"] == 1
        assert stub.seeds == [SEED]
        assert stub.closed == 1

    async def test_redirected_seed_keeps_the_requested_url(self):
        stub = StubCrawlberg([page("https://example.com/final", "# Final", depth=0)])
        with stub.installed():
            fetched = await _http().fetch_single(SEED, timeout=3)
        assert fetched.url == SEED
        assert fetched.markdown == "# Final"

    async def test_error_event_is_a_failed_page(self):
        stub = StubCrawlberg([error(SEED, "not_found: not_found: " + SEED), complete(0)])
        with stub.installed():
            fetched = await _http().fetch_single(SEED, timeout=3)
        assert fetched.success is False
        assert fetched.error == "not_found: not_found: " + SEED

    @pytest.mark.parametrize(
        "script",
        [[page(SEED, "   ", depth=0)], [page(SEED, None, depth=0)], [complete(0)]],
        ids=["blank-markdown", "no-markdown", "no-page"],
    )
    async def test_page_without_text_is_a_failure(self, script: list[Payload]):
        with StubCrawlberg(script).installed():
            fetched = await _http().fetch_single(SEED, timeout=3)
        assert fetched == FetchedPage(url=SEED, success=False, error="No content extracted")

    async def test_page_on_an_address_lilbee_refuses_is_not_returned(self, monkeypatch):
        def refuse(url: str) -> None:
            raise ValueError("private")

        monkeypatch.setattr(fetcher_mod.url_filter, "validate_crawl_url", refuse)
        with StubCrawlberg([page(SEED, "# Secret", depth=0)]).installed():
            fetched = await _http().fetch_single(SEED, timeout=3)
        assert fetched.success is False
        assert "Secret" not in fetched.markdown


class TestFetchRecursive:
    async def test_streams_pages_and_failures_in_order(self):
        script = [
            page("https://example.com/final", "# Home", depth=0),
            page("https://example.com/a", "# A"),
            error("https://example.com/gone", "not_found: gone"),
            {"type": "progress"},
            complete(2),
        ]
        with StubCrawlberg(script).installed():
            fetched = await _recursive(_http())
        assert fetched == [
            FetchedPage(url=SEED, markdown="# Home"),
            FetchedPage(url="https://example.com/a", markdown="# A"),
            FetchedPage(url="https://example.com/gone", success=False, error="not_found: gone"),
        ]

    async def test_link_refused_by_crawlberg_ssrf_is_dropped(self):
        script = [
            page(SEED, "# Home", depth=0),
            error("http://10.0.0.5/", f"{_SSRF_ERROR_CODE}: http://10.0.0.5/ - denied"),
        ]
        with StubCrawlberg(script).installed():
            fetched = await _recursive(_http())
        assert [f.url for f in fetched] == [SEED]

    async def test_seed_refused_by_crawlberg_ssrf_is_a_failed_page(self):
        refusal = f"{_SSRF_ERROR_CODE}: {SEED} - dns resolution failed"
        with StubCrawlberg([error(SEED, refusal)]).installed():
            fetched = await _recursive(_http())
        assert fetched == [FetchedPage(url=SEED, success=False, error=refusal)]

    async def test_page_on_an_address_lilbee_refuses_is_dropped(self, monkeypatch):
        def refuse_internal(url: str) -> None:
            if "internal" in url:
                raise ValueError("private")

        monkeypatch.setattr(fetcher_mod.url_filter, "validate_crawl_url", refuse_internal)
        script = [page(SEED, "# Home", depth=0), page("https://internal.example.com/", "# X")]
        with StubCrawlberg(script).installed():
            fetched = await _recursive(_http())
        assert [f.url for f in fetched] == [SEED]

    async def test_cancel_stops_the_stream_and_closes_it(self):
        cancel = threading.Event()
        served: list[int] = []

        async def endless() -> AsyncIterator[Payload]:
            for n in range(1000):
                served.append(n)
                if n == 2:
                    cancel.set()
                yield page(f"https://example.com/p{n}", f"# P{n}")

        stub = StubCrawlberg(endless)
        with stub.installed():
            fetched = await _recursive(_http(), cancel=cancel)
        assert [f.url for f in fetched] == ["https://example.com/p0", "https://example.com/p1"]
        assert len(served) == 3
        assert stub.closed == 1

    async def test_early_break_closes_the_stream(self):
        stub = StubCrawlberg([page(f"https://example.com/p{n}") for n in range(5)])
        with stub.installed():
            stream = _http().fetch_recursive(
                SEED,
                depth=None,
                max_pages=None,
                timeout=1,
                concurrency=ConcurrencySpec(),
                filters=FilterSpec(),
            )
            async for _fetched in stream:
                break
            await stream.aclose()
        assert stub.closed == 1

    async def test_engine_is_created_only_when_the_crawl_runs(self):
        stub = StubCrawlberg([page(SEED, depth=0)])
        fetcher = _http()
        with stub.installed():
            assert stub.configs == []
            await _recursive(fetcher)
        assert len(stub.configs) == 1


class TestLifecycle:
    async def test_context_manager_returns_the_fetcher(self):
        fetcher = _http()
        async with fetcher as entered:
            assert entered is fetcher


class TestCrawlerAvailable:
    def test_reports_whether_crawlberg_is_installed(self, monkeypatch):
        seen: list[str] = []

        def find_spec(name: str) -> object | None:
            seen.append(name)
            return None

        crawler_available.cache_clear()
        monkeypatch.setattr(fetcher_mod.importlib.util, "find_spec", find_spec)
        try:
            assert crawler_available() is False
        finally:
            crawler_available.cache_clear()
        assert seen == ["crawlberg"]
