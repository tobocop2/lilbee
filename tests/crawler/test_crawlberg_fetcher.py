"""Unit tests for the crawlberg fetcher, run against the stand-in crawlberg module."""

from __future__ import annotations

import asyncio
import ipaddress
import os
import threading
from collections.abc import AsyncIterator

import pytest

from lilbee.core.config.enums import CrawlRenderMode
from lilbee.crawler import crawlberg_fetcher as fetcher_mod
from lilbee.crawler.bootstrap import ChromiumMissingError, CrawlEngineRefusedError
from lilbee.crawler.crawlberg_fetcher import (
    _SSRF_ERROR_CODE,
    CRAWLBERG_DENIED_NETWORKS,
    CrawlbergFetcher,
    _queued_url,
    admitted_networks,
    crawler_available,
)
from lilbee.crawler.models import ConcurrencySpec, FetchedPage, FilterSpec
from lilbee.crawler.url_filter import _BLOCKED_NETWORKS
from tests._crawlberg_stub import (
    Payload,
    Recorded,
    Refusal,
    StubCrawlberg,
    complete,
    error,
    page,
)

SEED = "https://example.com/"
HTTP_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/116.0.0.0 Safari/537.36"
)
LOOPBACK = (ipaddress.ip_network("127.0.0.0/8"), ipaddress.ip_network("::1/128"))
# The client hints that belong to the browser user agent above.
BRAND_HINT = {"sec-ch-ua": '"Chromium";v="116", "Not_A Brand";v="8", "Google Chrome";v="116"'}
SECURE_ORIGIN_HINTS = {"sec-ch-ua-mobile": "?0", "sec-ch-ua-platform": '"Linux"'}


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
    depth: int = 2,
    seed: str = SEED,
) -> list[FetchedPage]:
    stream = fetcher.fetch_recursive(
        seed,
        depth=depth,
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
            "chrome_path": None,
            "chrome_args": [],
        }

    async def test_http_mode_names_the_http_user_agent(self):
        stub = StubCrawlberg([page(SEED, depth=0)])
        with stub.installed():
            await _recursive(_http())
            await _http().fetch_single(SEED, timeout=5)
        assert [config.kwargs["user_agent"] for config in stub.configs] == [HTTP_USER_AGENT] * 2

    async def test_browser_mode_names_the_browser_user_agent(self, monkeypatch, tmp_path):
        shell = tmp_path / "chrome-headless-shell"
        monkeypatch.setattr(fetcher_mod.bootstrap, "headless_shell_executable", lambda: shell)
        stub = StubCrawlberg([page(SEED, depth=0)])
        with stub.installed():
            await _recursive(_browser())
            await _browser().fetch_single(SEED, timeout=5)
        assert [config.kwargs["user_agent"] for config in stub.configs] == [BROWSER_USER_AGENT] * 2

    @pytest.mark.parametrize(
        "seed",
        [
            SEED,
            "http://127.0.0.1:8000/docs",
            "http://localhost:8000/",
            "http://docs.localhost/",
            "http://[::1]:8000/",
        ],
        ids=["https", "loopback-v4", "localhost", "localhost-subdomain", "loopback-v6"],
    )
    async def test_a_browser_crawl_of_a_secure_origin_sends_three_client_hints(
        self, monkeypatch, tmp_path, seed: str
    ):
        shell = tmp_path / "chrome-headless-shell"
        monkeypatch.setattr(fetcher_mod.bootstrap, "headless_shell_executable", lambda: shell)
        stub = StubCrawlberg([page(seed, depth=0)])
        with stub.installed():
            await _recursive(_browser(), seed=seed)
            await _browser().fetch_single(seed, timeout=5)
        sent = [config.kwargs["custom_headers"] for config in stub.configs]
        assert sent == [{**BRAND_HINT, **SECURE_ORIGIN_HINTS}] * 2

    @pytest.mark.parametrize(
        "seed",
        ["http://example.com/", "http://192.168.1.5/", "http://notlocalhost/"],
        ids=["http", "private-address", "name-that-ends-like-localhost"],
    )
    async def test_a_browser_crawl_of_any_other_origin_sends_the_brand_hint_alone(
        self, monkeypatch, tmp_path, seed: str
    ):
        shell = tmp_path / "chrome-headless-shell"
        monkeypatch.setattr(fetcher_mod.bootstrap, "headless_shell_executable", lambda: shell)
        stub = StubCrawlberg([page(seed, depth=0)])
        with stub.installed():
            await _recursive(_browser(), seed=seed)
            await _browser().fetch_single(seed, timeout=5)
        assert [config.kwargs["custom_headers"] for config in stub.configs] == [BRAND_HINT] * 2

    async def test_http_mode_sends_no_client_hints(self):
        stub = StubCrawlberg([page(SEED, depth=0)])
        with stub.installed():
            await _recursive(_http())
            await _http().fetch_single(SEED, timeout=5)
        assert [config.kwargs["custom_headers"] for config in stub.configs] == [{}, {}]
        assert [config.kwargs["user_agent"] for config in stub.configs] == [HTTP_USER_AGENT] * 2

    async def test_whole_urls_are_matched_and_query_urls_kept_apart_and_stripped_of_tracking(self):
        stub = StubCrawlberg()
        with stub.installed():
            await _recursive(_http())
        config = stub.config
        assert config["path_patterns_match_url"] is True
        assert "path_patterns_match_query" not in config
        assert config["dedup_include_query"] is True
        assert config["strip_tracking_params"] is True
        assert config["tracking_params"] == [
            "utm_*",
            "fbclid",
            "gclid",
            "msclkid",
            "yclid",
            "mc_cid",
            "mc_eid",
            "_hsenc",
            "_hsmi",
            "hsCtaTracking",
            "mkt_*",
            "trk",
            "trkInfo",
            "dm_i",
            "vero_id",
            "vero_conv",
            "oly_anon_id",
            "oly_enc_id",
            "igshid",
            "pk_*",
            "_ga",
            "affiliate",
            "aff_id",
            "aff_ref",
            "aff",
            "partner",
            "srsltid",
            "replytocom",
            "ref",
        ]

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

    @pytest.mark.parametrize(
        ("configured", "sent", "warnings"),
        [(20, 20, 0), (21, 20, 1), (1000, 20, 1)],
        ids=["at-the-limit", "one-over", "far-over"],
    )
    async def test_a_retry_count_over_crawlbergs_limit_is_sent_as_the_limit(
        self, caplog, configured: int, sent: int, warnings: int
    ):
        pacing = ConcurrencySpec(retry_on_rate_limit=True, retry_max_attempts=configured)
        stub = StubCrawlberg([page(SEED, depth=0)], refuse=_refuses_many_retries)
        with stub.installed(), caplog.at_level("WARNING", logger=fetcher_mod.__name__):
            fetched = await _recursive(_http(), concurrency=pacing)
        assert [one.url for one in fetched] == [SEED]
        assert stub.config["retry_count"] == sent
        capped = [r.getMessage() for r in caplog.records if "crawl_retry_max_attempts" in r.message]
        assert len(capped) == warnings
        assert all(str(configured) in message and "20" in message for message in capped)

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


CHROME_ENV = "CHROME"
SYSTEM_CHROME = "/usr/bin/system-chrome"
FLAGS = ["--lang=fr", "--disable-gpu"]


def _browser(chrome_args: list[str] | None = None) -> CrawlbergFetcher:
    return CrawlbergFetcher(render_mode=CrawlRenderMode.BROWSER, chrome_args=chrome_args or [])


class TestBrowserMode:
    async def test_the_crawl_names_the_headless_shell_and_the_launch_flags(
        self, monkeypatch, tmp_path
    ):
        shell = tmp_path / "chrome-headless-shell"
        monkeypatch.setattr(fetcher_mod.bootstrap, "headless_shell_executable", lambda: shell)
        stub = StubCrawlberg([page(SEED, depth=0)])
        with stub.installed():
            await _browser(FLAGS).fetch_single(SEED, timeout=5)
        assert stub.config["browser"].kwargs == {
            "mode": "always",
            "timeout": 5000,
            "overall_timeout": 10000,
            "shutdown_timeout": 5000,
            "chrome_path": str(shell),
            "chrome_args": FLAGS,
        }

    async def test_a_recursive_crawl_names_the_same_shell_and_flags(self, monkeypatch, tmp_path):
        shell = tmp_path / "chrome-headless-shell"
        monkeypatch.setattr(fetcher_mod.bootstrap, "headless_shell_executable", lambda: shell)
        stub = StubCrawlberg([page(SEED, depth=0), complete(1)])
        with stub.installed():
            pages = await _recursive(_browser(FLAGS))
        assert [fetched.url for fetched in pages] == [SEED]
        assert stub.config["browser"].kwargs["chrome_path"] == str(shell)
        assert stub.config["browser"].kwargs["chrome_args"] == FLAGS

    async def test_flags_given_to_one_fetcher_do_not_change_with_the_callers_list(
        self, monkeypatch, tmp_path
    ):
        shell = tmp_path / "chrome-headless-shell"
        monkeypatch.setattr(fetcher_mod.bootstrap, "headless_shell_executable", lambda: shell)
        flags = list(FLAGS)
        fetcher = _browser(flags)
        flags.append("--mute-audio")
        stub = StubCrawlberg([page(SEED, depth=0)])
        with stub.installed():
            await fetcher.fetch_single(SEED, timeout=5)
        assert stub.config["browser"].kwargs["chrome_args"] == FLAGS

    @pytest.mark.parametrize("before", [None, SYSTEM_CHROME], ids=["unset", "set"])
    async def test_browser_mode_leaves_the_chrome_variable_alone(
        self, monkeypatch, tmp_path, before: str | None
    ):
        shell = tmp_path / "chrome-headless-shell"
        monkeypatch.setattr(fetcher_mod.bootstrap, "headless_shell_executable", lambda: shell)
        monkeypatch.delenv(CHROME_ENV, raising=False)
        if before is not None:
            monkeypatch.setenv(CHROME_ENV, before)
        stub = StubCrawlberg([page(SEED, depth=0)])
        with stub.installed():
            await _browser().fetch_single(SEED, timeout=5)
        assert stub.config["browser"].kwargs["chrome_path"] == str(shell)
        assert os.environ.get(CHROME_ENV) == before

    async def test_http_mode_names_no_chrome_and_no_flags(self, monkeypatch, tmp_path):
        shell = tmp_path / "chrome-headless-shell"
        monkeypatch.setattr(fetcher_mod.bootstrap, "headless_shell_executable", lambda: shell)
        fetcher = CrawlbergFetcher(render_mode=CrawlRenderMode.HTTP, chrome_args=FLAGS)
        stub = StubCrawlberg([page(SEED, depth=0)])
        with stub.installed():
            await fetcher.fetch_single(SEED, timeout=5)
        assert stub.config["browser"].kwargs["mode"] == "never"
        assert stub.config["browser"].kwargs["chrome_path"] is None
        assert stub.config["browser"].kwargs["chrome_args"] == []

    async def test_http_mode_does_not_look_for_the_shell(self, monkeypatch):
        looked: list[bool] = []
        monkeypatch.setattr(
            fetcher_mod.bootstrap, "headless_shell_executable", lambda: looked.append(True)
        )
        stub = StubCrawlberg([page(SEED, "# Page", depth=0)])
        with stub.installed():
            fetched = await _http().fetch_single(SEED, timeout=5)
        assert fetched.markdown == "# Page"
        assert looked == []

    async def test_missing_shell_raises_before_any_engine_starts(self, monkeypatch):
        monkeypatch.setattr(fetcher_mod.bootstrap, "headless_shell_executable", lambda: None)
        stub = StubCrawlberg([page(SEED, depth=0)])
        with stub.installed(), pytest.raises(ChromiumMissingError, match="lilbee setup crawler"):
            await _browser().fetch_single(SEED, timeout=5)
        assert stub.configs == []


# The text crawlberg 1.10.2 raises for ``--headless=new``, for 21 retries, for a depth of 101
# and for a pattern its regex engine cannot parse.
HEADLESS_REASON = (
    "invalid_config: browser.chrome_args must not set --headless; crawlberg sets it to run Chrome"
)
RETRY_REASON = "invalid_config: retry_count must be <= 20 (got 21)"
DEPTH_REASON = "invalid_config: max_depth must be <= 100 (got 101)"
CONDITIONAL_PATTERN = "(?P<n>a)(?(n)b|c)"
# A pattern crawlberg refuses whose text holds the name crawlberg gives the launch flags.
FLAG_NAMING_PATTERN = "(?P<n>browser.chrome_args)(?(n)b|c)"
PATTERN_REASON = (
    "invalid_config: invalid exclude_path regex '{pattern}': regex parse error:\n"
    "    {pattern}\nerror: unrecognized flag"
)


def _refuses_headless(config: Recorded) -> str | None:
    """Refuse any config whose browser carries ``--headless=new``, as crawlberg does."""
    flags = config.kwargs["browser"].kwargs["chrome_args"]
    return HEADLESS_REASON if "--headless=new" in flags else None


def _refuses_many_retries(config: Recorded) -> str | None:
    """Refuse a crawl config with more than 20 retries, whatever its launch flags."""
    return RETRY_REASON if config.kwargs.get("retry_count", 0) > 20 else None


def _refuses_deep_crawls(config: Recorded) -> str | None:
    """Refuse a crawl config with a depth over 100, whatever its launch flags."""
    return DEPTH_REASON if config.kwargs["max_depth"] > 100 else None


def _refuses_conditional_patterns(config: Recorded) -> str | None:
    """Refuse the first exclude pattern that holds a conditional group, as crawlberg does."""
    refused = [pattern for pattern in config.kwargs["exclude_paths"] if "(?(" in pattern]
    return PATTERN_REASON.format(pattern=refused[0]) if refused else None


# What crawlberg 1.9.0 raises for a number over its integer width: not a ``RuntimeError``.
OVERSIZED_REASON = "int too big to convert"
ENGINE_ERRORS = [OverflowError(OVERSIZED_REASON), TypeError("argument 'config'"), KeyError("k")]


def _raises(exc: BaseException) -> Refusal:
    """An engine creation that raises *exc* itself, whatever its type."""

    def create(config: Recorded) -> str | None:
        raise exc

    return create


class TestRefusedEngine:
    @pytest.fixture(autouse=True)
    def _shell(self, monkeypatch, tmp_path):
        shell = tmp_path / "chrome-headless-shell"
        monkeypatch.setattr(fetcher_mod.bootstrap, "headless_shell_executable", lambda: shell)

    async def test_a_single_fetch_names_the_setting_and_gives_crawlbergs_reason(self):
        stub = StubCrawlberg([page(SEED, depth=0)], refuse=_refuses_headless)
        with stub.installed(), pytest.raises(CrawlEngineRefusedError) as refused:
            await _browser(["--lang=fr", "--headless=new"]).fetch_single(SEED, timeout=5)
        message = str(refused.value)
        assert message.startswith("The crawl_browser_extra_args setting")
        assert message.endswith(HEADLESS_REASON)
        assert stub.seeds == []

    async def test_a_recursive_fetch_names_the_setting_and_gives_crawlbergs_reason(self):
        stub = StubCrawlberg([page(SEED, depth=0), complete(1)], refuse=_refuses_headless)
        with stub.installed(), pytest.raises(CrawlEngineRefusedError) as refused:
            await _recursive(_browser(["--headless=new"]))
        assert "crawl_browser_extra_args" in str(refused.value)
        assert str(refused.value).endswith(HEADLESS_REASON)

    async def test_a_refusal_of_another_field_gives_crawlbergs_reason_and_names_no_setting(self):
        stub = StubCrawlberg([page(SEED, depth=0), complete(1)], refuse=_refuses_deep_crawls)
        with stub.installed(), pytest.raises(CrawlEngineRefusedError) as refused:
            await _recursive(_browser(["--lang=fr"]), depth=101)
        assert str(refused.value) == f"crawlberg refuses to start this crawl: {DEPTH_REASON}"
        assert stub.seeds == []

    @pytest.mark.parametrize("pattern", [CONDITIONAL_PATTERN, FLAG_NAMING_PATTERN])
    async def test_a_refused_exclude_pattern_names_the_pattern_setting(self, pattern: str):
        stub = StubCrawlberg(
            [page(SEED, depth=0), complete(1)], refuse=_refuses_conditional_patterns
        )
        filters = FilterSpec(exclude_patterns=[r"/ok/", pattern])
        with stub.installed(), pytest.raises(CrawlEngineRefusedError) as refused:
            await _recursive(_http(), filters=filters)
        message = str(refused.value)
        assert message.startswith(
            "The crawl_exclude_patterns setting holds a pattern that crawlberg refuses: "
        )
        assert message.endswith(PATTERN_REASON.format(pattern=pattern))
        assert "crawl_browser_extra_args" not in message
        assert stub.seeds == []

    async def test_a_refusal_keeps_crawlbergs_error_as_its_cause(self):
        stub = StubCrawlberg([page(SEED, depth=0)], refuse=_refuses_headless)
        with stub.installed(), pytest.raises(CrawlEngineRefusedError) as refused:
            await _browser(["--headless=new"]).fetch_single(SEED, timeout=5)
        cause = refused.value.__cause__
        assert type(cause) is RuntimeError
        assert str(cause) == HEADLESS_REASON

    @pytest.mark.parametrize("raised", ENGINE_ERRORS, ids=lambda exc: type(exc).__name__)
    async def test_an_engine_creation_error_of_any_type_is_the_refusal(self, raised: Exception):
        stub = StubCrawlberg([page(SEED, depth=0), complete(1)], refuse=_raises(raised))
        with stub.installed(), pytest.raises(CrawlEngineRefusedError) as refused:
            await _recursive(_http())
        assert str(refused.value) == f"crawlberg refuses to start this crawl: {raised}"
        assert refused.value.__cause__ is raised
        assert stub.seeds == []

    @pytest.mark.parametrize(
        ("delay", "raised", "text"),
        [
            (float("inf"), OverflowError, "cannot convert float infinity to integer"),
            (float("nan"), ValueError, "cannot convert float NaN to integer"),
        ],
        ids=["inf", "nan"],
    )
    async def test_an_error_while_lilbee_builds_the_config_is_not_a_refusal(
        self, delay: float, raised: type[Exception], text: str
    ):
        stub = StubCrawlberg([page(SEED, depth=0), complete(1)])
        with stub.installed(), pytest.raises(raised, match=text) as failed:
            await _recursive(_http(), concurrency=ConcurrencySpec(mean_delay=delay))
        assert type(failed.value) is raised
        assert stub.configs == []

    @pytest.mark.parametrize(
        "raised",
        [KeyboardInterrupt(), asyncio.CancelledError()],
        ids=lambda exc: type(exc).__name__,
    )
    async def test_an_interrupt_or_a_task_cancel_from_engine_creation_passes_through(
        self, raised: BaseException
    ):
        stub = StubCrawlberg([page(SEED, depth=0), complete(1)], refuse=_raises(raised))
        with stub.installed(), pytest.raises(type(raised)) as passed:
            await _recursive(_http())
        assert passed.value is raised
        assert stub.seeds == []

    async def test_a_refusal_wins_over_a_task_cancel_requested_while_the_engine_is_built(self):
        def cancel_then_refuse(config: Recorded) -> str | None:
            task = asyncio.current_task()
            assert task is not None
            task.cancel()
            raise OverflowError(OVERSIZED_REASON)

        stub = StubCrawlberg([page(SEED, depth=0), complete(1)], refuse=cancel_then_refuse)
        with stub.installed():
            crawl = asyncio.ensure_future(_recursive(_http()))
            with pytest.raises(CrawlEngineRefusedError) as refused:
                await crawl
        assert str(refused.value) == f"crawlberg refuses to start this crawl: {OVERSIZED_REASON}"
        assert not crawl.cancelled()

    async def test_an_error_from_the_call_that_opens_the_stream_is_not_a_refusal(self, monkeypatch):
        def refuse_to_open(engine: object, url: str) -> AsyncIterator[Payload]:
            raise OverflowError(OVERSIZED_REASON)

        stub = StubCrawlberg([page(SEED, depth=0), complete(1)])
        monkeypatch.setattr(stub.module, "crawl_stream", refuse_to_open)
        with stub.installed(), pytest.raises(OverflowError, match=OVERSIZED_REASON) as failed:
            await _recursive(_http())
        assert type(failed.value) is OverflowError
        assert len(stub.configs) == 1

    async def test_an_error_from_the_stream_is_not_a_refusal(self):
        async def dropped() -> AsyncIterator[Payload]:
            raise OverflowError(OVERSIZED_REASON)
            yield  # makes this an async generator

        stub = StubCrawlberg(dropped)
        with stub.installed(), pytest.raises(OverflowError, match=OVERSIZED_REASON) as failed:
            await _recursive(_http())
        assert type(failed.value) is OverflowError
        assert stub.seeds == [SEED]

    async def test_http_mode_does_not_send_a_refused_flag(self):
        fetcher = CrawlbergFetcher(render_mode=CrawlRenderMode.HTTP, chrome_args=["--headless=new"])
        stub = StubCrawlberg([page(SEED, "# Page", depth=0)], refuse=_refuses_headless)
        with stub.installed():
            fetched = await fetcher.fetch_single(SEED, timeout=5)
        assert fetched.markdown == "# Page"


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


# One address for each query parameter a campaign or analytics vendor adds to a link.
TRACKED_ADDRESSES = [
    f"https://x.dev/p?{name}=1"
    for name in (
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "fbclid",
        "gclid",
        "msclkid",
        "yclid",
        "mc_cid",
        "mc_eid",
        "_hsenc",
        "_hsmi",
        "hsCtaTracking",
        "mkt_tok",
        "trk",
        "trkInfo",
        "dm_i",
        "vero_id",
        "vero_conv",
        "oly_anon_id",
        "oly_enc_id",
        "igshid",
        "pk_campaign",
        "pk_source",
        "pk_medium",
        "_ga",
        "affiliate",
        "aff_id",
        "aff_ref",
        "aff",
        "partner",
        "srsltid",
        "replytocom",
        "ref",
    )
]


class TestQueuedUrl:
    @pytest.mark.parametrize("link", TRACKED_ADDRESSES)
    def test_a_tracking_parameter_is_stripped(self, link: str):
        assert _queued_url(link) == "https://x.dev/p"

    @pytest.mark.parametrize(
        "link",
        [
            "https://x.dev/p?share=twitter",
            "https://x.dev/p?reference=1",
            "https://x.dev/p?traffic=1",
            "https://x.dev/p?UTM_SOURCE=1",
            "https://x.dev/p?id=1&lang=en",
            "https://x.dev/p",
        ],
        ids=["share", "longer-name", "name-that-starts-like-trk", "other-case", "plain", "bare"],
    )
    def test_any_other_address_is_kept_as_written(self, link: str):
        assert _queued_url(link) == link

    @pytest.mark.parametrize(
        ("link", "queued"),
        [
            ("https://x.dev/p?id=1&utm_medium=mail&tab=2", "https://x.dev/p?id=1&tab=2"),
            ("https://x.dev/p?id=1#section", "https://x.dev/p?id=1"),
            ("https://x.dev/p#section", "https://x.dev/p"),
            ("https://x.dev/p?amp", "https://x.dev/p?amp="),
            ("https://x.dev/p?q=a%20b", "https://x.dev/p?q=a+b"),
            ("https://x.dev/p?", "https://x.dev/p"),
            ("https://x.dev/p?q=~x*y", "https://x.dev/p?q=%7Ex*y"),
            ("https://x.dev/p?a-b._c=d/e:f", "https://x.dev/p?a-b._c=d%2Fe%3Af"),
            ("https://x.dev/p?q=caf%C3%A9", "https://x.dev/p?q=caf%C3%A9"),
        ],
        ids=[
            "mixed",
            "fragment",
            "fragment-alone",
            "bare-name",
            "space",
            "empty-query",
            "tilde-and-star",
            "unreserved-and-reserved",
            "non-ascii",
        ],
    )
    def test_the_rest_of_the_query_is_kept_in_order_without_the_fragment(
        self, link: str, queued: str
    ):
        assert _queued_url(link) == queued


FEED = "https://example.com/feed/"
FEED_PATTERN = r"/feed/?$"
TAG_PATTERN = r"/tag/"
EXCLUDES = FilterSpec([FEED_PATTERN, TAG_PATTERN])
EXCLUDED_START = "Excluded "
SUMMARY_START = "Links of the crawl of "


def _summary(total: int, per_pattern: str) -> str:
    """The warning for a crawl of the seed that excluded *total* links."""
    return (
        f"Links of the crawl of {SEED} that an exclude pattern matched and the crawl did not "
        f"follow: {total}. {per_pattern}. "
        "The crawl_exclude_patterns setting holds the patterns. "
        "Run the crawl at the INFO log level to list every address."
    )


def _named(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The message of each log line that names one excluded address."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "INFO" and record.getMessage().startswith(EXCLUDED_START)
    ]


def _summaries(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The message of each warning that sums up the excluded addresses."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "WARNING" and record.getMessage().startswith(SUMMARY_START)
    ]


class TestExcludedLinks:
    @pytest.fixture(autouse=True)
    def _info_log(self, caplog):
        caplog.set_level("INFO", logger=fetcher_mod.__name__)

    async def test_an_address_two_pages_link_to_is_named_once_with_its_pattern(self, caplog):
        script = [
            page(SEED, depth=0, links=[FEED, "https://example.com/a"]),
            page("https://example.com/a", links=[FEED, "https://example.com/b"]),
        ]
        with StubCrawlberg(script).installed():
            await _recursive(_http(), filters=EXCLUDES)
        assert _named(caplog) == [f"Excluded {FEED}: it matches the exclude pattern {FEED_PATTERN}"]

    async def test_the_first_pattern_that_matches_is_the_one_named(self, caplog):
        link = "https://example.com/tag/feed/"
        script = [page(SEED, depth=0, links=[link])]
        with StubCrawlberg(script).installed():
            await _recursive(_http(), filters=FilterSpec([TAG_PATTERN, FEED_PATTERN]))
        assert _named(caplog) == [f"Excluded {link}: it matches the exclude pattern {TAG_PATTERN}"]

    async def test_an_address_is_matched_without_its_tracking_parameters(self, caplog):
        script = [page(SEED, depth=0, links=[f"{FEED}?utm_source=x#top"])]
        with StubCrawlberg(script).installed():
            await _recursive(_http(), filters=EXCLUDES)
        assert _named(caplog) == [f"Excluded {FEED}: it matches the exclude pattern {FEED_PATTERN}"]

    async def test_a_pattern_for_a_tracking_parameter_excludes_nothing(self, caplog):
        script = [page(SEED, depth=0, links=["https://example.com/news/?msclkid=9"])]
        with StubCrawlberg(script).installed():
            await _recursive(_http(), filters=FilterSpec([r"[?&]msclkid="]))
        assert _named(caplog) == []
        assert _summaries(caplog) == []

    async def test_two_links_that_strip_to_one_address_name_it_once(self, caplog):
        script = [
            page(SEED, depth=0, links=[FEED, f"{FEED}?utm_source=x", f"{FEED}#top"]),
            page("https://example.com/a", links=[f"{FEED}?utm_source=x", f"{FEED}?fbclid=y"]),
        ]
        with StubCrawlberg(script).installed():
            await _recursive(_http(), filters=EXCLUDES)
        assert _named(caplog) == [f"Excluded {FEED}: it matches the exclude pattern {FEED_PATTERN}"]
        assert _summaries(caplog) == [_summary(1, f"{FEED_PATTERN} matches 1, for example {FEED}")]

    @pytest.mark.parametrize("include_subdomains", [False, True])
    async def test_an_address_on_another_site_is_not_named(self, caplog, include_subdomains: bool):
        links = ["https://other.example/feed/", "https://notexample.com/feed/"]
        filters = FilterSpec([FEED_PATTERN], include_subdomains=include_subdomains)
        with StubCrawlberg([page(SEED, depth=0, links=links)]).installed():
            await _recursive(_http(), filters=filters)
        assert _named(caplog) == []

    async def test_an_address_on_a_subdomain_is_not_named_in_a_crawl_of_the_seed_host(self, caplog):
        script = [page(SEED, depth=0, links=["https://docs.example.com/feed/"])]
        with StubCrawlberg(script).installed():
            await _recursive(_http(), filters=FilterSpec([FEED_PATTERN]))
        assert _named(caplog) == []

    async def test_an_address_on_a_subdomain_is_named_in_a_crawl_that_follows_subdomains(
        self, caplog
    ):
        link = "https://docs.example.com/feed/"
        filters = FilterSpec([FEED_PATTERN], include_subdomains=True)
        with StubCrawlberg([page(SEED, depth=0, links=[link])]).installed():
            await _recursive(_http(), filters=filters)
        assert _named(caplog) == [f"Excluded {link}: it matches the exclude pattern {FEED_PATTERN}"]

    async def test_the_links_of_a_page_at_the_depth_limit_are_not_named(self, caplog):
        script = [
            page(SEED, depth=0, links=["https://example.com/a"]),
            page("https://example.com/a", depth=1, links=[FEED]),
        ]
        with StubCrawlberg(script).installed():
            await _recursive(_http(), filters=EXCLUDES, depth=1)
        assert _named(caplog) == []
        assert _summaries(caplog) == []

    async def test_a_crawl_with_no_depth_limit_names_links_at_any_depth(self, caplog):
        script = [page("https://example.com/deep", depth=50, links=[FEED])]
        with StubCrawlberg(script).installed():
            stream = _http().fetch_recursive(
                SEED,
                depth=None,
                max_pages=None,
                timeout=1,
                concurrency=ConcurrencySpec(),
                filters=EXCLUDES,
            )
            assert [fetched.url async for fetched in stream] == ["https://example.com/deep"]
        assert _named(caplog) == [f"Excluded {FEED}: it matches the exclude pattern {FEED_PATTERN}"]

    async def test_one_warning_gives_the_total_and_each_patterns_count_and_example(self, caplog):
        tags = [f"https://example.com/tag/{name}" for name in ("a", "b")]
        script = [
            page(SEED, depth=0, links=[tags[0], "https://example.com/kept", FEED]),
            page("https://example.com/kept", links=[tags[1], tags[0]]),
            complete(2),
        ]
        with StubCrawlberg(script).installed():
            await _recursive(_http(), filters=EXCLUDES)
        assert _summaries(caplog) == [
            _summary(
                3,
                f"{TAG_PATTERN} matches 2, for example {tags[0]}; "
                f"{FEED_PATTERN} matches 1, for example {FEED}",
            )
        ]

    async def test_a_crawl_that_excludes_nothing_logs_no_warning(self, caplog):
        script = [page(SEED, depth=0, links=["https://example.com/a"]), complete(1)]
        with StubCrawlberg(script).installed():
            await _recursive(_http(), filters=EXCLUDES)
        assert _named(caplog) == []
        assert _summaries(caplog) == []

    async def test_a_cancelled_crawl_still_gets_its_warning(self, caplog):
        cancel = threading.Event()

        async def cancelled_after_the_seed() -> AsyncIterator[Payload]:
            yield page(SEED, depth=0, links=[FEED])
            cancel.set()
            yield page("https://example.com/a", links=["https://example.com/tag/late"])

        with StubCrawlberg(cancelled_after_the_seed).installed():
            fetched = await _recursive(_http(), filters=EXCLUDES, cancel=cancel)
        assert [one.url for one in fetched] == [SEED]
        assert _summaries(caplog) == [_summary(1, f"{FEED_PATTERN} matches 1, for example {FEED}")]

    async def test_a_crawl_its_reader_closes_early_gets_one_warning(self, caplog):
        script = [page(SEED, depth=0, links=[FEED]), page("https://example.com/a")]
        with StubCrawlberg(script).installed():
            stream = _http().fetch_recursive(
                SEED,
                depth=2,
                max_pages=None,
                timeout=1,
                concurrency=ConcurrencySpec(),
                filters=EXCLUDES,
            )
            async for _fetched in stream:
                break
            await stream.aclose()
        assert _summaries(caplog) == [_summary(1, f"{FEED_PATTERN} matches 1, for example {FEED}")]

    async def test_a_crawl_whose_stream_fails_gets_one_warning(self, caplog):
        async def fails_after_the_seed() -> AsyncIterator[Payload]:
            yield page(SEED, depth=0, links=[FEED])
            raise OverflowError(OVERSIZED_REASON)

        with (
            StubCrawlberg(fails_after_the_seed).installed(),
            pytest.raises(OverflowError, match=OVERSIZED_REASON),
        ):
            await _recursive(_http(), filters=EXCLUDES)
        assert _summaries(caplog) == [_summary(1, f"{FEED_PATTERN} matches 1, for example {FEED}")]

    async def test_a_page_event_without_links_names_no_address(self, caplog):
        without_links = {"type": "page", "result": {"url": SEED, "depth": 0, "markdown": None}}
        with StubCrawlberg([without_links]).installed():
            fetched = await _recursive(_http(), filters=EXCLUDES)
        assert [one.url for one in fetched] == [SEED]
        assert _named(caplog) == []

    async def test_a_single_page_fetch_names_no_address(self, caplog):
        with StubCrawlberg([page(SEED, depth=0, links=[FEED])]).installed():
            await _http().fetch_single(SEED, timeout=3)
        assert _named(caplog) == []
        assert _summaries(caplog) == []


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
