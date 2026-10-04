"""Real crawlberg against a local site: SSRF, cancel, threads, limits, excludes, retries, Chrome."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import stat
import threading
import time
from collections.abc import Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from click.testing import Result
from typer.testing import CliRunner

crawlberg = pytest.importorskip("crawlberg")

from lilbee.cli.app import app  # noqa: E402
from lilbee.core.config import Config, cfg  # noqa: E402
from lilbee.core.config.enums import CrawlRenderMode  # noqa: E402
from lilbee.crawler import bootstrap, crawl_and_save, url_filter  # noqa: E402
from lilbee.crawler.bootstrap import CrawlEngineRefusedError  # noqa: E402
from lilbee.crawler.crawlberg_fetcher import (  # noqa: E402
    _SSRF_ERROR_CODE,
    CRAWLBERG_DENIED_NETWORKS,
    CrawlbergFetcher,
    admitted_networks,
)
from lilbee.crawler.models import ConcurrencySpec, FetchedPage, FilterSpec  # noqa: E402
from lilbee.crawler.task import CrawlTask, TaskStatus, run_crawl  # noqa: E402
from tests._private_mode import posix_only  # noqa: E402
from tests.integration._crawl_site import (  # noqa: E402
    REAL_BROWSERS_PATH,
    full_crawl_only,
    require_chromium,
    windows_proactor_loop,
)

LOOPBACK = (ipaddress.ip_network("127.0.0.0/8"), ipaddress.ip_network("::1/128"))
BROWSER = CrawlRenderMode.BROWSER
HTTP = CrawlRenderMode.HTTP
# lilbee blocks the whole NAT64 prefix; crawlberg refuses a NAT64 address only when the
# IPv4 address it carries is refused.
NAT64 = ipaddress.ip_network("64:ff9b::/96")
# IPv6 literals that carry the private IPv4 address 10.0.0.1 or loopback.
IPV6_FORMS_OF_PRIVATE_IPV4 = {
    "6to4": "[2002:7f00:1::]",
    "ipv4-translated": "[::ffff:0:10.0.0.1]",
    "ipv4-compatible": "[::10.0.0.1]",
    "nat64": "[64:ff9b::a00:1]",
    "local-use-nat64": "[64:ff9b:1::a00:1]",
}
# 6to4 of the public address 8.8.8.8.
IPV6_FORM_OF_PUBLIC_IPV4 = "[2002:808:808::]"
SLOW_PAGES = 40
SLOW_DELAY_S = 0.3
CANCEL_BOUND_S = 2.0
# A request sent just before the crawl stops can reach the site just after it.
LATE_START_MARGIN_S = 0.1
SETTLE_S = 1.0
RETRY_ATTEMPTS = 2
RETRY_DELAY_S = 0.05
RETRY_STATUSES = {
    "/retry/broken": HTTPStatus.INTERNAL_SERVER_ERROR,
    "/retry/busy": HTTPStatus.SERVICE_UNAVAILABLE,
}
SCOPE_LINKS = ("/scope/skip/a", "/scope/keep/b", "/scope/la/drop", "/scope/la/keepme")
# Flag lists crawlberg refuses, each with a part of the reason it gives.
REFUSED_FLAGS: tuple[tuple[list[str], str], ...] = (
    (["--headless=new"], "--headless"),
    (["--user-data-dir=/x"], "--user-data-dir"),
    (["disable-gpu"], '"disable-gpu"'),
    (["--Lang=fr"], '"--Lang=fr"'),
    (["--disable-gpu", "--disable-gpu"], "--disable-gpu more than once"),
)
DEFAULT_FLAGS = ["--disable-dev-shm-usage", "--disable-gpu"]
# Patterns Python's ``re`` compiles, so lilbee stores them, and crawlberg refuses. The last
# one holds the name crawlberg gives the launch flags.
REFUSED_PATTERNS = (
    "(?P<n>a)(?(n)b|c)",
    "(?#comment)a",
    "(?P<n>browser.chrome_args)(?(n)b|c)",
)
FLAG_SETTING = "The crawl_browser_extra_args setting holds a launch flag that crawlberg refuses: "
PATTERN_SETTING = "The crawl_exclude_patterns setting holds a pattern that crawlberg refuses: "
NO_SETTING = "crawlberg refuses to start this crawl: "
# A timeout in seconds whose milliseconds pass crawlberg's integer width, and what it raises.
OVERSIZED_TIMEOUT = 10**17
OVERSIZED_REASON = "int too big to convert"


QUERY_LINKS = (
    "/query/item?id=1",
    "/query/item?id=2",
    "/query/promo?utm_source=news",
    "/query/ad?gclid=abc&utm_term=spring",
    "/query/blog?p=42",
    "/moved",
)


def _query_page(path: str, filler: str) -> str:
    """The query listing, or a page whose text names the path and query it was served for."""
    if path == "/query/":
        links = "".join(f'<a href="{link}">{link}</a> ' for link in QUERY_LINKS)
        links += '<a href="/query/pair?a=1&amp;b=2">pair</a> <A HREF="/query/upper">UPPER</A> '
        return f"<html><head><title>Listing</title></head><body>{links}{filler}</body></html>"
    return f"<html><head><title>T</title></head><body><p>Served {path}.</p>{filler}</body></html>"


def _scope_or_retry_page(path: str, filler: str) -> tuple[int, str]:
    """The ``/scope/`` and ``/retry/`` listings, the two failing retry pages, or a scope page."""
    listings = {"/scope/": SCOPE_LINKS, "/retry/": tuple(RETRY_STATUSES)}
    if path in listings:
        links = "".join(f'<a href="{link}">{link}</a> ' for link in listings[path])
        return 200, f"<html><body><h1>Index</h1>{links}{filler}</body></html>"
    if path in RETRY_STATUSES:
        return RETRY_STATUSES[path], "<h1>failed</h1>"
    return 200, _query_page(path, filler)


class _Site:
    """A threaded local site: a wide listing, a slow listing, and a special page."""

    def __init__(self) -> None:
        self.requests: list[tuple[float, str]] = []
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def paths_since(self, since: float, prefix: str) -> list[str]:
        with self._lock:
            return [path for at, path in self.requests if at >= since and path.startswith(prefix)]

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _body(self, path: str) -> tuple[int, str]:
        filler = "<p>" + "Ordinary prose for a real page. " * 6 + "</p>"
        if path == "/moved":
            return 301, "/query/target"
        if path.startswith("/query/"):
            return 200, _query_page(path, filler)
        if path.startswith(("/scope/", "/retry/")):
            return _scope_or_retry_page(path, filler)
        if path in ("/wide/", "/slow/"):
            links = "".join(f'<a href="{path}p{n}">p{n}</a> ' for n in range(SLOW_PAGES))
            special = '<a href="/wiki/Special:Random">random</a><a href="/wiki/Home">home</a>'
            return 200, f"<html><body><h1>Index</h1>{links}{special}{filler}</body></html>"
        if path.startswith(("/wide/p", "/slow/p", "/wiki/")):
            if path.startswith("/slow/p"):
                time.sleep(SLOW_DELAY_S)
            return 200, f"<html><body><h1>{path}</h1>{filler}</body></html>"
        return 404, "<h1>missing</h1>"

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        site = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                """Silence the access log."""

            def do_GET(self) -> None:
                with site._lock:
                    site.requests.append((time.monotonic(), self.path))
                status, body = site._body(self.path)
                if status == HTTPStatus.MOVED_PERMANENTLY:
                    self.send_response(status)
                    self.send_header("Location", body)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                data = body.encode()
                self.send_response(status)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler


@pytest.fixture
def site() -> Iterator[_Site]:
    running = _Site()
    yield running
    running.close()


@pytest.fixture
def allow_loopback(monkeypatch):
    """The same loopback carve-out every local crawl test uses, read by both SSRF layers."""
    allowed = tuple(n for n in url_filter.get_blocked_networks() if n not in LOOPBACK)
    monkeypatch.setattr(url_filter, "get_blocked_networks", lambda: allowed)


@pytest.fixture
def isolated_env(tmp_path):
    snapshot = cfg.model_copy()
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir()
    cfg.data_dir = tmp_path / "data"
    cfg.data_dir.mkdir()
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.crawl_sync_interval = 0
    cfg.crawl_mean_delay = 0.0
    from lilbee.app.services import reset_services

    reset_services()
    yield tmp_path
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))
    reset_services()


def _representative(network: ipaddress.IPv4Network | ipaddress.IPv6Network) -> str:
    address = network[1] if network.num_addresses > 1 else network[0]
    return f"[{address}]" if address.version == 6 else str(address)


async def _first_error(policy: object, host: str) -> str:
    """The error crawlberg reports for a one-page crawl of *host* under *policy*, or ''."""
    config = crawlberg.CrawlConfig(
        max_depth=0, max_pages=1, request_timeout=1000, retry_count=0, ssrf=policy
    )
    engine = crawlberg.create_engine(config)
    async for event in crawlberg.crawl_stream(engine, f"http://{host}:9/"):
        payload = json.loads(str(event))
        if payload["type"] == "error":
            return str(payload["error"])
    return ""


class TestSsrfBlocklistsAgree:
    def test_lilbee_blocks_nothing_crawlberg_misses_but_nat64(self):
        uncovered = [
            net
            for net in url_filter.get_blocked_networks()
            if not any(
                net.version == denied.version and net.subnet_of(denied)  # type: ignore[arg-type]
                for denied in CRAWLBERG_DENIED_NETWORKS
            )
        ]
        assert uncovered == [NAT64]

    @pytest.mark.parametrize("network", CRAWLBERG_DENIED_NETWORKS, ids=str)
    async def test_crawlberg_refuses_each_network_in_the_copied_list(self, network):
        policy = crawlberg.SsrfPolicy(deny_private=True)
        assert (await _first_error(policy, _representative(network))).startswith(_SSRF_ERROR_CODE)

    @pytest.mark.parametrize(
        "network",
        [n for n in url_filter.get_blocked_networks() if n != NAT64],
        ids=str,
    )
    async def test_built_policy_refuses_every_network_lilbee_blocks(self, network):
        from lilbee.crawler.crawlberg_fetcher import _ssrf_policy

        error = await _first_error(_ssrf_policy(), _representative(network))
        assert error.startswith(_SSRF_ERROR_CODE), error

    @pytest.mark.parametrize(
        "host", IPV6_FORMS_OF_PRIVATE_IPV4.values(), ids=list(IPV6_FORMS_OF_PRIVATE_IPV4)
    )
    async def test_fetch_refuses_an_ipv6_address_that_carries_a_private_ipv4(self, host: str):
        fetcher = CrawlbergFetcher(render_mode=CrawlRenderMode.HTTP)
        fetched = await fetcher.fetch_single(f"http://{host}:9/", timeout=1)
        assert fetched.success is False
        assert fetched.error.startswith(_SSRF_ERROR_CODE), fetched.error

    async def test_fetch_tries_an_ipv6_address_that_carries_a_public_ipv4(self):
        fetcher = CrawlbergFetcher(render_mode=CrawlRenderMode.HTTP)
        fetched = await fetcher.fetch_single(f"http://{IPV6_FORM_OF_PUBLIC_IPV4}:9/", timeout=1)
        assert fetched.success is False
        assert fetched.error
        assert not fetched.error.startswith(_SSRF_ERROR_CODE), fetched.error

    async def test_built_policy_admits_what_lilbee_admits(self):
        from lilbee.crawler.crawlberg_fetcher import _ssrf_policy

        assert admitted_networks(url_filter.get_blocked_networks()) == [
            ipaddress.ip_network("224.0.0.0/4")
        ]
        error = await _first_error(_ssrf_policy(), "224.0.0.1")
        assert not error.startswith(_SSRF_ERROR_CODE), error


def _stream(fetcher: CrawlbergFetcher, url: str, **kwargs):
    return fetcher.fetch_recursive(
        url,
        depth=kwargs.get("depth", 1),
        max_pages=kwargs.get("max_pages"),
        timeout=10,
        concurrency=ConcurrencySpec(semaphore_count=3),
        filters=FilterSpec(),
        cancel=kwargs.get("cancel"),
    )


@pytest.mark.usefixtures("allow_loopback")
class TestCancel:
    async def test_cancel_stops_the_stream_and_no_request_starts_after_it(self, site):
        cancel = threading.Event()
        received: list[FetchedPage] = []
        fetcher = CrawlbergFetcher(render_mode=CrawlRenderMode.HTTP)
        cancelled_at = 0.0
        async for fetched in _stream(fetcher, site.url("/slow/"), cancel=cancel):
            received.append(fetched)
            if len(received) == 3:
                cancel.set()
                cancelled_at = time.monotonic()
        stopped_at = time.monotonic()
        assert stopped_at - cancelled_at < CANCEL_BOUND_S
        await asyncio.sleep(SETTLE_S)
        assert site.paths_since(stopped_at + LATE_START_MARGIN_S, "/slow/p") == []
        assert site.paths_since(0, "/slow/p")
        assert len(received) <= 4


@pytest.mark.usefixtures("allow_loopback")
class TestWorkerThread:
    def test_crawl_runs_on_a_worker_thread_with_its_own_loop(self, site):
        box: dict[str, object] = {}

        def worker() -> None:
            async def crawl() -> list[FetchedPage]:
                fetcher = CrawlbergFetcher(render_mode=CrawlRenderMode.HTTP)
                return [f async for f in _stream(fetcher, site.url("/wide/"), max_pages=3)]

            try:
                box["pages"] = asyncio.run(crawl())
            except BaseException as exc:  # PanicException derives from BaseException
                box["error"] = exc

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert "error" not in box, box.get("error")
        assert len(box["pages"]) == 3  # type: ignore[arg-type]


@pytest.mark.usefixtures("allow_loopback", "isolated_env")
class TestCrawlAndSave:
    async def test_max_pages_saves_exactly_that_many_pages(self, site):
        paths = await crawl_and_save(site.url("/wide/"), depth=1, max_pages=5)
        assert len(paths) == 5

    async def test_default_exclude_patterns_all_apply(self, site):
        """``/wiki/Special:`` is a default pattern that is not a regex anchor or digit class."""
        paths = await crawl_and_save(site.url("/wide/"), depth=1, max_pages=0)
        saved = {p.relative_to(cfg.documents_dir).as_posix() for p in paths}
        assert any(name.endswith("wiki/Home/index.md") for name in saved), saved
        assert not any("Special" in name for name in saved), saved


@pytest.mark.usefixtures("allow_loopback", "isolated_env")
class TestQueryUrlsAndRedirects:
    async def _saved_text(self, site: _Site) -> str:
        paths = await crawl_and_save(site.url("/query/"), depth=1, max_pages=0)
        return "\n".join(p.read_text(encoding="utf-8") for p in paths)

    async def test_pages_that_differ_only_by_query_are_each_saved(self, site):
        saved = await self._saved_text(site)
        assert "Served /query/item?id=1." in saved
        assert "Served /query/item?id=2." in saved

    async def test_tracking_parameters_are_stripped_before_the_fetch(self, site):
        saved = await self._saved_text(site)
        assert "Served /query/promo." in saved
        assert site.paths_since(0, "/query/promo") == ["/query/promo"]

    async def test_gclid_and_utm_term_are_stripped(self, site):
        saved = await self._saved_text(site)
        assert "Served /query/ad." in saved
        assert site.paths_since(0, "/query/ad") == ["/query/ad"]

    async def test_exclude_patterns_see_the_query(self, site):
        saved = await self._saved_text(site)
        assert "/query/blog?p=42." not in saved
        assert site.paths_since(0, "/query/blog") == []

    async def test_a_discovered_link_that_redirects_saves_its_target(self, site):
        saved = await self._saved_text(site)
        assert "Served /query/target." in saved

    async def test_a_link_with_an_encoded_ampersand_is_crawled_at_the_decoded_url(self, site):
        saved = await self._saved_text(site)
        assert "Served /query/pair?a=1&b=2." in saved
        assert site.paths_since(0, "/query/pair") == ["/query/pair?a=1&b=2"]

    async def test_a_link_in_uppercase_markup_is_followed(self, site):
        saved = await self._saved_text(site)
        assert "Served /query/upper." in saved

    async def test_saved_markdown_has_no_frontmatter(self, site):
        paths = await crawl_and_save(site.url("/query/"), depth=0)
        text = paths[0].read_text(encoding="utf-8")
        assert not text.startswith("---")
        assert "title: Listing" not in text


@pytest.mark.usefixtures("allow_loopback", "isolated_env")
class TestWholeUrlExcludePatterns:
    async def _crawled(self, site: _Site, patterns: list[str]) -> set[str]:
        cfg.crawl_exclude_patterns = patterns
        await crawl_and_save(site.url("/scope/"), depth=1, max_pages=0)
        return set(site.paths_since(0, "/scope/"))

    async def test_a_pattern_anchored_on_the_scheme_and_host_drops_its_pages(self, site):
        crawled = await self._crawled(site, [r"^http://127\.0\.0\.1:\d+/scope/skip/"])
        assert crawled == {"/scope/", "/scope/keep/b", "/scope/la/drop", "/scope/la/keepme"}

    async def test_a_look_ahead_pattern_keeps_the_page_it_names(self, site):
        crawled = await self._crawled(site, [r"^https?://[^/]+/scope/la/(?!keep)"])
        assert crawled == {"/scope/", "/scope/skip/a", "/scope/keep/b", "/scope/la/keepme"}

    async def test_a_pattern_for_another_host_drops_nothing(self, site):
        crawled = await self._crawled(site, [r"^https?://example\.com/scope/"])
        assert crawled == {"/scope/", *SCOPE_LINKS}


@pytest.mark.usefixtures("allow_loopback", "isolated_env")
class TestRetriedStatuses:
    async def test_only_a_rate_limit_status_is_retried(self, site):
        cfg.crawl_retry_on_rate_limit = True
        cfg.crawl_retry_max_attempts = RETRY_ATTEMPTS
        cfg.crawl_retry_base_delay_min = RETRY_DELAY_S
        cfg.crawl_retry_base_delay_max = RETRY_DELAY_S
        cfg.crawl_retry_max_backoff = RETRY_DELAY_S
        await crawl_and_save(site.url("/retry/"), depth=1, max_pages=0)
        assert len(site.paths_since(0, "/retry/busy")) == 1 + RETRY_ATTEMPTS
        assert site.paths_since(0, "/retry/broken") == ["/retry/broken"]


@pytest.mark.usefixtures("allow_loopback", "isolated_env")
class TestRefusedLaunchFlags:
    """crawlberg checks the flags when it builds the engine, before any Chrome starts."""

    @pytest.fixture(autouse=True)
    def _a_shell_that_is_never_started(self, monkeypatch, tmp_path):
        shell = tmp_path / "chrome-headless-shell"
        shell.write_text("", encoding="utf-8")
        monkeypatch.setattr(bootstrap, "chromium_installed", lambda: True)
        monkeypatch.setattr(bootstrap, "headless_shell_executable", lambda: shell)

    @pytest.mark.parametrize(("flags", "reason"), REFUSED_FLAGS, ids=repr)
    @pytest.mark.parametrize("depth", [0, 1], ids=["single-page", "recursive"])
    async def test_a_browser_crawl_names_the_setting_and_gives_crawlbergs_reason(
        self, site, flags: list[str], reason: str, depth: int
    ):
        cfg.crawl_browser_extra_args = flags
        with pytest.raises(CrawlEngineRefusedError) as refused:
            await crawl_and_save(site.url("/wiki/Home"), depth=depth, render_mode=BROWSER)
        message = str(refused.value)
        assert message.startswith("The crawl_browser_extra_args setting")
        assert "invalid_config: browser.chrome_args" in message
        assert reason in message
        assert site.paths_since(0, "/wiki/Home") == []

    def test_the_default_flags_build_an_engine(self):
        assert Config().crawl_browser_extra_args == DEFAULT_FLAGS
        browser = crawlberg.BrowserConfig(mode="always", chrome_args=DEFAULT_FLAGS)
        assert crawlberg.create_engine(crawlberg.CrawlConfig(browser=browser)) is not None

    async def test_an_http_crawl_with_a_refused_flag_fetches_the_page(self, site):
        cfg.crawl_browser_extra_args = ["--headless=new"]
        paths = await crawl_and_save(site.url("/wiki/Home"), depth=0, render_mode=HTTP)
        assert len(paths) == 1


@pytest.mark.usefixtures("allow_loopback", "isolated_env")
class TestRefusedExcludePatterns:
    """crawlberg parses the exclude patterns when it builds the engine, before it fetches a page."""

    @pytest.mark.parametrize("pattern", REFUSED_PATTERNS)
    @pytest.mark.parametrize("render_mode", [HTTP, BROWSER])
    async def test_a_recursive_crawl_names_the_setting_and_gives_crawlbergs_reason(
        self, site, monkeypatch, tmp_path, pattern: str, render_mode: CrawlRenderMode
    ):
        shell = tmp_path / "chrome-headless-shell"
        monkeypatch.setattr(bootstrap, "chromium_installed", lambda: True)
        monkeypatch.setattr(bootstrap, "headless_shell_executable", lambda: shell)
        assert Config(crawl_exclude_patterns=[pattern]).crawl_exclude_patterns == [pattern]
        cfg.crawl_exclude_patterns = [r"/scope/skip/", pattern]
        with pytest.raises(CrawlEngineRefusedError) as refused:
            await crawl_and_save(site.url("/scope/"), depth=1, max_pages=0, render_mode=render_mode)
        message = str(refused.value)
        assert message.startswith(PATTERN_SETTING + "invalid_config: invalid exclude_path regex")
        assert pattern in message
        assert site.paths_since(0, "/scope/") == []

    async def test_a_single_page_crawl_sends_no_pattern_and_fetches_the_page(self, site):
        cfg.crawl_exclude_patterns = [REFUSED_PATTERNS[0]]
        paths = await crawl_and_save(site.url("/wiki/Home"), depth=0, render_mode=HTTP)
        assert len(paths) == 1

    async def test_the_same_crawl_without_the_pattern_reaches_the_site(self, site):
        cfg.crawl_exclude_patterns = [r"/scope/skip/"]
        await crawl_and_save(site.url("/scope/"), depth=1, max_pages=0, render_mode=HTTP)
        assert "/scope/" in site.paths_since(0, "/scope/")


@pytest.mark.usefixtures("allow_loopback", "isolated_env")
class TestRefusedEngineOnTheCli:
    """``lilbee add`` ends a crawl crawlberg refuses with an error and a non-zero exit."""

    @pytest.fixture(autouse=True)
    def _a_shell_that_is_never_started(self, monkeypatch, tmp_path):
        shell = tmp_path / "chrome-headless-shell"
        shell.write_text("", encoding="utf-8")
        monkeypatch.setattr(bootstrap, "chromium_installed", lambda: True)
        monkeypatch.setattr(bootstrap, "headless_shell_executable", lambda: shell)

    def _add(self, *args: str) -> Result:
        return CliRunner().invoke(app, [*args, "--crawl", "--depth", "1"])

    def test_a_refused_exclude_pattern_exits_nonzero_and_names_the_setting(self, site):
        cfg.crawl_exclude_patterns = [REFUSED_PATTERNS[0]]
        cfg.crawl_render_mode = HTTP
        result = self._add("add", site.url("/scope/"))
        assert result.exit_code == 1
        assert "Error: " + PATTERN_SETTING + "invalid_config: invalid exclude_path" in result.output
        assert "Traceback" not in result.output
        assert "Crawled" not in result.output
        assert site.paths_since(0, "/scope/") == []

    def test_a_refused_flag_exits_nonzero_and_names_the_setting(self, site):
        cfg.crawl_browser_extra_args = ["--headless=new"]
        cfg.crawl_render_mode = BROWSER
        result = self._add("add", site.url("/scope/"))
        assert result.exit_code == 1
        assert "Error: " + FLAG_SETTING + "invalid_config: browser.chrome_args" in result.output
        assert "Traceback" not in result.output
        assert site.paths_since(0, "/scope/") == []

    def test_json_output_is_one_error_object(self, site):
        cfg.crawl_exclude_patterns = [REFUSED_PATTERNS[0]]
        cfg.crawl_render_mode = HTTP
        result = self._add("--json", "add", site.url("/scope/"))
        assert result.exit_code == 1
        objects = [json.loads(line) for line in result.output.splitlines() if line.startswith("{")]
        assert [list(found) for found in objects] == [["error"]]
        assert objects[0]["error"].startswith(PATTERN_SETTING)

    def test_a_depth_crawlberg_refuses_exits_nonzero_with_crawlbergs_reason(self, site):
        cfg.crawl_render_mode = HTTP
        result = CliRunner().invoke(app, ["add", site.url("/scope/"), "--crawl", "--depth", "101"])
        assert result.exit_code == 1
        assert f"Error: {NO_SETTING}invalid_config: max_depth must be <= 100 (got 101)" in (
            result.output
        )
        assert "Traceback" not in result.output
        assert "Crawled" not in result.output
        assert site.paths_since(0, "/scope/") == []

    def test_a_number_too_large_for_crawlberg_exits_nonzero_with_crawlbergs_reason(self, site):
        cfg.crawl_timeout = OVERSIZED_TIMEOUT
        cfg.crawl_render_mode = HTTP
        result = self._add("add", site.url("/scope/"))
        assert result.exit_code == 1
        assert f"Error: {NO_SETTING}{OVERSIZED_REASON}" in result.output
        assert "Traceback" not in result.output
        assert "Crawled" not in result.output
        assert site.paths_since(0, "/scope/") == []

    @pytest.mark.parametrize(
        ("timeout", "depth"),
        [(30, 0), (OVERSIZED_TIMEOUT, 0), (OVERSIZED_TIMEOUT, 1), (30, 101)],
        ids=["single-page", "timeout-single-page", "timeout", "depth"],
    )
    async def test_a_task_cancelled_before_the_fetch_ends_cancelled_not_failed(
        self, site, timeout: int, depth: int
    ):
        cfg.crawl_timeout = timeout
        task = CrawlTask(
            task_id="t1", url=site.url("/scope/"), depth=depth, max_pages=5, render_mode=HTTP
        )
        task.cancel.set()
        await run_crawl(task)
        assert task.status is TaskStatus.CANCELLED
        assert task.error is None
        assert task.pages_crawled == 0
        assert site.paths_since(0, "/scope/") == []

    @pytest.mark.parametrize(
        ("timeout", "depth"),
        [(OVERSIZED_TIMEOUT, 0), (OVERSIZED_TIMEOUT, 1), (30, 101)],
        ids=["timeout-single-page", "timeout", "depth"],
    )
    async def test_the_same_task_without_the_cancel_fails_with_crawlbergs_reason(
        self, site, timeout: int, depth: int
    ):
        cfg.crawl_timeout = timeout
        task = CrawlTask(
            task_id="t1", url=site.url("/scope/"), depth=depth, max_pages=5, render_mode=HTTP
        )
        await run_crawl(task)
        assert task.status is TaskStatus.FAILED
        assert task.error is not None
        assert task.error.startswith(NO_SETTING)

    @pytest.mark.parametrize("depth", [0, 1], ids=["single-page", "recursive"])
    async def test_crawlberg_raises_no_runtime_error_for_the_number_and_the_crawl_still_fails(
        self, site, depth: int
    ):
        cfg.crawl_timeout = OVERSIZED_TIMEOUT
        with pytest.raises(CrawlEngineRefusedError) as refused:
            await crawl_and_save(site.url("/wiki/Home"), depth=depth, render_mode=HTTP)
        assert str(refused.value) == NO_SETTING + OVERSIZED_REASON
        assert type(refused.value.__cause__) is OverflowError
        assert site.paths_since(0, "/wiki/Home") == []


def _logging_shell(directory: Path, log: Path) -> Path:
    """A script that records its arguments in *log*, then runs the real headless shell."""
    real = bootstrap.headless_shell_executable()
    assert real is not None
    script = directory / "chrome-headless-shell"
    script.write_text(f'#!/bin/sh\necho "$@" >> "{log}"\nexec "{real}" "$@"\n', encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


@full_crawl_only
@pytest.mark.usefixtures("allow_loopback", "isolated_env")
class TestBrowserLaunch:
    @pytest.fixture(autouse=True)
    def _real_browsers(self, monkeypatch):
        require_chromium()
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", REAL_BROWSERS_PATH)

    def _crawl(self, url: str) -> list[Path]:
        with windows_proactor_loop():
            return asyncio.run(crawl_and_save(url, depth=0, render_mode=CrawlRenderMode.BROWSER))

    def test_a_chrome_variable_that_names_no_binary_does_not_fail_the_crawl(
        self, site, monkeypatch, tmp_path
    ):
        missing = tmp_path / "no-such-chrome"
        monkeypatch.setenv("CHROME", str(missing))
        paths = self._crawl(site.url("/wiki/Home"))
        assert len(paths) == 1
        assert "/wiki/Home" in paths[0].read_text(encoding="utf-8")
        assert os.environ["CHROME"] == str(missing)

    @posix_only
    def test_chromium_starts_from_the_shell_lilbee_names_with_the_extra_flags(
        self, site, monkeypatch, tmp_path
    ):
        log = tmp_path / "launch.log"
        script = _logging_shell(tmp_path, log)
        monkeypatch.setattr(bootstrap, "headless_shell_executable", lambda: script)
        monkeypatch.setenv("CHROME", str(tmp_path / "no-such-chrome"))
        cfg.crawl_browser_extra_args = ["--lang=fr"]
        paths = self._crawl(site.url("/wiki/Home"))
        assert len(paths) == 1
        launch = log.read_text(encoding="utf-8")
        assert "--headless" in launch
        assert "--lang=fr" in launch
        assert "--lang=en_US" not in launch

    @posix_only
    def test_chromium_starts_with_the_default_flags(self, site, monkeypatch, tmp_path):
        log = tmp_path / "launch.log"
        script = _logging_shell(tmp_path, log)
        monkeypatch.setattr(bootstrap, "headless_shell_executable", lambda: script)
        cfg.crawl_browser_extra_args = Config().crawl_browser_extra_args
        paths = self._crawl(site.url("/wiki/Home"))
        assert len(paths) == 1
        launch = log.read_text(encoding="utf-8")
        for flag in DEFAULT_FLAGS:
            assert flag in launch
