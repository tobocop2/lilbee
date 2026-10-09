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
from lilbee.crawler import CrawlMeta, bootstrap, crawl_and_save, save, url_filter  # noqa: E402
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
# The most retries crawlberg 1.10.2 accepts.
CRAWLBERG_RETRY_LIMIT = 20
RETRY_DELAY_S = 0.05
RETRY_STATUSES = {
    "/retry/broken": HTTPStatus.INTERNAL_SERVER_ERROR,
    "/retry/busy": HTTPStatus.SERVICE_UNAVAILABLE,
}
SCOPE_LINKS = ("/scope/skip/a", "/scope/keep/b", "/scope/la/drop", "/scope/la/keepme")
# Links as a page writes them: a raw letter, a raw space, one already percent-encoded, a backslash.
SPELL_LINKS = ("/spell/naïve", "/spell/raw space", "/spell/caf%C3%A9", "/spell/back\\slash")
# The same four addresses as crawlberg reports and requests them.
SPELL_REQUESTS = (
    "/spell/na%C3%AFve",
    "/spell/raw%20space",
    "/spell/caf%C3%A9",
    "/spell/back/slash",
)
# Pairs of links a server answers with two pages: a re-crawl keeps each pair as two pages.
BACKSLASH_PAIR = ("/pair/b\\s", "/pair/b/s")
PAIR_LINKS = (
    BACKSLASH_PAIR,
    ("/pair/ca^ret", "/pair/ca%5Eret"),
    ("/pair/pi|pe", "/pair/pi%7Cpe"),
    ("/pair/a+b", "/pair/a%20b"),
    ("/pair/e%2Fs", "/pair/e/s"),
    ("/pair/d/%2e%2e/up", "/pair/up"),
    ("/pair/naïve", "/pair/na%C3%AFve"),
)
# A link with an escaped hyphen, and the equivalent address a library holds for it.
EQUIVALENT_LINK = "/same/x%2Dy"
EQUIVALENT_STORED = "/same/x-y"
# How the text of a stored copy starts before a crawl, and how a served page names its request.
STALE_TEXT = "stale text"
FRESH_TEXT = "Served"
# The links each listing page of written addresses holds.
LINK_LISTINGS = {
    "/spell/": SPELL_LINKS,
    "/pair/": tuple(link for pair in PAIR_LINKS for link in pair),
    "/same/": (EQUIVALENT_LINK,),
}
# Flag lists crawlberg refuses, each with a part of the reason it gives.
REFUSED_FLAGS: tuple[tuple[list[str], str], ...] = (
    (["--headless=new"], "--headless"),
    (["--user-data-dir=/x"], "--user-data-dir"),
    (["disable-gpu"], "must start with --"),
    (["--Lang=fr"], "must name the flag in lowercase"),
    (["--disable-gpu", "--disable-gpu"], "duplicates an earlier flag"),
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
# The user agents lilbee sent in each render mode before crawlberg, on every platform.
HTTP_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/116.0.0.0 Safari/537.36"
)
# The client hints lilbee sent with that browser user agent to a loopback site.
CLIENT_HINTS = {
    "sec-ch-ua": '"Chromium";v="116", "Not_A Brand";v="8", "Google Chrome";v="116"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Linux"',
}
# A page with a script and a link, the script, and the linked page.
HINT_PAGES = ("/hints/", "/hints/app.js", "/hints/next")
# Pages in the recursive browser crawl that must start Chromium once.
BROWSER_CRAWL_PAGES = 3
# A page whose image carries its own bytes in a ``data:`` address, here a one-pixel PNG.
INLINE_IMAGE_PAGE = "/inline/image"
INLINE_IMAGE_ALT = "Harbor lighthouse logo"
INLINE_IMAGE_PAYLOAD = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhf"
    "DwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


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
    if path in LINK_LISTINGS:
        links = "".join(f'<a href="{link}">{link}</a> ' for link in LINK_LISTINGS[path])
        head = '<head><meta charset="utf-8"><title>Spell</title></head>'
        return f"<html>{head}<body><h1>Index</h1>{links}{filler}</body></html>"
    served = f"<p>{FRESH_TEXT} {path}.</p>"
    return f"<html><head><title>T</title></head><body>{served}{filler}</body></html>"


def _scope_or_retry_page(path: str, filler: str) -> tuple[int, str]:
    """The ``/scope/`` and ``/retry/`` listings, the two failing retry pages, or a scope page."""
    listings = {"/scope/": SCOPE_LINKS, "/retry/": tuple(RETRY_STATUSES)}
    if path in listings:
        links = "".join(f'<a href="{link}">{link}</a> ' for link in listings[path])
        return 200, f"<html><body><h1>Index</h1>{links}{filler}</body></html>"
    if path in RETRY_STATUSES:
        return RETRY_STATUSES[path], "<h1>failed</h1>"
    return 200, _query_page(path, filler)


def _fixed_pages(filler: str) -> dict[str, tuple[int, str]]:
    """The status and body, or redirect target, of each page served at one exact path."""
    image = f"data:image/png;base64,{INLINE_IMAGE_PAYLOAD}"
    logo = f'<img alt="{INLINE_IMAGE_ALT}" src="{image}">'
    return {
        "/moved": (HTTPStatus.MOVED_PERMANENTLY, "/query/target"),
        INLINE_IMAGE_PAGE: (
            HTTPStatus.OK,
            f"<html><body><h1>Logo</h1>{logo}{filler}</body></html>",
        ),
        "/hints/": (
            HTTPStatus.OK,
            '<html><head><script src="/hints/app.js"></script></head><body><h1>Hints</h1>'
            f'<a href="/hints/next">next</a>{filler}</body></html>',
        ),
    }


class _Site:
    """A threaded local site: a wide listing, a slow listing, and a special page."""

    def __init__(self) -> None:
        self.requests: list[tuple[float, str]] = []
        self.agents: list[tuple[str, str]] = []
        self.hints: list[tuple[str, dict[str, str]]] = []
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

    def agents_for(self, prefix: str) -> list[str]:
        """The user agent of each request for a path under *prefix*."""
        with self._lock:
            return [agent for path, agent in self.agents if path.startswith(prefix)]

    def hints_for(self, prefix: str) -> dict[str, list[dict[str, str]]]:
        """The client hint headers of each request for a path under *prefix*, by path."""
        with self._lock:
            found: dict[str, list[dict[str, str]]] = {}
            for path, hints in self.hints:
                if path.startswith(prefix):
                    found.setdefault(path, []).append(hints)
            return found

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _body(self, path: str) -> tuple[int, str]:
        filler = "<p>" + "Ordinary prose for a real page. " * 6 + "</p>"
        fixed = _fixed_pages(filler)
        if path in fixed:
            return fixed[path]
        if path.startswith(("/query/", *LINK_LISTINGS)):
            return 200, _query_page(path, filler)
        if path.startswith(("/scope/", "/retry/")):
            return _scope_or_retry_page(path, filler)
        if path in ("/wide/", "/slow/"):
            links = "".join(f'<a href="{path}p{n}">p{n}</a> ' for n in range(SLOW_PAGES))
            special = '<a href="/wiki/Special:Random">random</a><a href="/wiki/Home">home</a>'
            return 200, f"<html><body><h1>Index</h1>{links}{special}{filler}</body></html>"
        if path.startswith(("/wide/p", "/slow/p", "/wiki/", "/hints/")):
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
                    site.agents.append((self.path, self.headers.get("User-Agent", "")))
                    hints = {
                        name.lower(): value
                        for name, value in self.headers.items()
                        if name.lower().startswith("sec-ch-")
                    }
                    site.hints.append((self.path, hints))
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

    async def test_an_inline_image_is_saved_as_its_alt_text_without_its_encoded_data(self, site):
        assert INLINE_IMAGE_PAYLOAD in site._body(INLINE_IMAGE_PAGE)[1]
        paths = await crawl_and_save(site.url(INLINE_IMAGE_PAGE), depth=0)
        assert len(paths) == 1
        text = paths[0].read_text(encoding="utf-8")
        assert "Ordinary prose for a real page." in text
        assert INLINE_IMAGE_ALT in text
        assert INLINE_IMAGE_PAYLOAD not in text
        assert "base64" not in text


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


def _library_with(*urls: str) -> list[Path]:
    """A library that holds each of *urls* as spelled, each with its own text no page serves."""
    entries = {}
    paths = []
    for url in urls:
        name = save.url_to_filename(url)
        path = cfg.documents_dir / "_web" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{STALE_TEXT} of {url}", encoding="utf-8")
        entries[url] = CrawlMeta(
            file=name, content_hash="stale", crawled_at="2026-01-01T00:00:00+00:00"
        )
        paths.append(path)
    save.save_crawl_metadata(entries)
    return paths


def _pair_library(site: _Site) -> list[tuple[Path, Path]]:
    """A library that holds both addresses of each pair of PAIR_LINKS: the two copies of each."""
    copies = _library_with(*(site.url(link) for pair in PAIR_LINKS for link in pair))
    assert len(set(copies)) == len(copies)
    return list(zip(copies[::2], copies[1::2], strict=True))


def _pair_texts(library: list[tuple[Path, Path]]) -> dict[tuple[str, str], tuple[str, str]]:
    """The text of the two copies of each pair of PAIR_LINKS."""
    return {
        pair: (one.read_text(encoding="utf-8"), other.read_text(encoding="utf-8"))
        for pair, (one, other) in zip(PAIR_LINKS, library, strict=True)
    }


def _assert_each_pair_is_two_pages(library: list[tuple[Path, Path]]) -> None:
    """Each pair has a copy the crawl rewrote, and no pair holds one text twice."""
    texts = _pair_texts(library)
    assert [pair for pair, both in texts.items() if FRESH_TEXT not in "".join(both)] == []
    assert [pair for pair, (one, other) in texts.items() if one == other] == []
    assert [pair for pair, both in texts.items() if not _own_pages(pair, both)] == []


def _own_pages(pair: tuple[str, str], texts: tuple[str, str]) -> bool:
    """Whether each text is the stale copy, or the page the site serves for its own address."""
    return all(
        text.startswith(STALE_TEXT) or f"{FRESH_TEXT} {link}." in text
        for link, text in zip(pair, texts, strict=True)
    )


@pytest.mark.usefixtures("allow_loopback", "isolated_env")
class TestStoredSpellingOfAnAddress:
    async def test_a_page_stored_under_an_equivalent_address_is_updated_in_place(self, site):
        stored = site.url(EQUIVALENT_STORED)
        (copy,) = _library_with(stored)
        await crawl_and_save(site.url("/same/"), depth=1, max_pages=0)
        assert site.paths_since(0, "/same/x") == [EQUIVALENT_LINK]
        assert f"{FRESH_TEXT} {EQUIVALENT_LINK}." in copy.read_text(encoding="utf-8")
        meta = save.load_crawl_metadata()
        assert set(meta) == {site.url("/same/"), stored}
        assert meta[stored].content_hash != "stale"
        assert len(list((cfg.documents_dir / "_web").rglob("*.md"))) == len(meta)

    async def test_a_page_stored_under_two_equivalent_addresses_is_rewritten_under_both(self, site):
        stored, as_written = site.url(EQUIVALENT_STORED), site.url(EQUIVALENT_LINK)
        other = site.url("/elsewhere/page")
        plain_copy, escaped_copy, other_copy = _library_with(stored, as_written, other)
        assert plain_copy != escaped_copy
        await crawl_and_save(site.url("/same/"), depth=1, max_pages=0)
        for copy in (plain_copy, escaped_copy):
            assert f"{FRESH_TEXT} {EQUIVALENT_LINK}." in copy.read_text(encoding="utf-8")
        assert other_copy.read_text(encoding="utf-8") == f"{STALE_TEXT} of {other}"
        meta = save.load_crawl_metadata()
        assert meta[stored].content_hash == meta[as_written].content_hash != "stale"
        assert meta[other].content_hash == "stale"

    @posix_only
    async def test_two_stored_pages_stay_two_pages_after_an_http_recrawl(self, site):
        library = _pair_library(site)
        await crawl_and_save(site.url("/pair/"), depth=1, max_pages=0)
        _assert_each_pair_is_two_pages(library)
        backslash_copy, _slash_copy = library[PAIR_LINKS.index(BACKSLASH_PAIR)]
        assert backslash_copy.read_text(encoding="utf-8").startswith(STALE_TEXT)
        assert len(save.load_crawl_metadata()) >= 1 + 2 * len(PAIR_LINKS)

    async def test_a_new_page_is_stored_as_crawlberg_reports_it(self, site):
        await crawl_and_save(site.url("/spell/"), depth=1, max_pages=0)
        reported = {site.url(path) for path in SPELL_REQUESTS}
        assert set(save.load_crawl_metadata()) == {site.url("/spell/"), *reported}


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

    async def test_a_retry_count_over_crawlbergs_limit_crawls_with_the_limit(self, site):
        cfg.crawl_retry_on_rate_limit = True
        cfg.crawl_retry_max_attempts = CRAWLBERG_RETRY_LIMIT + 5
        cfg.crawl_retry_base_delay_min = RETRY_DELAY_S
        cfg.crawl_retry_base_delay_max = RETRY_DELAY_S
        cfg.crawl_retry_max_backoff = RETRY_DELAY_S
        await crawl_and_save(site.url("/retry/"), depth=1, max_pages=0)
        assert len(site.paths_since(0, "/retry/busy")) == 1 + CRAWLBERG_RETRY_LIMIT


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
class TestShellThatCannotRun:
    """A headless shell path that cannot run is installed, and crawlberg refuses it."""

    def _shell_dir(self, tmp_path: Path) -> Path:
        """The folder of a finished headless shell install at revision 1, with no shell in it."""
        directory = tmp_path / f"{bootstrap._HEADLESS_SHELL_DIR_PREFIX}1"
        directory.mkdir()
        (directory / bootstrap._INSTALL_COMPLETE_MARKER).write_bytes(b"")
        return directory

    @posix_only
    @pytest.mark.parametrize("depth", [0, 1], ids=["single-page", "recursive"])
    @pytest.mark.parametrize("shape", ["dangling-link", "directory"])
    async def test_a_shell_path_that_is_no_file_is_named_and_starts_no_install(
        self, site, monkeypatch, tmp_path, depth: int, shape: str
    ):
        shell = self._shell_dir(tmp_path) / "chrome-headless-shell"
        if shape == "directory":
            shell.mkdir()
        else:
            shell.symlink_to(tmp_path / "no-such-target")
        monkeypatch.setattr(bootstrap, "_browsers_cache_path", lambda: tmp_path)
        monkeypatch.setattr(bootstrap, "_expected_chromium_revision", lambda: "1")
        installs: list[object] = []

        async def install(on_progress: object) -> None:
            installs.append(on_progress)

        monkeypatch.setattr(bootstrap, "_install_chromium", install)
        with pytest.raises(CrawlEngineRefusedError) as refused:
            await crawl_and_save(site.url("/wiki/Home"), depth=depth, render_mode=BROWSER)
        assert installs == []
        assert str(refused.value).startswith(f"{NO_SETTING}invalid_config: browser.chrome_path")
        assert f"'{shell}'" in str(refused.value)
        assert site.paths_since(0, "/wiki/Home") == []

    @posix_only
    @pytest.mark.parametrize("depth", [0, 1], ids=["single-page", "recursive"])
    async def test_a_browser_crawl_names_the_shell_and_starts_no_install(
        self, site, monkeypatch, tmp_path, depth: int
    ):
        shell = self._shell_dir(tmp_path) / "chrome-headless-shell"
        shell.write_text("#!/bin/sh\n", encoding="utf-8")
        shell.chmod(stat.S_IRUSR | stat.S_IWUSR)
        monkeypatch.setattr(bootstrap, "_browsers_cache_path", lambda: tmp_path)
        monkeypatch.setattr(bootstrap, "_expected_chromium_revision", lambda: "1")
        installs: list[object] = []

        async def install(on_progress: object) -> None:
            installs.append(on_progress)

        monkeypatch.setattr(bootstrap, "_install_chromium", install)
        with pytest.raises(CrawlEngineRefusedError) as refused:
            await crawl_and_save(site.url("/wiki/Home"), depth=depth, render_mode=BROWSER)
        assert installs == []
        assert str(refused.value) == (
            f"{NO_SETTING}invalid_config: browser.chrome_path '{shell}' is not executable"
        )
        assert site.paths_since(0, "/wiki/Home") == []


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

    def _crawl(self, url: str, depth: int = 0, max_pages: int | None = None) -> list[Path]:
        with windows_proactor_loop():
            return asyncio.run(
                crawl_and_save(url, depth=depth, max_pages=max_pages, render_mode=BROWSER)
            )

    @posix_only
    def test_two_stored_pages_stay_two_pages_after_a_browser_recrawl(self, site):
        library = _pair_library(site)
        self._crawl(site.url("/pair/"), depth=1, max_pages=0)
        _assert_each_pair_is_two_pages(library)

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

    @posix_only
    def test_a_recursive_crawl_of_several_pages_starts_chromium_once(
        self, site, monkeypatch, tmp_path
    ):
        log = tmp_path / "launch.log"
        script = _logging_shell(tmp_path, log)
        monkeypatch.setattr(bootstrap, "headless_shell_executable", lambda: script)
        paths = self._crawl(site.url("/wide/"), depth=1, max_pages=BROWSER_CRAWL_PAGES)
        assert len(paths) == BROWSER_CRAWL_PAGES
        assert len(log.read_text(encoding="utf-8").splitlines()) == 1

    def test_browser_requests_carry_the_browser_user_agent(self, site):
        paths = self._crawl(site.url("/wiki/Home"))
        assert len(paths) == 1
        assert set(site.agents_for("/wiki/Home")) == {BROWSER_USER_AGENT}

    def test_a_page_its_script_and_a_followed_page_carry_the_client_hints(self, site):
        paths = self._crawl(site.url("/hints/"), depth=1)
        assert len(paths) == 2
        hints = site.hints_for("/hints/")
        assert sorted(hints) == sorted(HINT_PAGES)
        for page in HINT_PAGES:
            assert hints[page] == [CLIENT_HINTS] * len(hints[page]), page


@pytest.mark.usefixtures("allow_loopback", "isolated_env")
class TestUserAgent:
    async def test_http_requests_carry_the_http_user_agent(self, site):
        paths = await crawl_and_save(site.url("/wide/"), depth=1, max_pages=3, render_mode=HTTP)
        assert len(paths) == 3
        agents = site.agents_for("/wide/")
        assert len(agents) >= 3
        assert set(agents) == {HTTP_USER_AGENT}

    async def test_a_single_page_request_carries_the_http_user_agent(self, site):
        await crawl_and_save(site.url("/wiki/Home"), depth=0, render_mode=HTTP)
        assert site.agents_for("/wiki/Home") == [HTTP_USER_AGENT]

    async def test_http_requests_carry_no_client_hints(self, site):
        await crawl_and_save(site.url("/hints/"), depth=1, render_mode=HTTP)
        assert site.hints_for("/hints/") == {"/hints/": [{}], "/hints/next": [{}]}
        assert set(site.agents_for("/hints/")) == {HTTP_USER_AGENT}
