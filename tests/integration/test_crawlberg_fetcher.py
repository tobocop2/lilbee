"""Real crawlberg against a local site: SSRF agreement, cancel, threads, page limits, excludes."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import threading
import time
from collections.abc import Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

crawlberg = pytest.importorskip("crawlberg")

from lilbee.core.config import cfg  # noqa: E402
from lilbee.core.config.enums import CrawlRenderMode  # noqa: E402
from lilbee.crawler import crawl_and_save, url_filter  # noqa: E402
from lilbee.crawler.crawlberg_fetcher import (  # noqa: E402
    _SSRF_ERROR_CODE,
    CRAWLBERG_DENIED_NETWORKS,
    CrawlbergFetcher,
    admitted_networks,
)
from lilbee.crawler.models import ConcurrencySpec, FetchedPage, FilterSpec  # noqa: E402

LOOPBACK = (ipaddress.ip_network("127.0.0.0/8"), ipaddress.ip_network("::1/128"))
# lilbee blocks the whole NAT64 prefix; crawlberg only refuses NAT64 forms of a
# private IPv4 address (a pending upstream security fix).
NAT64 = ipaddress.ip_network("64:ff9b::/96")
SLOW_PAGES = 40
SLOW_DELAY_S = 0.3
CANCEL_BOUND_S = 2.0


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
        return f"<html><head><title>Listing</title></head><body>{links}{filler}</body></html>"
    return f"<html><head><title>T</title></head><body><p>Served {path}.</p>{filler}</body></html>"


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
    async def test_cancel_stops_requests_within_the_bound(self, site):
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
        # crawlberg keeps fetching briefly after a crawl stops (xberg-io/crawlberg#77).
        await asyncio.sleep(CANCEL_BOUND_S + 1.0)
        assert site.paths_since(stopped_at + CANCEL_BOUND_S, "/slow/p") == []
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

    async def test_saved_markdown_has_no_frontmatter(self, site):
        paths = await crawl_and_save(site.url("/query/"), depth=0)
        text = paths[0].read_text(encoding="utf-8")
        assert not text.startswith("---")
        assert "title: Listing" not in text
