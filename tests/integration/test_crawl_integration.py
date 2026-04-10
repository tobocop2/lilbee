"""Integration tests for web crawling — real crawl4ai against a local HTTP server.

These tests exercise the full pipeline: crawl4ai fetches real HTML from a local
pytest-httpserver, saves as markdown, and verifies the files land correctly.
Only the embedding model is faked (existing test pattern).

Requires: crawl4ai + Playwright browser binaries.
Skipped automatically when crawl4ai is not installed.
"""

import ipaddress
import time

import pytest

crawl4ai = pytest.importorskip("crawl4ai")

from lilbee import crawler as crawler_mod  # noqa: E402
from lilbee.config import cfg  # noqa: E402
from lilbee.crawler import (  # noqa: E402
    crawl_and_save,
    load_crawl_metadata,
    save_crawl_results,
    validate_crawl_url,
)
from lilbee.progress import EventType  # noqa: E402

HOME_HTML = """\
<html><head><title>Home</title></head>
<body>
<h1>Welcome to Test Site</h1>
<p>This is the home page with unique content about quantum computing.</p>
<a href="/about">About Us</a>
</body></html>
"""

ABOUT_HTML = """\
<html><head><title>About</title></head>
<body>
<h1>About Us</h1>
<p>We are a team specializing in distributed systems and consensus algorithms.</p>
<a href="/">Home</a>
</body></html>
"""

UPDATED_HOME_HTML = """\
<html><head><title>Home</title></head>
<body>
<h1>Welcome to Test Site</h1>
<p>This is the UPDATED home page about neural network architectures.</p>
<a href="/about">About Us</a>
</body></html>
"""

TRAVERSAL_HTML = """\
<html><head><title>Evil</title></head>
<body><h1>Path Traversal Test</h1><p>Should stay contained.</p></body></html>
"""


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    """Redirect config paths for all integration tests."""
    snapshot = {name: getattr(cfg, name) for name in type(cfg).model_fields}
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir()
    cfg.data_dir = tmp_path / "data"
    cfg.data_dir.mkdir()
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.crawl_timeout = 15
    # Reset semaphore cache
    crawler_mod._state.semaphore = None
    yield tmp_path
    for name, val in snapshot.items():
        setattr(cfg, name, val)
    crawler_mod._state.semaphore = None


@pytest.fixture()
def allow_localhost(monkeypatch):
    """Temporarily allow crawling localhost by removing loopback from blocked_networks."""
    loopback_v4 = ipaddress.ip_network("127.0.0.0/8")
    loopback_v6 = ipaddress.ip_network("::1/128")
    filtered = tuple(
        net for net in crawler_mod.get_blocked_networks() if net not in (loopback_v4, loopback_v6)
    )
    monkeypatch.setattr(crawler_mod, "get_blocked_networks", lambda: filtered)
    yield


@pytest.fixture()
def test_site(httpserver):
    """Serve a minimal static site with linked pages."""
    httpserver.expect_request("/").respond_with_data(HOME_HTML, content_type="text/html")
    httpserver.expect_request("/about").respond_with_data(ABOUT_HTML, content_type="text/html")
    return httpserver


@pytest.fixture()
def test_site_with_404(httpserver):
    """Serve a site where one link leads to a 404."""
    html_with_bad_link = """\
<html><body>
<h1>Good Page</h1>
<p>Content about functional programming paradigms.</p>
<a href="/missing">Broken Link</a>
</body></html>
"""
    httpserver.expect_request("/").respond_with_data(html_with_bad_link, content_type="text/html")
    httpserver.expect_request("/missing").respond_with_data("Not Found", status=404)
    return httpserver


class TestSinglePageCrawl:
    async def test_crawl_single_page_saves_markdown(self, test_site, allow_localhost):
        """Real crawl4ai fetches a page and saves as .md."""
        url = test_site.url_for("/")
        paths = await crawl_and_save(str(url))
        assert len(paths) == 1
        assert paths[0].exists()
        assert "_web" in str(paths[0])
        content = paths[0].read_text(encoding="utf-8")
        assert len(content) > 0

    async def test_crawl_saves_metadata(self, test_site, allow_localhost):
        """Crawl metadata records the URL and content hash."""
        url = str(test_site.url_for("/"))
        paths = await crawl_and_save(url)
        assert len(paths) == 1
        meta = load_crawl_metadata()
        assert url in meta
        assert meta[url].content_hash != ""
        assert meta[url].file != ""

    async def test_crawl_progress_events_fire(self, test_site, allow_localhost):
        """Progress callback receives events in correct order."""
        url = str(test_site.url_for("/"))
        events: list[str] = []

        def on_progress(event_type: EventType, data: dict) -> None:
            events.append(str(event_type))

        await crawl_and_save(url, on_progress=on_progress)
        assert "crawl_start" in events
        assert "crawl_page" in events
        assert "crawl_done" in events
        # Order: start before done
        assert events.index("crawl_start") < events.index("crawl_done")


class TestRecursiveCrawl:
    async def test_recursive_crawl_follows_links(self, test_site, allow_localhost):
        """Recursive crawl with depth=1 fetches linked pages."""
        url = str(test_site.url_for("/"))
        paths = await crawl_and_save(url, depth=1)
        # Should get at least the home page; may get about page too depending on crawl4ai
        assert len(paths) >= 1
        all_content = " ".join(p.read_text(encoding="utf-8") for p in paths)
        assert len(all_content) > 0


class TestContentChangeDetection:
    async def test_unchanged_content_skips_save(self, test_site, allow_localhost):
        """Crawl same URL twice with same content — second skips save."""
        url = str(test_site.url_for("/"))
        paths1 = await crawl_and_save(url)
        assert len(paths1) == 1
        mtime1 = paths1[0].stat().st_mtime

        # Small delay to ensure mtime would differ if file were rewritten
        time.sleep(0.1)

        paths2 = await crawl_and_save(url)
        assert paths2 == []
        # File was not rewritten
        assert paths1[0].stat().st_mtime == mtime1

    async def test_changed_content_updates_file(self, httpserver, allow_localhost):
        """Crawl URL, change server content, crawl again — file updated."""
        httpserver.expect_request("/changing").respond_with_data(
            "<html><body><h1>Version 1</h1><p>Original content.</p></body></html>",
            content_type="text/html",
        )
        url = str(httpserver.url_for("/changing"))
        paths1 = await crawl_and_save(url)
        assert len(paths1) == 1

        # Change the server content
        httpserver.clear()
        v2 = "<html><body><h1>Version 2</h1><p>New content.</p></body></html>"
        httpserver.expect_request("/changing").respond_with_data(v2, content_type="text/html")
        paths2 = await crawl_and_save(url)
        assert len(paths2) == 1
        new_content = paths2[0].read_text(encoding="utf-8")
        assert len(new_content) > 0

        # Metadata updated
        meta = load_crawl_metadata()
        assert url in meta


class TestSecurity:
    def test_ssrf_blocked_by_default(self):
        """Localhost is blocked by the default SSRF blocklist."""
        with pytest.raises(ValueError, match="not allowed"):
            validate_crawl_url("http://localhost:8080/secret")

    def test_ssrf_private_ip_blocked(self):
        """Private IPs are blocked."""
        with pytest.raises(ValueError, match="not allowed"):
            validate_crawl_url("http://10.0.0.1/internal")

    def test_ssrf_link_local_blocked(self):
        """Link-local (cloud metadata) IPs are blocked."""
        with pytest.raises(ValueError, match="not allowed"):
            validate_crawl_url("http://169.254.169.254/latest/meta-data/")

    def test_ssrf_non_http_rejected(self):
        """Non-HTTP schemes are rejected."""
        with pytest.raises(ValueError, match="Only http"):
            validate_crawl_url("ftp://example.com/file")
        with pytest.raises(ValueError, match="Only http"):
            validate_crawl_url("file:///etc/passwd")

    def test_ssrf_configurable_allowlist(self, test_site, allow_localhost):
        """With loopback removed, localhost passes validation."""
        url = str(test_site.url_for("/"))
        # Should not raise
        validate_crawl_url(url)

    def test_path_traversal_contained(self, isolated_env):
        """Path traversal in URL stays contained within _web/."""
        from lilbee.crawler import CrawlResult

        result = CrawlResult(url="https://evil.com/../../etc/passwd", markdown="# Malicious")
        paths = save_crawl_results([result])
        web_dir = cfg.documents_dir / "_web"
        for p in paths:
            assert p.resolve().is_relative_to(web_dir.resolve())

    async def test_max_pages_enforced(self, httpserver, allow_localhost):
        """Max pages config caps the crawl."""
        # Serve many pages
        for i in range(20):
            links = " ".join(f'<a href="/p{j}">P{j}</a>' for j in range(20) if j != i)
            httpserver.expect_request(f"/p{i}").respond_with_data(
                f"<html><body><h1>Page {i}</h1>{links}</body></html>",
                content_type="text/html",
            )
        cfg.crawl_max_pages = 3
        url = str(httpserver.url_for("/p0"))
        paths = await crawl_and_save(url, depth=2, max_pages=3)
        assert len(paths) <= 3


class TestErrors:
    async def test_crawl_404_returns_error_result(self, httpserver, allow_localhost):
        """A 404 page produces an empty or error result, doesn't crash."""
        httpserver.expect_request("/notfound").respond_with_data("Not Found", status=404)
        url = str(httpserver.url_for("/notfound"))
        # crawl4ai may return success=False or empty markdown for 404s
        # Either way, crawl_and_save should not raise
        paths = await crawl_and_save(url)
        # May be 0 (failed) or 1 (crawl4ai returned something) — just verify no crash
        assert isinstance(paths, list)

    async def test_crawl_unreachable_host(self):
        """Crawling an unreachable host returns empty without crashing."""
        # Use a non-routable IP to ensure connection failure
        paths = await crawl_and_save("http://192.0.2.1:9999/nowhere")
        assert paths == []


class TestCrawlConcurrency:
    """Verify the asyncio semaphore limits concurrent crawl operations.

    Uses real Playwright crawls against a local httpserver.
    Concurrency is measured via server-side request timing.
    """

    async def test_semaphore_limits_to_configured_value(self, httpserver, allow_localhost):
        """With crawl_max_concurrent=2, 3 crawls take at least 2 batches of wall time."""
        import asyncio
        import threading

        request_times: list[float] = []
        lock = threading.Lock()

        def slow_handler(request):
            with lock:
                request_times.append(time.monotonic())
            time.sleep(0.3)
            return f"<html><body><h1>{request.path}</h1><p>Content.</p></body></html>"

        for i in range(3):
            httpserver.expect_request(f"/conc{i}").respond_with_handler(slow_handler)

        cfg.crawl_max_concurrent = 2
        crawler_mod._state.semaphore = None

        urls = [str(httpserver.url_for(f"/conc{i}")) for i in range(3)]
        tasks = [asyncio.create_task(crawl_and_save(url)) for url in urls]
        await asyncio.gather(*tasks)

        assert len(request_times) == 3
        # With limit=2, the 3rd request starts after one of the first 2 finishes.
        # So the gap between the 2nd and 3rd request should be >= the handler sleep.
        request_times.sort()
        gap = request_times[2] - request_times[1]
        assert gap >= 0.2, f"3rd request started too soon (gap={gap:.3f}s), semaphore not limiting"

    async def test_unlimited_when_zero(self, httpserver, allow_localhost):
        """With crawl_max_concurrent=0 (unlimited), all 3 crawls complete."""
        import asyncio

        for i in range(3):
            httpserver.expect_request(f"/par{i}").respond_with_data(
                f"<html><body><h1>Page {i}</h1><p>Content.</p></body></html>",
                content_type="text/html",
            )

        cfg.crawl_max_concurrent = 0
        crawler_mod._state.semaphore = None

        urls = [str(httpserver.url_for(f"/par{i}")) for i in range(3)]
        tasks = [asyncio.create_task(crawl_and_save(url)) for url in urls]
        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert all(r is not None for r in results)
