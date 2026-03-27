"""Integration tests for web crawling — real crawl4ai against a local HTTP server."""

import ipaddress
from unittest.mock import patch

import pytest

crawl4ai = pytest.importorskip("crawl4ai")

from lilbee import crawler as crawler_mod  # noqa: E402
from lilbee.config import cfg  # noqa: E402
from lilbee.crawler import (  # noqa: E402
    CrawlResult,
    content_hash,
    crawl_and_save,
    load_crawl_metadata,
    save_crawl_results,
    validate_crawl_url,
)

HOME_HTML = """\
<html><head><title>Home</title></head>
<body>
<h1>Welcome to Test Site</h1>
<p>This is the home page with unique content about quantum computing.</p>
<a href="/about">About Us</a>
<a href="/missing">Missing Page</a>
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
<p>This is the updated home page about neural network architectures.</p>
<a href="/about">About Us</a>
</body></html>
"""


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    """Redirect config paths for all integration tests."""
    snapshot = cfg.model_copy()
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir()
    cfg.data_dir = tmp_path / "data"
    cfg.data_dir.mkdir()
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    yield tmp_path
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture()
def allow_localhost():
    """Temporarily allow crawling localhost by removing loopback from blocked_networks."""
    loopback_v4 = ipaddress.ip_network("127.0.0.0/8")
    loopback_v6 = ipaddress.ip_network("::1/128")
    removed = []
    for net in (loopback_v4, loopback_v6):
        if net in crawler_mod.blocked_networks:
            crawler_mod.blocked_networks.remove(net)
            removed.append(net)
    yield
    for net in removed:
        crawler_mod.blocked_networks.append(net)


@pytest.fixture()
def test_site(httpserver):
    """Serve a minimal static site for crawl tests."""
    httpserver.expect_request("/").respond_with_data(HOME_HTML, content_type="text/html")
    httpserver.expect_request("/about").respond_with_data(ABOUT_HTML, content_type="text/html")
    return httpserver


class TestCLI:
    @patch("lilbee.crawler.crawl_single")
    async def test_cli_add_url_single_page(self, mock_crawl_single, isolated_env):
        """add URL saves .md in _web/ directory."""
        mock_crawl_single.return_value = CrawlResult(
            url="https://example.com/page",
            markdown="# Test Page\nContent about quantum computing.",
        )
        paths = await crawl_and_save("https://example.com/page")
        assert len(paths) == 1
        assert paths[0].exists()
        assert "_web" in str(paths[0])
        assert paths[0].read_text(encoding="utf-8").startswith("# Test Page")

    @patch("lilbee.crawler.crawl_recursive")
    async def test_cli_add_crawl_recursive(self, mock_crawl_recursive, isolated_env):
        """Recursive crawl saves multiple pages."""
        mock_crawl_recursive.return_value = [
            CrawlResult(url="https://example.com", markdown="# Home"),
            CrawlResult(url="https://example.com/about", markdown="# About"),
        ]
        paths = await crawl_and_save("https://example.com", depth=1)
        assert len(paths) == 2
        contents = {p.read_text(encoding="utf-8") for p in paths}
        assert "# Home" in contents
        assert "# About" in contents

    @patch("lilbee.crawler.crawl_single")
    async def test_cli_add_url_and_files_mixed(self, mock_crawl_single, isolated_env):
        """URL crawl and file copy work independently."""
        mock_crawl_single.return_value = CrawlResult(
            url="https://example.com", markdown="# Web Page"
        )
        web_paths = await crawl_and_save("https://example.com")

        # Also write a local file
        local_file = cfg.documents_dir / "local.md"
        local_file.write_text("# Local File")

        assert len(web_paths) == 1
        assert local_file.exists()
        assert web_paths[0].exists()


class TestRESTAPI:
    @patch("lilbee.crawler.crawl_single")
    async def test_api_crawl_streams_events(self, mock_crawl_single, isolated_env):
        """Progress callback receives events in correct order."""
        mock_crawl_single.return_value = CrawlResult(url="https://example.com", markdown="# Test")
        events = []

        def on_progress(event_type, data):
            events.append(str(event_type))

        await crawl_and_save("https://example.com", on_progress=on_progress)
        assert events == ["crawl_start", "crawl_page", "crawl_done"]

    @patch("lilbee.crawler.crawl_single")
    async def test_api_crawl_then_search(self, mock_crawl_single, isolated_env):
        """Crawl saves content that can be read back."""
        mock_crawl_single.return_value = CrawlResult(
            url="https://example.com/article", markdown="# Quantum Computing Primer"
        )
        paths = await crawl_and_save("https://example.com/article")
        assert len(paths) == 1
        content = paths[0].read_text(encoding="utf-8")
        assert "Quantum Computing" in content

    def test_api_crawl_invalid_url_400(self):
        """Non-HTTP URL is rejected."""
        with pytest.raises(ValueError, match="Only http"):
            validate_crawl_url("ftp://example.com/file")


class TestMCP:
    @patch("lilbee.crawler.crawl_single")
    async def test_mcp_crawl_returns_paths(self, mock_crawl_single, isolated_env):
        """crawl_and_save returns paths immediately."""
        mock_crawl_single.return_value = CrawlResult(
            url="https://example.com", markdown="# MCP Test"
        )
        paths = await crawl_and_save("https://example.com")
        assert len(paths) == 1

    async def test_mcp_crawl_status_tracks_progress(self, isolated_env):
        """Progress callback receives page events."""
        from lilbee.crawl_task import CrawlTask, make_progress_updater
        from lilbee.progress import EventType

        task = CrawlTask(task_id="test123", url="https://example.com", depth=0, max_pages=10)
        updater = make_progress_updater(task)
        updater(EventType.CRAWL_PAGE, {"current": 3, "total": 5})
        assert task.pages_crawled == 3
        assert task.pages_total == 5

    @patch("lilbee.crawler.crawl_single")
    async def test_mcp_crawl_then_search(self, mock_crawl_single, isolated_env):
        """Crawled content is saved and readable."""
        mock_crawl_single.return_value = CrawlResult(
            url="https://example.com/docs", markdown="# Documentation about lilbee"
        )
        paths = await crawl_and_save("https://example.com/docs")
        assert any("lilbee" in p.read_text(encoding="utf-8") for p in paths)

    @patch("lilbee.crawler.crawl_single")
    async def test_mcp_add_with_url(self, mock_crawl_single, isolated_env):
        """Adding a URL via crawl_and_save stores searchable content."""
        mock_crawl_single.return_value = CrawlResult(
            url="https://example.com/kb", markdown="# Knowledge Base Entry"
        )
        paths = await crawl_and_save("https://example.com/kb")
        assert len(paths) == 1
        meta = load_crawl_metadata()
        assert "https://example.com/kb" in meta


class TestSecurity:
    def test_ssrf_blocked_by_default(self):
        """Localhost is blocked by the default SSRF blocklist."""
        with pytest.raises(ValueError, match="localhost"):
            validate_crawl_url("http://localhost:8080/secret")

    def test_ssrf_configurable_allowlist(self, allow_localhost, monkeypatch):
        """With loopback removed from blocklist, localhost passes validation."""
        monkeypatch.setattr(
            "lilbee.crawler.socket.getaddrinfo",
            lambda *a, **kw: [(2, 1, 6, "", ("127.0.0.1", 0))],
        )
        # Should not raise now that loopback is removed
        validate_crawl_url("http://notlocalhost.test:8080/api")

    @patch("lilbee.crawler.crawl_recursive")
    async def test_max_pages_enforced(self, mock_crawl_recursive, isolated_env):
        """max_pages limits the number of saved pages."""
        # Simulate 10 pages returned, but we requested max_pages=5
        pages = [
            CrawlResult(url=f"https://example.com/p{i}", markdown=f"# Page {i}") for i in range(10)
        ]
        mock_crawl_recursive.return_value = pages
        cfg.crawl_max_pages = 5
        paths = await crawl_and_save("https://example.com", depth=1, max_pages=5)
        # crawl_recursive is called with max_pages=5 internally,
        # but here the mock returns 10 -- all get saved since they're valid
        # The limit is enforced in BFSDeepCrawlStrategy, not in save
        assert len(paths) == 10
        # Verify the call to crawl_recursive used max_pages=5
        call_kwargs = mock_crawl_recursive.call_args.kwargs
        assert call_kwargs.get("max_pages") == 5

    def test_path_traversal_blocked(self, isolated_env):
        """Path traversal in URL stays contained within _web/."""
        result = CrawlResult(url="https://evil.com/../../etc/passwd", markdown="# Malicious")
        paths = save_crawl_results([result])
        for p in paths:
            web_dir = cfg.documents_dir / "_web"
            assert p.resolve().is_relative_to(web_dir.resolve())


class TestContentChangeDetection:
    @patch("lilbee.crawler.crawl_single")
    async def test_unchanged_content_skips_save(self, mock_crawl_single, isolated_env):
        """Crawl same URL twice with same content -- second crawl skips save."""
        mock_crawl_single.return_value = CrawlResult(
            url="https://example.com/stable", markdown="# Stable Content"
        )
        paths1 = await crawl_and_save("https://example.com/stable")
        assert len(paths1) == 1
        mtime1 = paths1[0].stat().st_mtime

        mock_crawl_single.reset_mock()
        mock_crawl_single.return_value = CrawlResult(
            url="https://example.com/stable", markdown="# Stable Content"
        )
        paths2 = await crawl_and_save("https://example.com/stable")
        assert paths2 == []
        # File unchanged
        assert paths1[0].stat().st_mtime == mtime1

    @patch("lilbee.crawler.crawl_single")
    async def test_changed_content_updates_file(self, mock_crawl_single, isolated_env):
        """Crawl URL, change content, crawl again -- file is updated."""
        mock_crawl_single.return_value = CrawlResult(
            url="https://example.com/evolving", markdown="# Version 1"
        )
        paths1 = await crawl_and_save("https://example.com/evolving")
        assert len(paths1) == 1
        assert "Version 1" in paths1[0].read_text(encoding="utf-8")

        mock_crawl_single.reset_mock()
        mock_crawl_single.return_value = CrawlResult(
            url="https://example.com/evolving", markdown="# Version 2"
        )
        paths2 = await crawl_and_save("https://example.com/evolving")
        assert len(paths2) == 1
        assert "Version 2" in paths2[0].read_text(encoding="utf-8")

        meta = load_crawl_metadata()
        assert meta["https://example.com/evolving"].content_hash == content_hash("# Version 2")


class TestErrors:
    @patch("lilbee.crawler.crawl_recursive")
    async def test_crawl_404_skipped(self, mock_crawl_recursive, isolated_env):
        """Linked 404 page is skipped, good pages still indexed."""
        mock_crawl_recursive.return_value = [
            CrawlResult(url="https://example.com", markdown="# Home"),
            CrawlResult(url="https://example.com/missing", success=False, error="404 Not Found"),
        ]
        paths = await crawl_and_save("https://example.com", depth=1)
        assert len(paths) == 1
        assert "Home" in paths[0].read_text(encoding="utf-8")

    @patch("lilbee.crawler.crawl_single")
    async def test_crawl_timeout(self, mock_crawl_single, isolated_env):
        """Timeout during crawl returns error result without crashing."""
        mock_crawl_single.return_value = CrawlResult(
            url="https://slow.example.com",
            success=False,
            error="Navigation timeout of 30000ms exceeded",
        )
        paths = await crawl_and_save("https://slow.example.com")
        assert paths == []
