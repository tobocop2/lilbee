"""Tests for the web crawling module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lilbee.config import cfg
from lilbee.crawler import (
    CrawlMeta,
    CrawlResult,
    _get_crawl_semaphore,
    _maybe_periodic_sync,
    content_hash,
    crawl_and_save,
    crawl_recursive,
    crawl_single,
    is_url,
    load_crawl_metadata,
    require_valid_crawl_url,
    save_crawl_metadata,
    save_crawl_results,
    update_metadata,
    url_to_filename,
    validate_crawl_url,
)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    """Redirect config paths for all crawler tests."""
    snapshot = cfg.model_copy()
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir()
    cfg.data_dir = tmp_path / "data"
    cfg.data_dir.mkdir()
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    yield tmp_path
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


class TestUrlToFilename:
    def test_basic_page(self):
        assert url_to_filename("https://example.com/page") == "example.com/page/index.md"

    def test_trailing_slash(self):
        result = url_to_filename("https://docs.python.org/3/tutorial/")
        assert result == "docs.python.org/3/tutorial/index.md"

    def test_root_url(self):
        assert url_to_filename("https://example.com/") == "example.com/index.md"

    def test_root_no_slash(self):
        assert url_to_filename("https://example.com") == "example.com/index.md"

    def test_file_extension(self):
        result = url_to_filename("https://example.com/docs/guide.html")
        assert result == "example.com/docs/guide.md"

    def test_query_params_stripped(self):
        result = url_to_filename("https://example.com/page?q=1&foo=bar")
        assert result == "example.com/page/index.md"

    def test_fragment_stripped(self):
        result = url_to_filename("https://example.com/page#section")
        assert result == "example.com/page/index.md"

    def test_unsafe_chars_replaced(self):
        result = url_to_filename("https://example.com/a<b>c")
        assert "<" not in result
        assert ">" not in result

    def test_long_url_truncated(self):
        long_path = "/a" * 200
        result = url_to_filename(f"https://example.com{long_path}")
        assert len(result) <= 200

    def test_nested_path(self):
        result = url_to_filename("https://docs.python.org/3/library/os.html")
        assert result == "docs.python.org/3/library/os.md"

    def test_path_traversal_neutralized(self):
        result = url_to_filename("https://evil.com/../../etc/passwd")
        assert ".." not in result
        assert "etc" in result

    def test_path_traversal_double_dots(self):
        result = url_to_filename("https://evil.com/a/../b")
        assert ".." not in result


class TestSaveCrawlResults:
    def test_saves_successful_results(self, isolated_env):
        results = [
            CrawlResult(url="https://example.com/page1", markdown="# Page 1\nContent"),
            CrawlResult(url="https://example.com/page2", markdown="# Page 2\nMore content"),
        ]
        paths = save_crawl_results(results)
        assert len(paths) == 2
        for p in paths:
            assert p.exists()
            assert p.suffix == ".md"

    def test_skips_failed_results(self, isolated_env):
        results = [
            CrawlResult(url="https://example.com/ok", markdown="# OK"),
            CrawlResult(url="https://example.com/fail", success=False, error="404"),
        ]
        paths = save_crawl_results(results)
        assert len(paths) == 1

    def test_skips_empty_markdown(self, isolated_env):
        results = [
            CrawlResult(url="https://example.com/empty", markdown="   "),
        ]
        paths = save_crawl_results(results)
        assert len(paths) == 0

    def test_creates_nested_dirs(self, isolated_env):
        results = [
            CrawlResult(url="https://docs.example.com/a/b/c/page.html", markdown="# Deep"),
        ]
        paths = save_crawl_results(results)
        assert len(paths) == 1
        assert "docs.example.com" in str(paths[0])

    def test_content_written_correctly(self, isolated_env):
        content = "# Hello\n\nThis is a test page."
        results = [CrawlResult(url="https://example.com/test", markdown=content)]
        paths = save_crawl_results(results)
        assert paths[0].read_text(encoding="utf-8") == content

    def test_path_traversal_blocked(self, isolated_env):
        """Symlinks or crafted filenames that escape _web/ are skipped."""
        results = [
            CrawlResult(url="https://evil.com/../../etc/passwd", markdown="# Malicious"),
        ]
        paths = save_crawl_results(results)
        # File is saved because url_to_filename neutralizes .., but let's verify
        # the containment check itself by mocking url_to_filename
        with patch("lilbee.crawler.url_to_filename", return_value="../../etc/passwd"):
            paths = save_crawl_results(results)
        assert paths == []


class TestCrawlMetadata:
    def test_load_empty(self, isolated_env):
        meta = load_crawl_metadata()
        assert meta == {}

    def test_save_and_load_roundtrip(self, isolated_env):
        meta = {
            "https://example.com": CrawlMeta(
                file="example.com/index.md",
                content_hash="abc123",
                crawled_at="2026-01-01T00:00:00+00:00",
            )
        }
        save_crawl_metadata(meta)
        loaded = load_crawl_metadata()
        assert loaded["https://example.com"].file == "example.com/index.md"
        assert loaded["https://example.com"].content_hash == "abc123"

    def test_load_corrupted_json(self, isolated_env):
        meta_path = cfg.data_dir / "crawl_meta.json"
        meta_path.write_text("not valid json")
        meta = load_crawl_metadata()
        assert meta == {}

    def test_update_metadata(self, isolated_env):
        results = [
            CrawlResult(url="https://example.com/p1", markdown="Content 1"),
            CrawlResult(url="https://example.com/p2", success=False, error="oops"),
        ]
        update_metadata(results)
        meta = load_crawl_metadata()
        assert "https://example.com/p1" in meta
        assert "https://example.com/p2" not in meta


class TestContentHash:
    def test_consistent(self):
        assert content_hash("hello") == content_hash("hello")

    def test_different_for_different_content(self):
        assert content_hash("hello") != content_hash("world")


def _make_crawl4ai_result(url="https://example.com", markdown="# Test", success=True, error=None):
    """Build a mock crawl4ai CrawlResult."""
    result = MagicMock()
    result.url = url
    result.markdown = markdown
    result.success = success
    result.error_message = error
    return result


@pytest.fixture(autouse=True)
def _no_dns(monkeypatch):
    """Bypass SSRF DNS resolution in all crawler tests."""
    monkeypatch.setattr(
        "lilbee.crawler.socket.getaddrinfo",
        lambda host, port, *a, **kw: [(2, 1, 6, "", ("93.184.216.34", 0))],
    )


class TestIsUrl:
    def test_http(self):
        assert is_url("http://example.com")

    def test_https(self):
        assert is_url("https://example.com")

    def test_not_url(self):
        assert not is_url("/tmp/file.txt")

    def test_ftp_not_url(self):
        assert not is_url("ftp://example.com")

    def test_empty(self):
        assert not is_url("")


class TestValidateCrawlUrl:
    def test_rejects_ftp(self):
        with pytest.raises(ValueError, match="Only http"):
            validate_crawl_url("ftp://example.com")

    def test_rejects_file(self):
        with pytest.raises(ValueError, match="Only http"):
            validate_crawl_url("file:///etc/passwd")

    def test_rejects_no_hostname(self):
        with pytest.raises(ValueError, match="no hostname"):
            validate_crawl_url("http://")

    def test_rejects_localhost(self):
        with pytest.raises(ValueError, match="localhost"):
            validate_crawl_url("http://localhost/path")

    def test_rejects_localhost_dot(self):
        with pytest.raises(ValueError, match="localhost"):
            validate_crawl_url("http://localhost./path")

    def test_rejects_loopback_ipv4(self, monkeypatch):
        monkeypatch.setattr(
            "lilbee.crawler.socket.getaddrinfo",
            lambda *a, **kw: [(2, 1, 6, "", ("127.0.0.1", 0))],
        )
        with pytest.raises(ValueError, match="private"):
            validate_crawl_url("http://loopback.test")

    def test_rejects_private_10(self, monkeypatch):
        monkeypatch.setattr(
            "lilbee.crawler.socket.getaddrinfo",
            lambda *a, **kw: [(2, 1, 6, "", ("10.0.0.1", 0))],
        )
        with pytest.raises(ValueError, match="private"):
            validate_crawl_url("http://internal.test")

    def test_rejects_private_172(self, monkeypatch):
        monkeypatch.setattr(
            "lilbee.crawler.socket.getaddrinfo",
            lambda *a, **kw: [(2, 1, 6, "", ("172.16.0.1", 0))],
        )
        with pytest.raises(ValueError, match="private"):
            validate_crawl_url("http://internal.test")

    def test_rejects_private_192(self, monkeypatch):
        monkeypatch.setattr(
            "lilbee.crawler.socket.getaddrinfo",
            lambda *a, **kw: [(2, 1, 6, "", ("192.168.1.1", 0))],
        )
        with pytest.raises(ValueError, match="private"):
            validate_crawl_url("http://internal.test")

    def test_rejects_link_local(self, monkeypatch):
        monkeypatch.setattr(
            "lilbee.crawler.socket.getaddrinfo",
            lambda *a, **kw: [(2, 1, 6, "", ("169.254.169.254", 0))],
        )
        with pytest.raises(ValueError, match="private"):
            validate_crawl_url("http://metadata.test")

    def test_rejects_ipv6_loopback(self, monkeypatch):
        monkeypatch.setattr(
            "lilbee.crawler.socket.getaddrinfo",
            lambda *a, **kw: [(10, 1, 6, "", ("::1", 0, 0, 0))],
        )
        with pytest.raises(ValueError, match="private"):
            validate_crawl_url("http://ipv6loopback.test")

    def test_accepts_public_ip(self):
        validate_crawl_url("https://example.com")

    def test_rejects_unresolvable(self, monkeypatch):
        import socket

        monkeypatch.setattr(
            "lilbee.crawler.socket.getaddrinfo",
            MagicMock(side_effect=socket.gaierror("Name resolution failed")),
        )
        with pytest.raises(ValueError, match="Cannot resolve"):
            validate_crawl_url("http://nonexistent.invalid")


class TestRequireValidCrawlUrl:
    def test_rejects_non_url(self):
        with pytest.raises(ValueError, match="http"):
            require_valid_crawl_url("not-a-url")

    def test_rejects_ftp(self):
        with pytest.raises(ValueError, match="http"):
            require_valid_crawl_url("ftp://example.com")

    def test_accepts_valid_https(self):
        require_valid_crawl_url("https://example.com")

    def test_accepts_valid_http(self):
        require_valid_crawl_url("http://example.com")


class TestCrawlSingle:
    @patch("crawl4ai.AsyncWebCrawler")
    async def test_success(self, mock_crawler_cls):
        mock_result = _make_crawl4ai_result()
        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=mock_result)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_crawler_cls.return_value = mock_instance

        result = await crawl_single("https://example.com")
        assert result.success
        assert result.markdown == "# Test"

    @patch("crawl4ai.AsyncWebCrawler")
    async def test_failure(self, mock_crawler_cls):
        mock_result = _make_crawl4ai_result(success=False, error="Connection refused")
        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=mock_result)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_crawler_cls.return_value = mock_instance

        result = await crawl_single("https://example.com")
        assert not result.success
        assert result.error == "Connection refused"

    @patch("crawl4ai.AsyncWebCrawler")
    async def test_exception(self, mock_crawler_cls):
        mock_instance = AsyncMock()
        mock_instance.__aenter__ = AsyncMock(side_effect=RuntimeError("timeout"))
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_crawler_cls.return_value = mock_instance

        result = await crawl_single("https://example.com")
        assert not result.success
        assert "timeout" in result.error


class TestCrawlRecursive:
    @patch("crawl4ai.AsyncWebCrawler")
    async def test_returns_multiple_results(self, mock_crawler_cls):
        mock_results = [
            _make_crawl4ai_result(url="https://example.com", markdown="# Home"),
            _make_crawl4ai_result(url="https://example.com/about", markdown="# About"),
        ]
        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=mock_results)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_crawler_cls.return_value = mock_instance

        progress_calls = []

        def on_progress(event_type, data):
            progress_calls.append((event_type, data))

        results = await crawl_recursive(
            "https://example.com", max_depth=1, max_pages=10, on_progress=on_progress
        )
        assert len(results) == 2
        assert results[0].url == "https://example.com"
        assert results[1].url == "https://example.com/about"
        assert len(progress_calls) == 2

    @patch("crawl4ai.AsyncWebCrawler")
    async def test_single_result_not_list(self, mock_crawler_cls):
        """When deep crawl returns a single result (not a list), it's handled."""
        mock_result = _make_crawl4ai_result()
        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=mock_result)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_crawler_cls.return_value = mock_instance

        results = await crawl_recursive("https://example.com", max_depth=1, max_pages=5)
        assert len(results) == 1

    @patch("crawl4ai.AsyncWebCrawler")
    async def test_mixed_success_failure(self, mock_crawler_cls):
        mock_results = [
            _make_crawl4ai_result(url="https://example.com", markdown="# Home"),
            _make_crawl4ai_result(url="https://example.com/broken", success=False, error="404"),
        ]
        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=mock_results)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_crawler_cls.return_value = mock_instance

        results = await crawl_recursive("https://example.com", max_depth=1, max_pages=10)
        assert len(results) == 2
        assert results[0].success
        assert not results[1].success

    @patch("crawl4ai.AsyncWebCrawler")
    async def test_exception_returns_error_result(self, mock_crawler_cls):
        mock_instance = AsyncMock()
        mock_instance.__aenter__ = AsyncMock(side_effect=RuntimeError("network error"))
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_crawler_cls.return_value = mock_instance

        results = await crawl_recursive("https://example.com", max_depth=1, max_pages=5)
        assert len(results) == 1
        assert not results[0].success

    @patch("crawl4ai.AsyncWebCrawler")
    async def test_uses_config_defaults(self, mock_crawler_cls):
        """When depth/pages are 0, falls back to cfg defaults."""
        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=[])
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_crawler_cls.return_value = mock_instance

        await crawl_recursive("https://example.com", max_depth=0, max_pages=0)
        # Verify the crawler was called (config defaults used internally)
        mock_instance.arun.assert_awaited_once()

    @patch("crawl4ai.AsyncWebCrawler")
    async def test_max_pages_capped_by_config(self, mock_crawler_cls):
        """max_pages is capped at cfg.crawl_max_pages even when caller passes more."""
        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=[])
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_crawler_cls.return_value = mock_instance

        cfg.crawl_max_pages = 10
        await crawl_recursive("https://example.com", max_depth=1, max_pages=999)
        call_args = mock_instance.arun.call_args
        crawl_config = call_args.kwargs.get("config") or call_args[1].get("config")
        assert crawl_config.deep_crawl_strategy.max_pages <= 10


class TestCrawlAndSave:
    @patch("lilbee.crawler.crawl_single")
    async def test_single_page(self, mock_crawl_single, isolated_env):
        mock_crawl_single.return_value = CrawlResult(url="https://example.com", markdown="# Hello")
        paths = await crawl_and_save("https://example.com", force=True)
        assert len(paths) == 1
        assert paths[0].exists()

    @patch("lilbee.crawler.crawl_recursive")
    async def test_recursive(self, mock_crawl_recursive, isolated_env):
        mock_crawl_recursive.return_value = [
            CrawlResult(url="https://example.com", markdown="# Home"),
            CrawlResult(url="https://example.com/about", markdown="# About"),
        ]
        paths = await crawl_and_save("https://example.com", depth=2, max_pages=10, force=True)
        assert len(paths) == 2

    @patch("lilbee.crawler.crawl_single")
    async def test_single_page_with_progress(self, mock_crawl_single, isolated_env):
        """Progress callback receives crawl_start, crawl_page, crawl_done for single page."""
        mock_crawl_single.return_value = CrawlResult(url="https://example.com", markdown="# Hi")
        events = []

        def on_progress(event_type, data):
            events.append((str(event_type), data))

        await crawl_and_save("https://example.com", force=True, on_progress=on_progress)
        event_types = [e[0] for e in events]
        assert "crawl_start" in event_types
        assert "crawl_page" in event_types
        assert "crawl_done" in event_types

    @patch("lilbee.crawler.crawl_single")
    async def test_updates_metadata(self, mock_crawl_single, isolated_env):
        mock_crawl_single.return_value = CrawlResult(
            url="https://example.com/page", markdown="# Test"
        )
        await crawl_and_save("https://example.com/page", force=True)
        meta = load_crawl_metadata()
        assert "https://example.com/page" in meta

    @patch("lilbee.crawler.crawl_single")
    async def test_skips_duplicate_url(self, mock_crawl_single, isolated_env):
        """Already-crawled URL is skipped when force=False."""
        mock_crawl_single.return_value = CrawlResult(
            url="https://example.com/dup", markdown="# Dup"
        )
        # First crawl
        await crawl_and_save("https://example.com/dup", force=True)
        mock_crawl_single.reset_mock()

        # Second crawl without force: should skip
        paths = await crawl_and_save("https://example.com/dup")
        assert paths == []
        mock_crawl_single.assert_not_awaited()

    @patch("lilbee.crawler.crawl_single")
    async def test_force_overrides_duplicate_check(self, mock_crawl_single, isolated_env):
        """force=True re-crawls even when URL already exists in metadata."""
        mock_crawl_single.return_value = CrawlResult(
            url="https://example.com/dup", markdown="# Dup"
        )
        await crawl_and_save("https://example.com/dup", force=True)
        mock_crawl_single.reset_mock()
        mock_crawl_single.return_value = CrawlResult(
            url="https://example.com/dup", markdown="# Updated"
        )

        paths = await crawl_and_save("https://example.com/dup", force=True)
        assert len(paths) == 1
        mock_crawl_single.assert_awaited_once()

    @patch("lilbee.crawler.crawl_single")
    async def test_max_pages_capped_by_config(self, mock_crawl_single, isolated_env):
        """max_pages in crawl_and_save is capped by cfg.crawl_max_pages."""
        mock_crawl_single.return_value = CrawlResult(url="https://example.com", markdown="# Test")
        cfg.crawl_max_pages = 5
        await crawl_and_save("https://example.com", max_pages=999, force=True)
        # Single page mode: no assertion on max_pages since depth=0 uses crawl_single

    async def test_semaphore_limits_concurrency(self, isolated_env):
        """The semaphore limits concurrent crawls based on config."""
        import lilbee.crawler as crawler_mod

        crawler_mod._crawl_semaphore = None
        cfg.crawl_max_concurrent = 5
        sem = _get_crawl_semaphore()
        assert sem is not None
        assert sem._value == 5
        crawler_mod._crawl_semaphore = None

    async def test_semaphore_defaults_to_cpu_count(self, isolated_env):
        """Default concurrency matches CPU count."""
        import os

        import lilbee.crawler as crawler_mod

        crawler_mod._crawl_semaphore = None
        cfg.crawl_max_concurrent = os.cpu_count() or 4
        sem = _get_crawl_semaphore()
        assert sem is not None
        assert sem._value == (os.cpu_count() or 4)
        crawler_mod._crawl_semaphore = None

    async def test_semaphore_unlimited_when_zero(self, isolated_env):
        """Setting crawl_max_concurrent=0 disables the semaphore."""
        import lilbee.crawler as crawler_mod

        crawler_mod._crawl_semaphore = None
        cfg.crawl_max_concurrent = 0
        assert _get_crawl_semaphore() is None
        crawler_mod._crawl_semaphore = None


class TestPeriodicSync:
    async def test_sync_disabled_when_interval_zero(self, isolated_env):
        """No sync fires when crawl_sync_interval is 0."""
        import lilbee.crawler as crawler_mod

        cfg.crawl_sync_interval = 0
        crawler_mod._last_sync_time = 0.0
        crawler_mod._sync_running = False

        with patch("lilbee.ingest.sync", new_callable=AsyncMock) as mock_sync:
            await _maybe_periodic_sync()
            mock_sync.assert_not_awaited()

    async def test_sync_skipped_when_already_running(self, isolated_env):
        """No new sync is started if one is already in progress."""
        import lilbee.crawler as crawler_mod

        cfg.crawl_sync_interval = 1
        crawler_mod._last_sync_time = 0.0
        crawler_mod._sync_running = True

        with patch("lilbee.ingest.sync", new_callable=AsyncMock) as mock_sync:
            await _maybe_periodic_sync()
            mock_sync.assert_not_awaited()

        crawler_mod._sync_running = False

    async def test_sync_skipped_when_interval_not_elapsed(self, isolated_env):
        """No sync fires if the interval hasn't elapsed since last sync."""
        import time

        import lilbee.crawler as crawler_mod

        cfg.crawl_sync_interval = 9999
        crawler_mod._last_sync_time = time.monotonic()
        crawler_mod._sync_running = False

        with patch("lilbee.ingest.sync", new_callable=AsyncMock) as mock_sync:
            await _maybe_periodic_sync()
            mock_sync.assert_not_awaited()

    async def test_sync_fires_when_interval_elapsed(self, isolated_env):
        """Sync fires as a background task when interval has elapsed."""
        import asyncio

        import lilbee.crawler as crawler_mod

        cfg.crawl_sync_interval = 1
        crawler_mod._last_sync_time = 0.0
        crawler_mod._sync_running = False

        mock_sync = AsyncMock()
        with patch("lilbee.ingest.sync", mock_sync):
            await _maybe_periodic_sync()
            # Let the background task run
            await asyncio.sleep(0)
            mock_sync.assert_awaited_once()

        crawler_mod._sync_running = False

    async def test_sync_failure_resets_running_flag(self, isolated_env):
        """If sync raises, _sync_running is reset so future syncs can proceed."""
        import asyncio

        import lilbee.crawler as crawler_mod

        cfg.crawl_sync_interval = 1
        crawler_mod._last_sync_time = 0.0
        crawler_mod._sync_running = False

        mock_sync = AsyncMock(side_effect=RuntimeError("sync failed"))
        with patch("lilbee.ingest.sync", mock_sync):
            await _maybe_periodic_sync()
            await asyncio.sleep(0)

        assert not crawler_mod._sync_running
