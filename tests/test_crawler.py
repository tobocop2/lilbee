"""Tests for the web crawling module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lilbee.config import cfg
from lilbee.crawler import (
    CrawlMeta,
    CrawlResult,
    content_hash,
    crawl_and_save,
    crawl_recursive,
    crawl_single,
    load_crawl_metadata,
    save_crawl_metadata,
    save_crawl_results,
    update_metadata,
    url_to_filename,
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

        def on_progress(crawled, total, url):
            progress_calls.append((crawled, total, url))

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


class TestCrawlAndSave:
    @patch("lilbee.crawler.crawl_single")
    async def test_single_page(self, mock_crawl_single, isolated_env):
        mock_crawl_single.return_value = CrawlResult(url="https://example.com", markdown="# Hello")
        paths = await crawl_and_save("https://example.com")
        assert len(paths) == 1
        assert paths[0].exists()

    @patch("lilbee.crawler.crawl_recursive")
    async def test_recursive(self, mock_crawl_recursive, isolated_env):
        mock_crawl_recursive.return_value = [
            CrawlResult(url="https://example.com", markdown="# Home"),
            CrawlResult(url="https://example.com/about", markdown="# About"),
        ]
        paths = await crawl_and_save("https://example.com", depth=2, max_pages=10)
        assert len(paths) == 2

    @patch("lilbee.crawler.crawl_single")
    async def test_updates_metadata(self, mock_crawl_single, isolated_env):
        mock_crawl_single.return_value = CrawlResult(
            url="https://example.com/page", markdown="# Test"
        )
        await crawl_and_save("https://example.com/page")
        meta = load_crawl_metadata()
        assert "https://example.com/page" in meta
