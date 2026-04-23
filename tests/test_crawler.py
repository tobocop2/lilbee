"""Tests for the web crawling module."""

import asyncio
import sys
from pathlib import Path
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
    is_url,
    load_crawl_metadata,
    require_valid_crawl_url,
    save_crawl_metadata,
    url_to_filename,
    validate_crawl_url,
)
from lilbee.crawler.api import _get_crawl_semaphore, _maybe_periodic_sync
from lilbee.crawler.save import _save_single_result, _update_single_metadata
from lilbee.progress import EventType


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch, request):
    """Redirect config paths for all crawler tests.

    Also stubs :func:`chromium_installed` to return True so the
    pre-flight guard in ``_open_crawler`` doesn't trip in CI envs where
    Playwright's Chromium binary is absent. Tests marked
    ``real_browser_check`` (or in :class:`TestPlaywrightBrowserCheck`) get
    the real function so they can drive the check directly.
    """
    snapshot = cfg.model_copy()
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir()
    cfg.data_dir = tmp_path / "data"
    cfg.data_dir.mkdir()
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cls = request.cls.__name__ if request.cls else ""
    if cls != "TestPlaywrightBrowserCheck":
        monkeypatch.setattr("lilbee.crawler.bootstrap.chromium_installed", lambda: True)
    # Default the sitemap bound to "unknown" so tests don't hit the network.
    # Tests that exercise the sitemap hook directly (TestSitemapCounting)
    # opt out of this autopatch.
    if cls != "TestSitemapCounting":
        from lilbee.progress import CRAWL_TOTAL_UNKNOWN

        monkeypatch.setattr(
            "lilbee.crawler.sitemap._count_sitemap_urls",
            lambda *a, **kw: CRAWL_TOTAL_UNKNOWN,
        )
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

    def test_load_malformed_entry_skipped(self, isolated_env):
        """Entries that fail CrawlMeta(**data) are skipped with a warning."""
        import json

        meta_path = cfg.data_dir / "crawl_meta.json"
        meta_path.write_text(json.dumps({"https://bad.com": {"wrong_field": "value"}}))
        meta = load_crawl_metadata()
        assert meta == {}

    def test_save_atomic_write_cleans_up_on_error(self, isolated_env):
        """If the atomic write fails, the tmp file is removed and error re-raised."""
        meta = {
            "https://example.com": CrawlMeta(
                file="example.com/index.md",
                content_hash="abc",
                crawled_at="2026-01-01T00:00:00+00:00",
            )
        }
        with (
            patch("lilbee.crawler.save.Path.replace", side_effect=OSError("disk full")),
            pytest.raises(OSError, match="disk full"),
        ):
            save_crawl_metadata(meta)
        tmp_path = cfg.data_dir / "crawl_meta.tmp"
        assert not tmp_path.exists()


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
        "lilbee.crawler.url_filter.socket.getaddrinfo",
        lambda host, port, *a, **kw: [(2, 1, 6, "", ("93.184.216.34", 0))],
    )


class TestCrawlerAvailable:
    def test_returns_true_when_installed(self):
        from lilbee.crawler import crawler_available

        mock_crawl4ai = MagicMock()
        with patch.dict("sys.modules", {"crawl4ai": mock_crawl4ai}):
            assert crawler_available() is True

    def test_returns_false_when_not_installed(self):
        from lilbee.crawler import crawler_available

        with patch.dict("sys.modules", {"crawl4ai": None}):
            assert crawler_available() is False


class TestIsUrl:
    def test_http(self):
        assert is_url("http://example.com")

    def test_https(self):
        assert is_url("https://example.com")

    def test_not_url(self):
        assert not is_url("/some/file.txt")

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

    def test_rejects_localhost(self, monkeypatch):
        monkeypatch.setattr(
            "lilbee.crawler.url_filter.socket.getaddrinfo",
            lambda *a, **kw: [(2, 1, 6, "", ("127.0.0.1", 0))],
        )
        with pytest.raises(ValueError, match="not allowed"):
            validate_crawl_url("http://localhost/path")

    def test_rejects_localhost_dot(self, monkeypatch):
        monkeypatch.setattr(
            "lilbee.crawler.url_filter.socket.getaddrinfo",
            lambda *a, **kw: [(2, 1, 6, "", ("127.0.0.1", 0))],
        )
        with pytest.raises(ValueError, match="not allowed"):
            validate_crawl_url("http://localhost./path")

    def test_rejects_loopback_ipv4(self, monkeypatch):
        monkeypatch.setattr(
            "lilbee.crawler.url_filter.socket.getaddrinfo",
            lambda *a, **kw: [(2, 1, 6, "", ("127.0.0.1", 0))],
        )
        with pytest.raises(ValueError, match="private"):
            validate_crawl_url("http://loopback.test")

    def test_rejects_private_10(self, monkeypatch):
        monkeypatch.setattr(
            "lilbee.crawler.url_filter.socket.getaddrinfo",
            lambda *a, **kw: [(2, 1, 6, "", ("10.0.0.1", 0))],
        )
        with pytest.raises(ValueError, match="private"):
            validate_crawl_url("http://internal.test")

    def test_rejects_private_172(self, monkeypatch):
        monkeypatch.setattr(
            "lilbee.crawler.url_filter.socket.getaddrinfo",
            lambda *a, **kw: [(2, 1, 6, "", ("172.16.0.1", 0))],
        )
        with pytest.raises(ValueError, match="private"):
            validate_crawl_url("http://internal.test")

    def test_rejects_private_192(self, monkeypatch):
        monkeypatch.setattr(
            "lilbee.crawler.url_filter.socket.getaddrinfo",
            lambda *a, **kw: [(2, 1, 6, "", ("192.168.1.1", 0))],
        )
        with pytest.raises(ValueError, match="private"):
            validate_crawl_url("http://internal.test")

    def test_rejects_link_local(self, monkeypatch):
        monkeypatch.setattr(
            "lilbee.crawler.url_filter.socket.getaddrinfo",
            lambda *a, **kw: [(2, 1, 6, "", ("169.254.169.254", 0))],
        )
        with pytest.raises(ValueError, match="private"):
            validate_crawl_url("http://metadata.test")

    def test_rejects_ipv6_loopback(self, monkeypatch):
        monkeypatch.setattr(
            "lilbee.crawler.url_filter.socket.getaddrinfo",
            lambda *a, **kw: [(10, 1, 6, "", ("::1", 0, 0, 0))],
        )
        with pytest.raises(ValueError, match="private"):
            validate_crawl_url("http://ipv6loopback.test")

    def test_accepts_public_ip(self):
        validate_crawl_url("https://example.com")

    def test_rejects_unresolvable(self, monkeypatch):
        import socket

        monkeypatch.setattr(
            "lilbee.crawler.url_filter.socket.getaddrinfo",
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


def _mock_crawl4ai(mock_crawler_cls):
    """Install a fake crawl4ai module in sys.modules with the given AsyncWebCrawler."""
    mock_mod = MagicMock()
    mock_mod.AsyncWebCrawler = mock_crawler_cls
    mock_mod.CrawlerRunConfig = MagicMock()
    return mock_mod


class TestCrawlSingle:
    async def test_success(self):
        mock_result = _make_crawl4ai_result()
        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=mock_result)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_crawler_cls = MagicMock(return_value=mock_instance)
        mock_mod = _mock_crawl4ai(mock_crawler_cls)

        with patch.dict("sys.modules", {"crawl4ai": mock_mod}):
            result = await crawl_single("https://example.com")
        assert result.success
        assert result.markdown == "# Test"

    async def test_failure(self):
        mock_result = _make_crawl4ai_result(success=False, markdown="", error="Connection refused")
        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=mock_result)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_crawler_cls = MagicMock(return_value=mock_instance)
        mock_mod = _mock_crawl4ai(mock_crawler_cls)

        with patch.dict("sys.modules", {"crawl4ai": mock_mod}):
            result = await crawl_single("https://example.com")
        assert not result.success
        assert result.error == "Connection refused"

    async def test_exception(self):
        mock_instance = AsyncMock()
        mock_instance.__aenter__ = AsyncMock(side_effect=RuntimeError("timeout"))
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_crawler_cls = MagicMock(return_value=mock_instance)
        mock_mod = _mock_crawl4ai(mock_crawler_cls)

        with patch.dict("sys.modules", {"crawl4ai": mock_mod}):
            result = await crawl_single("https://example.com")
        assert not result.success
        assert "timeout" in result.error

    async def test_quiet_passes_verbose_false(self):
        """quiet=True passes verbose=False to AsyncWebCrawler."""
        mock_result = _make_crawl4ai_result()
        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=mock_result)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_crawler_cls = MagicMock(return_value=mock_instance)
        mock_mod = _mock_crawl4ai(mock_crawler_cls)

        with patch.dict("sys.modules", {"crawl4ai": mock_mod}):
            await crawl_single("https://example.com", quiet=True)
        mock_crawler_cls.assert_called_once_with(verbose=False)

    async def test_missing_chromium_raises_crawler_browser_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without Chromium installed, crawl_single raises a clean exception.

        Regression test for bb-60mj: without this guard Playwright prints
        a raw ASCII install banner into the TUI and the task lands as DONE.
        """
        from lilbee.crawler import CrawlerBrowserMissing

        # Stub crawl4ai so the test runs even when the `crawler` extra
        # isn't installed in the unit-test env.
        monkeypatch.setitem(__import__("sys").modules, "crawl4ai", _mock_crawl4ai(MagicMock()))
        monkeypatch.setattr("lilbee.crawler.bootstrap.chromium_installed", lambda: False)
        with pytest.raises(CrawlerBrowserMissing, match="Chromium"):
            await crawl_single("https://example.com")


class TestBootstrapChromium:
    """bb-wq8g: the subprocess wrapper that installs Playwright's Chromium."""

    async def test_short_circuits_when_already_installed(self, monkeypatch):
        """No subprocess, no stream events, when Chromium is already present."""
        from lilbee.crawler.bootstrap import bootstrap_chromium
        from lilbee.progress import EventType, SetupDoneEvent

        monkeypatch.setattr("lilbee.crawler.bootstrap.chromium_installed", lambda: True)
        events: list[tuple[EventType, object]] = []
        await bootstrap_chromium(on_progress=lambda e, d: events.append((e, d)))
        # Only a single setup_done (success) — no start/progress when we
        # short-circuit.
        assert len(events) == 1
        evt, payload = events[0]
        assert evt == EventType.SETUP_DONE
        assert isinstance(payload, SetupDoneEvent)
        assert payload.success is True

    async def test_parses_progress_from_fake_subprocess(self, monkeypatch):
        """Feed canned stdout through the subprocess to drive progress events."""
        from lilbee.crawler.bootstrap import bootstrap_chromium
        from lilbee.progress import EventType, SetupProgressEvent, SetupStartEvent

        monkeypatch.setattr("lilbee.crawler.bootstrap.chromium_installed", lambda: False)

        _bar = "\xe2\x96\xa0".encode("latin-1")  # three-byte UTF-8 for ■
        stdout_lines = [
            b"Downloading Chromium 145.0 ...\n",
            b"|" + _bar * 2 + b"        |  25% of 162.3 MiB\n",
            b"|" + _bar * 4 + b"    |  50% of 162.3 MiB\n",
            b"|" + _bar * 8 + b"| 100% of 162.3 MiB\n",
            b"",  # EOF
        ]

        class _Stream:
            def __init__(self, lines: list[bytes]) -> None:
                self._lines = list(lines)

            async def readline(self) -> bytes:
                return self._lines.pop(0) if self._lines else b""

        class _Proc:
            def __init__(self) -> None:
                self.stdout = _Stream(stdout_lines)
                self.stderr = _Stream([b""])

            async def wait(self) -> int:
                return 0

        async def _fake_create_subprocess_exec(*_args, **_kwargs):
            return _Proc()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

        events: list[tuple[EventType, object]] = []
        await bootstrap_chromium(on_progress=lambda e, d: events.append((e, d)))

        types = [e for e, _ in events]
        assert types[0] == EventType.SETUP_START
        assert types[-1] == EventType.SETUP_DONE
        progress_events = [d for e, d in events if e == EventType.SETUP_PROGRESS]
        assert len(progress_events) >= 1
        assert isinstance(events[0][1], SetupStartEvent)
        assert isinstance(progress_events[0], SetupProgressEvent)

    async def test_raises_crawler_browser_missing_on_subprocess_failure(self, monkeypatch):
        """Non-zero exit → CrawlerBrowserMissing with stderr tail."""
        from lilbee.crawler.bootstrap import CrawlerBrowserMissing, bootstrap_chromium
        from lilbee.progress import EventType, SetupDoneEvent

        monkeypatch.setattr("lilbee.crawler.bootstrap.chromium_installed", lambda: False)

        class _Stream:
            def __init__(self, lines: list[bytes]) -> None:
                self._lines = list(lines)

            async def readline(self) -> bytes:
                return self._lines.pop(0) if self._lines else b""

        class _Proc:
            def __init__(self) -> None:
                self.stdout = _Stream([b""])
                self.stderr = _Stream(
                    [b"error: network unreachable\n", b"cannot bind socket\n", b""]
                )

            async def wait(self) -> int:
                return 42

        async def _fake_create_subprocess_exec(*_args, **_kwargs):
            return _Proc()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

        events: list[tuple[EventType, object]] = []
        with pytest.raises(CrawlerBrowserMissing, match="exit 42"):
            await bootstrap_chromium(on_progress=lambda e, d: events.append((e, d)))
        final = events[-1]
        assert final[0] == EventType.SETUP_DONE
        assert isinstance(final[1], SetupDoneEvent)
        assert final[1].success is False
        assert "network unreachable" in (final[1].error or "")


class TestPlaywrightBrowserCheck:
    def test_detects_missing_browsers(self, tmp_path, monkeypatch):
        """Empty browsers path reports as not installed."""
        from lilbee.crawler.bootstrap import chromium_installed

        monkeypatch.setattr(
            "lilbee.crawler.bootstrap._browsers_cache_path", lambda: tmp_path / "empty"
        )
        assert not chromium_installed()

    def test_nonexistent_root_reports_missing(self, tmp_path, monkeypatch):
        """A root path that doesn't exist reports as missing (not a crash)."""
        from lilbee.crawler.bootstrap import chromium_installed

        monkeypatch.setattr(
            "lilbee.crawler.bootstrap._browsers_cache_path",
            lambda: tmp_path / "does" / "not" / "exist",
        )
        assert not chromium_installed()

    def test_detects_installed_chromium(self, tmp_path, monkeypatch):
        """A chromium-* subdirectory counts as installed."""
        from lilbee.crawler.bootstrap import chromium_installed

        browsers = tmp_path / "ms-playwright"
        browsers.mkdir()
        (browsers / "chromium-1234").mkdir()
        monkeypatch.setattr("lilbee.crawler.bootstrap._browsers_cache_path", lambda: browsers)
        assert chromium_installed()

    def test_path_respects_env_override(self, tmp_path, monkeypatch):
        """PLAYWRIGHT_BROWSERS_PATH overrides the platform default."""
        from lilbee.crawler.bootstrap import _browsers_cache_path

        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path / "custom"))
        assert _browsers_cache_path() == tmp_path / "custom"

    def test_path_darwin_default(self, monkeypatch):
        from lilbee.crawler.bootstrap import _browsers_cache_path

        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        monkeypatch.setattr("sys.platform", "darwin")
        parts = _browsers_cache_path().parts
        assert parts[-1] == "ms-playwright"
        assert "Library" in parts
        assert "Caches" in parts

    def test_path_linux_default(self, monkeypatch):
        from lilbee.crawler.bootstrap import _browsers_cache_path

        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        monkeypatch.setattr("sys.platform", "linux")
        path = _browsers_cache_path()
        assert path.name == "ms-playwright"
        assert ".cache" in str(path)

    def test_path_win32_default(self, monkeypatch):
        from lilbee.crawler.bootstrap import _browsers_cache_path

        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        monkeypatch.setenv("LOCALAPPDATA", "/tmp/localappdata")
        monkeypatch.setattr("sys.platform", "win32")
        assert _browsers_cache_path() == Path("/tmp/localappdata/ms-playwright")


class TestSetupEventHelpers:
    """bb-wq8g: _emit_setup_start / _emit_setup_done no-op when callback is None."""

    def test_emit_start_no_op_when_on_progress_none(self) -> None:
        from lilbee.crawler.bootstrap import _emit_setup_start

        _emit_setup_start(None)  # must not raise

    def test_emit_done_no_op_when_on_progress_none(self) -> None:
        from lilbee.crawler.bootstrap import _emit_setup_done

        _emit_setup_done(None, success=True, error=None)  # must not raise


class TestCrawlRecursive:
    def _setup_crawl4ai(self, mock_instance):
        """Create a fake crawl4ai module with the given crawler instance."""
        mock_crawler_cls = MagicMock(return_value=mock_instance)
        mock_bfs = MagicMock()
        mock_mod = _mock_crawl4ai(mock_crawler_cls)
        mock_deep = MagicMock()
        mock_deep.BFSDeepCrawlStrategy = mock_bfs
        # Recursive crawls build a SemaphoreDispatcher + RateLimiter when
        # cfg.crawl_retry_on_rate_limit is True (default). Stub both so
        # `from crawl4ai.async_dispatcher import ...` succeeds.
        mock_dispatcher_mod = MagicMock()
        mock_dispatcher_mod.RateLimiter = MagicMock()
        mock_dispatcher_mod.SemaphoreDispatcher = MagicMock()
        # Recursive crawls also build a FilterChain with URLPatternFilter to
        # exclude WordPress noise patterns. Stub both.
        mock_filters_mod = MagicMock()
        mock_filters_mod.FilterChain = MagicMock()
        mock_filters_mod.URLPatternFilter = MagicMock()
        return {
            "crawl4ai": mock_mod,
            "crawl4ai.deep_crawling": mock_deep,
            "crawl4ai.deep_crawling.filters": mock_filters_mod,
            "crawl4ai.async_dispatcher": mock_dispatcher_mod,
        }

    async def test_returns_multiple_results(self):
        mock_results = [
            _make_crawl4ai_result(url="https://example.com", markdown="# Home"),
            _make_crawl4ai_result(url="https://example.com/about", markdown="# About"),
        ]
        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=mock_results)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        progress_calls = []

        def on_progress(event_type, data):
            progress_calls.append((event_type, data))

        with patch.dict("sys.modules", self._setup_crawl4ai(mock_instance)):
            results = await crawl_recursive(
                "https://example.com", max_depth=1, max_pages=10, on_progress=on_progress
            )
        assert len(results) == 2
        assert results[0].url == "https://example.com"
        assert results[1].url == "https://example.com/about"
        assert len(progress_calls) == 2
        # Streaming semantics: total is unknown during BFS, counter advances per page.
        from lilbee.progress import CRAWL_TOTAL_UNKNOWN

        assert [c[1].current for c in progress_calls] == [1, 2]
        assert all(c[1].total == CRAWL_TOTAL_UNKNOWN for c in progress_calls)

    async def test_emits_events_before_stream_exhausted(self):
        """CRAWL_PAGE fires per page as it arrives, not only after the full list."""
        import asyncio as _asyncio

        observations: list[tuple[str, int]] = []

        async def _gen():
            for i in range(1, 4):
                await _asyncio.sleep(0)
                observations.append(("yielded", i))
                yield _make_crawl4ai_result(url=f"https://example.com/p{i}", markdown=f"# P{i}")

        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=_gen())
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        def on_progress(event_type, data):
            if event_type == EventType.CRAWL_PAGE:
                observations.append(("progress", data.current))

        with patch.dict("sys.modules", self._setup_crawl4ai(mock_instance)):
            await crawl_recursive(
                "https://example.com", max_depth=2, max_pages=100, on_progress=on_progress
            )

        # Each page's progress event must appear immediately after its yield,
        # before any subsequent yield. Pattern: yielded=1, progress=1, yielded=2, progress=2, ...
        for i in range(3):
            assert observations[2 * i] == ("yielded", i + 1)
            assert observations[2 * i + 1] == ("progress", i + 1)

    async def test_cancel_stops_mid_stream(self):
        """Setting the cancel event mid-stream stops further result collection."""
        import asyncio as _asyncio
        import threading

        cancel = threading.Event()

        async def _gen():
            for i in range(1, 6):
                await _asyncio.sleep(0)
                yield _make_crawl4ai_result(url=f"https://example.com/p{i}", markdown=f"# P{i}")
                if i == 2:
                    cancel.set()

        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=_gen())
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        with patch.dict("sys.modules", self._setup_crawl4ai(mock_instance)):
            results = await crawl_recursive(
                "https://example.com", max_depth=2, max_pages=100, cancel=cancel
            )

        assert len(results) <= 2

    async def test_single_result_not_list(self):
        """When deep crawl returns a single result (not a list), it's handled."""
        mock_result = _make_crawl4ai_result()
        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=mock_result)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        with patch.dict("sys.modules", self._setup_crawl4ai(mock_instance)):
            results = await crawl_recursive("https://example.com", max_depth=1, max_pages=5)
        assert len(results) == 1

    async def test_mixed_success_failure(self):
        mock_results = [
            _make_crawl4ai_result(url="https://example.com", markdown="# Home"),
            _make_crawl4ai_result(url="https://example.com/broken", success=False, error="404"),
        ]
        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=mock_results)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        with patch.dict("sys.modules", self._setup_crawl4ai(mock_instance)):
            results = await crawl_recursive("https://example.com", max_depth=1, max_pages=10)
        assert len(results) == 2
        assert results[0].success
        assert not results[1].success

    async def test_exception_returns_error_result(self):
        mock_instance = AsyncMock()
        mock_instance.__aenter__ = AsyncMock(side_effect=RuntimeError("network error"))
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        with patch.dict("sys.modules", self._setup_crawl4ai(mock_instance)):
            results = await crawl_recursive("https://example.com", max_depth=1, max_pages=5)
        assert len(results) == 1
        assert not results[0].success

    async def test_defaults_to_unbounded(self):
        """With no max_depth / max_pages and no cfg ceiling, strategy gets math.inf."""
        import math

        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=[])
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        cfg.crawl_max_depth = None
        cfg.crawl_max_pages = None
        modules = self._setup_crawl4ai(mock_instance)
        bfs = modules["crawl4ai.deep_crawling"].BFSDeepCrawlStrategy
        with patch.dict("sys.modules", modules):
            await crawl_recursive("https://example.com")
        kwargs = bfs.call_args.kwargs
        assert kwargs["max_depth"] == math.inf
        assert kwargs["max_pages"] == math.inf

    async def test_explicit_cap_overrides_cfg_ceiling(self):
        """An explicit int wins even when cfg sets a lower ceiling."""
        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=[])
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        cfg.crawl_max_pages = 10
        modules = self._setup_crawl4ai(mock_instance)
        bfs = modules["crawl4ai.deep_crawling"].BFSDeepCrawlStrategy
        with patch.dict("sys.modules", modules):
            await crawl_recursive("https://example.com", max_depth=1, max_pages=999)
        assert bfs.call_args.kwargs["max_pages"] == 999

    async def test_cfg_ceiling_applied_when_none_passed(self):
        """cfg.crawl_max_pages acts as ceiling when caller passes None."""
        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=[])
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        cfg.crawl_max_pages = 10
        modules = self._setup_crawl4ai(mock_instance)
        bfs = modules["crawl4ai.deep_crawling"].BFSDeepCrawlStrategy
        with patch.dict("sys.modules", modules):
            await crawl_recursive("https://example.com", max_depth=1, max_pages=None)
        assert bfs.call_args.kwargs["max_pages"] == 10

    async def test_zero_max_pages_raises(self):
        """max_pages=0 is invalid (callers should pass None for unbounded)."""
        with pytest.raises(ValueError, match="positive"):
            await crawl_recursive("https://example.com", max_depth=1, max_pages=0)

    async def test_quiet_passes_verbose_false(self):
        """quiet=True passes verbose=False to AsyncWebCrawler."""
        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=[])
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_crawler_cls = MagicMock(return_value=mock_instance)
        modules = self._setup_crawl4ai(mock_instance)
        modules["crawl4ai"].AsyncWebCrawler = mock_crawler_cls

        with patch.dict("sys.modules", modules):
            await crawl_recursive("https://example.com", max_depth=1, quiet=True)
        mock_crawler_cls.assert_called_once_with(verbose=False)

    async def test_reraises_browser_missing_from_crawler_open(self, monkeypatch):
        """CrawlerBrowserMissing raised inside the try block propagates past the broad except."""
        from lilbee.crawler import CrawlerBrowserMissing

        mock_instance = AsyncMock()
        mock_instance.__aenter__ = AsyncMock(side_effect=CrawlerBrowserMissing("chromium gone"))
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr("lilbee.crawler.bootstrap.chromium_installed", lambda: True)
        with (
            patch.dict("sys.modules", self._setup_crawl4ai(mock_instance)),
            pytest.raises(CrawlerBrowserMissing, match="chromium gone"),
        ):
            await crawl_recursive("https://example.com", max_depth=1, max_pages=5)

    async def test_propagates_crawler_browser_missing(self, monkeypatch):
        """bb-wq8g: crawl_recursive re-raises CrawlerBrowserMissing past its broad except."""
        import sys as _sys

        from lilbee.crawler import CrawlerBrowserMissing

        # Stub crawl4ai + the deep_crawling submodule so the test runs even
        # when the `crawler` extra isn't installed — crawl_recursive imports
        # both at the top of its body before _open_crawler can fire.
        monkeypatch.setitem(_sys.modules, "crawl4ai", _mock_crawl4ai(MagicMock()))
        monkeypatch.setitem(
            _sys.modules, "crawl4ai.deep_crawling", MagicMock(BFSDeepCrawlStrategy=MagicMock())
        )
        monkeypatch.setitem(
            _sys.modules,
            "crawl4ai.deep_crawling.filters",
            MagicMock(FilterChain=MagicMock(), URLPatternFilter=MagicMock()),
        )
        monkeypatch.setattr("lilbee.crawler.bootstrap.chromium_installed", lambda: False)
        with pytest.raises(CrawlerBrowserMissing, match="Chromium"):
            await crawl_recursive("https://example.com", max_depth=1)


class _StubURLFilter:
    def __init__(self) -> None:
        self.stats = MagicMock()

    def _update_stats(self, passed: bool) -> None:
        pass

    def apply(self, url: str) -> bool:
        return True


class _StubDomainFilter(_StubURLFilter):
    def __init__(self, allowed_domains: str) -> None:
        super().__init__()
        self.allowed_domains = allowed_domains

    def apply(self, url: str) -> bool:
        from urllib.parse import urlparse

        host = (urlparse(url).hostname or "").lower()
        allowed = self.allowed_domains.lower()
        return host == allowed or host.endswith(f".{allowed}")


@pytest.fixture
def _stub_crawl4ai_filters(monkeypatch):
    """Make ``crawl4ai.deep_crawling.filters`` importable with minimal stand-ins.

    CI installs without the ``crawler`` extra, so ``_host_scope_filter``'s inline
    ``from crawl4ai.deep_crawling.filters import ...`` would raise ImportError.
    """
    stub = MagicMock(URLFilter=_StubURLFilter, DomainFilter=_StubDomainFilter)
    monkeypatch.setitem(sys.modules, "crawl4ai", MagicMock())
    monkeypatch.setitem(sys.modules, "crawl4ai.deep_crawling", MagicMock())
    monkeypatch.setitem(sys.modules, "crawl4ai.deep_crawling.filters", stub)


class TestHostScopeFilter:
    """whole-site crawl must scope to the exact host by default."""

    def test_exact_host_rejects_other_subdomains(self, _stub_crawl4ai_filters):
        from lilbee.crawler.crawl4ai_fetcher import _host_scope_filter

        f = _host_scope_filter("https://en.wikipedia.org/wiki/X", include_subdomains=False)
        assert f.apply("https://en.wikipedia.org/wiki/Y") is True
        assert f.apply("https://af.wikipedia.org/wiki/Y") is False
        assert f.apply("https://wikipedia.org/wiki/Y") is False

    def test_include_subdomains_allows_siblings(self, _stub_crawl4ai_filters):
        from lilbee.crawler.crawl4ai_fetcher import _host_scope_filter

        f = _host_scope_filter("https://en.wikipedia.org/wiki/X", include_subdomains=True)
        assert f.apply("https://en.wikipedia.org/wiki/Y") is True
        assert f.apply("https://other.example.com/") is False

    def test_returns_none_when_host_missing(self, _stub_crawl4ai_filters):
        from lilbee.crawler.crawl4ai_fetcher import _host_scope_filter

        assert _host_scope_filter("not-a-url", include_subdomains=False) is None


class TestSitemapCounting:
    """best-effort sitemap lookup bounds the crawl progress total."""

    def test_returns_unknown_on_http_error(self, monkeypatch):
        import httpx

        from lilbee.crawler.sitemap import _count_sitemap_urls
        from lilbee.progress import CRAWL_TOTAL_UNKNOWN

        def _raise(*a, **kw):
            raise httpx.ConnectError("boom")

        monkeypatch.setattr("httpx.get", _raise)
        assert (
            _count_sitemap_urls("https://example.com/start", include_subdomains=False)
            == CRAWL_TOTAL_UNKNOWN
        )

    def test_returns_unknown_on_4xx(self, monkeypatch):
        from lilbee.crawler.sitemap import _count_sitemap_urls
        from lilbee.progress import CRAWL_TOTAL_UNKNOWN

        fake = MagicMock(status_code=404, text="")
        monkeypatch.setattr("httpx.get", lambda *a, **kw: fake)
        assert (
            _count_sitemap_urls("https://example.com/start", include_subdomains=False)
            == CRAWL_TOTAL_UNKNOWN
        )

    def test_counts_matching_urls_only(self, monkeypatch):
        from lilbee.crawler.sitemap import _count_sitemap_urls

        body = (
            "<urlset>"
            "<url><loc>https://example.com/a</loc></url>"
            "<url><loc>https://example.com/b</loc></url>"
            "<url><loc>https://other.com/c</loc></url>"
            "<url><loc>https://sub.example.com/d</loc></url>"
            "</urlset>"
        )
        fake = MagicMock(status_code=200, text=body)
        monkeypatch.setattr("httpx.get", lambda *a, **kw: fake)
        count = _count_sitemap_urls("https://example.com/start", include_subdomains=False)
        assert count == 2

    def test_include_subdomains_counts_children(self, monkeypatch):
        from lilbee.crawler.sitemap import _count_sitemap_urls

        body = (
            "<urlset>"
            "<url><loc>https://example.com/a</loc></url>"
            "<url><loc>https://sub.example.com/d</loc></url>"
            "<url><loc>https://other.com/c</loc></url>"
            "</urlset>"
        )
        fake = MagicMock(status_code=200, text=body)
        monkeypatch.setattr("httpx.get", lambda *a, **kw: fake)
        count = _count_sitemap_urls("https://example.com/start", include_subdomains=True)
        assert count == 2

    def test_returns_unknown_when_start_url_has_no_host(self):
        """A malformed start URL short-circuits before hitting the network."""
        from lilbee.crawler.sitemap import _count_sitemap_urls
        from lilbee.progress import CRAWL_TOTAL_UNKNOWN

        # file:///foo has no hostname, so the helper bails immediately.
        assert (
            _count_sitemap_urls("file:///not-a-real-host", include_subdomains=False)
            == CRAWL_TOTAL_UNKNOWN
        )

    def test_skips_entries_with_no_host(self, monkeypatch):
        """Sitemap entries whose loc has no hostname are skipped."""
        from lilbee.crawler.sitemap import _count_sitemap_urls

        body = (
            "<urlset>"
            "<url><loc>/relative/path</loc></url>"
            "<url><loc>https://example.com/a</loc></url>"
            "</urlset>"
        )
        fake = MagicMock(status_code=200, text=body)
        monkeypatch.setattr("httpx.get", lambda *a, **kw: fake)
        count = _count_sitemap_urls("https://example.com/start", include_subdomains=False)
        assert count == 1

    def test_caps_at_max_urls(self, monkeypatch):
        """A giant sitemap stops at _SITEMAP_MAX_URLS so the scan is bounded."""
        from lilbee.crawler import sitemap as sitemap_mod
        from lilbee.crawler.sitemap import _count_sitemap_urls

        monkeypatch.setattr(sitemap_mod, "_SITEMAP_MAX_URLS", 3)
        entries = "".join(f"<url><loc>https://example.com/{i}</loc></url>" for i in range(10))
        fake = MagicMock(status_code=200, text=f"<urlset>{entries}</urlset>")
        monkeypatch.setattr("httpx.get", lambda *a, **kw: fake)
        count = _count_sitemap_urls("https://example.com/start", include_subdomains=False)
        assert count == 3


class TestCrawlAndSave:
    @patch("lilbee.crawler.api.crawl_single")
    async def test_single_page(self, mock_crawl_single, isolated_env):
        mock_crawl_single.return_value = CrawlResult(url="https://example.com", markdown="# Hello")
        paths = await crawl_and_save("https://example.com", depth=0)
        assert len(paths) == 1
        assert paths[0].exists()

    @patch("lilbee.crawler.api.crawl_single")
    async def test_triggers_bootstrap_when_chromium_missing(
        self, mock_crawl_single, isolated_env, monkeypatch
    ):
        """bb-wq8g: crawl_and_save kicks off bootstrap_chromium on first use."""
        mock_crawl_single.return_value = CrawlResult(url="https://example.com", markdown="# Hi")
        monkeypatch.setattr("lilbee.crawler.bootstrap.chromium_installed", lambda: False)
        called: list[object] = []

        async def _fake_bootstrap(on_progress=None):
            called.append(on_progress)

        monkeypatch.setattr("lilbee.crawler.bootstrap.bootstrap_chromium", _fake_bootstrap)
        await crawl_and_save("https://example.com", depth=0)
        assert called == [None]

    @patch("lilbee.crawler.api.crawl_recursive")
    async def test_recursive(self, mock_crawl_recursive, isolated_env):
        streamed = [
            CrawlResult(url="https://example.com", markdown="# Home"),
            CrawlResult(url="https://example.com/about", markdown="# About"),
        ]

        async def _fake_recursive(*args, on_result=None, **kwargs):
            # Mimic the real streaming contract: invoke the flush callback
            # per-page before returning the full list.
            if on_result is not None:
                for r in streamed:
                    on_result(r)
            return streamed

        mock_crawl_recursive.side_effect = _fake_recursive
        paths = await crawl_and_save("https://example.com", depth=2, max_pages=10)
        assert len(paths) == 2

    @patch("lilbee.crawler.api.crawl_recursive")
    async def test_default_depth_is_recursive(self, mock_crawl_recursive, isolated_env):
        """No depth kwarg means recursive (whole-site) crawl, not single page."""
        mock_crawl_recursive.return_value = [
            CrawlResult(url="https://example.com", markdown="# Home"),
        ]
        await crawl_and_save("https://example.com")
        mock_crawl_recursive.assert_awaited_once()
        assert mock_crawl_recursive.await_args.kwargs["max_depth"] is None

    @patch("lilbee.crawler.api.crawl_single")
    async def test_quiet_forwarded_to_crawl_single(self, mock_crawl_single, isolated_env):
        """quiet=True is forwarded to crawl_single (depth=0 path)."""
        mock_crawl_single.return_value = CrawlResult(url="https://example.com", markdown="# Hi")
        await crawl_and_save("https://example.com", depth=0, quiet=True)
        mock_crawl_single.assert_awaited_once_with("https://example.com", quiet=True)

    @patch("lilbee.crawler.api.crawl_recursive")
    async def test_quiet_forwarded_to_crawl_recursive(self, mock_crawl_recursive, isolated_env):
        """quiet=True is forwarded to crawl_recursive."""
        mock_crawl_recursive.return_value = []
        await crawl_and_save("https://example.com", depth=2, quiet=True)
        call_kwargs = mock_crawl_recursive.call_args[1]
        assert call_kwargs["quiet"] is True

    @patch("lilbee.crawler.api.crawl_recursive")
    async def test_include_subdomains_defaults_false(self, mock_crawl_recursive, isolated_env):
        """whole-site default is exact-host scoping."""
        mock_crawl_recursive.return_value = []
        await crawl_and_save("https://example.com", depth=2)
        assert mock_crawl_recursive.await_args.kwargs["include_subdomains"] is False

    @patch("lilbee.crawler.api.crawl_recursive")
    async def test_include_subdomains_threaded_through(self, mock_crawl_recursive, isolated_env):
        """Opting into subdomain scope reaches the recursive crawl."""
        mock_crawl_recursive.return_value = []
        await crawl_and_save("https://example.com", depth=2, include_subdomains=True)
        assert mock_crawl_recursive.await_args.kwargs["include_subdomains"] is True

    @patch("lilbee.crawler.api.crawl_recursive")
    async def test_cancel_threaded_to_crawl_recursive(self, mock_crawl_recursive, isolated_env):
        """The cancel event is threaded through to crawl_recursive for BFS abort."""
        import threading

        mock_crawl_recursive.return_value = []
        cancel = threading.Event()
        await crawl_and_save("https://example.com", depth=2, cancel=cancel)
        assert mock_crawl_recursive.call_args[1]["cancel"] is cancel

    @patch("lilbee.crawler.api.crawl_single")
    async def test_single_page_with_progress(self, mock_crawl_single, isolated_env):
        """Progress callback receives crawl_start, crawl_page, crawl_done for single page."""
        mock_crawl_single.return_value = CrawlResult(url="https://example.com", markdown="# Hi")
        events = []

        def on_progress(event_type, data):
            events.append((str(event_type), data))

        await crawl_and_save("https://example.com", depth=0, on_progress=on_progress)
        event_types = [e[0] for e in events]
        assert "crawl_start" in event_types
        assert "crawl_page" in event_types
        assert "crawl_done" in event_types

    @patch("lilbee.crawler.api.crawl_single")
    async def test_updates_metadata(self, mock_crawl_single, isolated_env):
        mock_crawl_single.return_value = CrawlResult(
            url="https://example.com/page", markdown="# Test"
        )
        await crawl_and_save("https://example.com/page", depth=0)
        meta = load_crawl_metadata()
        assert "https://example.com/page" in meta

    @patch("lilbee.crawler.api.crawl_single")
    async def test_unchanged_content_skips_save(self, mock_crawl_single, isolated_env):
        """Same content on re-crawl skips saving (hash-based detection)."""
        mock_crawl_single.return_value = CrawlResult(
            url="https://example.com/dup", markdown="# Dup"
        )
        # First crawl saves the file
        paths1 = await crawl_and_save("https://example.com/dup", depth=0)
        assert len(paths1) == 1
        mock_crawl_single.reset_mock()

        # Second crawl with identical content: fetches but skips save
        mock_crawl_single.return_value = CrawlResult(
            url="https://example.com/dup", markdown="# Dup"
        )
        paths2 = await crawl_and_save("https://example.com/dup", depth=0)
        assert paths2 == []
        mock_crawl_single.assert_awaited_once()

    @patch("lilbee.crawler.api.crawl_single")
    async def test_changed_content_updates_file(self, mock_crawl_single, isolated_env):
        """Changed content on re-crawl saves updated file."""
        mock_crawl_single.return_value = CrawlResult(
            url="https://example.com/dup", markdown="# Dup"
        )
        await crawl_and_save("https://example.com/dup", depth=0)
        mock_crawl_single.reset_mock()
        mock_crawl_single.return_value = CrawlResult(
            url="https://example.com/dup", markdown="# Updated"
        )

        paths = await crawl_and_save("https://example.com/dup", depth=0)
        assert len(paths) == 1
        mock_crawl_single.assert_awaited_once()

    async def test_semaphore_limits_concurrency(self, isolated_env):
        """The semaphore limits concurrent crawls based on config."""
        from lilbee.crawler import api as crawler_mod

        crawler_mod._state.semaphore = None
        cfg.crawl_max_concurrent = 5
        sem = _get_crawl_semaphore()
        assert sem is not None
        assert sem._value == 5
        crawler_mod._state.semaphore = None

    async def test_semaphore_defaults_to_cpu_count(self, isolated_env):
        """Default concurrency matches CPU count."""
        import os

        from lilbee.crawler import api as crawler_mod

        crawler_mod._state.semaphore = None
        cfg.crawl_max_concurrent = os.cpu_count() or 4
        sem = _get_crawl_semaphore()
        assert sem is not None
        assert sem._value == (os.cpu_count() or 4)
        crawler_mod._state.semaphore = None

    async def test_semaphore_unlimited_when_zero(self, isolated_env):
        """Setting crawl_max_concurrent=0 disables the semaphore."""
        from lilbee.crawler import api as crawler_mod

        crawler_mod._state.semaphore = None
        cfg.crawl_max_concurrent = 0
        assert _get_crawl_semaphore() is None
        crawler_mod._state.semaphore = None

    @patch("lilbee.crawler.api.crawl_single")
    async def test_cancel_keeps_fetched_page(self, mock_crawl_single, isolated_env):
        """A single-URL crawl that gets cancelled still keeps the page it fetched.

        The new streaming-flush contract: anything already on disk stays on
        disk. For depth=0, crawl_single has already run by the time cancel is
        observed, so the page is flushed and returned.
        """
        import threading

        mock_crawl_single.return_value = CrawlResult(url="https://example.com", markdown="# Hello")
        cancel = threading.Event()
        cancel.set()
        paths = await crawl_and_save("https://example.com", depth=0, cancel=cancel)
        assert len(paths) == 1
        assert paths[0].exists()


class TestPeriodicSync:
    async def test_sync_disabled_when_interval_zero(self, isolated_env):
        """No sync fires when crawl_sync_interval is 0."""
        import threading

        from lilbee.crawler import api as crawler_mod

        cfg.crawl_sync_interval = 0
        crawler_mod._state.last_sync_time = 0.0
        crawler_mod._state.sync_running = threading.Lock()

        with patch("lilbee.ingest.sync", new_callable=AsyncMock) as mock_sync:
            await _maybe_periodic_sync()
            mock_sync.assert_not_awaited()

    async def test_sync_skipped_when_already_running(self, isolated_env):
        """No new sync is started if one is already in progress."""
        import threading

        from lilbee.crawler import api as crawler_mod

        cfg.crawl_sync_interval = 1
        crawler_mod._state.last_sync_time = 0.0
        lock = threading.Lock()
        lock.acquire()  # simulate already-running
        crawler_mod._state.sync_running = lock

        with patch("lilbee.ingest.sync", new_callable=AsyncMock) as mock_sync:
            await _maybe_periodic_sync()
            mock_sync.assert_not_awaited()

        lock.release()

    async def test_sync_skipped_when_interval_not_elapsed(self, isolated_env):
        """No sync fires if the interval hasn't elapsed since last sync."""
        import threading
        import time

        from lilbee.crawler import api as crawler_mod

        cfg.crawl_sync_interval = 9999
        crawler_mod._state.last_sync_time = time.monotonic()
        crawler_mod._state.sync_running = threading.Lock()

        with patch("lilbee.ingest.sync", new_callable=AsyncMock) as mock_sync:
            await _maybe_periodic_sync()
            mock_sync.assert_not_awaited()

    async def test_sync_fires_when_interval_elapsed(self, isolated_env):
        """Sync fires as a background task when interval has elapsed."""
        import asyncio
        import threading

        from lilbee.crawler import api as crawler_mod

        cfg.crawl_sync_interval = 1
        crawler_mod._state.last_sync_time = 0.0
        crawler_mod._state.sync_running = threading.Lock()

        mock_sync = AsyncMock()
        with patch("lilbee.ingest.sync", mock_sync):
            await _maybe_periodic_sync()
            # Let the background task run
            await asyncio.sleep(0)
            mock_sync.assert_awaited_once()

    async def test_sync_failure_resets_running_flag(self, isolated_env):
        """If sync raises, _sync_running lock is released so future syncs can proceed."""
        import asyncio
        import threading

        from lilbee.crawler import api as crawler_mod

        cfg.crawl_sync_interval = 1
        crawler_mod._state.last_sync_time = 0.0
        lock = threading.Lock()
        crawler_mod._state.sync_running = lock

        mock_sync = AsyncMock(side_effect=RuntimeError("sync failed"))
        with patch("lilbee.ingest.sync", mock_sync):
            await _maybe_periodic_sync()
            await asyncio.sleep(0)

        # Lock should be released after failure
        assert lock.acquire(blocking=False)
        lock.release()


class TestCrawlerStateReset:
    def test_reset_clears_all_state(self, isolated_env):
        """CrawlerState.reset() restores all fields to initial values."""
        from lilbee.crawler import api as crawler_mod

        state = crawler_mod._state
        state.semaphore = asyncio.Semaphore(3)
        state.semaphore_limit = 3
        state.last_sync_time = 99.0

        state.reset()

        assert state.semaphore is None
        assert state.semaphore_limit == 0
        assert state.last_sync_time == 0.0
        assert state.sync_running.acquire(blocking=False)
        state.sync_running.release()
        assert state.background_tasks == set()


class TestCrawlAndSaveSemaphore:
    @patch("lilbee.crawler.api.crawl_single")
    async def test_semaphore_acquired_and_released(self, mock_crawl_single, isolated_env):
        """When crawl_max_concurrent > 0, sem.acquire/release are called."""
        from lilbee.crawler import api as crawler_mod

        mock_crawl_single.return_value = CrawlResult(url="https://example.com", markdown="# Hello")
        cfg.crawl_max_concurrent = 2
        crawler_mod._state.semaphore = None

        paths = await crawl_and_save("https://example.com", depth=0)
        assert len(paths) == 1

        # Verify semaphore was created and is still available (released)
        sem = crawler_mod._state.semaphore
        assert sem is not None
        assert sem._value == 2
        crawler_mod._state.semaphore = None


class TestCrawlCancel:
    """Cancel path: the three stitches that were broken on the first pass."""

    def _setup_crawl4ai(self, mock_instance):
        mock_crawler_cls = MagicMock(return_value=mock_instance)
        mock_mod = _mock_crawl4ai(mock_crawler_cls)
        mock_bfs_cls = MagicMock()
        mock_deep = MagicMock()
        mock_deep.BFSDeepCrawlStrategy = mock_bfs_cls
        mock_dispatcher_mod = MagicMock()
        mock_dispatcher_mod.RateLimiter = MagicMock()
        mock_dispatcher_mod.SemaphoreDispatcher = MagicMock()
        mock_filters_mod = MagicMock()
        mock_filters_mod.FilterChain = MagicMock()
        mock_filters_mod.URLPatternFilter = MagicMock()
        return {
            "crawl4ai": mock_mod,
            "crawl4ai.deep_crawling": mock_deep,
            "crawl4ai.deep_crawling.filters": mock_filters_mod,
            "crawl4ai.async_dispatcher": mock_dispatcher_mod,
        }, mock_bfs_cls

    async def test_strategy_should_cancel_wired(self):
        """crawl_recursive passes should_cancel= to BFSDeepCrawlStrategy."""
        import threading as _threading

        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=[])
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        modules, bfs_cls = self._setup_crawl4ai(mock_instance)
        evt = _threading.Event()
        with patch.dict("sys.modules", modules):
            await crawl_recursive("https://example.com", max_depth=1, cancel=evt)

        kwargs = bfs_cls.call_args.kwargs
        assert "should_cancel" in kwargs
        cb = kwargs["should_cancel"]
        assert cb() is False
        evt.set()
        assert cb() is True

    async def test_strategy_cancel_called_on_event(self):
        """When cancel fires mid-stream, strategy.cancel() is invoked."""
        import asyncio as _asyncio
        import threading as _threading

        cancel = _threading.Event()

        async def _gen():
            for i in range(1, 5):
                await _asyncio.sleep(0)
                yield _make_crawl4ai_result(url=f"https://example.com/p{i}")
                if i == 1:
                    cancel.set()

        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=_gen())
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        modules, bfs_cls = self._setup_crawl4ai(mock_instance)
        strategy_instance = MagicMock()
        strategy_instance.cancel = MagicMock()
        bfs_cls.return_value = strategy_instance

        with patch.dict("sys.modules", modules):
            await crawl_recursive("https://example.com", max_depth=1, max_pages=10, cancel=cancel)

        strategy_instance.cancel.assert_called_once()

    async def test_stream_aclose_called_on_async_gen(self):
        """The async-generator stream is aclose()'d before the crawler context exits."""
        import threading as _threading

        aclose_called = []

        async def _gen():
            try:
                for i in range(1, 4):
                    yield _make_crawl4ai_result(url=f"https://example.com/p{i}")
            finally:
                aclose_called.append(True)

        gen = _gen()
        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=gen)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        modules, _ = self._setup_crawl4ai(mock_instance)
        cancel = _threading.Event()
        cancel.set()  # cancel immediately so the loop breaks after first result
        with patch.dict("sys.modules", modules):
            await crawl_recursive("https://example.com", max_depth=1, max_pages=10, cancel=cancel)
        # The generator's finally ran, proving aclose completed
        assert aclose_called == [True]

    async def test_stream_aclose_noop_for_list(self):
        """List-mode arun return (batch shape) doesn't trigger aclose."""
        mock_results = [_make_crawl4ai_result(url="https://example.com", markdown="# H")]
        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=mock_results)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        modules, _ = self._setup_crawl4ai(mock_instance)
        with patch.dict("sys.modules", modules):
            results = await crawl_recursive("https://example.com", max_depth=1, max_pages=5)
        assert len(results) == 1

    async def test_post_cancel_teardown_errors_logged_at_debug(self, caplog):
        """After cancel, BrowserContext-closed errors log at DEBUG, not WARNING."""
        import logging as _logging
        import threading as _threading

        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(side_effect=RuntimeError("BrowserContext.new_page: boom"))
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        modules, _ = self._setup_crawl4ai(mock_instance)
        cancel = _threading.Event()
        cancel.set()  # cancel fired before the exception happens
        caplog.set_level(_logging.DEBUG, logger="lilbee.crawler")
        with patch.dict("sys.modules", modules):
            results = await crawl_recursive("https://example.com", cancel=cancel)

        # No synthetic failure entry on the cancel path
        assert results == []
        # The teardown error is at DEBUG, not WARNING
        warnings = [r for r in caplog.records if r.levelno == _logging.WARNING]
        assert not warnings

    async def test_safe_strategy_cancel_missing_method(self):
        """_safe_strategy_cancel tolerates strategies without a cancel() method."""
        from lilbee.crawler.crawl4ai_fetcher import _safe_strategy_cancel

        _safe_strategy_cancel(object())  # object() has no cancel; must not raise

    async def test_safe_strategy_cancel_swallows_runtime_error(self, caplog):
        """_safe_strategy_cancel logs at debug when cancel() raises RuntimeError.

        Mirrors the real shape of ``BFSDeepCrawlStrategy.cancel()`` failing after
        the strategy's internal state was already torn down.
        """
        import logging as _logging

        from lilbee.crawler.crawl4ai_fetcher import _safe_strategy_cancel

        class _Strategy:
            def cancel(self) -> None:
                raise RuntimeError("strategy already closed")

        with caplog.at_level(_logging.DEBUG, logger="lilbee.crawler.crawl4ai_fetcher"):
            _safe_strategy_cancel(_Strategy())
        assert any("strategy.cancel() raised" in r.getMessage() for r in caplog.records)

    async def test_safe_aclose_noop_on_none(self):
        """_safe_aclose returns cleanly when stream is None (e.g. crawler never opened)."""
        from lilbee.crawler.crawl4ai_fetcher import _safe_aclose

        await _safe_aclose(None)  # must not raise

    async def test_hard_cap_on_visible_counter(self):
        """counter never exceeds the resolved max_pages, even if crawl4ai yields more.

        crawl4ai's BFS only counts successful pages toward max_pages, so failed
        or redirected pages can push our per-result counter past the cap. We
        break the loop explicitly when counter hits the cap.
        """
        import asyncio as _asyncio

        async def _gen():
            # yield 10 results even though cap will be 3
            for i in range(1, 11):
                await _asyncio.sleep(0)
                yield _make_crawl4ai_result(url=f"https://example.com/p{i}")

        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=_gen())
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        modules, bfs_cls = self._setup_crawl4ai(mock_instance)
        strategy_instance = MagicMock()
        strategy_instance.cancel = MagicMock()
        bfs_cls.return_value = strategy_instance

        events = []

        def on_progress(event_type, data):
            if event_type == EventType.CRAWL_PAGE:
                events.append(data.current)

        with patch.dict("sys.modules", modules):
            results = await crawl_recursive(
                "https://example.com", max_depth=1, max_pages=3, on_progress=on_progress
            )

        # User-visible counter stops at 3; no event announces current=4.
        assert events == [1, 2, 3]
        assert len(results) == 3
        # Strategy was asked to stop once the cap hit.
        strategy_instance.cancel.assert_called_once()


class TestCrawlDispatcher:
    """Rate-limit dispatcher wiring for the recursive path."""

    def _setup_crawl4ai(self, mock_instance):
        mock_crawler_cls = MagicMock(return_value=mock_instance)
        mock_mod = _mock_crawl4ai(mock_crawler_cls)
        mock_deep = MagicMock()
        mock_deep.BFSDeepCrawlStrategy = MagicMock()
        mock_dispatcher_mod = MagicMock()
        mock_rl = MagicMock()
        mock_sd = MagicMock()
        mock_dispatcher_mod.RateLimiter = mock_rl
        mock_dispatcher_mod.SemaphoreDispatcher = mock_sd
        mock_filters_mod = MagicMock()
        mock_filters_mod.FilterChain = MagicMock()
        mock_filters_mod.URLPatternFilter = MagicMock()
        return (
            {
                "crawl4ai": mock_mod,
                "crawl4ai.deep_crawling": mock_deep,
                "crawl4ai.deep_crawling.filters": mock_filters_mod,
                "crawl4ai.async_dispatcher": mock_dispatcher_mod,
            },
            mock_rl,
            mock_sd,
        )

    async def test_uniform_knobs_on_crawler_run_config(self):
        """mean_delay / max_range / semaphore_count come from cfg."""
        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=[])
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        modules, _, _ = self._setup_crawl4ai(mock_instance)
        crc = modules["crawl4ai"].CrawlerRunConfig

        cfg.crawl_mean_delay = 2.0
        cfg.crawl_max_delay_range = 1.0
        cfg.crawl_concurrent_requests = 7
        with patch.dict("sys.modules", modules):
            await crawl_recursive("https://example.com", max_depth=1, max_pages=5)

        kwargs = crc.call_args.kwargs
        assert kwargs["mean_delay"] == 2.0
        assert kwargs["max_range"] == 1.0
        assert kwargs["semaphore_count"] == 7

    async def test_rate_limiter_built_when_flag_on(self):
        """crawl_retry_on_rate_limit=True instantiates RateLimiter + SemaphoreDispatcher."""
        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=[])
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        modules, mock_rl, mock_sd = self._setup_crawl4ai(mock_instance)

        cfg.crawl_retry_on_rate_limit = True
        cfg.crawl_retry_base_delay_min = 1.0
        cfg.crawl_retry_base_delay_max = 3.0
        cfg.crawl_retry_max_backoff = 30.0
        cfg.crawl_retry_max_attempts = 3
        cfg.crawl_concurrent_requests = 3
        with patch.dict("sys.modules", modules):
            await crawl_recursive("https://example.com", max_depth=1, max_pages=5)

        rl_kwargs = mock_rl.call_args.kwargs
        assert rl_kwargs["base_delay"] == (1.0, 3.0)
        assert rl_kwargs["max_delay"] == 30.0
        assert rl_kwargs["max_retries"] == 3
        sd_kwargs = mock_sd.call_args.kwargs
        assert sd_kwargs["semaphore_count"] == 3
        assert sd_kwargs["rate_limiter"] is mock_rl.return_value

    async def test_rate_limiter_disabled_when_flag_off(self):
        """crawl_retry_on_rate_limit=False skips the dispatcher entirely."""
        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=[])
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        modules, mock_rl, mock_sd = self._setup_crawl4ai(mock_instance)

        cfg.crawl_retry_on_rate_limit = False
        with patch.dict("sys.modules", modules):
            await crawl_recursive("https://example.com", max_depth=1, max_pages=5)

        mock_rl.assert_not_called()
        mock_sd.assert_not_called()

    async def test_lilbee_async_crawler_forwards_default_dispatcher(self):
        """_LilbeeAsyncCrawler threads its default dispatcher into arun_many."""
        from lilbee.crawler.crawl4ai_fetcher import _LilbeeAsyncCrawler

        inner = MagicMock()
        inner.arun_many = AsyncMock()
        mock_awc = MagicMock(return_value=inner)
        with patch.dict("sys.modules", {"crawl4ai": MagicMock(AsyncWebCrawler=mock_awc)}):
            crawler = _LilbeeAsyncCrawler(verbose=False, dispatcher="DEFAULT")
            await crawler.arun_many(["u"], config="C")
        inner.arun_many.assert_awaited_once_with(["u"], config="C", dispatcher="DEFAULT")

    async def test_exclude_patterns_build_filter_chain(self):
        """cfg.crawl_exclude_patterns feeds URLPatternFilter into BFS strategy."""
        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=[])
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        modules, _, _ = self._setup_crawl4ai(mock_instance)
        bfs_cls = modules["crawl4ai.deep_crawling"].BFSDeepCrawlStrategy
        url_pattern_cls = modules["crawl4ai.deep_crawling.filters"].URLPatternFilter
        filter_chain_cls = modules["crawl4ai.deep_crawling.filters"].FilterChain

        cfg.crawl_exclude_patterns = ["/page/\\d+", "/tag/"]
        with patch.dict("sys.modules", modules):
            await crawl_recursive("https://example.com", max_depth=1, max_pages=5)

        # URLPatternFilter gets the patterns with reverse=True, use_glob=False
        url_pattern_cls.assert_called_once()
        call = url_pattern_cls.call_args
        assert call.args[0] == ["/page/\\d+", "/tag/"]
        assert call.kwargs.get("reverse") is True
        assert call.kwargs.get("use_glob") is False
        # FilterChain gets a list containing the pattern filter
        filter_chain_cls.assert_called_once()
        # BFS strategy receives the filter_chain
        assert "filter_chain" in bfs_cls.call_args.kwargs

    async def test_empty_exclude_patterns_uses_empty_filter_chain(self):
        """cfg.crawl_exclude_patterns=[] means no URLPatternFilter is constructed."""
        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=[])
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        modules, _, _ = self._setup_crawl4ai(mock_instance)
        url_pattern_cls = modules["crawl4ai.deep_crawling.filters"].URLPatternFilter

        cfg.crawl_exclude_patterns = []
        with patch.dict("sys.modules", modules):
            await crawl_recursive("https://example.com", max_depth=1, max_pages=5)
        url_pattern_cls.assert_not_called()

    async def test_lilbee_async_crawler_explicit_dispatcher_wins(self):
        """An explicit dispatcher= on arun_many beats the default."""
        from lilbee.crawler.crawl4ai_fetcher import _LilbeeAsyncCrawler

        inner = MagicMock()
        inner.arun_many = AsyncMock()
        mock_awc = MagicMock(return_value=inner)
        with patch.dict("sys.modules", {"crawl4ai": MagicMock(AsyncWebCrawler=mock_awc)}):
            crawler = _LilbeeAsyncCrawler(verbose=False, dispatcher="DEFAULT")
            await crawler.arun_many(["u"], dispatcher="EXPLICIT")
        inner.arun_many.assert_awaited_once_with(["u"], config=None, dispatcher="EXPLICIT")


class TestSaveSingleResult:
    """Unit tests for _save_single_result (per-page flush helper)."""

    def test_writes_new_content(self, isolated_env):
        from lilbee.crawler.save import _save_single_result

        meta: dict = {}
        result = CrawlResult(url="https://example.com/new", markdown="# New")
        outcome = _save_single_result(result, meta)
        assert outcome is not None
        assert outcome.path.exists()
        assert outcome.path.read_text(encoding="utf-8") == "# New"
        assert outcome.filename.endswith("index.md")
        assert outcome.content_hash == content_hash("# New")

    def test_returns_none_on_failure(self, isolated_env):
        from lilbee.crawler.save import _save_single_result

        result = CrawlResult(url="https://example.com/fail", success=False, error="oops")
        assert _save_single_result(result, {}) is None

    def test_returns_none_on_empty_markdown(self, isolated_env):
        from lilbee.crawler.save import _save_single_result

        result = CrawlResult(url="https://example.com/empty", markdown="   ")
        assert _save_single_result(result, {}) is None

    def test_hash_match_with_file_present_skips(self, isolated_env):
        """Prev metadata hash matches AND file exists -> skip."""
        from lilbee.crawler.save import _save_single_result

        url = "https://example.com/dup"
        markdown = "# Dup"
        # Write the file and seed metadata
        initial = CrawlResult(url=url, markdown=markdown)
        meta: dict = {}
        first = _save_single_result(initial, meta)
        assert first is not None
        meta[url] = CrawlMeta(
            file=str(first.path.relative_to(cfg.documents_dir / "_web")),
            content_hash=content_hash(markdown),
            crawled_at="2026-01-01T00:00:00+00:00",
        )
        # Identical content on re-save returns None
        again = CrawlResult(url=url, markdown=markdown)
        assert _save_single_result(again, meta) is None

    def test_hash_match_with_missing_file_rewrites(self, isolated_env):
        """Prev metadata hash matches but file was deleted -> re-write."""
        from lilbee.crawler.save import _save_single_result

        url = "https://example.com/gone"
        markdown = "# Gone"
        meta: dict = {
            url: CrawlMeta(
                file="example.com/gone/index.md",
                content_hash=content_hash(markdown),
                crawled_at="2026-01-01T00:00:00+00:00",
            )
        }
        # File does not exist yet
        result = CrawlResult(url=url, markdown=markdown)
        outcome = _save_single_result(result, meta)
        assert outcome is not None
        assert outcome.path.exists()

    def test_path_traversal_blocked(self, isolated_env):
        """A crafted filename escaping _web/ is skipped with a warning."""
        from lilbee.crawler.save import _save_single_result

        result = CrawlResult(url="https://evil.com/ok", markdown="# Evil")
        with patch("lilbee.crawler.save.url_to_filename", return_value="../../etc/passwd"):
            outcome = _save_single_result(result, {})
        assert outcome is None


class TestUpdateSingleMetadata:
    def test_updates_in_place(self, isolated_env):
        """Helper mutates the dict in place with the expected shape."""
        from lilbee.crawler.save import _save_single_result, _update_single_metadata

        meta: dict = {}
        result = CrawlResult(url="https://example.com/p", markdown="# P")
        outcome = _save_single_result(result, meta)
        assert outcome is not None
        now = "2026-04-20T00:00:00+00:00"
        _update_single_metadata(meta, result.url, outcome, now)
        assert "https://example.com/p" in meta
        entry = meta["https://example.com/p"]
        assert entry.crawled_at == now
        assert entry.content_hash == content_hash("# P")
        assert entry.file == "example.com/p/index.md"


class TestStreamingFlush:
    """End-to-end tests for the per-page flush contract in crawl_and_save."""

    def _setup_crawl4ai(self, mock_instance):
        mock_crawler_cls = MagicMock(return_value=mock_instance)
        mock_mod = _mock_crawl4ai(mock_crawler_cls)
        mock_deep = MagicMock()
        mock_deep.BFSDeepCrawlStrategy = MagicMock()
        mock_dispatcher_mod = MagicMock()
        mock_dispatcher_mod.RateLimiter = MagicMock()
        mock_dispatcher_mod.SemaphoreDispatcher = MagicMock()
        mock_filters_mod = MagicMock()
        mock_filters_mod.FilterChain = MagicMock()
        mock_filters_mod.URLPatternFilter = MagicMock()
        return {
            "crawl4ai": mock_mod,
            "crawl4ai.deep_crawling": mock_deep,
            "crawl4ai.deep_crawling.filters": mock_filters_mod,
            "crawl4ai.async_dispatcher": mock_dispatcher_mod,
        }

    async def test_cancel_preserves_written_pages(self, isolated_env):
        """A cancelled recursive crawl keeps the pages it already streamed."""
        import asyncio as _asyncio
        import threading

        cancel = threading.Event()

        async def _gen():
            yield _make_crawl4ai_result(url="https://example.com/p1", markdown="# P1")
            await _asyncio.sleep(0)
            yield _make_crawl4ai_result(url="https://example.com/p2", markdown="# P2")
            cancel.set()
            await _asyncio.sleep(0)
            yield _make_crawl4ai_result(url="https://example.com/p3", markdown="# P3")
            yield _make_crawl4ai_result(url="https://example.com/p4", markdown="# P4")

        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=_gen())
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        with patch.dict("sys.modules", self._setup_crawl4ai(mock_instance)):
            paths = await crawl_and_save(
                "https://example.com", depth=2, max_pages=10, cancel=cancel
            )

        # At least p1 and p2 were flushed before cancel stopped the stream.
        assert len(paths) >= 2
        for p in paths:
            assert p.exists()
        # Metadata reflects what's on disk.
        meta = load_crawl_metadata()
        written_urls = {
            f"https://example.com/p{i}"
            for i in range(1, 5)
            if (cfg.documents_dir / "_web" / f"example.com/p{i}/index.md").exists()
        }
        for url in written_urls:
            assert url in meta

    async def test_flushes_per_page_not_batched(self, isolated_env):
        """_save_single_result is invoked once per streamed page inside the loop."""
        import asyncio as _asyncio

        call_order: list[str] = []

        async def _gen():
            for i in range(1, 4):
                call_order.append(f"yield-{i}")
                await _asyncio.sleep(0)
                yield _make_crawl4ai_result(url=f"https://example.com/p{i}", markdown=f"# P{i}")

        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=_gen())
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        real_save = _save_single_result

        def _wrapped(result, meta):
            call_order.append(f"flush-{result.url.rsplit('/', 1)[-1]}")
            return real_save(result, meta)

        with (
            patch.dict("sys.modules", self._setup_crawl4ai(mock_instance)),
            patch("lilbee.crawler.save._save_single_result", side_effect=_wrapped),
        ):
            await crawl_and_save("https://example.com", depth=2, max_pages=10)

        # Each yield is followed by its own flush before the next yield starts.
        assert call_order == [
            "yield-1",
            "flush-p1",
            "yield-2",
            "flush-p2",
            "yield-3",
            "flush-p3",
        ]

    async def test_skips_unchanged_per_page(self, isolated_env):
        """Per-page flush skips URLs whose hash matches prior metadata."""
        import asyncio as _asyncio

        # Pre-seed: p1 was crawled last time with this exact content.
        from datetime import UTC as _UTC
        from datetime import datetime as _datetime

        seed = CrawlResult(url="https://example.com/p1", markdown="# Same")
        seed_meta: dict[str, CrawlMeta] = {}
        outcome = _save_single_result(seed, seed_meta)
        assert outcome is not None
        _update_single_metadata(seed_meta, seed.url, outcome, _datetime.now(_UTC).isoformat())
        save_crawl_metadata(seed_meta)

        async def _gen():
            yield _make_crawl4ai_result(url="https://example.com/p1", markdown="# Same")
            await _asyncio.sleep(0)
            yield _make_crawl4ai_result(url="https://example.com/p2", markdown="# NewPage")

        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=_gen())
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        with patch.dict("sys.modules", self._setup_crawl4ai(mock_instance)):
            paths = await crawl_and_save("https://example.com", depth=2, max_pages=10)

        # p1 is skipped (unchanged), p2 is written.
        assert len(paths) == 1
        assert paths[0].name == "index.md"
        assert paths[0].read_text(encoding="utf-8") == "# NewPage"

    @patch("lilbee.crawler.api.crawl_single")
    async def test_single_url_flushes_via_per_page_path(self, mock_crawl_single, isolated_env):
        """depth=0 uses the same flush_page callback, not the old batch helpers."""
        mock_crawl_single.return_value = CrawlResult(
            url="https://example.com/only", markdown="# Only"
        )
        with patch("lilbee.crawler.save._save_single_result", wraps=_save_single_result) as spy:
            paths = await crawl_and_save("https://example.com/only", depth=0)
        assert len(paths) == 1
        spy.assert_called_once()
        # Metadata written by the per-page path
        meta = load_crawl_metadata()
        assert "https://example.com/only" in meta

    async def test_full_success_matches_prior_behavior(self, isolated_env):
        """With no cancel, a full successful crawl produces the same on-disk result."""
        import asyncio as _asyncio

        urls_and_content = [
            ("https://example.com/a", "# A"),
            ("https://example.com/b", "# B"),
            ("https://example.com/c", "# C"),
        ]

        async def _gen():
            for url, md in urls_and_content:
                await _asyncio.sleep(0)
                yield _make_crawl4ai_result(url=url, markdown=md)

        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=_gen())
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        with patch.dict("sys.modules", self._setup_crawl4ai(mock_instance)):
            paths = await crawl_and_save("https://example.com", depth=2, max_pages=10)

        assert len(paths) == 3
        contents = {p.read_text(encoding="utf-8") for p in paths}
        assert contents == {"# A", "# B", "# C"}

        meta = load_crawl_metadata()
        for url, md in urls_and_content:
            assert url in meta
            assert meta[url].content_hash == content_hash(md)

    async def test_cancel_does_not_trigger_auto_sync(self, isolated_env):
        """On cancel, _maybe_periodic_sync is NOT awaited."""
        import asyncio as _asyncio
        import threading

        cancel = threading.Event()

        async def _gen():
            yield _make_crawl4ai_result(url="https://example.com/p1", markdown="# P1")
            cancel.set()
            await _asyncio.sleep(0)
            yield _make_crawl4ai_result(url="https://example.com/p2", markdown="# P2")

        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=_gen())
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.dict("sys.modules", self._setup_crawl4ai(mock_instance)),
            patch("lilbee.crawler.api._maybe_periodic_sync", new_callable=AsyncMock) as mock_sync,
        ):
            await crawl_and_save("https://example.com", depth=2, max_pages=10, cancel=cancel)
        mock_sync.assert_not_awaited()

    async def test_success_triggers_auto_sync(self, isolated_env):
        """On a clean (non-cancelled) crawl, _maybe_periodic_sync IS awaited."""
        import asyncio as _asyncio

        async def _gen():
            await _asyncio.sleep(0)
            yield _make_crawl4ai_result(url="https://example.com/p1", markdown="# P1")

        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=_gen())
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.dict("sys.modules", self._setup_crawl4ai(mock_instance)),
            patch("lilbee.crawler.api._maybe_periodic_sync", new_callable=AsyncMock) as mock_sync,
        ):
            await crawl_and_save("https://example.com", depth=2, max_pages=10)
        mock_sync.assert_awaited_once()

    async def test_metadata_flush_is_batched(self, isolated_env):
        """_flush_metadata fires every METADATA_FLUSH_INTERVAL pages, not every page."""
        import asyncio as _asyncio

        from lilbee.crawler import METADATA_FLUSH_INTERVAL

        total_pages = METADATA_FLUSH_INTERVAL + 3

        async def _gen():
            for i in range(1, total_pages + 1):
                await _asyncio.sleep(0)
                yield _make_crawl4ai_result(url=f"https://example.com/p{i}", markdown=f"# P{i}")

        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=_gen())
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.dict("sys.modules", self._setup_crawl4ai(mock_instance)),
            patch("lilbee.crawler.save.save_crawl_metadata") as mock_flush,
        ):
            await crawl_and_save("https://example.com", depth=2, max_pages=total_pages)

        # One flush at the interval boundary + one final flush after the loop.
        assert mock_flush.call_count == 2

    async def test_exact_interval_boundary_does_not_double_flush(self, isolated_env):
        """Ending on an exact METADATA_FLUSH_INTERVAL boundary runs one flush, not two."""
        import asyncio as _asyncio

        from lilbee.crawler import METADATA_FLUSH_INTERVAL

        async def _gen():
            for i in range(1, METADATA_FLUSH_INTERVAL + 1):
                await _asyncio.sleep(0)
                yield _make_crawl4ai_result(url=f"https://example.com/p{i}", markdown=f"# P{i}")

        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=_gen())
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.dict("sys.modules", self._setup_crawl4ai(mock_instance)),
            patch("lilbee.crawler.save.save_crawl_metadata") as mock_flush,
        ):
            await crawl_and_save("https://example.com", depth=2, max_pages=METADATA_FLUSH_INTERVAL)

        # Interval boundary triggers the only flush; post-loop skipped because
        # no entries remain pending after the counter reset.
        assert mock_flush.call_count == 1

    async def test_flush_callback_failure_does_not_fail_the_crawl(self, isolated_env):
        """An OSError from the flush callback is logged, not reraised as a crawl error."""
        import asyncio as _asyncio

        async def _gen():
            for i in range(1, 3):
                await _asyncio.sleep(0)
                yield _make_crawl4ai_result(url=f"https://example.com/p{i}", markdown=f"# P{i}")

        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=_gen())
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        call_count = {"n": 0}

        def failing_save(result, meta):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError("disk full")
            return None  # second page falls through unchanged

        with (
            patch.dict("sys.modules", self._setup_crawl4ai(mock_instance)),
            patch("lilbee.crawler.save._save_single_result", side_effect=failing_save),
        ):
            # Must not raise even though the first page write fails.
            paths = await crawl_and_save("https://example.com", depth=2, max_pages=10)
        assert paths == []

    @patch("lilbee.crawler.api.crawl_single")
    async def test_depth_zero_flush_failure_does_not_fail_the_crawl(
        self, mock_crawl_single, isolated_env
    ):
        """OSError from flush on the single-URL path is logged, not reraised."""
        mock_crawl_single.return_value = CrawlResult(
            url="https://example.com/only", markdown="# Only"
        )
        with patch("lilbee.crawler.save._save_single_result", side_effect=OSError("disk full")):
            # Must not raise even though the depth=0 flush fails.
            paths = await crawl_and_save("https://example.com/only", depth=0)
        assert paths == []

    async def test_final_flush_failure_does_not_fail_the_crawl(self, isolated_env):
        """OSError from the post-loop metadata flush is logged, not reraised."""
        import asyncio as _asyncio

        async def _gen():
            yield _make_crawl4ai_result(url="https://example.com/p1", markdown="# P1")
            await _asyncio.sleep(0)

        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=_gen())
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        real_flush = save_crawl_metadata
        call_count = {"n": 0}

        def flush_or_fail(meta):
            call_count["n"] += 1
            # Let interval-boundary flushes succeed; fail only the post-loop flush.
            if call_count["n"] == 1 and len(meta) >= 1:
                raise OSError("disk full")
            real_flush(meta)

        with (
            patch.dict("sys.modules", self._setup_crawl4ai(mock_instance)),
            patch("lilbee.crawler.save.save_crawl_metadata", side_effect=flush_or_fail),
        ):
            # Markdown was already written durably; the final flush must not
            # reraise since the caller has already consumed the stream.
            paths = await crawl_and_save("https://example.com", depth=2, max_pages=10)
        assert len(paths) == 1
        assert paths[0].exists()

    async def test_final_metadata_flush_fires_on_cancel(self, isolated_env):
        """Cancel still produces a final metadata flush so on-disk state stays consistent."""
        import asyncio as _asyncio
        import threading

        cancel = threading.Event()

        async def _gen():
            yield _make_crawl4ai_result(url="https://example.com/p1", markdown="# P1")
            cancel.set()
            await _asyncio.sleep(0)
            yield _make_crawl4ai_result(url="https://example.com/p2", markdown="# P2")

        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=_gen())
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.dict("sys.modules", self._setup_crawl4ai(mock_instance)),
            patch(
                "lilbee.crawler.save.save_crawl_metadata", wraps=save_crawl_metadata
            ) as spy_flush,
        ):
            await crawl_and_save("https://example.com", depth=2, max_pages=10, cancel=cancel)

        # One page written, below the interval, so the only flush is the final one.
        assert spy_flush.call_count == 1
        meta = load_crawl_metadata()
        assert "https://example.com/p1" in meta

    async def test_no_final_flush_when_nothing_written(self, isolated_env):
        """If zero pages were flushed, the post-loop metadata write is skipped."""
        import asyncio as _asyncio

        async def _gen():
            await _asyncio.sleep(0)
            # Empty markdown is skipped by _save_single_result
            yield _make_crawl4ai_result(url="https://example.com/empty", markdown="")

        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=_gen())
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.dict("sys.modules", self._setup_crawl4ai(mock_instance)),
            patch("lilbee.crawler.save.save_crawl_metadata") as mock_flush,
        ):
            paths = await crawl_and_save("https://example.com", depth=2, max_pages=10)
        assert paths == []
        mock_flush.assert_not_called()

    async def test_done_event_reports_partial_counts_on_cancel(self, isolated_env):
        """CRAWL_DONE fires even on cancel and reports what was actually flushed."""
        import asyncio as _asyncio
        import threading

        cancel = threading.Event()

        async def _gen():
            yield _make_crawl4ai_result(url="https://example.com/p1", markdown="# P1")
            cancel.set()
            await _asyncio.sleep(0)
            yield _make_crawl4ai_result(url="https://example.com/p2", markdown="# P2")

        mock_instance = AsyncMock()
        mock_instance.arun = AsyncMock(return_value=_gen())
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)

        done_events: list = []

        def on_progress(event_type, data):
            if event_type == EventType.CRAWL_DONE:
                done_events.append(data)

        with patch.dict("sys.modules", self._setup_crawl4ai(mock_instance)):
            paths = await crawl_and_save(
                "https://example.com",
                depth=2,
                max_pages=10,
                cancel=cancel,
                on_progress=on_progress,
            )

        assert len(done_events) == 1
        evt = done_events[0]
        assert evt.files_written == len(paths)
        assert evt.files_written >= 1
