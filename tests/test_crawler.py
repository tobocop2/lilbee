"""Tests for the web crawling module."""

import asyncio
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lilbee.app.services import get_services, reset_services
from lilbee.core.config import cfg
from lilbee.core.config.enums import CrawlRenderMode
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
from lilbee.crawler import bootstrap as bootstrap_mod
from lilbee.crawler.bootstrap import CrawlerBrowserError
from lilbee.crawler.runner import (
    _get_crawl_semaphore,
    _maybe_periodic_sync,
)
from lilbee.crawler.save import (
    _save_single_result,
    _update_single_metadata,
    normalize_crawled_markdown,
)
from lilbee.runtime.progress import EventType
from tests import _crawlberg_stub as cb
from tests._mock_effects import repeat_last


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
    # Disable periodic sync by default. The drain step in the runner's
    # finally-block waits for a real ingest to complete; on slow CI
    # runners (ubuntu 3.11) the embedding cold-start blows the
    # pytest-timeout. Tests that exercise periodic sync explicitly set
    # crawl_sync_interval to a non-zero value themselves.
    cfg.crawl_sync_interval = 0
    cls = request.cls.__name__ if request.cls else ""
    if cls not in ("TestPlaywrightBrowserCheck", "TestChromiumInstalledMatching"):
        monkeypatch.setattr("lilbee.crawler.bootstrap.chromium_installed", lambda: True)
        shell = tmp_path / "chrome-headless-shell"
        monkeypatch.setattr("lilbee.crawler.bootstrap.headless_shell_executable", lambda: shell)
    # The browser-mode fetcher points CHROME at the shell; restore it after each test.
    monkeypatch.delenv("CHROME", raising=False)
    # Stub crawler_available so crawl_and_save's pre-flight backend check passes
    # in CI envs where the [crawler] extra isn't installed. Tests that exercise
    # the missing-extra path opt out by re-patching to ``False``. Both the SDK
    # façade (used by callers) and the impl (used by tests that import directly)
    # are patched so the value is consistent across import paths. The
    # ``TestCrawlerAvailable`` class drives the real function directly via
    # ``sys.modules`` patching, so it opts out of the auto-stub.
    if cls != "TestCrawlerAvailable":
        monkeypatch.setattr("lilbee.crawler.crawler_available", lambda: True)
        monkeypatch.setattr("lilbee.crawler.crawlberg_fetcher.crawler_available", lambda: True)
    # Default the sitemap bound to "unknown" so tests don't hit the network.
    # Tests that exercise the sitemap hook directly (TestSitemapCounting)
    # opt out of this autopatch.
    if cls != "TestSitemapCounting":
        from lilbee.runtime.progress import CRAWL_TOTAL_UNKNOWN

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

    def test_distinct_queries_map_to_distinct_files(self):
        # ?q=1 and ?q=2 are different resources; they must not clobber each other.
        a = url_to_filename("https://example.com/page?q=1")
        b = url_to_filename("https://example.com/page?q=2")
        assert a != b
        assert a.startswith("example.com/page/") and a.endswith(".md")
        assert b.startswith("example.com/page/") and b.endswith(".md")

    def test_same_query_is_stable(self):
        url = "https://example.com/page?q=1&foo=bar"
        assert url_to_filename(url) == url_to_filename(url)

    def test_query_with_file_extension_disambiguated(self):
        a = url_to_filename("https://example.com/docs/guide.html?v=1")
        b = url_to_filename("https://example.com/docs/guide.html?v=2")
        assert a != b
        assert a.startswith("example.com/docs/guide") and a.endswith(".md")

    def test_dotted_non_extension_segment_still_disambiguates_query(self):
        # A path segment that is a bare "." has no real extension; it must still
        # route to .md so distinct queries don't collide on it.
        a = url_to_filename("https://example.com/.?a=1")
        b = url_to_filename("https://example.com/.?a=2")
        assert a != b
        assert a.endswith(".md") and b.endswith(".md")

    def test_fragment_stripped(self):
        # Fragments are client-side; same resource -> same file (and no query suffix).
        assert (
            url_to_filename("https://example.com/page#section")
            == url_to_filename("https://example.com/page")
            == "example.com/page/index.md"
        )

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

    def test_load_corrupted_json(self, isolated_env, caplog):
        meta_path = cfg.data_dir / "crawl_meta.json"
        meta_path.write_text("not valid json")
        with caplog.at_level("WARNING"):
            meta = load_crawl_metadata()
        assert meta == {}
        # A corrupt sidecar re-crawls everything; the operator gets told why.
        assert any("Discarding unreadable crawl metadata" in r.message for r in caplog.records)

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


@pytest.fixture(autouse=True)
def _no_dns(monkeypatch):
    """Bypass SSRF DNS resolution in all crawler tests."""
    monkeypatch.setattr(
        "lilbee.crawler.url_filter.socket.getaddrinfo",
        lambda host, port, *a, **kw: [(2, 1, 6, "", ("93.184.216.34", 0))],
    )


class TestCrawlerAvailable:
    """Exercises the un-patched ``crawler_available`` install probe.

    Uses ``importlib.util.find_spec`` so the check stays cheap on the UI
    thread; the Settings screen calls this synchronously during compose
    via ``_FEATURE_GATED_GROUPS``.
    """

    def test_returns_true_when_installed(self):
        from lilbee.crawler import crawler_available

        crawler_available.cache_clear()
        with patch(
            "importlib.util.find_spec",
            return_value=MagicMock(name="crawlberg_spec"),
        ):
            assert crawler_available() is True
        crawler_available.cache_clear()

    def test_returns_false_when_not_installed(self):
        from lilbee.crawler import crawler_available

        crawler_available.cache_clear()
        with patch("importlib.util.find_spec", return_value=None):
            assert crawler_available() is False
        crawler_available.cache_clear()

    def test_does_not_execute_module_init(self):
        """Regression guard: probe must not actually import ``crawlberg``.

        crawlberg loads a native library. The Settings screen calls this on
        the UI thread, so executing the module here would block the TUI.
        """
        import sys as _sys

        from lilbee.crawler import crawler_available

        crawler_available.cache_clear()
        with patch(
            "importlib.util.find_spec",
            return_value=MagicMock(name="crawlberg_spec"),
        ):
            had_module = "crawlberg" in _sys.modules
            crawler_available()
            assert ("crawlberg" in _sys.modules) is had_module
        crawler_available.cache_clear()


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


SEED = "https://example.com"


def _failing_engine(message: str):
    def create_engine(config):
        raise RuntimeError(message)

    return create_engine


class TestCrawlSingle:
    async def test_success(self):
        with cb.StubCrawlberg([cb.page(SEED, "# Test", depth=0)]).installed():
            result = await crawl_single(SEED)
        assert result.success
        assert result.markdown == "# Test"

    async def test_failure(self):
        with cb.StubCrawlberg([cb.error(SEED, "Connection refused")]).installed():
            result = await crawl_single(SEED)
        assert not result.success
        assert result.error == "Connection refused"

    async def test_exception(self):
        stub = cb.StubCrawlberg()
        stub.module.create_engine = _failing_engine("timeout")
        with stub.installed():
            result = await crawl_single(SEED)
        assert not result.success
        assert "timeout" in result.error

    async def test_bootstraps_and_retries_when_chromium_missing(self, monkeypatch, tmp_path):
        """A headless shell missing at launch is installed once, then the fetch retries."""
        shell = tmp_path / "chrome-headless-shell"
        monkeypatch.setattr(
            "lilbee.crawler.bootstrap.headless_shell_executable",
            MagicMock(side_effect=repeat_last(None, shell)),
        )
        bootstrapped: list[bool] = []

        async def fake_bootstrap(on_progress):
            bootstrapped.append(True)

        monkeypatch.setattr("lilbee.crawler.bootstrap.bootstrap_chromium", fake_bootstrap)
        with cb.StubCrawlberg([cb.page(SEED, "# Test", depth=0)]).installed():
            result = await crawl_single(SEED)
        assert result.success
        assert result.markdown == "# Test"
        assert bootstrapped == [True], "bootstrap_chromium should fire exactly once"

    async def test_retry_failure_is_returned_as_error_result(self, monkeypatch, tmp_path):
        """If the retry also fails, surface that error -- not the original."""
        shell = tmp_path / "chrome-headless-shell"
        monkeypatch.setattr(
            "lilbee.crawler.bootstrap.headless_shell_executable",
            MagicMock(side_effect=repeat_last(None, shell)),
        )

        async def fake_bootstrap(on_progress):
            pass

        monkeypatch.setattr("lilbee.crawler.bootstrap.bootstrap_chromium", fake_bootstrap)
        stub = cb.StubCrawlberg()
        stub.module.create_engine = _failing_engine("still broken after bootstrap")
        with stub.installed():
            result = await crawl_single(SEED)
        assert not result.success
        assert "still broken after bootstrap" in result.error

    async def test_missing_chromium_raises_crawler_browser_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without Chromium after a bootstrap attempt, crawl_single raises a clean exception.

        Regression test for bb-60mj: the task must land as failed, not done.
        """
        from lilbee.crawler import CrawlerBrowserError

        monkeypatch.setattr("lilbee.crawler.bootstrap.headless_shell_executable", lambda: None)
        bootstrapped: list[bool] = []

        async def fake_bootstrap(on_progress):
            bootstrapped.append(True)

        monkeypatch.setattr("lilbee.crawler.bootstrap.bootstrap_chromium", fake_bootstrap)
        with (
            cb.StubCrawlberg().installed(),
            pytest.raises(CrawlerBrowserError, match="Chromium"),
        ):
            await crawl_single(SEED)
        assert bootstrapped == [True]

    async def test_missing_crawler_extra_raises_backend_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without the ``crawler`` extra installed, crawl_single fails fast.

        Without this guard the lazy ``import crawlberg`` inside the fetcher
        would be swallowed by the broad ``except Exception`` at the bottom
        of crawl_single, turning a missing-extra install into a silent
        ``CrawlResult(success=False)`` and a ``crawl_done`` with
        ``files_written=0``: same silent failure the recursive path
        already guards against.
        """
        from lilbee.crawler import CrawlerBackendError

        monkeypatch.setattr("lilbee.crawler.crawler_available", lambda: False)
        monkeypatch.setattr("lilbee.crawler.crawlberg_fetcher.crawler_available", lambda: False)
        with pytest.raises(CrawlerBackendError, match="Web crawling is not available"):
            await crawl_single("https://example.com")


def _install_shell(root: Path, revision: str, monkeypatch) -> Path:
    """Lay out a headless shell binary the way Playwright installs it on a Linux x86_64 host."""
    monkeypatch.setattr(bootstrap_mod.sys, "platform", "linux")
    monkeypatch.setattr(bootstrap_mod.platform, "machine", lambda: "x86_64")
    executable = (
        root
        / f"chromium_headless_shell-{revision}"
        / "chrome-headless-shell-linux64"
        / "chrome-headless-shell"
    )
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"")
    return executable


class TestChromiumInstalledMatching:
    """``chromium_installed`` looks for the headless shell at the bundled Playwright's revision."""

    def test_matches_expected_revision_when_present(self, tmp_path, monkeypatch):
        from lilbee.crawler import bootstrap

        executable = _install_shell(tmp_path, "1208", monkeypatch)
        monkeypatch.setattr(bootstrap, "_browsers_cache_path", lambda: tmp_path)
        monkeypatch.setattr(bootstrap, "_expected_chromium_revision", lambda: "1208")

        assert bootstrap.chromium_installed() is True
        assert bootstrap.headless_shell_executable() == executable

    def test_rejects_wrong_revision(self, tmp_path, monkeypatch):
        from lilbee.crawler import bootstrap

        _install_shell(tmp_path, "1217", monkeypatch)
        monkeypatch.setattr(bootstrap, "_browsers_cache_path", lambda: tmp_path)
        monkeypatch.setattr(bootstrap, "_expected_chromium_revision", lambda: "1208")

        assert bootstrap.chromium_installed() is False

    def test_falls_back_to_any_revision_when_expected_unknown(self, tmp_path, monkeypatch):
        from lilbee.crawler import bootstrap

        _install_shell(tmp_path, "1217", monkeypatch)
        monkeypatch.setattr(bootstrap, "_browsers_cache_path", lambda: tmp_path)
        monkeypatch.setattr(bootstrap, "_expected_chromium_revision", lambda: None)

        assert bootstrap.chromium_installed() is True

    def test_full_chromium_build_does_not_count(self, tmp_path, monkeypatch):
        """Only the headless shell is launched, so the full build alone is not installed."""
        from lilbee.crawler import bootstrap

        _install_shell(tmp_path, "1208", monkeypatch).unlink()
        (tmp_path / "chromium-1208").mkdir()
        monkeypatch.setattr(bootstrap, "_browsers_cache_path", lambda: tmp_path)
        monkeypatch.setattr(bootstrap, "_expected_chromium_revision", lambda: "1208")

        assert bootstrap.chromium_installed() is False

    def test_platform_without_a_known_layout_has_no_shell(self, tmp_path, monkeypatch):
        from lilbee.crawler import bootstrap

        _install_shell(tmp_path, "1208", monkeypatch)
        monkeypatch.setattr(bootstrap.platform, "machine", lambda: "riscv64")
        monkeypatch.setattr(bootstrap, "_browsers_cache_path", lambda: tmp_path)

        assert bootstrap.headless_shell_executable() is None

    def test_returns_false_when_cache_root_missing(self, tmp_path, monkeypatch):
        from lilbee.crawler import bootstrap

        missing = tmp_path / "absent"
        monkeypatch.setattr(bootstrap, "_browsers_cache_path", lambda: missing)
        assert bootstrap.chromium_installed() is False

    def test_returns_false_when_no_chromium_dir_anywhere(self, tmp_path, monkeypatch):
        from lilbee.crawler import bootstrap

        (tmp_path / "firefox-1234").mkdir()
        monkeypatch.setattr(bootstrap, "_browsers_cache_path", lambda: tmp_path)
        monkeypatch.setattr(bootstrap, "_expected_chromium_revision", lambda: None)
        assert bootstrap.chromium_installed() is False


class TestReadChromiumRevision:
    """Pure-function helper that pulls the revision out of browsers.json."""

    def test_extracts_chromium_revision(self, tmp_path):
        from lilbee.crawler.bootstrap import _read_chromium_revision

        payload = (
            '{"browsers": [{"name": "chromium", "revision": 1300},'
            ' {"name": "chromium-headless-shell", "revision": 1208}]}'
        )
        path = tmp_path / "browsers.json"
        path.write_text(payload, encoding="utf-8")
        assert _read_chromium_revision(path) == "1208"

    def test_returns_none_when_chromium_entry_missing(self, tmp_path):
        from lilbee.crawler.bootstrap import _read_chromium_revision

        path = tmp_path / "browsers.json"
        path.write_text('{"browsers": [{"name": "firefox", "revision": 1}]}', encoding="utf-8")
        assert _read_chromium_revision(path) is None

    def test_returns_none_when_revision_is_missing(self, tmp_path):
        from lilbee.crawler.bootstrap import _read_chromium_revision

        path = tmp_path / "browsers.json"
        path.write_text('{"browsers": [{"name": "chromium-headless-shell"}]}', encoding="utf-8")
        assert _read_chromium_revision(path) is None

    def test_returns_none_on_malformed_json(self, tmp_path):
        from lilbee.crawler.bootstrap import _read_chromium_revision

        path = tmp_path / "browsers.json"
        path.write_text("not json", encoding="utf-8")
        assert _read_chromium_revision(path) is None

    def test_returns_none_when_file_missing(self, tmp_path):
        from lilbee.crawler.bootstrap import _read_chromium_revision

        assert _read_chromium_revision(tmp_path / "absent.json") is None


class TestExpectedChromiumRevision:
    """Walks the bundled Playwright install to find browsers.json."""

    def test_returns_revision_from_real_playwright(self):
        from lilbee.crawler.bootstrap import _expected_chromium_revision

        # In the dev/test env Playwright is installed via the extras;
        # the returned value should be a numeric string.
        revision = _expected_chromium_revision()
        assert revision is None or revision.isdigit()

    def test_returns_none_when_playwright_not_importable(self, monkeypatch):
        import sys

        from lilbee.crawler.bootstrap import _expected_chromium_revision

        # Drop playwright from sys.modules and block re-import.
        monkeypatch.setitem(sys.modules, "playwright", None)
        assert _expected_chromium_revision() is None

    def test_returns_none_when_browsers_json_missing(self, tmp_path, monkeypatch):
        import sys
        import types

        from lilbee.crawler import bootstrap

        # Stand up a fake playwright package whose path has no browsers.json.
        fake = types.ModuleType("playwright")
        fake.__file__ = str(tmp_path / "__init__.py")
        (tmp_path / "__init__.py").write_text("", encoding="utf-8")
        monkeypatch.setitem(sys.modules, "playwright", fake)
        assert bootstrap._expected_chromium_revision() is None

    def test_returns_revision_from_fake_playwright_browsers_json(self, tmp_path, monkeypatch):
        import sys
        import types

        from lilbee.crawler import bootstrap

        # Stand up a fake playwright package with a browsers.json that
        # rglob will discover. Exercises the read path end-to-end.
        pkg = tmp_path / "playwright"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "browsers.json").write_text(
            '{"browsers": [{"name": "chromium-headless-shell", "revision": 1208}]}',
            encoding="utf-8",
        )
        fake = types.ModuleType("playwright")
        fake.__file__ = str(pkg / "__init__.py")
        monkeypatch.setitem(sys.modules, "playwright", fake)
        assert bootstrap._expected_chromium_revision() == "1208"


class TestBootstrapChromium:
    """bb-wq8g: the subprocess wrapper that installs Playwright's Chromium."""

    async def test_short_circuits_when_already_installed(self, monkeypatch):
        """No subprocess, no stream events, when Chromium is already present."""
        from lilbee.crawler.bootstrap import bootstrap_chromium
        from lilbee.runtime.progress import EventType, SetupDoneEvent

        monkeypatch.setattr("lilbee.crawler.bootstrap.chromium_installed", lambda: True)
        events: list[tuple[EventType, object]] = []
        await bootstrap_chromium(on_progress=lambda e, d: events.append((e, d)))
        # Only a single setup_done (success): no start/progress when we
        # short-circuit.
        assert len(events) == 1
        evt, payload = events[0]
        assert evt == EventType.SETUP_DONE
        assert isinstance(payload, SetupDoneEvent)
        assert payload.success is True

    async def test_parses_progress_from_fake_subprocess(self, monkeypatch):
        """Feed canned stdout through the subprocess to drive progress events."""
        from lilbee.crawler.bootstrap import bootstrap_chromium
        from lilbee.runtime.progress import EventType, SetupProgressEvent, SetupStartEvent

        monkeypatch.setattr("lilbee.crawler.bootstrap.chromium_installed", lambda: False)
        # Stub the runner so the test doesn't need a real playwright install
        # (regular `test` job runs without crawler extras).
        monkeypatch.setattr(
            "lilbee.crawler.bootstrap._resolve_playwright_runner",
            lambda: (["/fake/node", "/fake/cli.js"], {}),
        )

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
                self.returncode: int | None = None

            async def wait(self) -> int:
                self.returncode = 0
                return self.returncode

        argv: list[tuple] = []

        async def _fake_create_subprocess_exec(*args, **_kwargs):
            argv.append(args)
            return _Proc()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

        events: list[tuple[EventType, object]] = []
        await bootstrap_chromium(on_progress=lambda e, d: events.append((e, d)))

        # Only the headless shell is downloaded: it is the one binary a crawl launches.
        assert argv == [("/fake/node", "/fake/cli.js", "install", "--only-shell", "chromium")]
        types = [e for e, _ in events]
        assert types[0] == EventType.SETUP_START
        assert types[-1] == EventType.SETUP_DONE
        progress_events = [d for e, d in events if e == EventType.SETUP_PROGRESS]
        assert len(progress_events) >= 1
        assert isinstance(events[0][1], SetupStartEvent)
        assert isinstance(progress_events[0], SetupProgressEvent)

    async def test_raises_crawler_browser_missing_on_subprocess_failure(self, monkeypatch):
        """Non-zero exit → CrawlerBrowserError with stderr tail."""
        from lilbee.crawler.bootstrap import CrawlerBrowserError, bootstrap_chromium
        from lilbee.runtime.progress import EventType, SetupDoneEvent

        monkeypatch.setattr("lilbee.crawler.bootstrap.chromium_installed", lambda: False)
        monkeypatch.setattr(
            "lilbee.crawler.bootstrap._resolve_playwright_runner",
            lambda: (["/fake/node", "/fake/cli.js"], {}),
        )

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
                self.returncode: int | None = None

            async def wait(self) -> int:
                self.returncode = 42
                return self.returncode

        async def _fake_create_subprocess_exec(*_args, **_kwargs):
            return _Proc()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

        events: list[tuple[EventType, object]] = []
        with pytest.raises(CrawlerBrowserError, match="exit 42"):
            await bootstrap_chromium(on_progress=lambda e, d: events.append((e, d)))
        final = events[-1]
        assert final[0] == EventType.SETUP_DONE
        assert isinstance(final[1], SetupDoneEvent)
        assert final[1].success is False
        assert "network unreachable" in (final[1].error or "")


class TestResolvePlaywrightRunner:
    """``_resolve_playwright_runner`` returns argv + env for the bundled driver (bb-sxsz)."""

    def _install_fake_driver_module(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        compute: object | None,
        env: dict[str, str] | None,
    ) -> None:
        """Inject a fake ``playwright._impl._driver`` into sys.modules.

        The regular ``test`` CI job runs without lilbee[crawler]; stubbing
        sys.modules keeps the test importable on every lane.
        """
        playwright = types.ModuleType("playwright")
        playwright_impl = types.ModuleType("playwright._impl")
        playwright_driver = types.ModuleType("playwright._impl._driver")
        playwright_driver.compute_driver_executable = compute  # type: ignore[attr-defined]
        playwright_driver.get_driver_env = (lambda: env) if env is not None else None  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "playwright", playwright)
        monkeypatch.setitem(sys.modules, "playwright._impl", playwright_impl)
        monkeypatch.setitem(sys.modules, "playwright._impl._driver", playwright_driver)

    def test_uses_compute_driver_executable(self, monkeypatch):
        """Returns argv from playwright's bundled driver lookup, with driver env."""
        fake_node = "/fake/site-packages/playwright/driver/node"
        fake_cli = "/fake/site-packages/playwright/driver/package/cli.js"
        self._install_fake_driver_module(
            monkeypatch,
            compute=lambda: (fake_node, fake_cli),
            env={"PW_LANG_NAME": "python", "PATH": "/usr/bin"},
        )

        argv, env = bootstrap_mod._resolve_playwright_runner()
        assert argv == [fake_node, fake_cli]
        assert env["PW_LANG_NAME"] == "python"
        assert env["PATH"] == "/usr/bin"

    def test_works_under_frozen_build(self, monkeypatch):
        """PyInstaller layout: same code path, no system Python required."""
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        fake_node = "/_MEI/playwright/driver/node"
        fake_cli = "/_MEI/playwright/driver/package/cli.js"
        self._install_fake_driver_module(
            monkeypatch,
            compute=lambda: (fake_node, fake_cli),
            env={},
        )

        argv, _env = bootstrap_mod._resolve_playwright_runner()
        assert argv == [fake_node, fake_cli]

    def test_compute_driver_failure_falls_back_when_not_frozen(self, monkeypatch):
        """compute_driver_executable raise on a wheel install falls back to sys.executable -m."""
        monkeypatch.delattr(sys, "frozen", raising=False)

        def _boom() -> tuple[str, str]:
            raise RuntimeError("playwright API drift")

        self._install_fake_driver_module(monkeypatch, compute=_boom, env={})

        argv, _env = bootstrap_mod._resolve_playwright_runner()
        assert argv == [sys.executable, "-m", "playwright"]

    def test_compute_driver_failure_under_frozen_propagates(self, monkeypatch):
        """compute_driver_executable raise under PyInstaller propagates (no fallback)."""
        monkeypatch.setattr(sys, "frozen", True, raising=False)

        def _boom() -> tuple[str, str]:
            raise RuntimeError("playwright API drift")

        self._install_fake_driver_module(monkeypatch, compute=_boom, env={})

        with pytest.raises(RuntimeError, match="API drift"):
            bootstrap_mod._resolve_playwright_runner()

    def test_compute_driver_failure_under_nuitka_propagates(self, monkeypatch):
        """Nuitka sets __compiled__ but never sys.frozen; still propagate (no -m fallback)."""
        from lilbee import _frozen

        monkeypatch.delattr(sys, "frozen", raising=False)
        monkeypatch.setattr(_frozen, "__compiled__", object(), raising=False)

        def _boom() -> tuple[str, str]:
            raise RuntimeError("playwright API drift")

        self._install_fake_driver_module(monkeypatch, compute=_boom, env={})

        with pytest.raises(RuntimeError, match="API drift"):
            bootstrap_mod._resolve_playwright_runner()

    def test_playwright_missing_raises_actionable_error(self, monkeypatch):
        """ImportError on playwright raises CrawlerBrowserError with install hint."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "playwright._impl._driver":
                raise ImportError("No module named 'playwright'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(CrawlerBrowserError) as ei:
            bootstrap_mod._resolve_playwright_runner()
        message = str(ei.value)
        # The error must name the install fix; bare 'not found' breaks the
        # user's recovery loop in the release executable.
        assert "lilbee[crawler]" in message or "playwright" in message

    async def test_bootstrap_chromium_emits_setup_done_when_playwright_missing(self, monkeypatch):
        """bootstrap_chromium fires setup_done(success=False) before re-raising."""
        import builtins

        from lilbee.runtime.progress import SetupDoneEvent

        monkeypatch.setattr("lilbee.crawler.bootstrap.chromium_installed", lambda: False)

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "playwright._impl._driver":
                raise ImportError("No module named 'playwright'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        events: list[tuple[EventType, object]] = []
        with pytest.raises(CrawlerBrowserError):
            await bootstrap_mod.bootstrap_chromium(on_progress=lambda e, d: events.append((e, d)))

        types = [e for e, _ in events]
        assert types[0] == EventType.SETUP_START
        assert types[-1] == EventType.SETUP_DONE
        last = events[-1][1]
        assert isinstance(last, SetupDoneEvent)
        assert last.success is False
        assert last.error is not None and "playwright" in last.error.lower()


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
        """When the expected revision is unknown, any installed headless shell counts."""
        from lilbee.crawler.bootstrap import chromium_installed

        browsers = tmp_path / "ms-playwright"
        _install_shell(browsers, "1234", monkeypatch)
        monkeypatch.setattr("lilbee.crawler.bootstrap._browsers_cache_path", lambda: browsers)
        monkeypatch.setattr("lilbee.crawler.bootstrap._expected_chromium_revision", lambda: None)
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
    async def test_returns_multiple_results(self):
        script = [cb.page(SEED, "# Home", depth=0), cb.page(f"{SEED}/about", "# About")]
        progress_calls = []

        def on_progress(event_type, data):
            progress_calls.append((event_type, data))

        with cb.StubCrawlberg(script).installed():
            results = await crawl_recursive(
                SEED, max_depth=1, max_pages=10, on_progress=on_progress
            )
        assert len(results) == 2
        assert results[0].url == SEED
        assert results[1].url == f"{SEED}/about"
        # Streaming semantics: total is unknown during BFS, counter advances per page.
        from lilbee.runtime.progress import CRAWL_TOTAL_UNKNOWN

        page_events = [c for c in progress_calls if c[0] == EventType.CRAWL_PAGE]
        assert [c[1].current for c in page_events] == [1, 2]
        assert all(c[1].total == CRAWL_TOTAL_UNKNOWN for c in page_events)
        # Setup events mean a Chromium install; with the shell present there is none.
        setup_types = [
            e for e, _ in progress_calls if e in (EventType.SETUP_START, EventType.SETUP_DONE)
        ]
        assert setup_types == []

    async def test_emits_events_before_stream_exhausted(self):
        """CRAWL_PAGE fires per page as it arrives, not only after the full list."""
        observations: list[tuple[str, int]] = []

        async def _gen():
            for i in range(1, 4):
                await asyncio.sleep(0)
                observations.append(("yielded", i))
                yield cb.page(f"{SEED}/p{i}", f"# P{i}")

        def on_progress(event_type, data):
            if event_type == EventType.CRAWL_PAGE:
                observations.append(("progress", data.current))

        with cb.StubCrawlberg(_gen).installed():
            await crawl_recursive(SEED, max_depth=2, max_pages=100, on_progress=on_progress)

        # Each page's progress event must appear immediately after its yield,
        # before any subsequent yield. Pattern: yielded=1, progress=1, yielded=2, progress=2, ...
        for i in range(3):
            assert observations[2 * i] == ("yielded", i + 1)
            assert observations[2 * i + 1] == ("progress", i + 1)

    async def test_cancel_stops_mid_stream(self):
        """Setting the cancel event mid-stream stops further result collection."""
        import threading

        cancel = threading.Event()

        async def _gen():
            for i in range(1, 6):
                await asyncio.sleep(0)
                yield cb.page(f"{SEED}/p{i}", f"# P{i}")
                if i == 2:
                    cancel.set()

        stub = cb.StubCrawlberg(_gen)
        with stub.installed():
            results = await crawl_recursive(SEED, max_depth=2, max_pages=100, cancel=cancel)

        assert len(results) <= 2
        assert stub.closed == 1

    async def test_hard_cap_on_visible_counter(self):
        """The visible counter stops at max_pages and the stream closes there."""

        async def _gen():
            for i in range(1, 11):
                await asyncio.sleep(0)
                yield cb.page(f"{SEED}/p{i}")

        events = []

        def on_progress(event_type, data):
            if event_type == EventType.CRAWL_PAGE:
                events.append(data.current)

        stub = cb.StubCrawlberg(_gen)
        with stub.installed():
            results = await crawl_recursive(SEED, max_depth=1, max_pages=3, on_progress=on_progress)
        assert events == [1, 2, 3]
        assert len(results) == 3
        assert stub.closed == 1

    async def test_error_during_cancel_teardown_is_not_a_failure(self, caplog):
        """A crawl that raises after cancel reports nothing rather than a failed page."""
        import threading

        cancel = threading.Event()

        async def _gen():
            yield cb.page(f"{SEED}/p1", "# P1")
            cancel.set()
            raise RuntimeError("stream dropped")

        with cb.StubCrawlberg(_gen).installed(), caplog.at_level("DEBUG"):
            results = await crawl_recursive(SEED, max_depth=1, max_pages=10, cancel=cancel)
        assert results == []
        assert "ended during cancel teardown" in caplog.text

    async def test_mixed_success_failure(self):
        script = [cb.page(SEED, "# Home", depth=0), cb.error(f"{SEED}/broken", "404")]
        with cb.StubCrawlberg(script).installed():
            results = await crawl_recursive(SEED, max_depth=1, max_pages=10)
        assert len(results) == 2
        assert results[0].success
        assert not results[1].success

    async def test_exception_returns_error_result(self):
        stub = cb.StubCrawlberg()
        stub.module.create_engine = _failing_engine("network error")
        with stub.installed():
            results = await crawl_recursive(SEED, max_depth=1, max_pages=5)
        assert len(results) == 1
        assert not results[0].success

    async def test_defaults_to_unbounded_depth_but_capped_pages(self):
        """No max_depth / max_pages: depth is unbounded, pages clamps to the safety ceiling."""
        cfg.crawl_max_depth = None
        cfg.crawl_max_pages = None
        cfg.crawl_safety_max_pages = 5_000
        stub = cb.StubCrawlberg()
        with stub.installed():
            await crawl_recursive(SEED)
        assert stub.config["max_depth"] is None
        assert stub.config["max_pages"] == 5_000

    async def test_explicit_cap_overrides_cfg_ceiling(self):
        """An explicit int wins even when cfg sets a lower ceiling."""
        cfg.crawl_max_pages = 10
        stub = cb.StubCrawlberg()
        with stub.installed():
            await crawl_recursive(SEED, max_depth=1, max_pages=999)
        assert stub.config["max_pages"] == 999

    async def test_cfg_ceiling_applied_when_none_passed(self):
        """cfg.crawl_max_pages acts as ceiling when caller passes None."""
        cfg.crawl_max_pages = 10
        stub = cb.StubCrawlberg()
        with stub.installed():
            await crawl_recursive(SEED, max_depth=1, max_pages=None)
        assert stub.config["max_pages"] == 10

    async def test_zero_max_pages_is_unlimited(self):
        """max_pages=0 (CRAWL_PAGES_UNLIMITED) is an explicit no-limit, even past cfg."""
        cfg.crawl_max_pages = 10
        stub = cb.StubCrawlberg()
        with stub.installed():
            await crawl_recursive(SEED, max_depth=1, max_pages=0)
        assert stub.config["max_pages"] is None

    async def test_reraises_browser_missing_from_crawler_open(self, monkeypatch):
        """CrawlerBrowserError raised inside the try block propagates past the broad except."""
        from lilbee.crawler import CrawlerBrowserError

        monkeypatch.setattr("lilbee.crawler.bootstrap.headless_shell_executable", lambda: None)
        with (
            cb.StubCrawlberg().installed(),
            pytest.raises(CrawlerBrowserError, match="Chromium"),
        ):
            await crawl_recursive(SEED, max_depth=1, max_pages=5)

    async def test_propagates_crawler_browser_missing(self, monkeypatch):
        """bb-wq8g: crawl_recursive refuses a browser-mode crawl when Chromium is missing."""
        from lilbee.crawler import CrawlerBrowserError

        monkeypatch.setattr("lilbee.crawler.bootstrap.chromium_installed", lambda: False)
        stub = cb.StubCrawlberg()
        with stub.installed(), pytest.raises(CrawlerBrowserError, match="Chromium"):
            await crawl_recursive(SEED, max_depth=1)
        assert stub.configs == []

    async def test_missing_crawler_extra_raises_backend_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without the ``crawler`` extra, crawl_recursive fails fast.

        Without this guard the BFS silently drops out and the crawl returns
        ``files_written=0`` because every lazy backend import inside the
        fetcher swallows the ImportError locally. The explicit raise lets
        the server's SSE layer surface ``event: error`` with a fix-it
        message instead of a zero-results ``crawl_done``.
        """
        from lilbee.crawler import CrawlerBackendError

        monkeypatch.setattr("lilbee.crawler.crawler_available", lambda: False)
        monkeypatch.setattr("lilbee.crawler.crawlberg_fetcher.crawler_available", lambda: False)
        with pytest.raises(CrawlerBackendError, match="Web crawling is not available"):
            await crawl_recursive("https://example.com", max_depth=1)


class TestSitemapCounting:
    """best-effort sitemap lookup bounds the crawl progress total."""

    @pytest.fixture(autouse=True)
    def _public_dns(self, monkeypatch):
        """The fetch re-validates the resolved sitemap URL; resolve to a public IP."""
        monkeypatch.setattr(
            "lilbee.crawler.url_filter.socket.getaddrinfo",
            lambda *a, **kw: [(2, 1, 6, "", ("93.184.216.34", 0))],
        )

    def test_returns_unknown_on_http_error(self, monkeypatch):
        import httpx

        from lilbee.crawler.sitemap import _count_sitemap_urls
        from lilbee.runtime.progress import CRAWL_TOTAL_UNKNOWN

        def _raise(*a, **kw):
            raise httpx.ConnectError("boom")

        monkeypatch.setattr("httpx.get", _raise)
        assert (
            _count_sitemap_urls("https://example.com/start", include_subdomains=False)
            == CRAWL_TOTAL_UNKNOWN
        )

    def test_returns_unknown_on_4xx(self, monkeypatch):
        from lilbee.crawler.sitemap import _count_sitemap_urls
        from lilbee.runtime.progress import CRAWL_TOTAL_UNKNOWN

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
        fake = MagicMock(status_code=200, text=body, url="https://example.com/sitemap.xml")
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
        fake = MagicMock(status_code=200, text=body, url="https://example.com/sitemap.xml")
        monkeypatch.setattr("httpx.get", lambda *a, **kw: fake)
        count = _count_sitemap_urls("https://example.com/start", include_subdomains=True)
        assert count == 2

    def test_returns_unknown_when_start_url_has_no_host(self):
        """A malformed start URL short-circuits before hitting the network."""
        from lilbee.crawler.sitemap import _count_sitemap_urls
        from lilbee.runtime.progress import CRAWL_TOTAL_UNKNOWN

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
        fake = MagicMock(status_code=200, text=body, url="https://example.com/sitemap.xml")
        monkeypatch.setattr("httpx.get", lambda *a, **kw: fake)
        count = _count_sitemap_urls("https://example.com/start", include_subdomains=False)
        assert count == 1

    def test_caps_at_max_urls(self, monkeypatch):
        """A giant sitemap stops at _SITEMAP_MAX_URLS so the scan is bounded."""
        from lilbee.crawler import sitemap as sitemap_mod
        from lilbee.crawler.sitemap import _count_sitemap_urls

        monkeypatch.setattr(sitemap_mod, "_SITEMAP_MAX_URLS", 3)
        entries = "".join(f"<url><loc>https://example.com/{i}</loc></url>" for i in range(10))
        fake = MagicMock(
            status_code=200,
            text=f"<urlset>{entries}</urlset>",
            url="https://example.com/sitemap.xml",
        )
        monkeypatch.setattr("httpx.get", lambda *a, **kw: fake)
        count = _count_sitemap_urls("https://example.com/start", include_subdomains=False)
        assert count == 3


class TestCrawlAndSave:
    @patch("lilbee.crawler.runner.crawl_single")
    async def test_single_page(self, mock_crawl_single, isolated_env):
        mock_crawl_single.return_value = CrawlResult(url="https://example.com", markdown="# Hello")
        paths = await crawl_and_save("https://example.com", depth=0)
        assert len(paths) == 1
        assert paths[0].exists()

    @patch("lilbee.crawler.runner.crawl_recursive")
    @patch("lilbee.crawler.runner.crawl_single")
    async def test_cfg_max_depth_zero_routes_to_single_page(
        self, mock_crawl_single, mock_crawl_recursive, isolated_env
    ):
        """bb-4j2p: a cfg.crawl_max_depth ceiling of 0 means seed-only, not an error.

        depth=None inherits the ceiling; 0 must route to the single-page path
        instead of raising in the recursive resolver.
        """
        cfg.crawl_max_depth = 0
        mock_crawl_single.return_value = CrawlResult(url="https://example.com", markdown="# Hi")
        paths = await crawl_and_save("https://example.com", depth=None)
        assert len(paths) == 1
        mock_crawl_single.assert_awaited_once()
        mock_crawl_recursive.assert_not_called()

    @patch("lilbee.crawler.runner.crawl_single")
    async def test_negative_depth_is_rejected(self, mock_crawl_single, isolated_env):
        """A negative depth is invalid (use None for unbounded)."""
        with pytest.raises(ValueError, match="seed only"):
            await crawl_and_save("https://example.com", depth=-1)
        mock_crawl_single.assert_not_called()

    @patch("lilbee.crawler.runner.crawl_recursive")
    @patch("lilbee.crawler.runner.crawl_single")
    async def test_max_pages_one_is_a_one_page_recursive_crawl(
        self, mock_crawl_single, mock_crawl_recursive, isolated_env
    ):
        """max_pages=1 runs the recursive crawl capped at one page."""
        mock_crawl_recursive.return_value = [CrawlResult(url="https://example.com")]
        await crawl_and_save("https://example.com", max_pages=1)
        mock_crawl_single.assert_not_called()
        assert mock_crawl_recursive.await_args.kwargs["max_pages"] == 1

    @patch("lilbee.crawler.runner.crawl_single")
    async def test_triggers_bootstrap_when_chromium_missing(
        self, mock_crawl_single, isolated_env, monkeypatch
    ):
        """bb-wq8g: browser-mode crawl_and_save kicks off bootstrap_chromium on first use."""
        mock_crawl_single.return_value = CrawlResult(url="https://example.com", markdown="# Hi")
        monkeypatch.setattr("lilbee.crawler.bootstrap.chromium_installed", lambda: False)
        called: list[object] = []

        async def _fake_bootstrap(on_progress=None):
            called.append(on_progress)

        monkeypatch.setattr("lilbee.crawler.bootstrap.bootstrap_chromium", _fake_bootstrap)
        await crawl_and_save("https://example.com", depth=0, render_mode=CrawlRenderMode.BROWSER)
        assert called == [None]

    @patch("lilbee.crawler.runner.crawl_single")
    async def test_http_mode_skips_chromium_bootstrap(
        self, mock_crawl_single, isolated_env, monkeypatch
    ):
        """HTTP mode (the default) needs no browser, so it never bootstraps Chromium."""
        mock_crawl_single.return_value = CrawlResult(url="https://example.com", markdown="# Hi")
        monkeypatch.setattr("lilbee.crawler.bootstrap.chromium_installed", lambda: False)
        called: list[object] = []

        async def _fake_bootstrap(on_progress=None):
            called.append(on_progress)

        monkeypatch.setattr("lilbee.crawler.bootstrap.bootstrap_chromium", _fake_bootstrap)
        # Default render mode is http; bootstrap must not fire even with Chromium absent.
        await crawl_and_save("https://example.com", depth=0)
        assert called == []

    async def test_raises_backend_missing_before_bootstrap(self, isolated_env, monkeypatch):
        """Without [crawler] extra, fail fast: never trigger Chromium bootstrap."""
        from lilbee.crawler import CrawlerBackendError

        monkeypatch.setattr("lilbee.crawler.crawler_available", lambda: False)
        monkeypatch.setattr("lilbee.crawler.crawlberg_fetcher.crawler_available", lambda: False)
        monkeypatch.setattr("lilbee.crawler.bootstrap.chromium_installed", lambda: False)
        bootstrap_called: list[object] = []

        async def _fake_bootstrap(on_progress=None):
            bootstrap_called.append(on_progress)

        monkeypatch.setattr("lilbee.crawler.bootstrap.bootstrap_chromium", _fake_bootstrap)
        with pytest.raises(CrawlerBackendError, match="Web crawling is not available"):
            await crawl_and_save("https://example.com", depth=0)
        assert bootstrap_called == []

    @patch("lilbee.crawler.runner.crawl_recursive")
    async def test_recursive(self, mock_crawl_recursive, isolated_env):
        streamed = [
            CrawlResult(url="https://example.com", markdown="# Home"),
            CrawlResult(url="https://example.com/about", markdown="# About"),
        ]

        async def _fake_recursive(*args, on_result=None, **kwargs):
            # Mimic the real streaming contract: invoke the flush callback
            # per-page before returning the full list.
            if on_result is not None:
                import inspect

                for r in streamed:
                    rv = on_result(r)
                    if inspect.isawaitable(rv):
                        await rv
            return streamed

        mock_crawl_recursive.side_effect = _fake_recursive
        paths = await crawl_and_save("https://example.com", depth=2, max_pages=10)
        assert len(paths) == 2

    @patch("lilbee.crawler.runner.crawl_recursive")
    async def test_default_depth_is_recursive(self, mock_crawl_recursive, isolated_env):
        """No depth kwarg means recursive (whole-site) crawl, not single page."""
        mock_crawl_recursive.return_value = [
            CrawlResult(url="https://example.com", markdown="# Home"),
        ]
        await crawl_and_save("https://example.com")
        mock_crawl_recursive.assert_awaited_once()
        assert mock_crawl_recursive.await_args.kwargs["max_depth"] is None

    @patch("lilbee.crawler.runner.crawl_recursive")
    async def test_include_subdomains_defaults_false(self, mock_crawl_recursive, isolated_env):
        """whole-site default is exact-host scoping."""
        mock_crawl_recursive.return_value = []
        await crawl_and_save("https://example.com", depth=2)
        assert mock_crawl_recursive.await_args.kwargs["include_subdomains"] is False

    @patch("lilbee.crawler.runner.crawl_recursive")
    async def test_include_subdomains_threaded_through(self, mock_crawl_recursive, isolated_env):
        """Opting into subdomain scope reaches the recursive crawl."""
        mock_crawl_recursive.return_value = []
        await crawl_and_save("https://example.com", depth=2, include_subdomains=True)
        assert mock_crawl_recursive.await_args.kwargs["include_subdomains"] is True

    @patch("lilbee.crawler.runner.crawl_recursive")
    async def test_cancel_threaded_to_crawl_recursive(self, mock_crawl_recursive, isolated_env):
        """The cancel event is threaded through to crawl_recursive for BFS abort."""
        import threading

        mock_crawl_recursive.return_value = []
        cancel = threading.Event()
        await crawl_and_save("https://example.com", depth=2, cancel=cancel)
        assert mock_crawl_recursive.call_args[1]["cancel"] is cancel

    @patch("lilbee.crawler.runner.crawl_single")
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

    @patch("lilbee.crawler.runner.crawl_single")
    async def test_single_page_failure_emits_page_failed(self, mock_crawl_single, isolated_env):
        """A failed single-page fetch emits crawl_page_failed naming the error."""
        mock_crawl_single.return_value = CrawlResult(
            url="https://example.com", success=False, error="403 Forbidden"
        )
        events = []

        def on_progress(event_type, data):
            events.append((str(event_type), data))

        await crawl_and_save("https://example.com", depth=0, on_progress=on_progress)
        failed = [d for t, d in events if t == "crawl_page_failed"]
        assert [(e.url, e.reason) for e in failed] == [("https://example.com", "403 Forbidden")]

    @patch("lilbee.crawler.runner.crawl_single")
    async def test_updates_metadata(self, mock_crawl_single, isolated_env):
        mock_crawl_single.return_value = CrawlResult(
            url="https://example.com/page", markdown="# Test"
        )
        await crawl_and_save("https://example.com/page", depth=0)
        meta = load_crawl_metadata()
        assert "https://example.com/page" in meta

    @patch("lilbee.crawler.runner.crawl_single")
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

    @patch("lilbee.crawler.runner.crawl_single")
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
        cfg.crawl_max_concurrent = 5
        reset_services()
        sem = _get_crawl_semaphore()
        assert sem is not None
        assert sem._value == 5
        reset_services()

    async def test_semaphore_defaults_to_cpu_count(self, isolated_env):
        """Default concurrency matches CPU count."""
        import os

        cfg.crawl_max_concurrent = os.cpu_count() or 4
        reset_services()
        sem = _get_crawl_semaphore()
        assert sem is not None
        assert sem._value == (os.cpu_count() or 4)
        reset_services()

    async def test_semaphore_unlimited_when_zero(self, isolated_env):
        """Setting crawl_max_concurrent=0 disables the semaphore."""
        cfg.crawl_max_concurrent = 0
        reset_services()
        assert _get_crawl_semaphore() is None
        reset_services()

    @patch("lilbee.crawler.runner.crawl_single")
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
        cfg.crawl_sync_interval = 0
        reset_services()
        sync_state = get_services().crawler_sync_state
        sync_state.last_run = 0.0

        tasks: set[asyncio.Task[None]] = set()
        with patch("lilbee.data.ingest.sync", new_callable=AsyncMock) as mock_sync:
            await _maybe_periodic_sync(tasks)
            mock_sync.assert_not_awaited()
        assert not tasks

    async def test_sync_skipped_when_already_running(self, isolated_env):
        """No new sync is started if one is already in progress."""
        cfg.crawl_sync_interval = 1
        reset_services()
        sync_state = get_services().crawler_sync_state
        sync_state.last_run = 0.0
        sync_state.lock.acquire()  # simulate already-running

        tasks: set[asyncio.Task[None]] = set()
        try:
            with patch("lilbee.data.ingest.sync", new_callable=AsyncMock) as mock_sync:
                await _maybe_periodic_sync(tasks)
                mock_sync.assert_not_awaited()
        finally:
            sync_state.lock.release()

    async def test_sync_skipped_when_interval_not_elapsed(self, isolated_env):
        """No sync fires if the interval hasn't elapsed since last sync."""
        import time

        cfg.crawl_sync_interval = 9999
        reset_services()
        sync_state = get_services().crawler_sync_state
        sync_state.last_run = time.monotonic()

        tasks: set[asyncio.Task[None]] = set()
        with patch("lilbee.data.ingest.sync", new_callable=AsyncMock) as mock_sync:
            await _maybe_periodic_sync(tasks)
            mock_sync.assert_not_awaited()

    async def test_sync_fires_when_interval_elapsed(self, isolated_env):
        """Sync fires as a background task when interval has elapsed."""
        cfg.crawl_sync_interval = 1
        reset_services()
        sync_state = get_services().crawler_sync_state
        sync_state.last_run = 0.0

        tasks: set[asyncio.Task[None]] = set()
        mock_sync = AsyncMock()
        with patch("lilbee.data.ingest.sync", mock_sync):
            await _maybe_periodic_sync(tasks)
            # Drain the spawned task so the awaited assertion is deterministic.
            await asyncio.gather(*tasks, return_exceptions=True)
            mock_sync.assert_awaited_once()

    async def test_sync_failure_resets_running_flag(self, isolated_env):
        """If sync raises, the sync lock is released so future syncs can proceed."""
        cfg.crawl_sync_interval = 1
        reset_services()
        sync_state = get_services().crawler_sync_state
        sync_state.last_run = 0.0

        tasks: set[asyncio.Task[None]] = set()
        mock_sync = AsyncMock(side_effect=RuntimeError("sync failed"))
        with patch("lilbee.data.ingest.sync", mock_sync):
            await _maybe_periodic_sync(tasks)
            await asyncio.gather(*tasks, return_exceptions=True)

        # Lock should be released after failure
        assert sync_state.lock.acquire(blocking=False)
        sync_state.lock.release()


class TestServicesCrawlerSemaphore:
    def test_services_crawler_semaphore_rebuilds_on_reset(self, isolated_env):
        """``reset_services`` produces a fresh semaphore matching the cfg limit."""
        cfg.crawl_max_concurrent = 3
        reset_services()
        first = get_services().crawler_semaphore
        assert first is not None
        assert first._value == 3
        reset_services()
        second = get_services().crawler_semaphore
        assert second is not None
        assert second._value == 3
        assert first is not second

    def test_services_crawler_semaphore_none_when_unlimited(self, isolated_env):
        """``crawl_max_concurrent <= 0`` leaves ``crawler_semaphore`` as ``None``."""
        cfg.crawl_max_concurrent = 0
        reset_services()
        assert get_services().crawler_semaphore is None


class TestCrawlAndSaveSemaphore:
    @patch("lilbee.crawler.runner.crawl_single")
    async def test_semaphore_acquired_and_released(self, mock_crawl_single, isolated_env):
        """When crawl_max_concurrent > 0, sem.acquire/release are called."""
        mock_crawl_single.return_value = CrawlResult(url="https://example.com", markdown="# Hello")
        cfg.crawl_max_concurrent = 2
        reset_services()

        paths = await crawl_and_save("https://example.com", depth=0)
        assert len(paths) == 1

        # Verify semaphore is still at full capacity (released).
        sem = get_services().crawler_semaphore
        assert sem is not None
        assert sem._value == 2
        reset_services()


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

    def test_normalize_crawled_markdown_collapses_reference_links(self):
        ref = "[[2]](https://en.wikipedia.org/wiki/Foo#cite_note-2)"
        md = f"Car of the Year.{ref} Production ended."
        # Double brackets collapse to single; text and URL are preserved.
        expected = (
            "Car of the Year.[2](https://en.wikipedia.org/wiki/Foo#cite_note-2) Production ended."
        )
        assert normalize_crawled_markdown(md) == expected
        # Ordinary single-bracket links are left untouched.
        keep = 'See the [trim package](https://en.wikipedia.org/wiki/Trim "title") here.'
        assert normalize_crawled_markdown(keep) == keep

    def test_writes_normalized_markdown(self, isolated_env):
        from lilbee.crawler.save import _save_single_result

        raw = "9C1[[1]](https://en.wikipedia.org/wiki/Caprice#cite_note-1) introduced for 1986."
        expected = "9C1[1](https://en.wikipedia.org/wiki/Caprice#cite_note-1) introduced for 1986."
        outcome = _save_single_result(CrawlResult(url="https://example.com/ref", markdown=raw), {})
        assert outcome is not None
        assert outcome.path.read_text(encoding="utf-8") == expected
        assert outcome.content_hash == content_hash(expected)

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

    async def test_cancel_preserves_written_pages(self, isolated_env):
        """A cancelled recursive crawl keeps the pages it already streamed."""
        import asyncio as _asyncio
        import threading

        cancel = threading.Event()

        async def _gen():
            yield cb.page("https://example.com/p1", "# P1")
            await _asyncio.sleep(0)
            yield cb.page("https://example.com/p2", "# P2")
            cancel.set()
            await _asyncio.sleep(0)
            yield cb.page("https://example.com/p3", "# P3")
            yield cb.page("https://example.com/p4", "# P4")

        stub = cb.StubCrawlberg(_gen)

        with stub.installed():
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
                yield cb.page(f"https://example.com/p{i}", f"# P{i}")

        stub = cb.StubCrawlberg(_gen)

        real_save = _save_single_result

        def _wrapped(result, meta):
            call_order.append(f"flush-{result.url.rsplit('/', 1)[-1]}")
            return real_save(result, meta)

        with (
            stub.installed(),
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
            yield cb.page("https://example.com/p1", "# Same")
            await _asyncio.sleep(0)
            yield cb.page("https://example.com/p2", "# NewPage")

        stub = cb.StubCrawlberg(_gen)

        with stub.installed():
            paths = await crawl_and_save("https://example.com", depth=2, max_pages=10)

        # p1 is skipped (unchanged), p2 is written.
        assert len(paths) == 1
        assert paths[0].name == "index.md"
        assert paths[0].read_text(encoding="utf-8") == "# NewPage"

    @patch("lilbee.crawler.runner.crawl_single")
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
                yield cb.page(url, md)

        stub = cb.StubCrawlberg(_gen)

        with stub.installed():
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
            yield cb.page("https://example.com/p1", "# P1")
            cancel.set()
            await _asyncio.sleep(0)
            yield cb.page("https://example.com/p2", "# P2")

        stub = cb.StubCrawlberg(_gen)

        with (
            stub.installed(),
            patch(
                "lilbee.crawler.runner._maybe_periodic_sync", new_callable=AsyncMock
            ) as mock_sync,
        ):
            await crawl_and_save("https://example.com", depth=2, max_pages=10, cancel=cancel)
        mock_sync.assert_not_awaited()

    async def test_success_triggers_auto_sync(self, isolated_env):
        """On a clean (non-cancelled) crawl, _maybe_periodic_sync IS awaited."""
        import asyncio as _asyncio

        async def _gen():
            await _asyncio.sleep(0)
            yield cb.page("https://example.com/p1", "# P1")

        stub = cb.StubCrawlberg(_gen)

        with (
            stub.installed(),
            patch(
                "lilbee.crawler.runner._maybe_periodic_sync", new_callable=AsyncMock
            ) as mock_sync,
        ):
            await crawl_and_save("https://example.com", depth=2, max_pages=10)
        mock_sync.assert_awaited_once()

    async def test_crawl_and_save_drains_background_tasks(self, isolated_env):
        """crawl_and_save awaits its locally-spawned periodic-sync task before returning."""

        async def _gen():
            await asyncio.sleep(0)
            yield cb.page("https://example.com/p1", "# P1")

        stub = cb.StubCrawlberg(_gen)

        sync_completed = asyncio.Event()

        async def _short_sync(*_args, **_kwargs):
            await asyncio.sleep(0.05)
            sync_completed.set()

        cfg.crawl_sync_interval = 1
        reset_services()
        sync_state = get_services().crawler_sync_state
        sync_state.last_run = 0.0

        # Snapshot tasks currently on this loop so we can assert no
        # crawler-spawned task survives the call.
        before = set(asyncio.all_tasks())

        with (
            stub.installed(),
            patch("lilbee.data.ingest.sync", _short_sync),
        ):
            await crawl_and_save("https://example.com", depth=0)

        # crawl_and_save must not return before its periodic-sync task finished.
        assert sync_completed.is_set()
        # No new pending tasks leak past the call.
        leaked = {t for t in asyncio.all_tasks() if t not in before and not t.done()}
        assert not leaked

    async def test_concurrent_crawl_and_save_calls_isolated_tasks(self, isolated_env):
        """Two concurrent crawl_and_save calls each own their own local task set."""
        cfg.crawl_max_concurrent = 0  # don't serialize the two concurrent calls
        cfg.crawl_sync_interval = 1  # spawn a periodic-sync per call
        reset_services()
        sync_state = get_services().crawler_sync_state
        sync_state.last_run = 0.0

        async def _short_sync(*_args, **_kwargs):
            await asyncio.sleep(0)

        async def _fake_single(url: str, *, on_progress=None, render_mode=None) -> CrawlResult:
            await asyncio.sleep(0)
            return CrawlResult(url=url, markdown=f"# {url}")

        before = set(asyncio.all_tasks())

        with (
            patch("lilbee.crawler.runner.crawl_single", side_effect=_fake_single),
            patch("lilbee.data.ingest.sync", _short_sync),
        ):
            results = await asyncio.gather(
                crawl_and_save("https://example.com/a", depth=0),
                crawl_and_save("https://example.com/b", depth=0),
            )
        assert len(results) == 2
        assert all(len(paths) == 1 for paths in results)
        # Each call's local task set was drained before returning, so no
        # crawler-spawned task leaks past either invocation.
        leaked = {t for t in asyncio.all_tasks() if t not in before and not t.done()}
        assert not leaked

    async def test_metadata_flush_is_batched(self, isolated_env):
        """_flush_metadata fires every METADATA_FLUSH_INTERVAL pages, not every page."""
        import asyncio as _asyncio

        from lilbee.crawler import METADATA_FLUSH_INTERVAL

        total_pages = METADATA_FLUSH_INTERVAL + 3

        async def _gen():
            for i in range(1, total_pages + 1):
                await _asyncio.sleep(0)
                yield cb.page(f"https://example.com/p{i}", f"# P{i}")

        stub = cb.StubCrawlberg(_gen)

        with (
            stub.installed(),
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
                yield cb.page(f"https://example.com/p{i}", f"# P{i}")

        stub = cb.StubCrawlberg(_gen)

        with (
            stub.installed(),
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
                yield cb.page(f"https://example.com/p{i}", f"# P{i}")

        stub = cb.StubCrawlberg(_gen)

        call_count = {"n": 0}

        def failing_save(result, meta):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError("disk full")
            return  # second page falls through unchanged

        with (
            stub.installed(),
            patch("lilbee.crawler.save._save_single_result", side_effect=failing_save),
        ):
            # Must not raise even though the first page write fails.
            paths = await crawl_and_save("https://example.com", depth=2, max_pages=10)
        assert paths == []

    @patch("lilbee.crawler.runner.crawl_single")
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
            yield cb.page("https://example.com/p1", "# P1")
            await _asyncio.sleep(0)

        stub = cb.StubCrawlberg(_gen)

        real_flush = save_crawl_metadata
        call_count = {"n": 0}

        def flush_or_fail(meta):
            call_count["n"] += 1
            # Let interval-boundary flushes succeed; fail only the post-loop flush.
            if call_count["n"] == 1 and len(meta) >= 1:
                raise OSError("disk full")
            real_flush(meta)

        with (
            stub.installed(),
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
            yield cb.page("https://example.com/p1", "# P1")
            cancel.set()
            await _asyncio.sleep(0)
            yield cb.page("https://example.com/p2", "# P2")

        stub = cb.StubCrawlberg(_gen)

        with (
            stub.installed(),
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
            yield cb.page("https://example.com/empty", "")

        stub = cb.StubCrawlberg(_gen)

        with (
            stub.installed(),
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
            yield cb.page("https://example.com/p1", "# P1")
            cancel.set()
            await _asyncio.sleep(0)
            yield cb.page("https://example.com/p2", "# P2")

        stub = cb.StubCrawlberg(_gen)

        done_events: list = []

        def on_progress(event_type, data):
            if event_type == EventType.CRAWL_DONE:
                done_events.append(data)

        with stub.installed():
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


class _FakeStream:
    async def readline(self) -> bytes:
        return b""  # immediate EOF, so the drains finish


class _FakeProc:
    """Minimal asyncio subprocess stand-in whose wait() can be made to hang."""

    def __init__(self, *, hang_wait: bool = False, returncode: int = 0) -> None:
        self.stdout = _FakeStream()
        self.stderr = _FakeStream()
        self._returncode: int | None = None if hang_wait else returncode
        self._exit = asyncio.Event()
        self.wait_entered = asyncio.Event()  # set once wait() is actually awaited
        self.terminated = False
        self.killed = False
        if not hang_wait:
            self._exit.set()

    @property
    def returncode(self) -> int | None:
        return self._returncode

    def terminate(self) -> None:
        self.terminated = True
        self._returncode = 0
        self._exit.set()

    def kill(self) -> None:
        self.killed = True
        self._returncode = -9
        self._exit.set()

    async def wait(self) -> int | None:
        self.wait_entered.set()
        await self._exit.wait()
        return self._returncode


class TestChromiumBootstrapTermination:
    async def test_terminate_process_terminates_running(self):
        from lilbee.crawler.bootstrap import _terminate_process

        proc = _FakeProc(hang_wait=True)
        await _terminate_process(proc)
        assert proc.terminated
        assert not proc.killed

    async def test_terminate_process_noop_when_already_exited(self):
        from lilbee.crawler.bootstrap import _terminate_process

        proc = _FakeProc(returncode=0)
        await _terminate_process(proc)
        assert not proc.terminated

    async def test_terminate_process_kills_when_terminate_ignored(self, monkeypatch):
        from lilbee.crawler import bootstrap

        monkeypatch.setattr(bootstrap, "_TERMINATE_TIMEOUT_S", 0.05)

        class _Stubborn(_FakeProc):
            def terminate(self) -> None:
                self.terminated = True  # ignores the signal; stays running

        proc = _Stubborn(hang_wait=True)
        await bootstrap._terminate_process(proc)
        assert proc.terminated
        assert proc.killed

    async def test_bootstrap_cancel_terminates_install(self, monkeypatch):
        # Cancelling the bootstrap (SSE client disconnect) must not
        # leave the chromium install running orphaned.
        from lilbee.crawler import bootstrap

        monkeypatch.setattr(bootstrap, "chromium_installed", lambda: False)
        monkeypatch.setattr(bootstrap, "_resolve_playwright_runner", lambda: (["pw"], {}))
        proc = _FakeProc(hang_wait=True)

        async def _fake_exec(*_args, **_kwargs):
            return proc

        monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

        task = asyncio.create_task(bootstrap.bootstrap_chromium())
        # Cancel only once the task is genuinely parked inside proc.wait() (the
        # finally that must terminate it has been entered), not merely scheduled.
        await asyncio.wait_for(proc.wait_entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert proc.terminated  # the orphaned install was terminated
