"""Tests for the web crawling module."""

import asyncio
import contextlib
import hashlib
import json
import os
import sys
import threading
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lilbee.app.services import get_services, reset_services, set_services
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
    save,
    url_to_filename,
    validate_crawl_url,
)
from lilbee.crawler import bootstrap as bootstrap_mod
from lilbee.crawler.bootstrap import CrawlEngineRefusedError, CrawlerBrowserError
from lilbee.crawler.runner import (
    _get_crawl_semaphore,
    _maybe_periodic_sync,
)
from lilbee.crawler.save import (
    CrawlRecord,
    _claim,
    _replace_atomically,
    canonical_address,
    load_crawl_record,
    normalize_crawled_markdown,
    save_page,
)
from lilbee.crawler.task import CrawlTask, TaskStatus, run_crawl
from lilbee.runtime.progress import EventType
from tests import _crawlberg_stub as cb
from tests._crawl_meta import write_crawl_metadata
from tests._mock_effects import repeat_last
from tests._private_mode import posix_only
from tests.conftest import make_mock_services


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


def _named(readable: str, url: str) -> str:
    """The saved path of *url*: *readable*, ``~`` and the hash of its canonical address."""
    digest = hashlib.sha256(canonical_address(url).encode("utf-8")).hexdigest()[:16]
    return f"{readable}~{digest}.md"


# An address and the readable part of the path it is saved under.
SAVED_NAMES = [
    pytest.param("https://example.com/page", "example.com/page", id="page"),
    pytest.param(
        "https://docs.python.org/3/tutorial/", "docs.python.org/3/tutorial/index", id="slash"
    ),
    pytest.param("https://example.com/", "example.com/index", id="root"),
    pytest.param("https://example.com", "example.com/index", id="root-no-slash"),
    pytest.param("https://example.com/docs/guide.html", "example.com/docs/guide.html", id="ext"),
    pytest.param("https://example.com/page?q=1", "example.com/page", id="query"),
    pytest.param("https://example.com/page#section", "example.com/page", id="fragment"),
    pytest.param("https://example.com/t/caf%C3%A9", "example.com/t/café", id="escaped-letter"),
    pytest.param("https://example.com/t/café", "example.com/t/café", id="decomposed"),
    pytest.param("https://example.com/t/raw space", "example.com/t/raw-space", id="space"),
    pytest.param("https://example.com/t/a<b>:c", "example.com/t/a-b-c", id="reserved"),
    pytest.param("https://example.com/t/..x..", "example.com/t/x", id="dot-ends"),
    pytest.param("https://example.com/t/%E2%80%AE", "example.com/t/_", id="nothing-readable"),
    pytest.param(
        "https://example.com/con/Com1.txt/nul", "example.com/_con/_Com1.txt/_nul", id="device"
    ),
    pytest.param("https://example.com/a//b", "example.com/a/b", id="empty-segment"),
    pytest.param("https://example.com/s/a.md%3F/z", "example.com/s/a-md/z", id="folder-suffix"),
    pytest.param("https://example.com/s/A.MD./z", "example.com/s/A-MD/z", id="folder-suffix-case"),
    pytest.param("https://example.com/s/a.md", "example.com/s/a.md", id="page-suffix"),
    pytest.param("https://evil.com/../../etc/passwd", "evil.com/etc/passwd", id="dot-segments"),
    pytest.param("https://bücher.example/t", "bücher.example/t", id="unicode-host"),
    pytest.param("https://xn--bcher-kva.example/t", "bücher.example/t", id="punycode-host"),
    pytest.param("https://[0:0::1]:8443/t", "0-0-0-0-0-0-0-1-8443/t", id="ipv6-and-port"),
    pytest.param("http://[2001:DB8::a]/t", "2001-db8-0-0-0-0-0-a/t", id="ipv6"),
    pytest.param("not a url", "unknown/index", id="not-a-url"),
]


class TestUrlToFilename:
    @pytest.mark.parametrize(("url", "readable"), SAVED_NAMES)
    def test_an_address_is_saved_under_its_readable_name_and_hash(self, url: str, readable: str):
        assert url_to_filename(url) == _named(readable, url)

    def test_distinct_queries_map_to_distinct_files(self):
        a = url_to_filename("https://example.com/page?q=1")
        b = url_to_filename("https://example.com/page?q=2")
        assert a != b

    def test_a_long_part_is_cut_to_80_bytes(self):
        name = url_to_filename("https://example.com/" + "漢" * 40)
        assert name == _named("example.com/" + "漢" * 26, "https://example.com/" + "漢" * 40)

    def test_the_deepest_folders_go_first_when_the_path_is_too_long(self):
        url = "https://example.com/" + "folder/" * 40 + "page"
        name = url_to_filename(url)
        assert len(name.encode("utf-8")) <= 200
        assert name.startswith("example.com/folder/")
        assert name.endswith(_named("folder/page", url))

    def test_the_hash_is_the_start_of_the_sha256_of_the_canonical_address(self):
        url = "https://example.com/T/x?b=2&a=1#top"
        digest = hashlib.sha256(b"https://example.com/T/x?a=1&b=2").hexdigest()[:16]
        assert url_to_filename(url) == f"example.com/T/x~{digest}.md"


# Two spellings of one page, as crawlberg 1.11.1 keys it: (one spelling, the key).
SAME_PAGE = [
    pytest.param("https://example.com/t/a%2Db", "https://example.com/t/a-b", id="hyphen"),
    pytest.param("https://example.com/t/a%2db", "https://example.com/t/a-b", id="lower-hex"),
    pytest.param("https://example.com/t/%41%7a%30", "https://example.com/t/Az0", id="alnum"),
    pytest.param(
        "https://example.com/t/a%2Eb%5Fc%7Ed", "https://example.com/t/a.b_c~d", id="marks"
    ),
    pytest.param(
        "https://example.com/t/caf%c3%a9", "https://example.com/t/caf%C3%A9", id="hex-case"
    ),
    pytest.param(
        "https://example.com/t/q?n=%61%2fb", "https://example.com/t/q?n=a%2Fb", id="query"
    ),
    pytest.param("https://example.com/t/%2e%2e/up", "https://example.com/up", id="dots"),
    pytest.param(
        "https://example.com/t/naïve", "https://example.com/t/na%C3%AFve", id="raw-letter"
    ),
    pytest.param(
        "https://example.com/t/raw space", "https://example.com/t/raw%20space", id="raw-space"
    ),
    pytest.param("https://example.com/t/a\x00b", "https://example.com/t/a%00b", id="raw-nul"),
    pytest.param("https://example.com/t/b\\s", "https://example.com/t/b/s", id="backslash"),
    pytest.param("https://example.com/t/a{b}", "https://example.com/t/a%7Bb%7D", id="braces"),
    pytest.param("https://Example.COM:443/t/x#top", "https://example.com/t/x", id="host-port"),
    pytest.param("https://bücher.example/t", "https://xn--bcher-kva.example/t", id="idn"),
    pytest.param("https://[0:0::1]/t", "https://[::1]/t", id="ipv6"),
    pytest.param(
        "https://example.com/t/q?b=2&a=1&a=0", "https://example.com/t/q?a=1&a=0&b=2", id="sort"
    ),
    pytest.param(
        "https://example.com/t/ca%5eret", "https://example.com/t/ca%5Eret", id="caret-escape"
    ),
    pytest.param("not a url", "not a url", id="not-a-url"),
]
# Two addresses crawlberg 1.11.1 keys as two pages: no rule makes them one.
DIFFERENT_PAGES = [
    pytest.param("https://example.com/t/ca^ret", "https://example.com/t/ca%5Eret", id="caret"),
    pytest.param("https://example.com/t/pi|pe", "https://example.com/t/pi%7Cpe", id="pipe"),
    pytest.param("https://example.com/t/a+b", "https://example.com/t/a%20b", id="plus-space"),
    pytest.param("https://example.com/t/a+b", "https://example.com/t/a%2Bb", id="plus-escape"),
    pytest.param("https://example.com/t/e%2Fs", "https://example.com/t/e/s", id="encoded-slash"),
    pytest.param("https://example.com/t/a%3Ab", "https://example.com/t/a:b", id="kept-escape"),
    pytest.param("https://example.com/t/ts", "https://example.com/t/ts/", id="trailing-slash"),
    pytest.param("https://example.com/t//x", "https://example.com/t/x", id="empty-segment"),
    pytest.param("https://example.com/t/Case", "https://example.com/t/case", id="letter-case"),
    pytest.param("https://example.com/t/doc;v=1", "https://example.com/t/doc;v=2", id="params"),
    pytest.param("https://example.com:8443/t", "https://example.com/t", id="port"),
    pytest.param("http://example.com/t/scheme", "https://example.com/t/scheme", id="scheme"),
    pytest.param("https://example.com/t/q?", "https://example.com/t/q", id="empty-query"),
    pytest.param("https://user@example.com/t/user", "https://example.com/t/user", id="user-info"),
    pytest.param("https://example.com/t/x%a", "https://example.com/t/x%A", id="one-hex-digit"),
    pytest.param("https://example.com/t/x%zz", "https://example.com/t/x%ZZ", id="not-hex"),
    pytest.param("https://example.com/t/%%341", "https://example.com/t/%41", id="escape-once"),
    pytest.param("https://example.com/t/q?a=1&b=2", "https://example.com/t/q?a=2&b=1", id="values"),
]


class TestCanonicalAddress:
    @pytest.mark.parametrize(("spelling", "key"), SAME_PAGE)
    def test_a_spelling_takes_the_key_crawlberg_gives_its_page(self, spelling: str, key: str):
        assert canonical_address(spelling) == key
        assert canonical_address(key) == key

    @pytest.mark.parametrize(("one", "other"), DIFFERENT_PAGES)
    def test_two_pages_keep_two_keys_and_two_names(self, one: str, other: str):
        assert canonical_address(one) != canonical_address(other)
        assert url_to_filename(one).lower() != url_to_filename(other).lower()


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
        write_crawl_metadata(meta)
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

    def test_atomic_write_cleans_up_on_error(self, isolated_env):
        """If the atomic write fails, the tmp file is removed and error re-raised."""
        with (
            patch("lilbee.crawler.save.Path.replace", side_effect=OSError("disk full")),
            pytest.raises(OSError, match="disk full"),
        ):
            _replace_atomically(cfg.data_dir / "crawl_pages.jsonl", "{}\n")
        assert list(cfg.data_dir.iterdir()) == []

    def test_atomic_write_error_before_the_temporary_file_exists_is_raised(self, isolated_env):
        """A failure to make the temporary file leaves the target untouched and re-raises."""
        with (
            patch("lilbee.crawler.save.tempfile.NamedTemporaryFile", side_effect=OSError("full")),
            pytest.raises(OSError, match="full"),
        ):
            _replace_atomically(cfg.data_dir / "crawl_pages.jsonl", "{}\n")
        assert list(cfg.data_dir.iterdir()) == []


STORED_TEXT = "# Stored"


def _stored(*urls: str) -> dict[str, CrawlMeta]:
    """Earlier crawl metadata of *urls*, each under its own earlier name, with the stored text."""
    return {
        url: CrawlMeta(
            file=f"earlier/{hashlib.sha256(url.encode()).hexdigest()[:8]}/index.md",
            content_hash=content_hash(STORED_TEXT),
            crawled_at="2026-01-01T00:00:00+00:00",
        )
        for url in urls
    }


def _save_files(meta: dict[str, CrawlMeta], text: bytes = STORED_TEXT.encode()) -> None:
    """Write a file under ``_web`` for each entry of *meta*."""
    for entry in meta.values():
        path = cfg.documents_dir / "_web" / entry.file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text)


def _write_web(name: str, text: str) -> Path:
    """Write *text* at *name* under ``_web``; the path."""
    path = cfg.documents_dir / "_web" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _record_lines(*entries: tuple[str, CrawlMeta]) -> str:
    """Page record text with one line per (address, entry)."""
    return "".join(
        json.dumps({"url": address, **vars(entry)}, ensure_ascii=False) + "\n"
        for address, entry in entries
    )


def _pages_file() -> Path:
    """The page record file."""
    return cfg.data_dir / "crawl_pages.jsonl"


@pytest.fixture
def mock_services():
    """A mock Services container whose store records removed index sources."""
    services = make_mock_services()
    set_services(services)
    return services


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


def _failing_stream(message: str):
    """A crawl script whose stream raises before its first event, as a dropped fetch does."""

    async def script():
        raise RuntimeError(message)
        yield  # makes this an async generator

    return script


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
        with cb.StubCrawlberg(_failing_stream("timeout")).installed():
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
        with cb.StubCrawlberg(_failing_stream("still broken after bootstrap")).installed():
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


# The directory and file each Playwright build unpacks the headless shell into.
_LINUX_X64_SHELL = ("chrome-headless-shell-linux64", "chrome-headless-shell")
_PLAYWRIGHT_SHELLS = [
    pytest.param(_LINUX_X64_SHELL, id="linux-x64"),
    pytest.param(("chrome-linux", "headless_shell"), id="linux-arm64"),
    pytest.param(("chrome-headless-shell-mac-x64", "chrome-headless-shell"), id="mac-x64"),
    pytest.param(("chrome-headless-shell-mac-arm64", "chrome-headless-shell"), id="mac-arm64"),
    pytest.param(("chrome-headless-shell-win64", "chrome-headless-shell.exe"), id="win-x64"),
]


def _install_shell(
    root: Path,
    revision: str,
    layout: tuple[str, ...] = _LINUX_X64_SHELL,
    *,
    executable: bool = True,
    complete: bool = True,
) -> Path:
    """Lay out a headless shell binary the way Playwright installs it."""
    directory = root / f"{bootstrap_mod._HEADLESS_SHELL_DIR_PREFIX}{revision}"
    shell = directory.joinpath(*layout)
    shell.parent.mkdir(parents=True)
    shell.write_bytes(b"")
    shell.chmod(0o755 if executable else 0o644)
    if complete:
        (directory / bootstrap_mod._INSTALL_COMPLETE_MARKER).write_bytes(b"")
    return shell


class TestChromiumInstalledMatching:
    """``chromium_installed`` looks for the headless shell at the bundled Playwright's revision."""

    def test_matches_expected_revision_when_present(self, tmp_path, monkeypatch):
        from lilbee.crawler import bootstrap

        executable = _install_shell(tmp_path, "1208")
        monkeypatch.setattr(bootstrap, "_browsers_cache_path", lambda: tmp_path)
        monkeypatch.setattr(bootstrap, "_expected_chromium_revision", lambda: "1208")

        assert bootstrap.chromium_installed() is True
        assert bootstrap.headless_shell_executable() == executable

    def test_rejects_wrong_revision(self, tmp_path, monkeypatch):
        from lilbee.crawler import bootstrap

        _install_shell(tmp_path, "1217")
        monkeypatch.setattr(bootstrap, "_browsers_cache_path", lambda: tmp_path)
        monkeypatch.setattr(bootstrap, "_expected_chromium_revision", lambda: "1208")

        assert bootstrap.chromium_installed() is False

    def test_falls_back_to_any_revision_when_expected_unknown(self, tmp_path, monkeypatch):
        from lilbee.crawler import bootstrap

        _install_shell(tmp_path, "1217")
        monkeypatch.setattr(bootstrap, "_browsers_cache_path", lambda: tmp_path)
        monkeypatch.setattr(bootstrap, "_expected_chromium_revision", lambda: None)

        assert bootstrap.chromium_installed() is True

    def test_full_chromium_build_does_not_count(self, tmp_path, monkeypatch):
        """Only the headless shell is launched, so the full build alone is not installed."""
        from lilbee.crawler import bootstrap

        _install_shell(tmp_path, "1208").unlink()
        (tmp_path / "chromium-1208").mkdir()
        monkeypatch.setattr(bootstrap, "_browsers_cache_path", lambda: tmp_path)
        monkeypatch.setattr(bootstrap, "_expected_chromium_revision", lambda: "1208")

        assert bootstrap.chromium_installed() is False

    @pytest.mark.parametrize("layout", _PLAYWRIGHT_SHELLS)
    def test_finds_the_shell_in_every_playwright_build(self, tmp_path, monkeypatch, layout):
        """Found by file name, whichever build Playwright installed on the host."""
        from lilbee.crawler import bootstrap

        shell = _install_shell(tmp_path, "1208", layout)
        (shell.parent / "libEGL.so").write_bytes(b"")
        monkeypatch.setattr(bootstrap, "_browsers_cache_path", lambda: tmp_path)
        monkeypatch.setattr(bootstrap, "_expected_chromium_revision", lambda: "1208")

        assert bootstrap.headless_shell_executable() == shell

    @posix_only
    def test_a_shell_file_that_cannot_run_counts_as_installed(self, tmp_path, monkeypatch):
        from lilbee.crawler import bootstrap

        shell = _install_shell(tmp_path, "1208", executable=False)
        monkeypatch.setattr(bootstrap, "_browsers_cache_path", lambda: tmp_path)
        monkeypatch.setattr(bootstrap, "_expected_chromium_revision", lambda: "1208")

        assert not os.access(shell, os.X_OK)
        assert bootstrap.headless_shell_executable() == shell
        assert bootstrap.chromium_installed() is True

    @posix_only
    @pytest.mark.parametrize("shape", ["dangling-link", "directory"])
    def test_a_shell_path_that_is_no_file_counts_as_installed(self, tmp_path, monkeypatch, shape):
        from lilbee.crawler import bootstrap

        shell = _install_shell(tmp_path, "1208")
        shell.unlink()
        if shape == "directory":
            shell.mkdir()
        else:
            shell.symlink_to(tmp_path / "no-such-target")
        monkeypatch.setattr(bootstrap, "_browsers_cache_path", lambda: tmp_path)
        monkeypatch.setattr(bootstrap, "_expected_chromium_revision", lambda: "1208")

        assert not shell.is_file()
        assert bootstrap.headless_shell_executable() == shell
        assert bootstrap.chromium_installed() is True

    def test_an_install_that_never_finished_does_not_count(self, tmp_path, monkeypatch):
        """Playwright marks a finished install; a shell without the marker may be half unpacked."""
        from lilbee.crawler import bootstrap

        _install_shell(tmp_path, "1208", complete=False)
        monkeypatch.setattr(bootstrap, "_browsers_cache_path", lambda: tmp_path)
        monkeypatch.setattr(bootstrap, "_expected_chromium_revision", lambda: "1208")

        assert bootstrap.headless_shell_executable() is None

    def test_an_empty_cache_has_no_shell(self, tmp_path, monkeypatch):
        from lilbee.crawler import bootstrap

        monkeypatch.setattr(bootstrap, "_browsers_cache_path", lambda: tmp_path)
        monkeypatch.setattr(bootstrap, "_expected_chromium_revision", lambda: "1208")

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
        _install_shell(browsers, "1234")
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
        with cb.StubCrawlberg(_failing_stream("network error")).installed():
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
    @patch("lilbee.crawler.runner.crawl_recursive")
    async def test_a_page_is_saved_and_recorded_under_its_canonical_address(
        self, mock_crawl_recursive, isolated_env, mock_services
    ):
        stored = "https://example.com/t/a-b"
        same = "https://example.com/t/caf%C3%A9"
        new = "https://example.com/t/new%2Fpage"
        before = _stored(stored, same)
        write_crawl_metadata(before)
        _save_files(before)
        web = cfg.documents_dir / "_web"
        reported = [
            CrawlResult(url="https://example.com/t/a%2Db", markdown="# Hyphen"),
            CrawlResult(url="https://example.com/t/café", markdown="# Cafe"),
            CrawlResult(url=new, markdown="# New"),
        ]

        async def crawl(url: str, **kwargs: object) -> list[CrawlResult]:
            for result in reported:
                await kwargs["on_result"](result)  # type: ignore[operator]
            return reported

        mock_crawl_recursive.side_effect = crawl
        paths = await crawl_and_save("https://example.com/", depth=1)
        assert paths == [web / url_to_filename(url) for url in (stored, same, new)]
        assert paths[0].read_text(encoding="utf-8") == "# Hyphen"
        pages = load_crawl_record().pages
        assert set(pages) == {stored, same, new}
        assert pages[stored].content_hash == content_hash("# Hyphen")
        assert pages[stored].file == url_to_filename(stored)
        assert not any((web / entry.file).exists() for entry in before.values())

    @patch("lilbee.crawler.runner.crawl_recursive")
    async def test_a_page_stored_under_several_spellings_keeps_one_entry_and_one_file(
        self, mock_crawl_recursive, isolated_env, mock_services
    ):
        reported = "https://example.com/t/tri%2db"
        spellings = ["https://example.com/t/tri-b", "https://example.com/t/tri%2Db", reported]
        other = "https://example.com/t/other"
        before = _stored(*spellings, other)
        write_crawl_metadata(before)
        _save_files(before)

        async def crawl(url: str, **kwargs: object) -> list[CrawlResult]:
            result = CrawlResult(url=reported, markdown="# Tri")
            await kwargs["on_result"](result)  # type: ignore[operator]
            return [result]

        mock_crawl_recursive.side_effect = crawl
        paths = await crawl_and_save("https://example.com/", depth=1)
        web = cfg.documents_dir / "_web"
        assert paths == [web / url_to_filename(reported)]
        assert paths[0].read_text(encoding="utf-8") == "# Tri"
        pages = load_crawl_record().pages
        assert set(pages) == {"https://example.com/t/tri-b"}
        assert pages["https://example.com/t/tri-b"].content_hash == content_hash("# Tri")
        assert not any((web / before[spelling].file).exists() for spelling in spellings)
        assert (web / before[other].file).read_text(encoding="utf-8") == STORED_TEXT

    @pytest.mark.parametrize("fetched", [0, 1], ids=["first-fetched", "second-fetched"])
    @pytest.mark.parametrize(("one", "other"), DIFFERENT_PAGES)
    @patch("lilbee.crawler.runner.crawl_recursive")
    async def test_a_fetched_page_never_replaces_the_stored_copy_of_a_different_page(
        self, mock_crawl_recursive, isolated_env, one: str, other: str, fetched: int
    ):
        pair = (one, other)
        reported, kept = pair[fetched], pair[1 - fetched]
        before = {canonical_address(kept): CrawlMeta(url_to_filename(kept), "old", "t")}
        write_crawl_metadata(before)
        _save_files(before)

        async def crawl(url: str, **kwargs: object) -> list[CrawlResult]:
            result = CrawlResult(url=reported, markdown="# Fetched")
            await kwargs["on_result"](result)  # type: ignore[operator]
            return [result]

        mock_crawl_recursive.side_effect = crawl
        paths = await crawl_and_save("https://example.com/", depth=1)
        web = cfg.documents_dir / "_web"
        assert paths == [web / url_to_filename(reported)]
        pages = load_crawl_record().pages
        assert set(pages) == {canonical_address(reported)}
        assert pages[canonical_address(reported)].content_hash == content_hash("# Fetched")
        assert (web / url_to_filename(kept)).read_bytes() == b"# Stored"

    @patch("lilbee.crawler.runner.crawl_single")
    async def test_a_page_saved_under_an_earlier_name_is_saved_at_its_name_and_the_copy_retired(
        self, mock_crawl_single, isolated_env, mock_services
    ):
        url = "https://example.com/t/Special:Random"
        earlier = "example.com/t/Special_Random/index.md"
        web = cfg.documents_dir / "_web"
        _write_web(earlier, "# Old")
        write_crawl_metadata({url: CrawlMeta(earlier, content_hash("# Old"), "x")})
        mock_crawl_single.return_value = CrawlResult(url=url, markdown="# New")
        paths = await crawl_and_save(url, depth=0)
        assert paths == [web / url_to_filename(url)]
        assert paths[0].read_text(encoding="utf-8") == "# New"
        assert not (web / "example.com/t/Special_Random").exists()
        assert load_crawl_record().pages[url].file == url_to_filename(url)
        mock_services.store.remove_documents.assert_called_once_with([f"_web/{earlier}"])

    @patch("lilbee.crawler.runner.crawl_single")
    async def test_an_earlier_name_outside_the_web_folder_is_left_alone(
        self, mock_crawl_single, isolated_env
    ):
        url = "https://example.com/t/page"
        outside = cfg.documents_dir / "outside.md"
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.write_text("# Mine", encoding="utf-8")
        meta = CrawlMeta(file="../outside.md", content_hash=content_hash("# Mine"), crawled_at="x")
        write_crawl_metadata({url: meta})
        mock_crawl_single.return_value = CrawlResult(url=url, markdown="# New")
        paths = await crawl_and_save(url, depth=0)
        assert len(paths) == 1
        assert paths[0].read_text(encoding="utf-8") == "# New"
        assert outside.read_text(encoding="utf-8") == "# Mine"

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
    async def test_records_the_saved_page(self, mock_crawl_single, isolated_env):
        mock_crawl_single.return_value = CrawlResult(
            url="https://example.com/page", markdown="# Test"
        )
        await crawl_and_save("https://example.com/page", depth=0)
        entry = load_crawl_record().pages["https://example.com/page"]
        assert entry.file == url_to_filename("https://example.com/page")
        assert (cfg.documents_dir / "_web" / entry.file).read_text(encoding="utf-8") == "# Test"
        assert not (cfg.data_dir / "crawl_meta.json").exists()

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
        """A single-URL crawl cancelled while its page is fetched keeps that page."""
        cancel = threading.Event()

        async def fetch_then_cancel(url: str, *, render_mode: CrawlRenderMode) -> CrawlResult:
            cancel.set()
            return CrawlResult(url=url, markdown="# Hello")

        mock_crawl_single.side_effect = fetch_then_cancel
        paths = await crawl_and_save("https://example.com", depth=0, cancel=cancel)
        assert len(paths) == 1
        assert paths[0].exists()


BROWSER_FLAGS = ["--lang=fr", "--disable-gpu"]


class TestBrowserLaunchFlags:
    """``crawl_browser_extra_args`` reaches the Chrome launch of every crawl path."""

    @pytest.fixture(autouse=True)
    def _admit_every_url(self, monkeypatch):
        monkeypatch.setattr("lilbee.crawler.url_filter.validate_crawl_url", lambda url: None)

    async def test_single_page_browser_crawl_launches_with_the_setting(self, isolated_env):
        cfg.crawl_browser_extra_args = list(BROWSER_FLAGS)
        stub = cb.StubCrawlberg([cb.page(SEED, "# Page", depth=0)])
        with stub.installed():
            paths = await crawl_and_save(SEED, depth=0, render_mode=CrawlRenderMode.BROWSER)
        assert len(paths) == 1
        browser = stub.config["browser"].kwargs
        assert browser["chrome_args"] == BROWSER_FLAGS
        assert browser["chrome_path"] == str(isolated_env / "chrome-headless-shell")

    async def test_recursive_browser_crawl_launches_with_the_setting(self, isolated_env):
        cfg.crawl_browser_extra_args = list(BROWSER_FLAGS)
        stub = cb.StubCrawlberg([cb.page(SEED, "# Page", depth=0), cb.complete(1)])
        with stub.installed():
            paths = await crawl_and_save(SEED, depth=1, render_mode=CrawlRenderMode.BROWSER)
        assert len(paths) == 1
        browser = stub.config["browser"].kwargs
        assert browser["chrome_args"] == BROWSER_FLAGS
        assert browser["chrome_path"] == str(isolated_env / "chrome-headless-shell")

    async def test_default_setting_launches_with_the_default_flags(self, isolated_env):
        stub = cb.StubCrawlberg([cb.page(SEED, "# Page", depth=0)])
        with stub.installed():
            await crawl_and_save(SEED, depth=0, render_mode=CrawlRenderMode.BROWSER)
        assert stub.config["browser"].kwargs["chrome_args"] == [
            "--disable-dev-shm-usage",
            "--disable-gpu",
        ]

    async def test_http_crawl_carries_no_launch_flags(self, isolated_env):
        cfg.crawl_browser_extra_args = list(BROWSER_FLAGS)
        stub = cb.StubCrawlberg([cb.page(SEED, "# Page", depth=0)])
        with stub.installed():
            paths = await crawl_and_save(SEED, depth=0, render_mode=CrawlRenderMode.HTTP)
        assert len(paths) == 1
        assert stub.config["browser"].kwargs["chrome_args"] == []
        assert stub.config["browser"].kwargs["chrome_path"] is None


REFUSED_FLAG_REASON = (
    "invalid_config: browser.chrome_args must not set --headless; crawlberg sets it to run Chrome"
)
REFUSED_PATTERN = "(?P<n>a)(?(n)b|c)"
REFUSED_PATTERN_REASON = (
    f"invalid_config: invalid exclude_path regex '{REFUSED_PATTERN}': regex parse error:\n"
    f"    {REFUSED_PATTERN}\nerror: unrecognized flag"
)
# A refusal whose text names no field lilbee knows, as after crawlberg rewords an error.
REWORDED_FLAG = "--user-data-dir=/elsewhere"
REWORDED_REASON = "invalid_config: the Chrome launch flags must not set --user-data-dir"
FLAG_REFUSAL = (
    "The crawl_browser_extra_args setting holds a launch flag that crawlberg refuses: "
    + REFUSED_FLAG_REASON
)
PATTERN_REFUSAL = (
    "The crawl_exclude_patterns setting holds a pattern that crawlberg refuses: "
    + REFUSED_PATTERN_REASON
)
REWORDED_REFUSAL = "crawlberg refuses to start this crawl: " + REWORDED_REASON


def _refuse_like_crawlberg(config: cb.Recorded) -> str | None:
    """Refuse a headless flag and a conditional pattern as crawlberg 1.9.0 does, and one more."""
    kwargs = config.kwargs
    flags = kwargs["browser"].kwargs["chrome_args"]
    if "--headless=new" in flags:
        return REFUSED_FLAG_REASON
    if REWORDED_FLAG in flags:
        return REWORDED_REASON
    if REFUSED_PATTERN in kwargs["exclude_paths"]:
        return REFUSED_PATTERN_REASON
    return None


OVERSIZED_REASON = "int too big to convert"
OVERSIZED_TIMEOUT = 10**17


def _oversized_number(config: cb.Recorded) -> str | None:
    """Raise what crawlberg 1.9.0 raises for a number over its integer width."""
    raise OverflowError(OVERSIZED_REASON)


def _oversized_setting() -> None:
    cfg.crawl_timeout = OVERSIZED_TIMEOUT


def _refuse_or_overflow(config: cb.Recorded) -> str | None:
    """Refuse as crawlberg does, with its overflow error for the oversized timeout."""
    if config.kwargs["request_timeout"] == OVERSIZED_TIMEOUT * 1000:
        raise OverflowError(OVERSIZED_REASON)
    return _refuse_like_crawlberg(config)


def _refused_flag() -> None:
    cfg.crawl_browser_extra_args = ["--lang=fr", "--headless=new"]


def _refused_pattern() -> None:
    cfg.crawl_exclude_patterns = [r"/ok/", REFUSED_PATTERN]


def _reworded_refusal() -> None:
    cfg.crawl_browser_extra_args = [REWORDED_FLAG]


def _default_settings() -> None:
    """Leave every crawl setting at its default, so crawlberg refuses nothing."""


BROWSER = CrawlRenderMode.BROWSER
# Each refused setting, the render mode and depth that send it to crawlberg, and the error.
# A single-page crawl follows no link, so it sends no exclude pattern.
REFUSALS = [
    pytest.param(_refused_flag, BROWSER, 0, FLAG_REFUSAL, id="flag-single-page"),
    pytest.param(_refused_flag, BROWSER, 1, FLAG_REFUSAL, id="flag-recursive"),
    pytest.param(_refused_pattern, CrawlRenderMode.HTTP, 1, PATTERN_REFUSAL, id="pattern"),
    pytest.param(_refused_pattern, BROWSER, 1, PATTERN_REFUSAL, id="pattern-browser"),
    pytest.param(_reworded_refusal, BROWSER, 0, REWORDED_REFUSAL, id="reworded-single-page"),
    pytest.param(_reworded_refusal, BROWSER, 1, REWORDED_REFUSAL, id="reworded-recursive"),
]
DEPTHS = pytest.mark.parametrize("depth", [0, 1], ids=["single-page", "recursive"])
# What ``lilbee add URL`` takes to crawl at each depth.
CLI_FLAGS: dict[int, tuple[str, ...]] = {0: (), 1: ("--crawl", "--depth", "1")}
REFUSED = pytest.mark.parametrize(("configure", "render_mode", "depth", "message"), REFUSALS)
# The settings a cancelled HTTP crawl never sends, at each depth; a single-page
# crawl sends no exclude pattern, so the pattern refusal exists at depth 1 only.
CANCELLED_AT_START = pytest.mark.parametrize(
    ("configure", "depth"),
    [
        pytest.param(_default_settings, 0, id="single-page"),
        pytest.param(_default_settings, 1, id="recursive"),
        pytest.param(_oversized_setting, 0, id="oversized-single-page"),
        pytest.param(_oversized_setting, 1, id="oversized-recursive"),
        pytest.param(_refused_pattern, 1, id="pattern-recursive"),
    ],
)
REFUSED_BY_HTTP = pytest.mark.parametrize(
    ("configure", "depth"),
    [
        pytest.param(_oversized_setting, 0, id="oversized-single-page"),
        pytest.param(_oversized_setting, 1, id="oversized-recursive"),
        pytest.param(_refused_pattern, 1, id="pattern-recursive"),
    ],
)


def _refusing_stub() -> cb.StubCrawlberg:
    return cb.StubCrawlberg([cb.page(SEED, "# Page", depth=0)], refuse=_refuse_like_crawlberg)


def _saved_pages(root: Path) -> list[Path]:
    return list((root / "documents").rglob("*.md"))


class TestRefusedEngine:
    """Any refusal by crawlberg to build the engine is the crawl's error, not a failed page."""

    @pytest.fixture(autouse=True)
    def _admit_every_url(self, monkeypatch):
        monkeypatch.setattr("lilbee.crawler.url_filter.validate_crawl_url", lambda url: None)

    @REFUSED
    async def test_the_crawl_raises_crawlbergs_reason(
        self, isolated_env, configure, render_mode: CrawlRenderMode, depth: int, message: str
    ):
        configure()
        events: list[EventType] = []
        stub = _refusing_stub()
        with stub.installed(), pytest.raises(CrawlEngineRefusedError) as refused:
            await crawl_and_save(
                SEED,
                depth=depth,
                render_mode=render_mode,
                on_progress=lambda kind, data: events.append(kind),
            )
        assert str(refused.value) == message
        assert events == [EventType.CRAWL_START]
        assert stub.seeds == []
        assert _saved_pages(isolated_env) == []

    @DEPTHS
    async def test_a_page_that_fails_after_the_engine_is_built_stays_a_failed_page(
        self, isolated_env, depth: int
    ):
        events: list[EventType] = []
        with cb.StubCrawlberg(_failing_stream("invalid_config: stream dropped")).installed():
            paths = await crawl_and_save(
                SEED,
                depth=depth,
                render_mode=CrawlRenderMode.HTTP,
                on_progress=lambda kind, data: events.append(kind),
            )
        assert paths == []
        assert events[-1] is EventType.CRAWL_DONE

    @DEPTHS
    async def test_an_engine_creation_error_that_is_no_runtime_error_fails_the_crawl(
        self, isolated_env, depth: int
    ):
        events: list[EventType] = []
        stub = cb.StubCrawlberg([cb.page(SEED, "# Page", depth=0)], refuse=_oversized_number)
        with stub.installed(), pytest.raises(CrawlEngineRefusedError) as refused:
            await crawl_and_save(
                SEED,
                depth=depth,
                render_mode=CrawlRenderMode.HTTP,
                on_progress=lambda kind, data: events.append(kind),
            )
        assert str(refused.value) == f"crawlberg refuses to start this crawl: {OVERSIZED_REASON}"
        assert type(refused.value.__cause__) is OverflowError
        assert events == [EventType.CRAWL_START]
        assert stub.seeds == []
        assert _saved_pages(isolated_env) == []

    @CANCELLED_AT_START
    async def test_a_task_cancelled_before_the_fetch_ends_cancelled_and_builds_no_engine(
        self, isolated_env, configure, depth: int
    ):
        configure()
        task = CrawlTask(
            task_id="t1", url=SEED, depth=depth, max_pages=10, render_mode=CrawlRenderMode.HTTP
        )
        task.cancel.set()
        stub = cb.StubCrawlberg([cb.page(SEED, "# Page", depth=0)], refuse=_refuse_or_overflow)
        with stub.installed():
            await run_crawl(task)
        assert task.status is TaskStatus.CANCELLED
        assert task.error is None
        assert task.pages_crawled == 0
        assert stub.configs == []
        assert _saved_pages(isolated_env) == []

    @REFUSED_BY_HTTP
    async def test_the_same_task_without_the_cancel_fails_with_crawlbergs_reason(
        self, isolated_env, configure, depth: int
    ):
        configure()
        task = CrawlTask(
            task_id="t1", url=SEED, depth=depth, max_pages=10, render_mode=CrawlRenderMode.HTTP
        )
        stub = cb.StubCrawlberg([cb.page(SEED, "# Page", depth=0)], refuse=_refuse_or_overflow)
        with stub.installed():
            await run_crawl(task)
        assert task.status is TaskStatus.FAILED
        assert task.error is not None and "crawlberg refuses" in task.error
        assert len(stub.configs) == 1

    @DEPTHS
    async def test_a_crawl_cancelled_before_it_starts_fetches_nothing(
        self, isolated_env, depth: int
    ):
        cancel = threading.Event()
        cancel.set()
        events: list[EventType] = []
        stub = cb.StubCrawlberg([cb.page(SEED, "# Page", depth=0), cb.complete(1)])
        with stub.installed():
            paths = await crawl_and_save(
                SEED,
                depth=depth,
                render_mode=CrawlRenderMode.HTTP,
                cancel=cancel,
                on_progress=lambda kind, data: events.append(kind),
            )
        assert paths == []
        assert events == [EventType.CRAWL_START, EventType.CRAWL_DONE]
        assert stub.configs == []
        assert stub.seeds == []

    @DEPTHS
    async def test_the_same_crawl_without_the_cancel_fetches_the_page(
        self, isolated_env, depth: int
    ):
        stub = cb.StubCrawlberg([cb.page(SEED, "# Page", depth=0), cb.complete(1)])
        with stub.installed():
            paths = await crawl_and_save(SEED, depth=depth, render_mode=CrawlRenderMode.HTTP)
        assert len(paths) == 1
        assert len(stub.configs) == 1
        assert stub.seeds == [SEED]

    async def test_an_error_while_lilbee_builds_the_config_stays_a_failed_crawl(self, isolated_env):
        cfg.crawl_mean_delay = float("inf")
        events: list[EventType] = []
        stub = cb.StubCrawlberg([cb.page(SEED, "# Page", depth=0)])
        with stub.installed():
            paths = await crawl_and_save(
                SEED,
                depth=1,
                render_mode=CrawlRenderMode.HTTP,
                on_progress=lambda kind, data: events.append(kind),
            )
        assert paths == []
        assert events[-1] is EventType.CRAWL_DONE
        assert stub.configs == []

    async def test_a_single_page_crawl_sends_no_exclude_pattern(self, isolated_env):
        _refused_pattern()
        with _refusing_stub().installed():
            paths = await crawl_and_save(SEED, depth=0, render_mode=CrawlRenderMode.HTTP)
        assert len(paths) == 1

    async def test_an_http_crawl_ignores_the_refused_flag(self, isolated_env):
        cfg.crawl_browser_extra_args = ["--headless=new"]
        with _refusing_stub().installed():
            paths = await crawl_and_save(SEED, depth=0, render_mode=CrawlRenderMode.HTTP)
        assert len(paths) == 1


class TestRefusedEngineOnEachEntryPoint:
    """The CLI, the HTTP stream and the MCP task each end a refused crawl as an error."""

    @pytest.fixture(autouse=True)
    def _admit_every_url(self, monkeypatch):
        monkeypatch.setattr("lilbee.crawler.url_filter.validate_crawl_url", lambda url: None)
        monkeypatch.setattr("lilbee.crawler.url_filter.require_valid_crawl_url", lambda url: None)

    @REFUSED
    def test_cli_add_exits_nonzero_with_the_reason_and_no_traceback(
        self, isolated_env, configure, render_mode: CrawlRenderMode, depth: int, message: str
    ):
        from typer.testing import CliRunner

        from lilbee.cli.app import app

        configure()
        cfg.crawl_render_mode = render_mode
        with _refusing_stub().installed():
            result = CliRunner().invoke(app, ["add", SEED, *CLI_FLAGS[depth]])
        assert result.exit_code == 1
        assert f"Error: {message}" in result.output
        assert "Traceback" not in result.output
        assert "Crawled" not in result.output
        assert _saved_pages(isolated_env) == []

    @REFUSED
    def test_cli_json_add_prints_one_error_object(
        self, isolated_env, configure, render_mode: CrawlRenderMode, depth: int, message: str
    ):
        import json

        from typer.testing import CliRunner

        from lilbee.cli.app import app

        configure()
        cfg.crawl_render_mode = render_mode
        with _refusing_stub().installed():
            result = CliRunner().invoke(app, ["--json", "add", SEED, *CLI_FLAGS[depth]])
        assert result.exit_code == 1
        objects = [json.loads(line) for line in result.output.splitlines() if line.startswith("{")]
        assert objects == [{"error": message}]

    @REFUSED
    async def test_http_stream_ends_with_an_error_frame(
        self, isolated_env, configure, render_mode: CrawlRenderMode, depth: int, message: str
    ):
        import json

        from lilbee.server import handlers

        configure()
        with _refusing_stub().installed():
            frames = [
                frame
                async for frame in handlers.crawl_stream(SEED, depth=depth, render_mode=render_mode)
            ]
        names = [frame.split("\n", 1)[0] for frame in frames]
        assert names == ["event: crawl_start", "event: error"]
        payload = frames[-1].split("data: ", 1)[1]
        assert json.loads(payload)["message"] == message

    @REFUSED
    async def test_mcp_crawl_task_fails_with_the_reason(
        self, isolated_env, configure, render_mode: CrawlRenderMode, depth: int, message: str
    ):
        from lilbee.crawler.task import clear_tasks, get_task
        from lilbee.mcp_server import crawl, crawl_status

        configure()
        clear_tasks()
        with _refusing_stub().installed():
            started = await crawl(url=SEED, depth=depth, render_mode=render_mode)
            task = get_task(started["task_id"])
            assert task is not None and task._async_task is not None
            await asyncio.wait_for(task._async_task, timeout=10)
        status = crawl_status(started["task_id"])
        assert status["status"] == "failed"
        assert status["error"] == message
        assert status["pages_crawled"] == 0

    @pytest.mark.parametrize(
        ("configure", "message"),
        [(_refused_flag, FLAG_REFUSAL), (_reworded_refusal, REWORDED_REFUSAL)],
        ids=["flag", "reworded"],
    )
    async def test_mcp_add_raises_the_reason_for_the_one_page_it_fetches(
        self, isolated_env, configure, message: str
    ):
        from lilbee.mcp_server import add

        configure()
        with _refusing_stub().installed(), pytest.raises(CrawlEngineRefusedError) as refused:
            await add(paths=[SEED], render_mode=BROWSER)
        assert str(refused.value) == message
        assert _saved_pages(isolated_env) == []


class TestPeriodicSync:
    async def test_sync_disabled_when_interval_zero(self, isolated_env):
        """No sync fires when crawl_sync_interval is 0."""
        cfg.crawl_sync_interval = 0
        reset_services()
        sync_state = get_services().crawler_sync_state
        sync_state.last_run = 0.0

        tasks: set[asyncio.Task[None]] = set()
        with patch("lilbee.data.ingest.sync", new_callable=AsyncMock) as mock_sync:
            await _maybe_periodic_sync(tasks, None)
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
                await _maybe_periodic_sync(tasks, None)
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
            await _maybe_periodic_sync(tasks, None)
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
            await _maybe_periodic_sync(tasks, None)
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
            await _maybe_periodic_sync(tasks, None)
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


class TestSavePage:
    """Unit tests for save_page, the per-page save at the page's name."""

    def test_writes_new_content_and_records_it(self, isolated_env):
        record = load_crawl_record()
        result = CrawlResult(url="https://example.com/new", markdown="# New")
        path = save_page(result, record, MagicMock())
        name = url_to_filename("https://example.com/new")
        assert path == cfg.documents_dir / "_web" / name
        assert path.read_text(encoding="utf-8") == "# New"
        entry = record.pages["https://example.com/new"]
        assert (entry.file, entry.content_hash) == (name, content_hash("# New"))
        assert name in record.claimed
        assert load_crawl_record().pages == {"https://example.com/new": entry}

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
        raw = "9C1[[1]](https://en.wikipedia.org/wiki/Caprice#cite_note-1) introduced for 1986."
        expected = "9C1[1](https://en.wikipedia.org/wiki/Caprice#cite_note-1) introduced for 1986."
        record = CrawlRecord()
        result = CrawlResult(url="https://example.com/ref", markdown=raw)
        path = save_page(result, record, MagicMock())
        assert path is not None
        assert path.read_text(encoding="utf-8") == expected
        assert record.pages["https://example.com/ref"].content_hash == content_hash(expected)

    def test_returns_none_on_failure(self, isolated_env):
        result = CrawlResult(url="https://example.com/fail", success=False, error="oops")
        assert save_page(result, CrawlRecord(), MagicMock()) is None
        assert not _pages_file().exists()

    def test_returns_none_on_empty_markdown(self, isolated_env):
        result = CrawlResult(url="https://example.com/empty", markdown="   ")
        assert save_page(result, CrawlRecord(), MagicMock()) is None
        assert not _pages_file().exists()

    def test_unchanged_page_is_not_rewritten(self, isolated_env, caplog):
        record = load_crawl_record()
        url = "https://example.com/dup"
        first = save_page(CrawlResult(url=url, markdown="# Dup"), record, MagicMock())
        assert first is not None
        lines_after_first = _pages_file().read_text(encoding="utf-8")
        with caplog.at_level("INFO", logger="lilbee.crawler.save"):
            again = save_page(CrawlResult(url=url, markdown="# Dup"), record, MagicMock())
        assert again is None
        assert _pages_file().read_text(encoding="utf-8") == lines_after_first
        assert any("Content unchanged" in r.message for r in caplog.records)

    def test_unchanged_text_under_another_spelling_of_the_address_skips(self, isolated_env):
        address = "https://example.com/t/page"
        name = url_to_filename(address)
        _write_web(name, "# Page")
        record = CrawlRecord(pages={address: CrawlMeta(name, content_hash("# Page"), "t")})
        again = CrawlResult(url="https://EXAMPLE.com/t/pag%65#top", markdown="# Page")
        assert save_page(again, record, MagicMock()) is None

    def test_recorded_page_with_a_missing_file_is_rewritten(self, isolated_env):
        url = "https://example.com/gone"
        name = url_to_filename(url)
        record = CrawlRecord(pages={url: CrawlMeta(name, content_hash("# Gone"), "t")})
        path = save_page(CrawlResult(url=url, markdown="# Gone"), record, MagicMock())
        assert path == cfg.documents_dir / "_web" / name
        assert path.read_text(encoding="utf-8") == "# Gone"

    def test_path_traversal_blocked(self, isolated_env, caplog):
        result = CrawlResult(url="https://evil.com/ok", markdown="# Evil")
        with (
            caplog.at_level("WARNING", logger="lilbee.crawler.save"),
            patch("lilbee.crawler.save.page_name", return_value="../../etc/passwd"),
        ):
            assert save_page(result, CrawlRecord(), MagicMock()) is None
        assert any("Path traversal blocked" in r.message for r in caplog.records)
        assert not _pages_file().exists()

    def test_the_record_line_is_written_before_the_file(self, isolated_env):
        url = "https://example.com/p"
        with (
            patch("lilbee.crawler.save.Path.write_text", side_effect=OSError("disk full")),
            pytest.raises(OSError, match="disk full"),
        ):
            save_page(CrawlResult(url=url, markdown="# P"), CrawlRecord(), MagicMock())
        assert load_crawl_record().pages[url].content_hash == content_hash("# P")

    def test_a_cut_record_line_does_not_join_the_next_line(self, isolated_env, caplog):
        _pages_file().write_text('{"url": "https://example.com/cut", "fi', encoding="utf-8")
        url = "https://example.com/p"
        save_page(CrawlResult(url=url, markdown="# P"), load_crawl_record(), MagicMock())
        with caplog.at_level("WARNING", logger="lilbee.crawler.save"):
            pages = load_crawl_record().pages
        assert set(pages) == {url}
        assert any("malformed crawl record line" in r.message for r in caplog.records)

    def test_a_name_held_by_a_file_the_record_does_not_give_the_page_is_not_written(
        self, isolated_env, caplog
    ):
        url = "https://example.com/mine"
        mine = _write_web(url_to_filename(url), "# My notes")
        with caplog.at_level("WARNING", logger="lilbee.crawler.save"):
            result = save_page(CrawlResult(url=url, markdown="# Page"), CrawlRecord(), MagicMock())
        assert result is None
        assert mine.read_text(encoding="utf-8") == "# My notes"
        assert [r.message for r in caplog.records if "is not saved" in r.message] != []
        assert not _pages_file().exists()

    def test_a_recorded_page_whose_name_holds_a_folder_is_not_written(self, isolated_env):
        url = "https://example.com/p"
        name = url_to_filename(url)
        (cfg.documents_dir / "_web" / name).mkdir(parents=True)
        record = CrawlRecord(pages={url: CrawlMeta(name, "old", "t")})
        assert save_page(CrawlResult(url=url, markdown="# P"), record, MagicMock()) is None
        assert (cfg.documents_dir / "_web" / name).is_dir()
        assert record.pages[url].content_hash == "old"


class TestCrawlRecord:
    """The append-only page record and the earlier crawl metadata it carries."""

    def test_a_missing_record_has_no_pages(self, isolated_env):
        record = load_crawl_record()
        assert (record.pages, record.claimed, record.earlier) == ({}, set(), {})
        assert not _pages_file().exists()

    def test_a_malformed_line_is_skipped_with_a_warning(self, isolated_env, caplog):
        good = CrawlMeta("example.com/a~1.md", "h", "t")
        _pages_file().write_text(
            "not json\n" + _record_lines(("https://example.com/a", good)) + '{"file": "x"}\n',
            encoding="utf-8",
        )
        with caplog.at_level("WARNING", logger="lilbee.crawler.save"):
            pages = load_crawl_record().pages
        assert pages == {"https://example.com/a": good}
        warned = [r for r in caplog.records if "malformed crawl record line" in r.message]
        assert len(warned) == 2

    def test_duplicate_lines_are_compacted_to_one_per_address(self, isolated_env):
        old_a = CrawlMeta("example.com/a~1.md", "old", "t1")
        b = CrawlMeta("example.com/b~2.md", "b", "t2")
        new_a = CrawlMeta("example.com/a~1.md", "new", "t3")
        a_url, b_url = "https://example.com/a", "https://example.com/b"
        _pages_file().write_text(
            "\n" + _record_lines((a_url, old_a), (b_url, b), (a_url, new_a)), encoding="utf-8"
        )
        pages = load_crawl_record().pages
        assert pages == {a_url: new_a, b_url: b}
        assert _pages_file().read_text(encoding="utf-8") == _record_lines(
            (a_url, new_a), (b_url, b)
        )

    def test_the_append_and_the_compaction_each_write_under_the_record_lock(
        self, isolated_env, monkeypatch
    ):
        held: list[tuple[str, str, str]] = []

        @contextlib.contextmanager
        def lock(path, timeout_s):
            before = path.read_text(encoding="utf-8") if path.exists() else ""
            yield
            held.append((path.name, before, path.read_text(encoding="utf-8")))

        monkeypatch.setattr("lilbee.crawler.save.file_lock_or_warn", lock)
        page = CrawlMeta("example.com/a~1.md", "h", "t")
        url = "https://example.com/a"
        _pages_file().write_text("\n" + _record_lines((url, page), (url, page)), encoding="utf-8")
        record = load_crawl_record()
        _claim(url, CrawlMeta("example.com/a~1.md", "h2", "t2"), record)
        compacted, appended = held
        assert compacted[0] == appended[0] == "crawl_pages.jsonl"
        assert compacted[2] == _record_lines((url, page))
        assert appended[1] == compacted[2]
        assert appended[2].count("\n{") == 1 + compacted[2].count("\n{")

    def test_a_record_with_one_line_per_page_is_not_rewritten(self, isolated_env):
        text = "\n" + _record_lines(("https://example.com/a", CrawlMeta("a.md", "h", "t")))
        _pages_file().write_text(text, encoding="utf-8")
        load_crawl_record()
        assert _pages_file().read_text(encoding="utf-8") == text

    def test_earlier_metadata_is_carried_as_copies_by_canonical_address(self, isolated_env):
        earlier = CrawlMeta("example.com/t/page/index.md", "h", "t")
        write_crawl_metadata({"https://EXAMPLE.com/t/pag%65#top": earlier})
        page = CrawlMeta("example.com/x~1.md", "x", "t")
        _pages_file().write_text(_record_lines(("https://example.com/x", page)), encoding="utf-8")
        record = load_crawl_record()
        assert record.earlier == {"https://example.com/t/page": [earlier]}
        assert record.earlier_by_text == {"h": [("https://example.com/t/page", earlier)]}
        assert record.claimed == {"example.com/x~1.md"}


class TestRetireEarlier:
    """Earlier crawl metadata copies are deleted once the page is saved at its name."""

    URL = "https://example.com/t/page"
    EARLIER = "example.com/t/page/index.md"

    def _record_with_copy(self, text: str, entry_text: str | None = None) -> CrawlRecord:
        """A record whose earlier copy of URL holds *text*, its entry the hash of *entry_text*."""
        _write_web(self.EARLIER, text)
        entry_hash = content_hash(text if entry_text is None else entry_text)
        write_crawl_metadata({self.URL: CrawlMeta(self.EARLIER, entry_hash, "t")})
        return load_crawl_record()

    def test_a_first_save_with_the_same_text_removes_the_index_source(self, isolated_env):
        record = self._record_with_copy("# Page")
        remove_sources = MagicMock()
        path = save_page(CrawlResult(url=self.URL, markdown="# Page"), record, remove_sources)
        assert path is not None
        assert not (cfg.documents_dir / "_web" / self.EARLIER).exists()
        remove_sources.assert_called_once_with([f"_web/{self.EARLIER}"])

    def test_the_index_source_is_removed_before_the_copy_is_deleted(self, isolated_env):
        record = self._record_with_copy("# Old")
        copy = cfg.documents_dir / "_web" / self.EARLIER
        present_at_removal: list[bool] = []
        remove_sources = MagicMock(side_effect=lambda _: present_at_removal.append(copy.exists()))
        save_page(CrawlResult(url=self.URL, markdown="# New"), record, remove_sources)
        assert not copy.exists()
        assert present_at_removal == [True]
        remove_sources.assert_called_once_with([f"_web/{self.EARLIER}"])

    def test_an_unchanged_page_is_not_rewritten_but_its_earlier_copy_is_retired(self, isolated_env):
        page = _write_web(url_to_filename(self.URL), "# Page")
        before = page.stat().st_mtime_ns
        record = self._record_with_copy("# Page")
        record.pages[self.URL] = CrawlMeta(url_to_filename(self.URL), content_hash("# Page"), "t")
        remove_sources = MagicMock()
        path = save_page(CrawlResult(url=self.URL, markdown="# Page"), record, remove_sources)
        assert path is None
        assert page.stat().st_mtime_ns == before
        assert not (cfg.documents_dir / "_web" / self.EARLIER).exists()
        remove_sources.assert_called_once_with([f"_web/{self.EARLIER}"])

    def test_an_earlier_copy_whose_text_differs_from_its_entry_is_kept(self, isolated_env):
        record = self._record_with_copy("# Edited by the user", entry_text="# Crawled")
        remove_sources = MagicMock()
        save_page(CrawlResult(url=self.URL, markdown="# New"), record, remove_sources)
        copy = cfg.documents_dir / "_web" / self.EARLIER
        assert copy.read_text(encoding="utf-8") == "# Edited by the user"
        remove_sources.assert_not_called()

    def test_an_earlier_copy_named_by_a_page_line_is_kept(self, isolated_env):
        record = self._record_with_copy("# Old")
        other = CrawlMeta(self.EARLIER, content_hash("# Old"), "t")
        record.pages["https://example.com/other"] = other
        record.claimed.add(self.EARLIER)
        remove_sources = MagicMock()
        save_page(CrawlResult(url=self.URL, markdown="# New"), record, remove_sources)
        assert (cfg.documents_dir / "_web" / self.EARLIER).read_text(encoding="utf-8") == "# Old"
        remove_sources.assert_not_called()

    def test_an_earlier_copy_shared_with_another_pages_same_text_entry_is_kept(self, isolated_env):
        _write_web(self.EARLIER, "# Shared")
        entry = CrawlMeta(self.EARLIER, content_hash("# Shared"), "t")
        write_crawl_metadata({self.URL: entry, "https://example.com/t/other": entry})
        record = load_crawl_record()
        remove_sources = MagicMock()
        save_page(CrawlResult(url=self.URL, markdown="# New"), record, remove_sources)
        copy = cfg.documents_dir / "_web" / self.EARLIER
        assert copy.read_text(encoding="utf-8") == "# Shared"
        remove_sources.assert_not_called()

    def test_a_missing_earlier_copy_is_skipped(self, isolated_env):
        write_crawl_metadata({self.URL: CrawlMeta(self.EARLIER, content_hash("# Old"), "t")})
        remove_sources = MagicMock()
        path = save_page(
            CrawlResult(url=self.URL, markdown="# New"), load_crawl_record(), remove_sources
        )
        assert path is not None
        remove_sources.assert_not_called()

    def test_same_file_is_false_when_a_name_does_not_exist(self, isolated_env):
        _write_web("a.md", "# A")
        assert save._same_file("a.md", "a.md") is True
        assert save._same_file("a.md", "missing/b.md") is False

    def test_empty_folders_are_removed_up_to_the_web_folder(self, isolated_env):
        web = cfg.documents_dir / "_web"
        (web / "a" / "b").mkdir(parents=True)
        save._remove_empty_folders(web / "a" / "b")
        assert not (web / "a").exists()
        assert web.is_dir()

    def test_a_folder_that_still_holds_a_file_is_kept(self, isolated_env):
        web = cfg.documents_dir / "_web"
        kept = _write_web("a/kept.md", "# Kept")
        (web / "a" / "b").mkdir()
        save._remove_empty_folders(web / "a" / "b")
        assert not (web / "a" / "b").exists()
        assert kept.is_file()

    def test_disk_name_returns_the_name_as_the_disk_spells_it(self, isolated_env):
        _write_web("Host/Page.md", "# Page")
        assert save._disk_name("Host/Page.md") == "Host/Page.md"
        if not (cfg.documents_dir / "_web" / "host" / "page.md").exists():
            pytest.skip("the disk keeps letter case apart")
        assert save._disk_name("host/page.md") == "Host/Page.md"


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
        # The page record names exactly the pages on disk.
        recorded = {cfg.documents_dir / "_web" / e.file for e in load_crawl_record().pages.values()}
        assert recorded == set(paths)

    async def test_flushes_per_page_not_batched(self, isolated_env):
        """save_page is invoked once per streamed page inside the loop."""
        import asyncio as _asyncio

        call_order: list[str] = []

        async def _gen():
            for i in range(1, 4):
                call_order.append(f"yield-{i}")
                await _asyncio.sleep(0)
                yield cb.page(f"https://example.com/p{i}", f"# P{i}")

        stub = cb.StubCrawlberg(_gen)

        real_save = save_page

        def _wrapped(result, record, remove_sources):
            call_order.append(f"flush-{result.url.rsplit('/', 1)[-1]}")
            return real_save(result, record, remove_sources)

        with (
            stub.installed(),
            patch("lilbee.crawler.save.save_page", side_effect=_wrapped),
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
        """Per-page flush skips URLs whose hash matches the page record."""
        import asyncio as _asyncio

        # Pre-seed: p1 was crawled last time with this exact content.
        seed = CrawlResult(url="https://example.com/p1", markdown="# Same")
        assert save_page(seed, load_crawl_record(), MagicMock()) is not None

        async def _gen():
            yield cb.page("https://example.com/p1", "# Same")
            await _asyncio.sleep(0)
            yield cb.page("https://example.com/p2", "# NewPage")

        stub = cb.StubCrawlberg(_gen)

        with stub.installed():
            paths = await crawl_and_save("https://example.com", depth=2, max_pages=10)

        # p1 is skipped (unchanged), p2 is written.
        assert len(paths) == 1
        assert paths[0] == cfg.documents_dir / "_web" / url_to_filename("https://example.com/p2")
        assert paths[0].read_text(encoding="utf-8") == "# NewPage"

    @patch("lilbee.crawler.runner.crawl_single")
    async def test_single_url_flushes_via_per_page_path(self, mock_crawl_single, isolated_env):
        """depth=0 uses the same flush_page callback, not the old batch helpers."""
        mock_crawl_single.return_value = CrawlResult(
            url="https://example.com/only", markdown="# Only"
        )
        with patch("lilbee.crawler.save.save_page", wraps=save_page) as spy:
            paths = await crawl_and_save("https://example.com/only", depth=0)
        assert len(paths) == 1
        spy.assert_called_once()
        assert "https://example.com/only" in load_crawl_record().pages

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

        pages = load_crawl_record().pages
        for url, md in urls_and_content:
            assert pages[url].content_hash == content_hash(md)
            assert pages[url].file == url_to_filename(url)

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

    async def test_a_crawl_cancel_stops_its_periodic_sync(self, isolated_env):
        """The periodic sync a crawl starts stops on that crawl's cancel."""
        import threading

        cfg.crawl_sync_interval = 1
        reset_services()
        sync_state = get_services().crawler_sync_state
        sync_state.last_run = 0.0
        cancel = threading.Event()
        started = asyncio.Event()
        seen: list[object] = []

        async def _sync_until_cancelled(**kwargs):
            seen.append(kwargs.get("cancel"))
            started.set()
            for _ in range(200):
                if cancel.is_set() and kwargs.get("cancel") is cancel:
                    raise asyncio.CancelledError
                await asyncio.sleep(0.01)
            raise AssertionError("the periodic sync never saw the crawl's cancel")

        async def _fake_single(
            url: str, *, quiet: bool = False, on_progress=None, render_mode=None
        ) -> CrawlResult:
            return CrawlResult(url=url, markdown="# page")

        with (
            patch("lilbee.crawler.runner.crawl_single", side_effect=_fake_single),
            patch("lilbee.data.ingest.sync", _sync_until_cancelled),
        ):
            crawl = asyncio.create_task(
                crawl_and_save("https://example.com/c", depth=0, cancel=cancel)
            )
            await asyncio.wait_for(started.wait(), timeout=5)
            cancel.set()
            paths = await asyncio.wait_for(crawl, timeout=5)
        assert len(paths) == 1
        assert seen == [cancel]
        # The cancelled sync still releases the lock for the next one.
        assert sync_state.lock.acquire(blocking=False)
        sync_state.lock.release()

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

    async def test_the_record_is_read_once_per_crawl(self, isolated_env):
        """The page record is loaded once, and every page is saved against it."""
        import asyncio as _asyncio

        async def _gen():
            for i in range(1, 4):
                await _asyncio.sleep(0)
                yield cb.page(f"https://example.com/p{i}", f"# P{i}")

        stub = cb.StubCrawlberg(_gen)

        with (
            stub.installed(),
            patch("lilbee.crawler.save.load_crawl_record", wraps=load_crawl_record) as load_spy,
            patch("lilbee.crawler.save.save_page", wraps=save_page) as save_spy,
        ):
            await crawl_and_save("https://example.com", depth=2, max_pages=10)

        load_spy.assert_called_once()
        records = {id(c.args[1]) for c in save_spy.call_args_list}
        assert save_spy.call_count == 3
        assert len(records) == 1

    async def test_each_saved_page_is_recorded_before_the_crawl_ends(self, isolated_env):
        """A page's record line is on disk as soon as the page is saved."""
        import asyncio as _asyncio

        recorded_mid_crawl: list[set[str]] = []

        async def _gen():
            yield cb.page("https://example.com/p1", "# P1")
            await _asyncio.sleep(0)
            recorded_mid_crawl.append(set(load_crawl_record().pages))
            yield cb.page("https://example.com/p2", "# P2")

        stub = cb.StubCrawlberg(_gen)

        with stub.installed():
            await crawl_and_save("https://example.com", depth=2, max_pages=10)

        assert recorded_mid_crawl == [{"https://example.com/p1"}]

    async def test_the_runner_passes_the_store_to_retire_earlier_copies(
        self, isolated_env, mock_services
    ):
        """The crawl removes an earlier copy's index source through the store."""
        earlier = "example.com/old/p1.md"
        _write_web(earlier, "# Old")
        write_crawl_metadata(
            {"https://example.com/p1": CrawlMeta(earlier, content_hash("# Old"), "t")}
        )

        async def _gen():
            yield cb.page("https://example.com/p1", "# P1")

        with cb.StubCrawlberg(_gen).installed():
            await crawl_and_save("https://example.com", depth=2, max_pages=10)

        mock_services.store.remove_documents.assert_called_once_with([f"_web/{earlier}"])
        assert not (cfg.documents_dir / "_web" / "example.com" / "old").exists()

    async def test_flush_callback_failure_does_not_fail_the_crawl(self, isolated_env):
        """An OSError from the flush callback is logged, not reraised as a crawl error."""
        import asyncio as _asyncio

        async def _gen():
            for i in range(1, 3):
                await _asyncio.sleep(0)
                yield cb.page(f"https://example.com/p{i}", f"# P{i}")

        stub = cb.StubCrawlberg(_gen)

        call_count = {"n": 0}

        def failing_save(result, record, remove_sources):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError("disk full")
            return  # second page falls through unchanged

        with (
            stub.installed(),
            patch("lilbee.crawler.save.save_page", side_effect=failing_save),
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
        with patch("lilbee.crawler.save.save_page", side_effect=OSError("disk full")):
            # Must not raise even though the depth=0 flush fails.
            paths = await crawl_and_save("https://example.com/only", depth=0)
        assert paths == []

    async def test_cancel_keeps_the_record_of_saved_pages(self, isolated_env):
        """A cancelled crawl leaves the record of each page it saved."""
        import asyncio as _asyncio
        import threading

        cancel = threading.Event()

        async def _gen():
            yield cb.page("https://example.com/p1", "# P1")
            cancel.set()
            await _asyncio.sleep(0)
            yield cb.page("https://example.com/p2", "# P2")

        stub = cb.StubCrawlberg(_gen)

        with stub.installed():
            paths = await crawl_and_save(
                "https://example.com", depth=2, max_pages=10, cancel=cancel
            )

        pages = load_crawl_record().pages
        assert "https://example.com/p1" in pages
        assert {cfg.documents_dir / "_web" / e.file for e in pages.values()} == set(paths)

    async def test_no_record_line_when_nothing_written(self, isolated_env):
        """A page with empty markdown is not saved and writes no record line."""
        import asyncio as _asyncio

        async def _gen():
            await _asyncio.sleep(0)
            yield cb.page("https://example.com/empty", "")

        stub = cb.StubCrawlberg(_gen)

        with stub.installed():
            paths = await crawl_and_save("https://example.com", depth=2, max_pages=10)
        assert paths == []
        assert not _pages_file().exists()
        assert not (cfg.data_dir / "crawl_meta.json").exists()

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
