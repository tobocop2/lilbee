"""Tests for the backend-agnostic URL filter / sitemap helpers.

These extract the pure-Python pieces of the old monolith (SSRF-safe
URL validation, host scope checks, sitemap counting) so a future
adapter can rely on the same primitives.
"""

from __future__ import annotations

import socket
from unittest.mock import MagicMock

import pytest

from lilbee.crawler.sitemap import _count_sitemap_urls, _fetch_sitemap_text
from lilbee.crawler.url_filter import (
    get_blocked_networks,
    host_in_scope,
    is_url,
    require_valid_crawl_url,
    validate_crawl_url,
)
from lilbee.runtime.progress import CRAWL_TOTAL_UNKNOWN


@pytest.fixture(autouse=True)
def _bypass_dns(monkeypatch):
    """All URL validators use socket.getaddrinfo; stub it to a fixed public IP."""
    monkeypatch.setattr(
        "lilbee.crawler.url_filter.socket.getaddrinfo",
        lambda host, port, *a, **kw: [(2, 1, 6, "", ("93.184.216.34", 0))],
    )


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
    def test_accepts_http(self):
        validate_crawl_url("http://example.com")

    def test_accepts_https(self):
        validate_crawl_url("https://example.com")

    def test_rejects_ftp(self):
        with pytest.raises(ValueError, match="Only http"):
            validate_crawl_url("ftp://example.com")

    def test_rejects_file(self):
        with pytest.raises(ValueError, match="Only http"):
            validate_crawl_url("file:///etc/passwd")

    def test_rejects_missing_hostname(self):
        with pytest.raises(ValueError, match="no hostname"):
            validate_crawl_url("http://")

    def test_rejects_private_ip(self, monkeypatch):
        monkeypatch.setattr(
            "lilbee.crawler.url_filter.socket.getaddrinfo",
            lambda *a, **kw: [(2, 1, 6, "", ("10.0.0.5", 0))],
        )
        with pytest.raises(ValueError, match="private/reserved"):
            validate_crawl_url("http://internal.example.com")

    def test_rejects_unresolvable_host(self, monkeypatch):
        def _fail(*a, **kw):
            raise socket.gaierror("nxdomain")

        monkeypatch.setattr("lilbee.crawler.url_filter.socket.getaddrinfo", _fail)
        with pytest.raises(ValueError, match="Cannot resolve"):
            validate_crawl_url("http://nope.example.com")

    def test_rejects_ipv4_mapped_ipv6_metadata(self, monkeypatch):
        # ::ffff:169.254.169.254 is the cloud metadata IP smuggled as IPv6.
        monkeypatch.setattr(
            "lilbee.crawler.url_filter.socket.getaddrinfo",
            lambda *a, **kw: [(10, 1, 6, "", ("::ffff:169.254.169.254", 0, 0, 0))],
        )
        with pytest.raises(ValueError, match="private/reserved"):
            validate_crawl_url("http://metadata.example.com")

    def test_rejects_unspecified_zero_host(self, monkeypatch):
        monkeypatch.setattr(
            "lilbee.crawler.url_filter.socket.getaddrinfo",
            lambda *a, **kw: [(2, 1, 6, "", ("0.0.0.0", 0))],
        )
        with pytest.raises(ValueError, match="private/reserved"):
            validate_crawl_url("http://zero.example.com")


class TestRequireValidCrawlUrl:
    def test_rejects_non_url(self):
        with pytest.raises(ValueError, match="http"):
            require_valid_crawl_url("/etc/passwd")

    def test_accepts_http(self):
        require_valid_crawl_url("http://example.com")


class TestBlockedNetworks:
    def test_contains_loopback(self):
        import ipaddress as _ip

        networks = get_blocked_networks()
        assert any(_ip.ip_address("127.0.0.1") in n for n in networks)

    def test_contains_private(self):
        import ipaddress as _ip

        networks = get_blocked_networks()
        assert any(_ip.ip_address("10.1.2.3") in n for n in networks)

    @pytest.mark.parametrize(
        "addr",
        [
            "fe80::1",  # link-local (fe80::/10)
            "fc00::1",  # unique local address (fc00::/7)
            "fd12:3456:789a::1",  # ULA inside fc00::/7
            "ff02::1",  # multicast (ff00::/8)
        ],
    )
    def test_contains_ipv6_reserved_ranges(self, addr):
        import ipaddress as _ip

        networks = get_blocked_networks()
        assert any(_ip.ip_address(addr) in n for n in networks)

    @pytest.mark.parametrize(
        "addr",
        ["fe80::dead:beef", "fc00::5", "ff00::abcd"],
    )
    def test_rejects_ipv6_reserved_targets(self, monkeypatch, addr):
        monkeypatch.setattr(
            "lilbee.crawler.url_filter.socket.getaddrinfo",
            lambda *a, **kw: [(10, 1, 6, "", (addr, 0, 0, 0))],
        )
        with pytest.raises(ValueError, match="private/reserved"):
            validate_crawl_url("http://ipv6.example.com")


class TestHostInScope:
    def test_exact_match(self):
        assert host_in_scope("example.com", "example.com", include_subdomains=False) is True

    def test_empty_link_host(self):
        assert host_in_scope("", "example.com", include_subdomains=True) is False

    def test_subdomain_included_when_flag_set(self):
        assert host_in_scope("sub.example.com", "example.com", include_subdomains=True) is True

    def test_subdomain_rejected_when_flag_clear(self):
        assert host_in_scope("sub.example.com", "example.com", include_subdomains=False) is False

    def test_unrelated_host(self):
        assert host_in_scope("other.org", "example.com", include_subdomains=True) is False


class TestSitemapFetch:
    def test_returns_none_on_http_error(self, monkeypatch):
        import httpx

        def _raise(*a, **kw):
            raise httpx.ConnectError("boom")

        monkeypatch.setattr("httpx.get", _raise)
        assert _fetch_sitemap_text("https://example.com/start") is None

    def test_returns_none_on_4xx(self, monkeypatch):
        fake = MagicMock(status_code=404, text="")
        monkeypatch.setattr("httpx.get", lambda *a, **kw: fake)
        assert _fetch_sitemap_text("https://example.com/start") is None

    def test_returns_body_on_success(self, monkeypatch):
        fake = MagicMock(status_code=200, text="<urlset></urlset>")
        fake.url = "https://example.com/sitemap.xml"
        monkeypatch.setattr("httpx.get", lambda *a, **kw: fake)
        assert _fetch_sitemap_text("https://example.com/start") == "<urlset></urlset>"

    def test_does_not_follow_redirects(self, monkeypatch):
        """A 30x is not followed; an unfollowed redirect yields no sitemap (SSRF-safe)."""
        fake = MagicMock(status_code=301, text="<urlset></urlset>")
        captured = {}

        def _get(url, *a, **kw):
            captured.update(kw)
            return fake

        monkeypatch.setattr("httpx.get", _get)
        assert _fetch_sitemap_text("https://example.com/start") is None
        assert captured.get("follow_redirects") is False

    def test_validates_seed_before_any_fetch(self, monkeypatch):
        """A private/metadata seed host is rejected without making a request."""
        monkeypatch.setattr(
            "lilbee.crawler.url_filter.socket.getaddrinfo",
            lambda *a, **kw: [(2, 1, 6, "", ("169.254.169.254", 0))],
        )
        called = []
        monkeypatch.setattr("httpx.get", lambda *a, **kw: called.append(1))
        assert _fetch_sitemap_text("http://metadata.internal/start") is None
        assert called == []


class TestSitemapCount:
    def test_missing_host_short_circuits(self):
        assert _count_sitemap_urls("file:///foo", include_subdomains=False) == CRAWL_TOTAL_UNKNOWN

    def test_missing_body_returns_unknown(self, monkeypatch):
        monkeypatch.setattr("httpx.get", lambda *a, **kw: MagicMock(status_code=500, text=""))
        assert (
            _count_sitemap_urls("https://example.com/start", include_subdomains=False)
            == CRAWL_TOTAL_UNKNOWN
        )

    def test_counts_matching_hosts(self, monkeypatch):
        body = (
            "<urlset>"
            "<url><loc>https://example.com/a</loc></url>"
            "<url><loc>https://example.com/b</loc></url>"
            "<url><loc>https://other.com/c</loc></url>"
            "</urlset>"
        )
        monkeypatch.setattr(
            "httpx.get",
            lambda *a, **kw: MagicMock(
                status_code=200, text=body, url="https://example.com/sitemap.xml"
            ),
        )
        count = _count_sitemap_urls("https://example.com/start", include_subdomains=False)
        assert count == 2

    def test_include_subdomains_counts_children(self, monkeypatch):
        body = (
            "<urlset>"
            "<url><loc>https://example.com/a</loc></url>"
            "<url><loc>https://sub.example.com/d</loc></url>"
            "</urlset>"
        )
        monkeypatch.setattr(
            "httpx.get",
            lambda *a, **kw: MagicMock(
                status_code=200, text=body, url="https://example.com/sitemap.xml"
            ),
        )
        count = _count_sitemap_urls("https://example.com/start", include_subdomains=True)
        assert count == 2

    def test_no_matching_entries_returns_unknown(self, monkeypatch):
        body = "<urlset><url><loc>https://other.com/a</loc></url></urlset>"
        monkeypatch.setattr("httpx.get", lambda *a, **kw: MagicMock(status_code=200, text=body))
        assert (
            _count_sitemap_urls("https://example.com/start", include_subdomains=False)
            == CRAWL_TOTAL_UNKNOWN
        )

    def test_cap_overrides_via_module_attribute(self, monkeypatch):
        """``_SITEMAP_MAX_URLS`` on the sitemap module bounds the scan."""
        from lilbee.crawler import sitemap as sitemap_mod

        monkeypatch.setattr(sitemap_mod, "_SITEMAP_MAX_URLS", 2)
        body = "".join(f"<url><loc>https://example.com/{i}</loc></url>" for i in range(10))
        monkeypatch.setattr(
            "httpx.get",
            lambda *a, **kw: MagicMock(
                status_code=200,
                text=f"<urlset>{body}</urlset>",
                url="https://example.com/sitemap.xml",
            ),
        )
        count = _count_sitemap_urls("https://example.com/start", include_subdomains=False)
        assert count == 2
