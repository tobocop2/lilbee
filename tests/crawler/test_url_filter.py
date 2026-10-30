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

    @pytest.mark.parametrize(
        "addr",
        [
            "::",  # IPv6 unspecified
            "100.64.0.1",  # RFC 6598 CGNAT
            "64:ff9b::a9fe:a9fe",  # NAT64 embedding 169.254.169.254
            "2002:a9fe:a9fe::",  # 6to4 embedding 169.254.169.254
            "::169.254.169.254",  # IPv4-compatible embedding metadata
            "::ffff:0:7f00:1",  # IPv4-translated/SIIT embedding 127.0.0.1
        ],
    )
    def test_rejects_reserved_via_embedding_or_range(self, monkeypatch, addr):
        family = 10 if ":" in addr else 2
        sockaddr = (addr, 0, 0, 0) if family == 10 else (addr, 0)
        monkeypatch.setattr(
            "lilbee.crawler.url_filter.socket.getaddrinfo",
            lambda *a, **kw: [(family, 1, 6, "", sockaddr)],
        )
        with pytest.raises(ValueError, match="private/reserved"):
            validate_crawl_url("http://sneaky.example.com")

    @pytest.mark.parametrize(
        "addr",
        [
            "2001:4860:4860::8888",  # public IPv6 (Google DNS)
            "2002:808:808::",  # public 6to4 embedding 8.8.8.8
        ],
    )
    def test_accepts_public_ipv6(self, monkeypatch, addr):
        # Guards against an over-eager embedded_ipv4 edit blocking public v6.
        monkeypatch.setattr(
            "lilbee.crawler.url_filter.socket.getaddrinfo",
            lambda *a, **kw: [(10, 1, 6, "", (addr, 0, 0, 0))],
        )
        validate_crawl_url("http://public.example.com")  # must not raise

    @pytest.mark.parametrize(
        "addr",
        [
            "2001:0:4136:e378:8000:1:5601:5601",  # Teredo client 169.254.169.254
            "2001:0:4136:e378:8000:1:80ff:fffe",  # Teredo client 127.0.0.1
        ],
    )
    def test_rejects_a_teredo_address_whose_client_is_blocked(self, monkeypatch, addr):
        monkeypatch.setattr(
            "lilbee.crawler.url_filter.socket.getaddrinfo",
            lambda *a, **kw: [(10, 1, 6, "", (addr, 0, 0, 0))],
        )
        with pytest.raises(ValueError, match="private/reserved"):
            validate_crawl_url("http://teredo.example.com")

    def test_accepts_a_teredo_address_whose_client_is_public(self, monkeypatch):
        monkeypatch.setattr(
            "lilbee.crawler.url_filter.socket.getaddrinfo",
            lambda *a, **kw: [(10, 1, 6, "", ("2001:0:4136:e378:8000:1:f7f7:f7f7", 0, 0, 0))],
        )
        validate_crawl_url("http://teredo.example.com")  # client 8.8.8.8, must not raise

    def test_rejects_a_nat64_address_with_a_public_ipv4(self, monkeypatch):
        monkeypatch.setattr(
            "lilbee.crawler.url_filter.socket.getaddrinfo",
            lambda *a, **kw: [(10, 1, 6, "", ("64:ff9b::808:808", 0, 0, 0))],
        )
        with pytest.raises(ValueError, match="private/reserved"):
            validate_crawl_url("http://nat64.example.com")


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

    @pytest.mark.parametrize("addr", ["240.0.0.1", "255.255.255.255"])
    def test_contains_ipv4_reserved_range(self, addr):
        import ipaddress as _ip

        networks = get_blocked_networks()
        assert any(_ip.ip_address(addr) in n for n in networks)

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
        assert _fetch_sitemap_text("https://example.com/start", respect_robots_txt=False) is None

    def test_returns_none_on_4xx(self, monkeypatch):
        fake = MagicMock(status_code=404, text="")
        monkeypatch.setattr("httpx.get", lambda *a, **kw: fake)
        assert _fetch_sitemap_text("https://example.com/start", respect_robots_txt=False) is None

    def test_returns_body_on_success(self, monkeypatch):
        fake = MagicMock(status_code=200, text="<urlset></urlset>")
        fake.url = "https://example.com/sitemap.xml"
        monkeypatch.setattr("httpx.get", lambda *a, **kw: fake)
        assert (
            _fetch_sitemap_text("https://example.com/start", respect_robots_txt=False)
            == "<urlset></urlset>"
        )

    @pytest.mark.parametrize(
        ("user_agent", "headers"),
        [("Probe/1.0", {"User-Agent": "Probe/1.0"}), ("", None)],
        ids=["override", "blank"],
    )
    def test_the_request_names_the_crawl_user_agent(self, monkeypatch, user_agent, headers):
        sent: list[object] = []

        def _get(url: str, **kwargs: object) -> MagicMock:
            sent.append(kwargs["headers"])
            return MagicMock(status_code=404, text="")

        monkeypatch.setattr("httpx.get", _get)
        _count_sitemap_urls(
            "https://example.com/start",
            include_subdomains=False,
            respect_robots_txt=False,
            user_agent=user_agent,
        )
        assert sent == [headers]

    @pytest.mark.parametrize(
        ("seed", "sent_url", "auth"),
        [
            (
                "https://alice:s3%3Acr%40t@example.com/start",
                "https://example.com/sitemap.xml",
                ("alice", "s3:cr@t"),
            ),
            ("https://alice@example.com/start", "https://example.com/sitemap.xml", ("alice", "")),
            ("http://example.com:8080/start", "http://example.com:8080/sitemap.xml", None),
        ],
        ids=["username-and-password", "username-only", "no-userinfo"],
    )
    def test_the_seed_credentials_go_as_basic_auth_never_in_the_url(
        self, monkeypatch, seed, sent_url, auth
    ):
        sent: list[tuple[str, object]] = []

        def _get(url: str, **kwargs: object) -> MagicMock:
            sent.append((url, kwargs["auth"]))
            return MagicMock(status_code=200, text="<urlset></urlset>")

        monkeypatch.setattr("httpx.get", _get)
        assert _fetch_sitemap_text(seed, respect_robots_txt=False) == "<urlset></urlset>"
        assert sent == [(sent_url, auth)]

    def test_a_seed_that_does_not_parse_makes_no_request(self, monkeypatch):
        called: list[int] = []
        monkeypatch.setattr("httpx.get", lambda *a, **kw: called.append(1))
        assert (
            _fetch_sitemap_text("http://alice:se/cret@example.com/start", respect_robots_txt=False)
            is None
        )
        assert called == []

    def test_does_not_follow_redirects(self, monkeypatch):
        """A 30x is not followed; an unfollowed redirect yields no sitemap (SSRF-safe)."""
        fake = MagicMock(status_code=301, text="<urlset></urlset>")
        captured = {}

        def _get(url, *a, **kw):
            captured.update(kw)
            return fake

        monkeypatch.setattr("httpx.get", _get)
        assert _fetch_sitemap_text("https://example.com/start", respect_robots_txt=False) is None
        assert captured.get("follow_redirects") is False

    def test_validates_seed_before_any_fetch(self, monkeypatch):
        """A private/metadata seed host is rejected without making a request."""
        monkeypatch.setattr(
            "lilbee.crawler.url_filter.socket.getaddrinfo",
            lambda *a, **kw: [(2, 1, 6, "", ("169.254.169.254", 0))],
        )
        called = []
        monkeypatch.setattr("httpx.get", lambda *a, **kw: called.append(1))
        assert (
            _fetch_sitemap_text("http://metadata.internal/start", respect_robots_txt=False) is None
        )
        assert called == []


ROBOTS_URL = "https://example.com/robots.txt"
SITEMAP_URL = "https://example.com/sitemap.xml"
SITEMAP_BODY = "<urlset><url><loc>https://example.com/a</loc></url></urlset>"


def _serve(monkeypatch, answers: dict[str, MagicMock]) -> list[str]:
    """Route httpx.get to *answers* by URL; a URL not listed raises a connection error."""
    import httpx

    asked: list[str] = []

    def _get(url: str, **kwargs: object) -> MagicMock:
        asked.append(url)
        if url not in answers:
            raise httpx.ConnectError("refused")
        return answers[url]

    monkeypatch.setattr("httpx.get", _get)
    return asked


class TestSitemapRobots:
    def test_a_disallow_all_robots_txt_stops_the_sitemap_request(self, monkeypatch):
        robots = MagicMock(status_code=200, text="User-agent: *\nDisallow: /\n")
        asked = _serve(monkeypatch, {ROBOTS_URL: robots})
        assert _fetch_sitemap_text("https://example.com/start", respect_robots_txt=True) is None
        assert asked == [ROBOTS_URL]

    def test_an_allowing_robots_txt_lets_the_sitemap_through(self, monkeypatch):
        robots = MagicMock(status_code=200, text="User-agent: *\nDisallow: /private/\n")
        sitemap = MagicMock(status_code=200, text=SITEMAP_BODY)
        asked = _serve(monkeypatch, {ROBOTS_URL: robots, SITEMAP_URL: sitemap})
        assert (
            _fetch_sitemap_text("https://example.com/start", respect_robots_txt=True)
            == SITEMAP_BODY
        )
        assert asked == [ROBOTS_URL, SITEMAP_URL]

    @pytest.mark.parametrize(
        ("rules", "allowed"),
        [
            ("User-agent: *\nDisallow: /sitemap.xml\n", False),
            ("User-agent: *\nDisallow: /start\n", True),
        ],
        ids=["sitemap-path-disallowed", "seed-path-disallowed"],
    )
    def test_robots_txt_is_checked_against_the_sitemap_path(self, monkeypatch, rules, allowed):
        robots = MagicMock(status_code=200, text=rules)
        sitemap = MagicMock(status_code=200, text=SITEMAP_BODY)
        _serve(monkeypatch, {ROBOTS_URL: robots, SITEMAP_URL: sitemap})
        fetched = _fetch_sitemap_text("https://example.com/start", respect_robots_txt=True)
        assert fetched == (SITEMAP_BODY if allowed else None)

    @pytest.mark.parametrize("status", [404, 410])
    def test_a_missing_robots_txt_lets_the_sitemap_through(self, monkeypatch, status):
        robots = MagicMock(status_code=status, text="Not Found")
        sitemap = MagicMock(status_code=200, text=SITEMAP_BODY)
        asked = _serve(monkeypatch, {ROBOTS_URL: robots, SITEMAP_URL: sitemap})
        assert (
            _fetch_sitemap_text("https://example.com/start", respect_robots_txt=True)
            == SITEMAP_BODY
        )
        assert asked == [ROBOTS_URL, SITEMAP_URL]

    @pytest.mark.parametrize("status", [301, 401, 403, 429, 500, 503])
    def test_a_robots_txt_that_answers_with_an_error_stops_the_sitemap_request(
        self, monkeypatch, status
    ):
        robots = MagicMock(status_code=status, text="User-agent: *\nAllow: /\n")
        sitemap = MagicMock(status_code=200, text=SITEMAP_BODY)
        asked = _serve(monkeypatch, {ROBOTS_URL: robots, SITEMAP_URL: sitemap})
        assert _fetch_sitemap_text("https://example.com/start", respect_robots_txt=True) is None
        assert asked == [ROBOTS_URL]

    def test_a_robots_txt_that_cannot_be_reached_stops_the_sitemap_request(self, monkeypatch):
        sitemap = MagicMock(status_code=200, text=SITEMAP_BODY)
        asked = _serve(monkeypatch, {SITEMAP_URL: sitemap})
        assert _fetch_sitemap_text("https://example.com/start", respect_robots_txt=True) is None
        assert asked == [ROBOTS_URL]

    def test_with_the_setting_off_robots_txt_is_never_requested(self, monkeypatch):
        robots = MagicMock(status_code=200, text="User-agent: *\nDisallow: /\n")
        sitemap = MagicMock(status_code=200, text=SITEMAP_BODY)
        asked = _serve(monkeypatch, {ROBOTS_URL: robots, SITEMAP_URL: sitemap})
        assert (
            _fetch_sitemap_text("https://example.com/start", respect_robots_txt=False)
            == SITEMAP_BODY
        )
        assert asked == [SITEMAP_URL]

    def test_the_user_agent_picks_the_robots_txt_group(self, monkeypatch):
        rules = "User-agent: *\nAllow: /\n\nUser-agent: probebot\nDisallow: /\n"
        robots = MagicMock(status_code=200, text=rules)
        sitemap = MagicMock(status_code=200, text=SITEMAP_BODY)
        _serve(monkeypatch, {ROBOTS_URL: robots, SITEMAP_URL: sitemap})
        assert (
            _fetch_sitemap_text(
                "https://example.com/start", user_agent="probebot/1.0", respect_robots_txt=True
            )
            is None
        )
        assert (
            _fetch_sitemap_text("https://example.com/start", respect_robots_txt=True)
            == SITEMAP_BODY
        )

    def test_the_robots_request_carries_the_credentials_and_the_user_agent(self, monkeypatch):
        sent: list[tuple[str, object, object]] = []

        def _get(url: str, **kwargs: object) -> MagicMock:
            sent.append((url, kwargs["auth"], kwargs["headers"]))
            return MagicMock(status_code=404, text="")

        monkeypatch.setattr("httpx.get", _get)
        _fetch_sitemap_text(
            "https://alice:pw@example.com/start", user_agent="Probe/1.0", respect_robots_txt=True
        )
        assert sent[0] == (ROBOTS_URL, ("alice", "pw"), {"User-Agent": "Probe/1.0"})

    def test_the_robots_request_does_not_follow_redirects(self, monkeypatch):
        captured: list[object] = []

        def _get(url: str, **kwargs: object) -> MagicMock:
            captured.append(kwargs["follow_redirects"])
            return MagicMock(status_code=301, text="")

        monkeypatch.setattr("httpx.get", _get)
        assert _fetch_sitemap_text("https://example.com/start", respect_robots_txt=True) is None
        assert captured == [False]

    def test_a_private_seed_makes_no_robots_request(self, monkeypatch):
        monkeypatch.setattr(
            "lilbee.crawler.url_filter.socket.getaddrinfo",
            lambda *a, **kw: [(2, 1, 6, "", ("169.254.169.254", 0))],
        )
        asked = _serve(monkeypatch, {})
        assert (
            _fetch_sitemap_text("http://metadata.internal/start", respect_robots_txt=True) is None
        )
        assert asked == []


class TestSitemapCount:
    def test_missing_host_short_circuits(self):
        assert (
            _count_sitemap_urls("file:///foo", include_subdomains=False, respect_robots_txt=False)
            == CRAWL_TOTAL_UNKNOWN
        )

    def test_missing_body_returns_unknown(self, monkeypatch):
        monkeypatch.setattr("httpx.get", lambda *a, **kw: MagicMock(status_code=500, text=""))
        assert (
            _count_sitemap_urls(
                "https://example.com/start", include_subdomains=False, respect_robots_txt=False
            )
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
        count = _count_sitemap_urls(
            "https://example.com/start", include_subdomains=False, respect_robots_txt=False
        )
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
        count = _count_sitemap_urls(
            "https://example.com/start", include_subdomains=True, respect_robots_txt=False
        )
        assert count == 2

    def test_no_matching_entries_returns_unknown(self, monkeypatch):
        body = "<urlset><url><loc>https://other.com/a</loc></url></urlset>"
        monkeypatch.setattr("httpx.get", lambda *a, **kw: MagicMock(status_code=200, text=body))
        assert (
            _count_sitemap_urls(
                "https://example.com/start", include_subdomains=False, respect_robots_txt=False
            )
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
        count = _count_sitemap_urls(
            "https://example.com/start", include_subdomains=False, respect_robots_txt=False
        )
        assert count == 2
