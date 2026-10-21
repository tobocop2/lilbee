"""URL validation, blocked-network checks, and host-scope helpers."""

from __future__ import annotations

import ipaddress
import re
import socket
from typing import TYPE_CHECKING
from urllib.parse import ParseResult, urlparse

# ada-url is in the optional ``crawler`` extra, so each function that needs it imports it on call.
if TYPE_CHECKING:
    import ada_url

_BLOCKED_NETWORKS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    ipaddress.ip_network("0.0.0.0/8"),  # "this host" range; 0.0.0.0 routes to localhost
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("100.64.0.0/10"),  # RFC 6598 shared / CGNAT
    ipaddress.ip_network("240.0.0.0/4"),  # RFC 1112 reserved; holds broadcast 255.255.255.255
    ipaddress.ip_network("::/128"),  # IPv6 unspecified
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
    ipaddress.ip_network("fc00::/7"),  # IPv6 unique-local (ULA)
    ipaddress.ip_network("ff00::/8"),  # IPv6 multicast
    ipaddress.ip_network("64:ff9b::/96"),  # NAT64 well-known prefix
)

_NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")
_IPV4_TRANSLATED = ipaddress.ip_network("::ffff:0:0:0/96")  # RFC 6052 SIIT
# The mark that ends a username and password in the host part of an address.
_USERINFO_END = "@"
# A WHATWG parser escapes a raw "^" in a path; the url crate 2.5.8 crawlberg 1.11.1 uses keeps it.
_CARET = "^"
_CARET_ESCAPE = "%5E"
_CARET_ESCAPE_LOWER = "%5e"
_ANY_CARET_ESCAPE = re.compile(_CARET_ESCAPE, re.IGNORECASE)
# What is shown for an address with an ``@`` that does not parse: where a password ends is unknown.
UNPARSED_ADDRESS = "(an address that does not parse)"
# The refusal of a port that is not a number; the port text can hold part of a password.
INVALID_PORT_MESSAGE = "URL has an invalid port"


def get_blocked_networks() -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    """Return blocked network list. Override in tests via monkeypatch."""
    return _BLOCKED_NETWORKS


def _embedded_ipv4(ip: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    """Return the IPv4 an IPv6 address embeds, if any.

    Covers IPv4-mapped (``::ffff:a.b.c.d``), 6to4 (``2002::``), the NAT64
    well-known prefix, the IPv4-translated/SIIT prefix (``::ffff:0:a.b.c.d``),
    and the deprecated IPv4-compatible (``::a.b.c.d``) form. Each can reach the
    same host as its bare IPv4, so the embedded address must face the blocklist.
    """
    if ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    if ip.sixtofour is not None:
        return ip.sixtofour
    low32 = int(ip) & 0xFFFFFFFF
    if ip in _NAT64_PREFIX or ip in _IPV4_TRANSLATED:
        return ipaddress.IPv4Address(low32)
    # IPv4-compatible ::a.b.c.d: top 96 bits zero, excluding :: and ::1.
    if int(ip) >> 32 == 0 and low32 > 1:
        return ipaddress.IPv4Address(low32)
    return None


def is_url(value: str) -> bool:
    """Check if a string is an HTTP/HTTPS URL."""
    return value.startswith(("http://", "https://"))


def parsed_url(url: str) -> ada_url.URL | None:
    """*url* parsed by the WHATWG URL standard, or None when it does not parse."""
    import ada_url

    try:
        return ada_url.URL(url)
    except ValueError:
        return None


def parsed_address(url: str) -> str | None:
    """*url* without its fragment as crawlberg 1.11.1's URL parser writes it; None if it fails."""
    parsed = parsed_url(_ANY_CARET_ESCAPE.sub(_CARET_ESCAPE_LOWER, url))
    if parsed is None:
        return None
    parsed.hash = ""
    return parsed.href.replace(_CARET_ESCAPE, _CARET)


def without_userinfo(url: str) -> str:
    """*url* without the username and password of its host part, as the URL standard reads it.

    An address with an ``@`` that does not parse is shown as ``UNPARSED_ADDRESS``.
    """
    if _USERINFO_END not in url:
        return url
    parsed = parsed_url(url)
    if parsed is None:
        return UNPARSED_ADDRESS
    if not parsed.username and not parsed.password:
        return url
    parsed.username = ""
    parsed.password = ""
    return parsed.href


def _checked_port(parsed: ParseResult) -> int | None:
    """The port of *parsed*; a ValueError naming no part of the address when it is no number."""
    try:
        return parsed.port
    except ValueError as exc:
        raise ValueError(INVALID_PORT_MESSAGE) from exc


def validate_crawl_url(url: str) -> None:
    """Validate a URL for crawling. Raises ValueError for unsafe URLs.
    Rejects private IPs, loopback, link-local, and non-HTTP schemes.
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"Only http:// and https:// URLs are allowed, got {scheme}://")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL has no hostname")
    _checked_port(parsed)

    try:
        addr_infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"Cannot resolve hostname: {hostname}") from exc

    networks = get_blocked_networks()
    for _family, _type, _proto, _canonname, sockaddr in addr_infos:
        ip = ipaddress.ip_address(sockaddr[0])
        # An IPv6 address can embed an IPv4 (mapped, 6to4, NAT64, compatible)
        # that reaches the same host but would slip past the IPv4 checks, so
        # test both the address and any embedded IPv4 against the blocklist.
        candidates: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = [ip]
        if isinstance(ip, ipaddress.IPv6Address):
            embedded = _embedded_ipv4(ip)
            if embedded is not None:
                candidates.append(embedded)
        for candidate in candidates:
            for network in networks:
                if candidate in network:
                    raise ValueError(f"Crawling private/reserved IP {candidate} is not allowed")


def require_valid_crawl_url(url: str) -> None:
    """Validate URL for crawling. Raises ValueError if invalid."""
    if not is_url(url):
        raise ValueError("URL must start with http:// or https://")
    validate_crawl_url(url)


def host_in_scope(link_host: str, host: str, *, include_subdomains: bool) -> bool:
    """Return True when ``link_host`` should be followed during a whole-site crawl."""
    if not link_host:
        return False
    if link_host == host:
        return True
    return include_subdomains and link_host.endswith(f".{host}")
