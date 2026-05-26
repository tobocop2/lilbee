"""URL validation, blocked-network checks, and host-scope helpers."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

_BLOCKED_NETWORKS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
    ipaddress.ip_network("fc00::/7"),  # IPv6 unique-local (ULA)
    ipaddress.ip_network("ff00::/8"),  # IPv6 multicast
)


def get_blocked_networks() -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    """Return blocked network list. Override in tests via monkeypatch."""
    return _BLOCKED_NETWORKS


def is_url(value: str) -> bool:
    """Check if a string is an HTTP/HTTPS URL."""
    return value.startswith(("http://", "https://"))


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

    try:
        addr_infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"Cannot resolve hostname: {hostname}") from exc

    for _family, _type, _proto, _canonname, sockaddr in addr_infos:
        ip = ipaddress.ip_address(sockaddr[0])
        for network in get_blocked_networks():
            if ip in network:
                raise ValueError(f"Crawling private/reserved IP {ip} is not allowed")


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
