"""The networks an outbound address is checked against, and the IPv4 an IPv6 address embeds."""

from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urlsplit

IpNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network
IpAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

IPV4_MULTICAST = ipaddress.ip_network("224.0.0.0/4")
_NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")
_IPV4_TRANSLATED = ipaddress.ip_network("::ffff:0:0:0/96")  # RFC 6052 SIIT

# No outbound request may reach these, whatever asked for it.
ALWAYS_BLOCKED_NETWORKS: tuple[IpNetwork, ...] = (
    ipaddress.ip_network("0.0.0.0/8"),  # "this host" range; 0.0.0.0 routes to localhost
    ipaddress.ip_network("169.254.0.0/16"),  # link-local, holds the cloud metadata address
    IPV4_MULTICAST,
    ipaddress.ip_network("240.0.0.0/4"),  # RFC 1112 reserved; holds broadcast 255.255.255.255
    ipaddress.ip_network("::/128"),  # IPv6 unspecified
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
    ipaddress.ip_network("ff00::/8"),  # IPv6 multicast
    _NAT64_PREFIX,
)

# The user's own machine and network: a local server may live here, a crawled page may not.
PRIVATE_NETWORKS: tuple[IpNetwork, ...] = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("100.64.0.0/10"),  # RFC 6598 shared / CGNAT
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),  # IPv6 unique-local (ULA)
)


def embedded_ipv4(ip: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    """Return the IPv4 an IPv6 address embeds, if any.

    Covers IPv4-mapped (``::ffff:a.b.c.d``), 6to4 (``2002::``), the Teredo
    client (``2001::/32``), the NAT64 well-known prefix, the IPv4-translated/SIIT
    prefix (``::ffff:0:a.b.c.d``), and the deprecated IPv4-compatible
    (``::a.b.c.d``) form. Each can reach the same host as its bare IPv4, so the
    embedded address must face the blocklist.
    """
    if ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    if ip.sixtofour is not None:
        return ip.sixtofour
    if ip.teredo is not None:
        return ip.teredo[1]
    low32 = int(ip) & 0xFFFFFFFF
    if ip in _NAT64_PREFIX or ip in _IPV4_TRANSLATED:
        return ipaddress.IPv4Address(low32)
    # IPv4-compatible ::a.b.c.d: top 96 bits zero, excluding :: and ::1.
    if int(ip) >> 32 == 0 and low32 > 1:
        return ipaddress.IPv4Address(low32)
    return None


def blocked_form(ip: IpAddress, networks: tuple[IpNetwork, ...]) -> IpAddress | None:
    """The first of *ip* and the IPv4 it embeds that lies in *networks*, or None."""
    candidates: list[IpAddress] = [ip]
    if isinstance(ip, ipaddress.IPv6Address):
        embedded = embedded_ipv4(ip)
        if embedded is not None:
            candidates.append(embedded)
    for candidate in candidates:
        if any(candidate in network for network in networks):
            return candidate
    return None


_DEFAULT_PORTS = {"http": 80, "https": 443}
# What a base URL may hold: printable ASCII, less a space and a backslash. A URL standard
# parser reads these differently from urlsplit, or folds them into an address.
_URL_PATTERN = re.compile(r"[!-\[\]-~]+")
_IPV4_NUMBER = re.compile(r"0[xX][0-9a-fA-F]*|[0-9]+")
_IPV4_MAX_PARTS = 4
_IPV4_PART_BASE = 256
_LOCAL_NAT64_PREFIX = ipaddress.ip_network("64:ff9b:1::/48")  # RFC 8215
_ISATAP_INTERFACE_PREFIXES = (0x00005EFE, 0x02005EFE)  # RFC 5214: bits 32 to 63 of the address
_NOT_AN_IPV4 = "The URL host is not a valid IPv4 address."
_NOT_AN_IPV6 = "The URL host is not a valid IPv6 address."


def _ipv4_part(part: str) -> int:
    """The value of one part of a numeric host by the URL standard."""
    if part[:2].lower() == "0x":
        return int(part[2:] or "0", 16)
    return int(part, 8 if len(part) > 1 and part.startswith("0") else 10)


def _numeric_host(host: str) -> ipaddress.IPv4Address | None:
    """The IPv4 a host that ends in a number names by the URL standard; None for a host name.

    A connection library reads ``2852039166``, ``0xa9.254.0xa9fe`` and ``0251.0376.0251.0376``
    as addresses without asking DNS. ValueError when the host ends in a number and names none.
    """
    parts = host.removesuffix(".").split(".")
    *_, final_part = parts
    if not _IPV4_NUMBER.fullmatch(final_part):
        return None
    if len(parts) > _IPV4_MAX_PARTS or not all(_IPV4_NUMBER.fullmatch(part) for part in parts):
        raise ValueError(_NOT_AN_IPV4)
    try:
        *leading, last = [_ipv4_part(part) for part in parts]
    except ValueError:  # an octal part with an 8 or a 9
        raise ValueError(_NOT_AN_IPV4) from None
    # The last part fills every byte the parts before it leave.
    room = _IPV4_PART_BASE ** (_IPV4_MAX_PARTS - len(leading))
    if last >= room or any(part >= _IPV4_PART_BASE for part in leading):
        raise ValueError(_NOT_AN_IPV4)
    value = 0
    for part in leading:
        value = value * _IPV4_PART_BASE + part
    return ipaddress.IPv4Address(value * room + last)


def url_host(url: str) -> str:
    """The host of a well-formed base URL; ValueError that repeats no part of *url*."""
    if not _URL_PATTERN.fullmatch(url):
        raise ValueError(
            "A server URL needs printable ASCII with no space or backslash. "
            "Write an international host name in its xn-- form."
        )
    if "?" in url or "#" in url:
        raise ValueError("A server URL cannot hold a query or a fragment.")
    try:
        parts = urlsplit(url)
        if parts.port == 0:
            raise ValueError
    except ValueError:
        raise ValueError("The server URL does not parse, or its port is not valid.") from None
    if parts.scheme.lower() not in _DEFAULT_PORTS:
        raise ValueError("A server URL must use the http or https scheme.")
    if "@" in parts.netloc:
        raise ValueError("A server URL cannot hold a user name or password. Use the API key.")
    if "%" in parts.netloc:
        raise ValueError("A server URL cannot hold a percent sign or a zone id in its host.")
    if not parts.hostname:
        raise ValueError("A server URL needs a host.")
    if parts.netloc.startswith("[") and ":" not in parts.hostname:
        raise ValueError(_NOT_AN_IPV6)  # urlsplit admits a bracketed [v1.abc]; a connection cannot
    return parts.hostname


def url_origin(url: str) -> tuple[str, str, int] | None:
    """The scheme, host and port that *url* connects to; None when *url* is not well formed."""
    try:
        host = url_host(url)
    except ValueError:
        return None
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    return scheme, host, _DEFAULT_PORTS[scheme] if parts.port is None else parts.port


def _literal_address(host: str) -> IpAddress | None:
    """The address *host* names without DNS, or None for a host name."""
    if ":" in host:
        try:
            return ipaddress.IPv6Address(host)
        except ValueError:
            raise ValueError(_NOT_AN_IPV6) from None
    return _numeric_host(host)


def _tunnels_to_ipv4(address: IpAddress) -> bool:
    """True for a local-use NAT64, Teredo or ISATAP address: each can carry a blocked IPv4."""
    if not isinstance(address, ipaddress.IPv6Address):
        return False
    return (
        address in _LOCAL_NAT64_PREFIX
        or address.teredo is not None
        or (int(address) >> 32) & 0xFFFFFFFF in _ISATAP_INTERFACE_PREFIXES
    )


def _refuse_blocked(address: IpAddress) -> None:
    """Refuse *address* when it, or the IPv4 it embeds, lies in an always-blocked network."""
    blocked = blocked_form(address, ALWAYS_BLOCKED_NETWORKS)
    if blocked is None and _tunnels_to_ipv4(address):
        blocked = address
    if blocked is not None:
        raise ValueError(f"A server preset cannot use the blocked address {blocked}.")


def _resolved_addresses(host: str) -> list[IpAddress]:
    """Every address DNS gives for *host*; empty when it gives none."""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return []
    except UnicodeError:
        raise ValueError("The URL host is not a valid host name.") from None
    return [ipaddress.ip_address(sockaddr[0]) for *_, sockaddr in infos]


def check_url_form(url: str) -> str | None:
    """Check all of *url* that needs no DNS; return its host name, or None for an address."""
    host = url_host(url)
    literal = _literal_address(host)
    if literal is None:
        return host
    _refuse_blocked(literal)
    return None


def validate_preset_url(url: str) -> None:
    """Refuse a URL no preset may save; a host with no DNS answer is accepted."""
    host = check_url_form(url)
    if host is not None:
        for address in _resolved_addresses(host):
            _refuse_blocked(address)


def check_request_address(url: str) -> None:
    """Refuse a request unless every address of the URL's host is allowed at this moment.

    The SDK resolves the host again when it connects, so a DNS change between this check
    and the connection is not refused.
    """
    host = check_url_form(url)
    if host is None:
        return
    addresses = _resolved_addresses(host)
    if not addresses:
        raise ValueError("The server host has no DNS answer.")
    for address in addresses:
        _refuse_blocked(address)
