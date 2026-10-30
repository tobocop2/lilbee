"""The networks an outbound address is checked against, and the IPv4 an IPv6 address embeds."""

from __future__ import annotations

import ipaddress

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
