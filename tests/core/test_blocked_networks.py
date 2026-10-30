"""The shared blocked networks and the IPv4 an IPv6 address embeds."""

import ipaddress

import pytest

from lilbee.core.blocked_networks import (
    ALWAYS_BLOCKED_NETWORKS,
    PRIVATE_NETWORKS,
    blocked_form,
    embedded_ipv4,
)
from lilbee.crawler.url_filter import get_blocked_networks

# Typed from the list the crawler held before the sets moved, so a dropped network fails here.
CRAWLER_NETWORKS = {
    "0.0.0.0/8",
    "127.0.0.0/8",
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "169.254.0.0/16",
    "100.64.0.0/10",
    "240.0.0.0/4",
    "::/128",
    "::1/128",
    "fe80::/10",
    "fc00::/7",
    "ff00::/8",
    "64:ff9b::/96",
}


def _texts(networks):
    return {str(network) for network in networks}


class TestSets:
    def test_the_two_sets_share_no_network(self):
        assert not _texts(ALWAYS_BLOCKED_NETWORKS) & _texts(PRIVATE_NETWORKS)

    def test_the_crawler_list_is_both_sets_less_ipv4_multicast(self):
        assert _texts(get_blocked_networks()) == CRAWLER_NETWORKS
        both = _texts(ALWAYS_BLOCKED_NETWORKS) | _texts(PRIVATE_NETWORKS)
        assert both - _texts(get_blocked_networks()) == {"224.0.0.0/4"}

    @pytest.mark.parametrize(
        "network", ["169.254.0.0/16", "224.0.0.0/4", "240.0.0.0/4", "64:ff9b::/96"]
    )
    def test_the_always_blocked_set_holds(self, network):
        assert network in _texts(ALWAYS_BLOCKED_NETWORKS)

    @pytest.mark.parametrize("network", ["127.0.0.0/8", "10.0.0.0/8", "fc00::/7", "::1/128"])
    def test_a_local_server_network_stays_out_of_the_always_blocked_set(self, network):
        assert network in _texts(PRIVATE_NETWORKS)
        assert network not in _texts(ALWAYS_BLOCKED_NETWORKS)


class TestEmbeddedIpv4:
    @pytest.mark.parametrize(
        ("address", "embedded"),
        [
            ("::ffff:169.254.169.254", "169.254.169.254"),
            ("2002:a9fe:a9fe::", "169.254.169.254"),
            ("64:ff9b::a9fe:a9fe", "169.254.169.254"),
            ("::ffff:0:a9fe:a9fe", "169.254.169.254"),
            ("::a9fe:a9fe", "169.254.169.254"),
            ("2001:0:4136:e378:8000:1:5601:5601", "169.254.169.254"),  # Teredo client
            ("2001:0:4136:e378:8000:1:f7f7:f7f7", "8.8.8.8"),  # Teredo client
            ("2002:808:808::", "8.8.8.8"),
        ],
    )
    def test_each_form_gives_the_address_it_embeds(self, address, embedded):
        assert embedded_ipv4(ipaddress.IPv6Address(address)) == ipaddress.IPv4Address(embedded)

    @pytest.mark.parametrize("address", ["::", "::1", "2606:4700:4700::1111"])
    def test_an_address_that_embeds_none_gives_none(self, address):
        assert embedded_ipv4(ipaddress.IPv6Address(address)) is None


class TestBlockedForm:
    def test_a_blocked_ipv4_is_returned(self):
        address = ipaddress.ip_address("169.254.169.254")
        assert blocked_form(address, ALWAYS_BLOCKED_NETWORKS) == address

    def test_an_allowed_ipv4_gives_none(self):
        assert blocked_form(ipaddress.ip_address("8.8.8.8"), ALWAYS_BLOCKED_NETWORKS) is None

    def test_the_embedded_form_is_the_blocked_one(self):
        address = ipaddress.ip_address("::ffff:169.254.169.254")
        assert blocked_form(address, ALWAYS_BLOCKED_NETWORKS) == ipaddress.IPv4Address(
            "169.254.169.254"
        )

    def test_a_blocked_ipv6_with_no_embedded_address_is_returned(self):
        address = ipaddress.ip_address("fe80::1")
        assert blocked_form(address, ALWAYS_BLOCKED_NETWORKS) == address

    @pytest.mark.parametrize(
        "address",
        [
            "2001:0:4136:e378:8000:1:5601:5601",  # Teredo client 169.254.169.254
            "2002:a9fe:a9fe::",  # 6to4 for 169.254.169.254
        ],
    )
    def test_a_tunnel_form_returns_the_blocked_ipv4_it_carries(self, address):
        assert blocked_form(
            ipaddress.ip_address(address), ALWAYS_BLOCKED_NETWORKS
        ) == ipaddress.IPv4Address("169.254.169.254")

    @pytest.mark.parametrize(
        "address",
        [
            "2001:0:4136:e378:8000:1:f7f7:f7f7",  # Teredo client 8.8.8.8
            "2002:808:808::",  # 6to4 for 8.8.8.8
        ],
    )
    def test_a_tunnel_form_with_a_public_ipv4_gives_none(self, address):
        assert blocked_form(ipaddress.ip_address(address), ALWAYS_BLOCKED_NETWORKS) is None

    def test_an_allowed_ipv6_with_an_allowed_embedded_address_gives_none(self):
        address = ipaddress.ip_address("::ffff:8.8.8.8")
        assert blocked_form(address, ALWAYS_BLOCKED_NETWORKS) is None
