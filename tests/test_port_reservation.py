"""Picking a port is a reservation until the child actually binds it.

llama-swap binds a member port lazily, on that member's first request, so every
port lilbee hands out is unbound for an unbounded stretch. Probing with a socket
and closing it proves the port was free at that instant and reserves nothing, so
a picker that always searches from the same place hands the next caller exactly
what it gave the last one.
"""

from __future__ import annotations

from lilbee.providers.fleet import swap_manager as sm


class TestConsecutivePicksDoNotOverlap:
    def test_two_groups_started_in_a_row_get_different_ports(self) -> None:
        first = sm._pick_free_ports(3)
        second = sm._pick_free_ports(3)
        try:
            assert not set(first) & set(second)
        finally:
            sm.release_reserved_ports(first + second)

    def test_a_released_port_becomes_available_again(self) -> None:
        first = sm._pick_free_ports(2)
        sm.release_reserved_ports(first)
        second = sm._pick_free_ports(2)
        try:
            assert set(first) == set(second)  # nothing else took them meanwhile
        finally:
            sm.release_reserved_ports(second)

    def test_every_pick_is_internally_distinct(self) -> None:
        ports = sm._pick_free_ports(6)
        try:
            assert len(set(ports)) == 6
        finally:
            sm.release_reserved_ports(ports)

    def test_exhausting_the_window_still_yields_ports(self, monkeypatch) -> None:
        # With the sub-ephemeral window fully reserved the picker must fall back
        # to letting the OS choose rather than failing the fleet start.
        monkeypatch.setattr(sm, "_PORT_SEARCH_ATTEMPTS", 2)
        held = sm._pick_free_ports(2)
        try:
            extra = sm._pick_free_ports(2)
            assert len(set(extra)) == 2
            assert not set(held) & set(extra)
            sm.release_reserved_ports(extra)
        finally:
            sm.release_reserved_ports(held)


def test_separate_processes_do_not_start_at_the_same_offset(monkeypatch) -> None:
    # Cross-process collision cannot be reserved away, so the search start is
    # spread by pid instead of every process scanning from the same floor.
    starts = set()
    for pid in (1000, 1001, 1002, 1003):
        monkeypatch.setattr(sm.os, "getpid", lambda p=pid: p)
        starts.add(sm._search_start((32768, 60999)))
    assert len(starts) > 1
