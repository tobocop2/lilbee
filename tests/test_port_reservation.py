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

    def test_a_released_port_becomes_available_again(self, monkeypatch) -> None:
        # Stated against a known range rather than the host's. Where the kernel's
        # range cannot be read, which is every Windows host, the picker hands the
        # choice to the OS and reserves nothing, so reservation is not a property
        # of every platform; it is a property of the sub-ephemeral path, and that
        # is the path worth pinning.
        monkeypatch.setattr(sm, "_ephemeral_range", lambda: (32768, 60999))
        first = sm._pick_free_ports(2)
        assert all(p in sm._reserved_ports for p in first)
        sm.release_reserved_ports(first)
        assert not any(p in sm._reserved_ports for p in first)
        second = sm._pick_free_ports(2)
        sm.release_reserved_ports(second)

    def test_an_unknown_range_reserves_nothing_and_still_works(self, monkeypatch) -> None:
        # The Windows shape: no range to read, so the OS chooses and there is
        # nothing to reserve. A fleet still starts, which is the point.
        monkeypatch.setattr(sm, "_ephemeral_range", lambda: None)
        ports = sm._pick_free_ports(2)
        assert len(set(ports)) == 2
        assert not any(p in sm._reserved_ports for p in ports)

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


class TestTheSubEphemeralSearch:
    """The branches that only run when a range is known and the window is busy."""

    def test_a_reserved_port_is_skipped_rather_than_rebound(self, monkeypatch) -> None:
        # Two picks in a row must not collide, which means the search has to step
        # over what it already handed out rather than offering it again.
        monkeypatch.setattr(sm, "_ephemeral_range", lambda: (32768, 60999))
        held = sm._pick_free_ports(3)
        try:
            nxt = sm._pick_free_ports(3)
            assert not set(held) & set(nxt)
            sm.release_reserved_ports(nxt)
        finally:
            sm.release_reserved_ports(held)

    def test_a_sysctl_range_is_read_when_procfs_is_absent(self, monkeypatch, tmp_path) -> None:
        # The macOS path: no /proc, so the range comes from sysctl's two numbers.
        import subprocess

        monkeypatch.setattr(sm, "_PROC_PORT_RANGE", tmp_path / "missing")
        monkeypatch.setattr(
            sm.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="49152 65535\n", stderr=""),
        )
        assert sm._ephemeral_range() == (49152, 65535)

    def test_a_sysctl_that_answers_nonsense_is_no_answer(self, monkeypatch, tmp_path) -> None:
        import subprocess

        monkeypatch.setattr(sm, "_PROC_PORT_RANGE", tmp_path / "missing")
        monkeypatch.setattr(
            sm.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="not numbers\n", stderr=""),
        )
        assert sm._ephemeral_range() is None

    def test_a_port_another_process_holds_is_stepped_over(self, monkeypatch) -> None:
        # Not reserved by us and not free either: somebody else has it. The search
        # has to keep walking rather than fail the fleet start.
        #
        # The refusal is injected rather than staged by binding a real port. A
        # real blocker races every other worker in a parallel run for the same
        # number, which leaves this branch covered on one machine and uncovered
        # on the next, failing the coverage gate rather than the test.
        import socket

        # The reserved set is a module global, so whatever ran earlier in this
        # worker decides whether the first candidate is already reserved. If it
        # is, the loop short-circuits before the bind and this branch never runs,
        # which leaves it covered alone and uncovered in a parallel suite.
        monkeypatch.setattr(sm, "_reserved_ports", set())
        monkeypatch.setattr(sm, "_ephemeral_range", lambda: (32768, 60999))
        first_candidate = sm._search_start((32768, 60999))
        real_bind = socket.socket.bind

        def _refuse_the_first_candidate(sock, address):
            if address[1] == first_candidate:
                raise OSError(48, "Address already in use")
            return real_bind(sock, address)

        monkeypatch.setattr(socket.socket, "bind", _refuse_the_first_candidate)
        ports = sm._pick_free_ports(1)
        try:
            assert ports[0] != first_candidate
        finally:
            sm.release_reserved_ports(ports)
