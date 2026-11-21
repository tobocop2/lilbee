"""A test reads a fixed host unless it asks for the live read."""

from __future__ import annotations

import pytest

from lilbee.catalog import runnable
from lilbee.providers.fleet import host_caps
from lilbee.providers.fleet.host_caps import HostOS


def _count_cpu_probes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace the CPU probe, which starts a child process, with a counter."""
    probes: list[int] = []

    def _probe() -> dict[str, object]:
        probes.append(1)
        return {"flags": ["avx2"]}

    monkeypatch.setattr(host_caps, "_cpu_info", _probe)
    return probes


def test_the_default_host_read_starts_no_cpu_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    probes = _count_cpu_probes(monkeypatch)
    host = runnable.session_host()
    assert probes == []
    assert (host.os, host.avx2.present, host.gpus) == (HostOS.LINUX, True, ())


def _fixed_linux_host() -> host_caps.HostCaps:
    """The fixed host; a live Linux read lists accelerators, so its NPU list is known."""
    return host_caps.collect(
        system="Linux", machine="x86_64", cpu_info={"flags": ["avx2"]}, nvidia=None
    )


def test_the_default_host_is_the_fixed_linux_host() -> None:
    host = runnable.session_host()
    assert host == _fixed_linux_host()
    assert host.npu_known is False


@pytest.mark.real_host_read
def test_a_real_host_read_test_runs_the_live_read(monkeypatch: pytest.MonkeyPatch) -> None:
    probes = _count_cpu_probes(monkeypatch)
    runnable.session_host()
    assert probes == [1]


def test_a_refresh_after_an_install_reads_the_host_again(monkeypatch: pytest.MonkeyPatch) -> None:
    before = runnable.session_host()
    assert before == _fixed_linux_host()
    after = host_caps.collect(
        system="Darwin", machine="arm64", cpu_info={}, nvidia=None, mac_ver="14.5"
    )
    monkeypatch.setattr(runnable, "detect", lambda: after)
    assert runnable.session_host() is before
    assert runnable.refresh_session_host() is after
    assert runnable.session_host() is after
