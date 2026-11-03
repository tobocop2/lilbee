"""The macOS version in the host answer: parsed, absent off macOS, unknown when unreadable."""

from __future__ import annotations

import platform

import pytest

from lilbee.providers.fleet import host_caps
from lilbee.providers.fleet.host_caps import collect, detect


def _mac(mac_ver: str) -> host_caps.HostCaps:
    return collect(system="Darwin", machine="arm64", cpu_info={}, nvml=None, mac_ver=mac_ver)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("13.6", (13, 6)),
        ("14.0", (14, 0)),
        ("15.2", (15, 2)),
        ("14", (14,)),
        ("26.0.1", (26, 0, 1)),
    ],
)
def test_a_readable_macos_version_is_parsed(raw: str, expected: tuple[int, ...]) -> None:
    caps = _mac(raw)
    assert caps.macos_version == expected
    assert caps.macos_known is True
    assert caps.macos_reason == ""


@pytest.mark.parametrize("raw", ["", "unknown", "x.y"])
def test_an_unreadable_macos_version_is_unknown_with_a_reason(raw: str) -> None:
    caps = _mac(raw)
    assert caps.macos_version is None
    assert caps.macos_known is False
    assert caps.macos_reason == "The macOS version of this computer could not be read."


@pytest.mark.parametrize(("system", "machine"), [("Linux", "x86_64"), ("Windows", "AMD64")])
def test_a_host_that_is_not_macos_has_no_version_and_nothing_unknown(
    system: str, machine: str
) -> None:
    caps = collect(system=system, machine=machine, cpu_info={}, nvml=None, mac_ver="14.0")
    assert caps.macos_version is None
    assert caps.macos_known is True


def test_detect_reads_the_macos_version_from_the_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(platform, "mac_ver", lambda: ("15.2", ("", "", ""), "arm64"))
    monkeypatch.setattr(host_caps, "read_nvml", lambda: None)
    assert detect().macos_version == (15, 2)
