"""NPU detection answers from fixtures; an NPU list that cannot be read is unknown, not absent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lilbee.providers.fleet import host_caps
from lilbee.providers.fleet.engines.host_gate import npu_gate
from lilbee.providers.fleet.host_caps import HostCaps, Npu, NpuVendor, collect

FIXTURES = Path(__file__).parent / "fixtures" / "host_caps"


def _host(name: str) -> HostCaps:
    data = json.loads((FIXTURES / f"{name}.accel.json").read_text(encoding="utf-8"))
    assert data["source"] == "doc"
    accel = data["accel"]
    return collect(
        system=data["system"],
        machine=data["machine"],
        cpu_info={},
        nvml=None,
        accel_devices=None if accel is None else [tuple(pair) for pair in accel],
    )


def test_an_intel_accelerator_reads_as_an_intel_npu() -> None:
    caps = _host("intel-core-ultra-linux")
    assert caps.npu_known
    assert caps.npus == (Npu(NpuVendor.INTEL, "Intel NPU"),)
    assert npu_gate(caps, NpuVendor.INTEL).auto_selectable


def test_no_accelerator_on_linux_is_no_npu() -> None:
    caps = _host("no-accel-linux")
    assert (caps.npu_known, caps.npus) == (True, ())
    gate = npu_gate(caps, NpuVendor.INTEL)
    assert (gate.ok, gate.known) == (False, True)


def test_l5_s1_windows_answers_unknown_never_no_npu() -> None:
    caps = _host("windows-amd64")
    assert (caps.npu_known, caps.npus) == (False, ())
    assert caps.npu_reason == "lilbee cannot read the NPU list on windows."
    gate = npu_gate(caps, NpuVendor.AMD)
    assert (gate.ok, gate.known, gate.auto_selectable) == (True, False, False)
    assert "windows" in gate.reason


def test_s1_macos_answers_unknown() -> None:
    gate = npu_gate(_host("mac-arm64"), NpuVendor.INTEL)
    assert (gate.ok, gate.known) == (True, False)
    assert "macos" in gate.reason


def test_s2_a_qualcomm_accelerator_on_linux_is_refused_by_name() -> None:
    caps = _host("qualcomm-datacentre-linux")
    gate = npu_gate(caps, NpuVendor.QUALCOMM)
    assert gate.ok is False
    assert gate.reason == "Qualcomm accelerator found, which is not a Snapdragon X NPU."


def test_s3_an_amd_npu_of_unread_generation_is_unknown_and_explicit_install_stands() -> None:
    caps = _host("amd-generation-unread-linux")
    gate = npu_gate(caps, NpuVendor.AMD, "XDNA2")
    assert (gate.ok, gate.known, gate.auto_selectable) == (True, False, False)
    assert gate.reason == "The generation of the AMD NPU could not be read."


def test_an_amd_xdna2_device_passes_the_xdna2_gate_and_fails_for_xdna() -> None:
    caps = _host("amd-xdna2-linux")
    assert npu_gate(caps, NpuVendor.AMD, "XDNA2").auto_selectable
    refused = npu_gate(caps, NpuVendor.AMD, "XDNA")
    assert (refused.ok, refused.reason) == (False, "AMD NPU is not XDNA.")


def test_an_unread_list_on_linux_is_unknown() -> None:
    caps = collect(system="Linux", machine="x86_64", cpu_info={}, nvml=None)
    assert (caps.npu_known, caps.npu_reason) == (False, host_caps.NPU_UNREAD)


def test_the_sysfs_reader_pairs_vendor_and_device_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = tmp_path / "accel0" / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text("0x1022\n", encoding="utf-8")
    (device / "device").write_text("0x17f0\n", encoding="utf-8")
    (tmp_path / "accel1" / "device").mkdir(parents=True)
    monkeypatch.setattr(host_caps, "_SYSFS_ACCEL_DIR", tmp_path)
    assert host_caps._accel_devices() == [("0x1022", "0x17f0")]


def test_detect_reads_sysfs_only_on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host_caps, "_accel_devices", lambda: [("0x8086", "0x7d1d")])
    monkeypatch.setattr(host_caps, "_cpu_info", lambda: {})
    monkeypatch.setattr(host_caps, "read_nvml", lambda: None)
    monkeypatch.setattr(host_caps.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(host_caps.platform, "system", lambda: "Linux")
    assert host_caps.detect().npus == (Npu(NpuVendor.INTEL, "Intel NPU"),)
    monkeypatch.setattr(host_caps.platform, "system", lambda: "Windows")
    assert host_caps.detect().npu_known is False
