"""Host capability detection: recorded hosts, NVML readers and the live CPU."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from lilbee.providers.fleet import host_caps
from lilbee.providers.fleet.host_caps import (
    HostArch,
    HostOS,
    Npu,
    NpuVendor,
    collect,
    detect,
    nvml_device_totals,
    read_nvml,
)

FIXTURES = Path(__file__).parent / "fixtures" / "host_caps"


def _recorded(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.probe.json").read_text(encoding="utf-8"))


def _collect_recorded(name: str, npu_devices: list[str] | None = None):
    probe = _recorded(name)
    cpu = {k: probe[k] for k in ("arch", "arch_string_raw", "brand_raw", "flags")}
    return collect(
        system=probe["system"],
        machine=probe["machine"],
        mac_ver=probe["mac_ver"],
        cpu_info=cpu,
        nvml=probe.get("nvml"),
        npu_devices=npu_devices or [],
    )


class _NvmlError(Exception):
    pass


def _fake_nvml(*, init_error: bool = False, gpus: int = 1) -> types.ModuleType:
    mod = types.ModuleType("pynvml")
    mod.NVMLError = _NvmlError
    calls: list[str] = []
    mod.calls = calls

    def init() -> None:
        calls.append("init")
        if init_error:
            raise _NvmlError("no library")

    mod.nvmlInit = init
    mod.nvmlShutdown = lambda: calls.append("shutdown")
    mod.nvmlDeviceGetCount = lambda: gpus
    mod.nvmlDeviceGetHandleByIndex = lambda i: i
    mod.nvmlDeviceGetUUID = lambda h: f"GPU-{h}".encode() if h else f"GPU-{h}"
    mod.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(total=(h + 1) * 1000)
    mod.nvmlDeviceGetName = lambda h: b"Test GPU"
    mod.nvmlDeviceGetCudaComputeCapability = lambda h: (8, 6)
    mod.nvmlSystemGetDriverVersion = lambda: "570.1"
    mod.nvmlSystemGetCudaDriverVersion = lambda: 12080
    return mod


class TestRecordedHosts:
    def test_apple_silicon_has_neon_from_the_architecture_and_no_avx2(self) -> None:
        caps = _collect_recorded("mac-m1pro")
        assert _recorded("mac-m1pro")["flags"] is None
        assert caps.neon.present
        assert not caps.avx2.present
        assert caps.avx2.reason == "not an x86 CPU"
        assert (caps.os, caps.arch, caps.macos_version) == (
            HostOS.MACOS,
            HostArch.ARM64,
            (14, 6, 1),
        )
        assert caps.gpus == ()
        assert caps.nvidia_driver_cuda is None
        assert caps.nvidia_reason == "no NVIDIA driver"

    def test_old_xeon_lacks_avx2_and_names_the_cpu(self) -> None:
        caps = _collect_recorded("home-xeon-1070ti")
        assert not caps.avx2.present
        assert "E5-2609" in caps.avx2.reason
        assert not caps.neon.present
        assert caps.neon.reason == "not an ARM CPU"
        assert [(g.name, g.compute) for g in caps.gpus] == [("NVIDIA GeForce GTX 1070 Ti", (6, 1))]
        assert caps.nvidia_driver_cuda == (12, 4)
        assert caps.nvidia_reason == ""
        assert caps.macos_version is None

    def test_epyc_without_gpu_has_avx2(self) -> None:
        caps = _collect_recorded("gcp-epyc-nogpu")
        assert caps.avx2.present
        assert caps.avx2.reason == ""
        assert (caps.os, caps.arch) == (HostOS.LINUX, HostArch.X86_64)
        assert caps.nvidia_reason == "no NVIDIA driver"


class TestCollect:
    def test_x86_cpu_with_unreadable_flags_is_absent_with_a_reason(self) -> None:
        caps = collect(
            system="Linux",
            machine="x86_64",
            mac_ver="",
            cpu_info={"flags": None, "brand_raw": "Odd CPU"},
            nvml=None,
            npu_devices=[],
        )
        assert not caps.avx2.present
        assert "Odd CPU" in caps.avx2.reason

    def test_windows_arm_and_amd_machines_map_to_one_architecture_each(self) -> None:
        arm = collect(
            system="Windows", machine="ARM64", mac_ver="", cpu_info={}, nvml=None, npu_devices=[]
        )
        amd = collect(
            system="Windows",
            machine="AMD64",
            mac_ver="",
            cpu_info={"flags": ["avx2"]},
            nvml=None,
            npu_devices=[],
        )
        assert (arm.os, arm.arch, arm.neon.present) == (HostOS.WINDOWS, HostArch.ARM64, True)
        assert (amd.arch, amd.avx2.present) == (HostArch.X86_64, True)

    def test_linux_aarch64_is_arm64(self) -> None:
        caps = collect(
            system="Linux", machine="aarch64", mac_ver="", cpu_info={}, nvml=None, npu_devices=[]
        )
        assert caps.arch is HostArch.ARM64

    def test_npu_names_map_to_vendors_and_unknown_names_are_dropped(self) -> None:
        caps = collect(
            system="Linux",
            machine="x86_64",
            mac_ver="",
            cpu_info={"flags": []},
            nvml=None,
            npu_devices=["Intel NPU 3720", "Qualcomm Hexagon", "AMD XDNA", "Mystery"],
        )
        assert caps.npus == (
            Npu(NpuVendor.INTEL, "Intel NPU 3720"),
            Npu(NpuVendor.QUALCOMM, "Qualcomm Hexagon"),
            Npu(NpuVendor.AMD, "AMD XDNA"),
        )

    @pytest.mark.parametrize(
        ("code", "expected"), [(12040, (12, 4)), (12080, (12, 8)), (11080, (11, 8))]
    )
    def test_cuda_driver_version_decodes_the_nvml_integer(self, code, expected) -> None:
        nvml = {"driver": "1", "cuda_driver": code, "gpus": [{"name": "G", "cc": [8, 0]}]}
        caps = collect(
            system="Linux",
            machine="x86_64",
            mac_ver="",
            cpu_info={"flags": []},
            nvml=nvml,
            npu_devices=[],
        )
        assert caps.nvidia_driver_cuda == expected

    def test_second_gpu_keeps_its_own_compute_capability(self) -> None:
        nvml = {
            "driver": "1",
            "cuda_driver": 12080,
            "gpus": [{"name": "A", "cc": [6, 1]}, {"name": "B", "cc": [9, 0]}],
        }
        caps = collect(
            system="Linux",
            machine="x86_64",
            mac_ver="",
            cpu_info={"flags": []},
            nvml=nvml,
            npu_devices=[],
        )
        assert [(g.name, g.compute) for g in caps.gpus] == [("A", (6, 1)), ("B", (9, 0))]


class TestNvmlReaders:
    def test_read_nvml_returns_driver_cuda_and_gpus_then_shuts_down(self, monkeypatch) -> None:
        fake = _fake_nvml()
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        assert read_nvml() == {
            "driver": "570.1",
            "cuda_driver": 12080,
            "gpus": [{"name": "Test GPU", "cc": [8, 6]}],
        }
        assert fake.calls == ["init", "shutdown"]

    def test_totals_decode_bytes_uuids_and_keep_driver_order(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "pynvml", _fake_nvml(gpus=2))
        assert nvml_device_totals() == [("GPU-0", 1000), ("GPU-1", 2000)]

    def test_totals_are_none_when_the_driver_lists_no_gpu(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "pynvml", _fake_nvml(gpus=0))
        assert nvml_device_totals() is None

    def test_a_failed_init_reads_as_no_driver_and_skips_shutdown(self, monkeypatch) -> None:
        fake = _fake_nvml(init_error=True)
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        assert read_nvml() is None
        assert nvml_device_totals() is None
        assert "shutdown" not in fake.calls

    def test_a_driver_error_mid_read_gives_no_answer_and_still_shuts_down(
        self, monkeypatch
    ) -> None:
        fake = _fake_nvml(gpus=2)

        def fail(handle):
            if handle == 1:
                raise _NvmlError("device lost")
            return types.SimpleNamespace(total=1000)

        fake.nvmlDeviceGetMemoryInfo = fail
        fake.nvmlDeviceGetName = lambda h: (_ for _ in ()).throw(_NvmlError("device lost"))
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        assert nvml_device_totals() is None
        assert read_nvml() is None
        assert fake.calls == ["init", "shutdown", "init", "shutdown"]

    def test_a_missing_binding_reads_as_no_driver(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "pynvml", None)
        assert read_nvml() is None
        assert nvml_device_totals() is None


class TestAccelDevices:
    def test_vendor_ids_map_to_npu_names(self, tmp_path, monkeypatch) -> None:
        for index, vendor in enumerate(["0x8086", "0x17CB", "0x10de"]):
            device = tmp_path / f"accel{index}" / "device"
            device.mkdir(parents=True)
            (device / "vendor").write_text(vendor + "\n", encoding="utf-8")
        monkeypatch.setattr(host_caps, "_SYSFS_ACCEL_DIR", tmp_path)
        assert host_caps._accel_device_names() == ["Intel NPU", "Qualcomm NPU"]

    def test_no_accel_class_lists_nothing(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(host_caps, "_SYSFS_ACCEL_DIR", tmp_path / "absent")
        assert host_caps._accel_device_names() == []


class TestDetect:
    def test_detect_wires_the_platform_cpu_and_nvml_readings(self, monkeypatch) -> None:
        import cpuinfo

        monkeypatch.setattr(host_caps.platform, "system", lambda: "Linux")
        monkeypatch.setattr(host_caps.platform, "machine", lambda: "x86_64")
        monkeypatch.setattr(host_caps.platform, "mac_ver", lambda: ("", ("", "", ""), ""))
        monkeypatch.setattr(cpuinfo, "get_cpu_info", lambda: {"flags": ["avx2"], "brand_raw": "X"})
        monkeypatch.setattr(host_caps, "read_nvml", lambda: None)
        monkeypatch.setattr(host_caps, "_accel_device_names", lambda: ["AMD NPU"])
        caps = detect()
        assert caps.avx2.present
        assert caps.nvidia_reason == "no NVIDIA driver"
        assert caps.npus == (Npu(NpuVendor.AMD, "AMD NPU"),)

    @pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/cpuinfo")
    def test_live_avx2_agrees_with_proc_cpuinfo(self) -> None:
        raw = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        flag_lines = [line for line in raw.splitlines() if line.startswith("flags")]
        assert flag_lines
        in_proc = any(" avx2" in line for line in flag_lines)
        if host_caps.platform.machine() == "x86_64":
            assert detect().avx2.present is in_proc
