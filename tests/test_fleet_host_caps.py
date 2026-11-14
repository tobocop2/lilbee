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


class _NvmlLibraryNotFound(_NvmlError):
    pass


class _NvmlDriverNotLoaded(_NvmlError):
    pass


class _NvmlMismatch(_NvmlError):
    pass


def _fake_nvml(*, init_error: type[Exception] | None = None, gpus: int = 1) -> types.ModuleType:
    mod = types.ModuleType("pynvml")
    mod.NVMLError = _NvmlError
    mod.NVMLError_LibraryNotFound = _NvmlLibraryNotFound
    mod.NVMLError_DriverNotLoaded = _NvmlDriverNotLoaded
    calls: list[str] = []
    mod.calls = calls

    def init() -> None:
        calls.append("init")
        if init_error is not None:
            raise init_error("init failed")

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

    @pytest.mark.parametrize("error", [_NvmlLibraryNotFound, _NvmlDriverNotLoaded])
    def test_a_missing_library_or_driver_reads_as_no_driver_and_skips_shutdown(
        self, monkeypatch, error
    ) -> None:
        fake = _fake_nvml(init_error=error)
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        assert read_nvml() is None
        assert nvml_device_totals() is None
        assert "shutdown" not in fake.calls

    @pytest.mark.parametrize("error", [_NvmlMismatch, _NvmlError])
    def test_any_other_init_error_reads_as_unknown_not_no_driver(self, monkeypatch, error) -> None:
        fake = _fake_nvml(init_error=error)
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        answer = read_nvml()
        assert answer == {"driver": None, "cuda_driver": None, "gpus": None}
        caps = _collect(nvml=answer)
        assert not caps.nvidia_known
        assert caps.nvidia_reason == host_caps.NVIDIA_UNREADABLE
        assert nvml_device_totals() is None
        assert "shutdown" not in fake.calls

    def test_a_driver_error_mid_read_gives_no_totals_and_still_shuts_down(
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
        assert read_nvml()["gpus"] is None
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

    def test_a_glob_that_fails_lists_nothing(self, monkeypatch) -> None:
        class FailingDir:
            def glob(self, pattern: str):
                raise PermissionError("denied")

        monkeypatch.setattr(host_caps, "_SYSFS_ACCEL_DIR", FailingDir())
        assert host_caps._accel_device_names() == []

    def test_a_vendor_file_that_is_not_text_is_skipped(self, tmp_path, monkeypatch) -> None:
        for index, content in enumerate([b"\xff\xfe\x00", b"0x8086\n"]):
            device = tmp_path / f"accel{index}" / "device"
            device.mkdir(parents=True)
            (device / "vendor").write_bytes(content)
        monkeypatch.setattr(host_caps, "_SYSFS_ACCEL_DIR", tmp_path)
        assert host_caps._accel_device_names() == ["Intel NPU"]

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

    def test_detect_reports_the_gpus_the_nvml_binding_lists(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "pynvml", _fake_nvml())
        monkeypatch.setattr(host_caps, "_accel_device_names", list)
        caps = detect()
        assert caps.nvidia_known
        assert caps.gpus[0].compute == (8, 6)
        assert caps.nvidia_driver_cuda == (12, 8)

    @pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/cpuinfo")
    def test_live_avx2_agrees_with_proc_cpuinfo(self) -> None:
        raw = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        flag_lines = [line for line in raw.splitlines() if line.startswith("flags")]
        assert flag_lines
        in_proc = any(" avx2" in line for line in flag_lines)
        if host_caps.platform.machine() == "x86_64":
            assert detect().avx2.present is in_proc


def _collect(**overrides):
    args = {
        "system": "Linux",
        "machine": "x86_64",
        "mac_ver": "",
        "cpu_info": {"flags": ["avx2"], "brand_raw": "X"},
        "nvml": None,
        "npu_devices": [],
    }
    return collect(**{**args, **overrides})


def _nvml_answer(**overrides) -> dict:
    answer = {"driver": "570.1", "cuda_driver": 12080, "gpus": [{"name": "G", "cc": [8, 6]}]}
    return {**answer, **overrides}


class TestUnknownIsNotAbsent:
    @pytest.mark.parametrize(
        ("system", "machine"),
        [("FreeBSD", "amd64"), ("Linux", "riscv64"), ("Linux", "armv7l"), ("Linux", "ppc64le")],
    )
    def test_unrecognised_platform_reports_unknown_and_keeps_other_fields(
        self, system, machine
    ) -> None:
        caps = _collect(system=system, machine=machine, nvml=_nvml_answer())
        if system == "FreeBSD":
            assert caps.os is None
        else:
            assert caps.os is HostOS.LINUX
        assert caps.arch is None
        assert not caps.avx2.known
        assert "architecture" in caps.avx2.reason
        assert not caps.neon.known
        assert caps.nvidia_driver_cuda == (12, 8)
        assert caps.gpus[0].compute == (8, 6)

    def test_unreadable_x86_flags_are_unknown_not_absent(self) -> None:
        caps = _collect(cpu_info={"flags": None, "brand_raw": "Odd CPU"})
        assert not caps.avx2.known
        assert "Odd CPU" in caps.avx2.reason
        assert "could not be read" in caps.avx2.reason

    def test_flags_that_are_not_a_collection_are_unknown(self) -> None:
        assert not _collect(cpu_info={"flags": 7}).avx2.known

    def test_flags_given_as_one_string_are_unknown(self) -> None:
        avx2 = _collect(cpu_info={"flags": "avx2 sse2", "brand_raw": "Odd"}).avx2
        assert (avx2.present, avx2.known) == (False, False)

    @pytest.mark.parametrize("system", ["Linux", "Windows"])
    def test_a_mac_version_is_read_only_on_macos(self, system) -> None:
        assert _collect(system=system, mac_ver="14.2").macos_version is None
        assert _collect(system="Darwin", machine="arm64", mac_ver="14.2").macos_version == (14, 2)

    def test_a_cpu_without_avx2_is_known_absent(self) -> None:
        avx2 = _collect(cpu_info={"flags": ["sse2"], "brand_raw": "Old"}).avx2
        assert (avx2.present, avx2.known) == (False, True)

    def test_a_present_feature_is_known(self) -> None:
        assert _collect().avx2.known

    @pytest.mark.parametrize("code", [None, "junk", -1, 0, 12.5, True])
    def test_junk_cuda_version_is_unknown_and_gpus_stay(self, code) -> None:
        caps = _collect(nvml=_nvml_answer(cuda_driver=code))
        assert caps.nvidia_driver_cuda is None
        assert not caps.nvidia_known
        assert caps.nvidia_reason == host_caps.NVIDIA_UNREADABLE
        assert caps.gpus[0].compute == (8, 6)

    @pytest.mark.parametrize(
        "gpu",
        [
            {"name": "G", "cc": [None, None]},
            {"name": "G", "cc": ["8", "6"]},
            {"name": "G", "cc": [8.0, 6.0]},
            {"name": "G", "cc": [True, False]},
            {"name": None, "cc": [8, 6]},
            {"name": 7, "cc": [8, 6]},
        ],
    )
    def test_a_gpu_with_a_junk_field_is_unknown_and_cuda_version_stays(self, gpu) -> None:
        caps = _collect(nvml=_nvml_answer(gpus=[gpu]))
        assert caps.gpus == ()
        assert not caps.nvidia_known
        assert caps.nvidia_driver_cuda == (12, 8)

    @pytest.mark.parametrize("gpus", [None, [{"name": "G", "cc": []}], [{"name": "G"}], 5])
    def test_junk_gpu_list_is_unknown_and_cuda_version_stays(self, gpus) -> None:
        caps = _collect(nvml=_nvml_answer(gpus=gpus))
        assert caps.gpus == ()
        assert not caps.nvidia_known
        assert caps.nvidia_driver_cuda == (12, 8)

    def test_no_driver_is_known_and_has_its_own_reason(self) -> None:
        caps = _collect(nvml=None)
        assert caps.nvidia_known
        assert caps.nvidia_reason == host_caps.NO_NVIDIA_DRIVER

    def test_a_readable_driver_is_known_with_no_reason(self) -> None:
        caps = _collect(nvml=_nvml_answer())
        assert (caps.nvidia_known, caps.nvidia_reason) == (True, "")

    def test_unparsable_macos_version_is_unknown(self) -> None:
        caps = _collect(system="Darwin", machine="arm64", mac_ver="14.x")
        assert caps.os is HostOS.MACOS
        assert caps.macos_version is None


class TestNvmlReadsNeverRaise:
    def test_oserror_at_init_reads_unknown_for_detection_and_none_for_totals(
        self, monkeypatch
    ) -> None:
        fake = _fake_nvml()

        def boom() -> None:
            raise OSError("library cannot load")

        fake.nvmlInit = boom
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        assert read_nvml() == {"driver": None, "cuda_driver": None, "gpus": None}
        assert nvml_device_totals() is None
        assert "shutdown" not in fake.calls

    def test_attribute_error_mid_read_is_none_for_totals(self, monkeypatch) -> None:
        fake = _fake_nvml()
        del fake.nvmlDeviceGetCount
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        assert nvml_device_totals() is None
        assert fake.calls == ["init", "shutdown"]

    def test_memory_info_none_is_none_for_totals(self, monkeypatch) -> None:
        fake = _fake_nvml()
        fake.nvmlDeviceGetMemoryInfo = lambda h: None
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        assert nvml_device_totals() is None

    def test_shutdown_error_keeps_the_totals_and_the_reading(self, monkeypatch) -> None:
        fake = _fake_nvml()

        def boom() -> None:
            raise _NvmlError("shutdown failed")

        fake.nvmlShutdown = boom
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        assert nvml_device_totals() == [("GPU-0", 1000)]
        assert read_nvml()["cuda_driver"] == 12080

    def test_error_on_the_cuda_version_call_leaves_the_other_fields(self, monkeypatch) -> None:
        fake = _fake_nvml()

        def boom() -> int:
            raise _NvmlError("function not found")

        fake.nvmlSystemGetCudaDriverVersion = boom
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        answer = read_nvml()
        assert answer == {
            "driver": "570.1",
            "cuda_driver": None,
            "gpus": [{"name": "Test GPU", "cc": [8, 6]}],
        }
        caps = _collect(nvml=answer)
        assert not caps.nvidia_known
        assert caps.gpus[0].compute == (8, 6)

    def test_junk_cuda_version_from_the_binding_reads_unknown(self, monkeypatch) -> None:
        fake = _fake_nvml()
        fake.nvmlSystemGetCudaDriverVersion = lambda: None
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        assert read_nvml()["cuda_driver"] is None

    def test_driver_error_reading_the_gpus_leaves_the_version(self, monkeypatch) -> None:
        fake = _fake_nvml()
        fake.nvmlDeviceGetName = lambda h: (_ for _ in ()).throw(_NvmlError("device lost"))
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        answer = read_nvml()
        assert answer["gpus"] is None
        assert answer["cuda_driver"] == 12080

    def test_empty_compute_capability_from_the_binding_is_unknown_gpus(self, monkeypatch) -> None:
        fake = _fake_nvml()
        fake.nvmlDeviceGetCudaComputeCapability = lambda h: ()
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        caps = _collect(nvml=read_nvml())
        assert caps.gpus == ()
        assert not caps.nvidia_known


class TestDetectNeverRaises:
    def test_unreadable_npu_vendor_file_is_skipped(self, tmp_path, monkeypatch) -> None:
        good = tmp_path / "accel0" / "device"
        good.mkdir(parents=True)
        (good / "vendor").write_text("0x8086\n", encoding="utf-8")
        bad = tmp_path / "accel1" / "device" / "vendor"
        bad.mkdir(parents=True)
        monkeypatch.setattr(host_caps, "_SYSFS_ACCEL_DIR", tmp_path)
        assert host_caps._accel_device_names() == ["Intel NPU"]

    def test_failing_cpu_probe_reads_unknown_avx2(self, monkeypatch) -> None:
        import cpuinfo

        def boom() -> dict:
            raise RuntimeError("probe failed")

        monkeypatch.setattr(host_caps.platform, "system", lambda: "Linux")
        monkeypatch.setattr(host_caps.platform, "machine", lambda: "x86_64")
        monkeypatch.setattr(cpuinfo, "get_cpu_info", boom)
        monkeypatch.setattr(host_caps, "read_nvml", lambda: None)
        monkeypatch.setattr(host_caps, "_accel_device_names", list)
        caps = detect()
        assert not caps.avx2.known

    def test_an_nvml_answer_with_missing_keys_is_unknown(self) -> None:
        caps = _collect(nvml={"driver": "570.1"})
        assert not caps.nvidia_known
        assert caps.gpus == ()


_CARD_TOTAL = 24 << 30
_CARD_FREE = 20 << 30


class _FakeDriver:
    """The calls of the driver library the card read uses."""

    def __init__(self, count: int = 1, memory_error: bool = False) -> None:
        self._count = count
        self._memory_error = memory_error

    def nvmlDeviceGetCount(self) -> int:
        return self._count

    def nvmlDeviceGetHandleByIndex(self, index: int) -> int:
        return index

    def nvmlDeviceGetMemoryInfo(self, _handle: int) -> types.SimpleNamespace:
        if self._memory_error:
            raise RuntimeError("driver gone")
        return types.SimpleNamespace(total=_CARD_TOTAL, free=_CARD_FREE)

    def nvmlDeviceGetUUID(self, _handle: int) -> bytes:
        return b"GPU-a"

    def nvmlDeviceGetName(self, _handle: int) -> str:
        return "card0"

    def nvmlShutdown(self) -> None:
        return None


def test_the_card_read_reports_the_driver_figures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host_caps, "_open_nvml", _FakeDriver)
    assert host_caps.read_card_memory() == [
        host_caps.CardMemory("GPU-a", "card0", _CARD_TOTAL, _CARD_FREE)
    ]


def test_the_card_read_is_empty_when_the_driver_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raises() -> None:
        raise OSError("no library")

    monkeypatch.setattr(host_caps, "_open_nvml", _raises)
    assert host_caps.read_card_memory() == []
    monkeypatch.setattr(host_caps, "_open_nvml", lambda: None)
    assert host_caps.read_card_memory() == []
    monkeypatch.setattr(host_caps, "_open_nvml", lambda: _FakeDriver(memory_error=True))
    assert host_caps.read_card_memory() == []
