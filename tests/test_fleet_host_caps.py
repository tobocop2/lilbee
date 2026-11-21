"""Host capability detection: recorded hosts, the NVIDIA answer and the live CPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from lilbee.providers.fleet import host_caps
from lilbee.providers.fleet.gpu_backends import nvidia as nvidia_mod
from lilbee.providers.fleet.host_caps import (
    HostArch,
    HostOS,
    collect,
    detect,
)
from tests._nvidia_smi import prints

FIXTURES = Path(__file__).parent / "fixtures" / "host_caps"


def _smi_prints(_tool: str, args: list[str], _timeout: float = 5.0) -> str:
    """What a host with one 'Test GPU' prints to each nvidia-smi query."""
    if "-x" in args:
        return "<nvidia_smi_log><cuda_version>12.8</cuda_version></nvidia_smi_log>\n"
    if any("compute_cap" in a for a in args):
        return "0, 8.6\n"
    return "0, GPU-0, 1000, 900, Test GPU\n"


def _recorded(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.probe.json").read_text(encoding="utf-8"))


def _collect_recorded(name: str):
    probe = _recorded(name)
    cpu = {k: probe[k] for k in ("arch", "arch_string_raw", "brand_raw", "flags")}
    return collect(
        system=probe["system"],
        machine=probe["machine"],
        cpu_info=cpu,
        nvidia=probe.get("nvml"),
    )


class TestRecordedHosts:
    def test_apple_silicon_has_neon_from_the_architecture_and_no_avx2(self) -> None:
        caps = _collect_recorded("mac-m1pro")
        assert _recorded("mac-m1pro")["flags"] is None
        assert caps.neon.present
        assert not caps.avx2.present
        assert caps.avx2.reason == "This is not an x86 CPU."
        assert (caps.os, caps.arch) == (HostOS.MACOS, HostArch.ARM64)
        assert caps.gpus == ()
        assert caps.nvidia_driver_cuda is None
        assert caps.nvidia_reason == "no NVIDIA driver"

    def test_old_xeon_lacks_avx2_and_names_the_cpu(self) -> None:
        caps = _collect_recorded("home-xeon-1070ti")
        assert not caps.avx2.present
        assert "E5-2609" in caps.avx2.reason
        assert not caps.neon.present
        assert caps.neon.reason == "This is not an ARM CPU."
        assert [(g.name, g.compute) for g in caps.gpus] == [("NVIDIA GeForce GTX 1070 Ti", (6, 1))]
        assert caps.nvidia_driver_cuda == (12, 4)
        assert caps.nvidia_reason == ""

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
            cpu_info={"flags": None, "brand_raw": "Odd CPU"},
            nvidia=None,
        )
        assert not caps.avx2.present
        assert "Odd CPU" in caps.avx2.reason

    def test_windows_arm_and_amd_machines_map_to_one_architecture_each(self) -> None:
        arm = collect(system="Windows", machine="ARM64", cpu_info={}, nvidia=None)
        amd = collect(
            system="Windows",
            machine="AMD64",
            cpu_info={"flags": ["avx2"]},
            nvidia=None,
        )
        assert (arm.os, arm.arch, arm.neon.present) == (HostOS.WINDOWS, HostArch.ARM64, True)
        assert (amd.arch, amd.avx2.present) == (HostArch.X86_64, True)

    def test_linux_aarch64_is_arm64(self) -> None:
        caps = collect(system="Linux", machine="aarch64", cpu_info={}, nvidia=None)
        assert caps.arch is HostArch.ARM64

    @pytest.mark.parametrize(
        ("code", "expected"), [(12040, (12, 4)), (12080, (12, 8)), (11080, (11, 8))]
    )
    def test_cuda_driver_version_decodes_the_encoded_integer(self, code, expected) -> None:
        nvidia = {"driver": "1", "cuda_driver": code, "gpus": [{"name": "G", "cc": [8, 0]}]}
        caps = collect(
            system="Linux",
            machine="x86_64",
            cpu_info={"flags": []},
            nvidia=nvidia,
        )
        assert caps.nvidia_driver_cuda == expected

    def test_second_gpu_keeps_its_own_compute_capability(self) -> None:
        nvidia = {
            "driver": "1",
            "cuda_driver": 12080,
            "gpus": [{"name": "A", "cc": [6, 1]}, {"name": "B", "cc": [9, 0]}],
        }
        caps = collect(
            system="Linux",
            machine="x86_64",
            cpu_info={"flags": []},
            nvidia=nvidia,
        )
        assert [(g.name, g.compute) for g in caps.gpus] == [("A", (6, 1)), ("B", (9, 0))]


class TestDetect:
    def test_detect_wires_the_platform_cpu_and_driver_readings(self, monkeypatch) -> None:
        import cpuinfo

        monkeypatch.setattr(host_caps.platform, "system", lambda: "Linux")
        monkeypatch.setattr(host_caps.platform, "machine", lambda: "x86_64")
        monkeypatch.setattr(cpuinfo, "get_cpu_info", lambda: {"flags": ["avx2"], "brand_raw": "X"})
        monkeypatch.setattr(host_caps, "host_facts", lambda: None)
        caps = detect()
        assert caps.avx2.present
        assert caps.nvidia_reason == "no NVIDIA driver"

    def test_detect_reports_the_gpus_nvidia_smi_lists(self, monkeypatch) -> None:
        monkeypatch.setattr(nvidia_mod, "run_tool", prints(_smi_prints))
        caps = detect()
        assert caps.nvidia_known
        assert [(g.name, g.compute) for g in caps.gpus] == [("Test GPU", (8, 6))]
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
        "cpu_info": {"flags": ["avx2"], "brand_raw": "X"},
        "nvidia": None,
    }
    return collect(**{**args, **overrides})


def _nvidia_answer(**overrides) -> dict:
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
        caps = _collect(system=system, machine=machine, nvidia=_nvidia_answer())
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

    def test_a_cpu_without_avx2_is_known_absent(self) -> None:
        avx2 = _collect(cpu_info={"flags": ["sse2"], "brand_raw": "Old"}).avx2
        assert (avx2.present, avx2.known) == (False, True)

    def test_a_present_feature_is_known(self) -> None:
        assert _collect().avx2.known

    @pytest.mark.parametrize("code", [None, "junk", -1, 0, 12.5, True])
    def test_junk_cuda_version_is_unknown_and_gpus_stay(self, code) -> None:
        caps = _collect(nvidia=_nvidia_answer(cuda_driver=code))
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
        caps = _collect(nvidia=_nvidia_answer(gpus=[gpu]))
        assert caps.gpus == ()
        assert not caps.nvidia_known
        assert caps.nvidia_driver_cuda == (12, 8)

    @pytest.mark.parametrize("gpus", [None, [{"name": "G", "cc": []}], [{"name": "G"}], 5])
    def test_junk_gpu_list_is_unknown_and_cuda_version_stays(self, gpus) -> None:
        caps = _collect(nvidia=_nvidia_answer(gpus=gpus))
        assert caps.gpus == ()
        assert not caps.nvidia_known
        assert caps.nvidia_driver_cuda == (12, 8)

    def test_no_driver_is_known_and_has_its_own_reason(self) -> None:
        caps = _collect(nvidia=None)
        assert caps.nvidia_known
        assert caps.nvidia_reason == host_caps.NO_NVIDIA_DRIVER

    def test_a_readable_driver_is_known_with_no_reason(self) -> None:
        caps = _collect(nvidia=_nvidia_answer())
        assert (caps.nvidia_known, caps.nvidia_reason) == (True, "")


class TestDetectNeverRaises:
    def test_failing_cpu_probe_reads_unknown_avx2(self, monkeypatch) -> None:
        import cpuinfo

        def boom() -> dict:
            raise RuntimeError("probe failed")

        monkeypatch.setattr(host_caps.platform, "system", lambda: "Linux")
        monkeypatch.setattr(host_caps.platform, "machine", lambda: "x86_64")
        monkeypatch.setattr(cpuinfo, "get_cpu_info", boom)
        monkeypatch.setattr(host_caps, "host_facts", lambda: None)
        caps = detect()
        assert not caps.avx2.known

    def test_an_nvidia_answer_with_missing_keys_is_unknown(self) -> None:
        caps = _collect(nvidia={"driver": "570.1"})
        assert not caps.nvidia_known
        assert caps.gpus == ()
