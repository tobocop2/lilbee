"""Tests for binary-native GPU enumeration and per-backend device pinning."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from lilbee.providers.fleet import devices as dev_mod
from lilbee.providers.fleet.devices import (
    FleetDevice,
    probe_devices,
    visible_env,
)

_CUDA_LISTING = """\
Available devices:
  CUDA0: NVIDIA GeForce RTX 3090 (24268 MiB, 23500 MiB free)
  CUDA1: NVIDIA GeForce RTX 4090 (24564 MiB, 24000 MiB free)
"""
_MIB = 1024 * 1024


def _fake_run(stdout: str):
    def _run(*_a: object, **_k: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")

    return _run


_TOPO_NVLINK = """\
\tGPU0\tGPU1\tCPU Affinity\tNUMA Affinity
GPU0\t X \tNV4\t0-31\t0
GPU1\tNV4\t X \t0-31\t0
"""
_TOPO_PCIE = """\
\tGPU0\tGPU1\tCPU Affinity
GPU0\t X \tPHB\t0-31
GPU1\tPHB\t X \t0-31
"""


# Captured from a live 3xH100 SXM pod: nvidia-smi underlines the header via SGR
# escapes even when stdout is not a tty, and adds NIC columns plus a legend.
_TOPO_REAL_H100 = (
    "\t\x1b[4mGPU0\tGPU1\tGPU2\tNIC0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m\n"
    "GPU0\t X \tNV18\tNV18\tSYS\t0,2,4,6,8,10\t0\t\tN/A\n"
    "GPU1\tNV18\t X \tNV18\tNODE\t1,3,5,7,9,11\t1\t\tN/A\n"
    "GPU2\tNV18\tNV18\t X \tNODE\t1,3,5,7,9,11\t1\t\tN/A\n"
    "NIC0\tSYS\tNODE\tNODE\t X \t\t\t\t\n"
    "\nLegend:\n\n  X    = Self\n  SYS  = Connection traversing PCIe\n"
)


class TestNvlinkTopology:
    def test_real_h100_output_parses_despite_ansi_escapes(self) -> None:
        # The SGR-wrapped header must still yield the GPU columns; without the
        # strip, no pairs parse and an NVLinked host is mis-flagged as PCIe-only.
        gpu_rows, pairs = dev_mod._parse_topo_matrix(_TOPO_REAL_H100)
        assert gpu_rows == {0, 1, 2}
        assert pairs == {frozenset({0, 1}), frozenset({0, 2}), frozenset({1, 2})}

    def test_real_h100_host_has_nvlink(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dev_mod.subprocess, "run", _fake_run(_TOPO_REAL_H100))
        assert dev_mod.host_lacks_nvlink() is False

    def test_parse_finds_nvlink_pair(self) -> None:
        gpu_rows, pairs = dev_mod._parse_topo_matrix(_TOPO_NVLINK)
        assert gpu_rows == {0, 1}
        assert pairs == {frozenset({0, 1})}

    def test_parse_pcie_only_has_no_pairs(self) -> None:
        gpu_rows, pairs = dev_mod._parse_topo_matrix(_TOPO_PCIE)
        assert gpu_rows == {0, 1}
        assert pairs == set()

    def test_lacks_nvlink_true_for_pcie_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dev_mod.subprocess, "run", _fake_run(_TOPO_PCIE))
        assert dev_mod.host_lacks_nvlink() is True

    def test_lacks_nvlink_false_when_linked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dev_mod.subprocess, "run", _fake_run(_TOPO_NVLINK))
        assert dev_mod.host_lacks_nvlink() is False

    def test_unparseable_topo_makes_no_claim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Garbage output parses zero GPU rows: stay silent rather than mis-warn.
        monkeypatch.setattr(dev_mod.subprocess, "run", _fake_run("some unrelated output\n"))
        assert dev_mod.host_lacks_nvlink() is False

    def test_single_gpu_host_is_not_flagged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        single = "\tGPU0\tCPU Affinity\nGPU0\t X \t0-31\n"
        monkeypatch.setattr(dev_mod.subprocess, "run", _fake_run(single))
        assert dev_mod.host_lacks_nvlink() is False

    def test_probe_failure_is_silent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(*_a: object, **_k: object) -> object:
            raise OSError("no nvidia-smi")

        monkeypatch.setattr(dev_mod.subprocess, "run", _boom)
        assert dev_mod.host_lacks_nvlink() is False


def test_probe_parses_cuda_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dev_mod.subprocess, "run", _fake_run(_CUDA_LISTING))
    devices = probe_devices(Path("/bin/llama-server"))
    assert devices == [
        FleetDevice("CUDA", 0, "NVIDIA GeForce RTX 3090", 24268 * _MIB, 23500 * _MIB),
        FleetDevice("CUDA", 1, "NVIDIA GeForce RTX 4090", 24564 * _MIB, 24000 * _MIB),
    ]


def test_probe_drops_cpu_and_keeps_gpu_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    listing = (
        "  CUDA0: NVIDIA (24268 MiB, 23000 MiB free)\n"
        "  CPU0: some cpu (64000 MiB)\n"
        "  Vulkan0: NVIDIA (24268 MiB, 23000 MiB free)\n"
    )
    monkeypatch.setattr(dev_mod.subprocess, "run", _fake_run(listing))
    devices = probe_devices(Path("/bin/llama-server"))
    # CUDA outranks Vulkan; CPU dropped entirely.
    assert [d.backend for d in devices] == ["CUDA"]


def test_probe_defaults_free_to_total_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dev_mod.subprocess, "run", _fake_run("  Vulkan0: AMD (16000 MiB)\n"))
    (device,) = probe_devices(Path("/bin/llama-server"))
    assert device.free_bytes == device.total_bytes == 16000 * _MIB


def test_probe_returns_a_single_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    # Even if a listing somehow names two GPU backends, pin exactly one (no mixed
    # index spaces). CUDA and ROCm tie on rank; the tie breaks deterministically.
    listing = (
        "  CUDA0: NVIDIA (24268 MiB, 23000 MiB free)\n  ROCm0: AMD (24268 MiB, 23000 MiB free)\n"
    )
    monkeypatch.setattr(dev_mod.subprocess, "run", _fake_run(listing))
    backends = {d.backend for d in probe_devices(Path("/bin/llama-server"))}
    assert len(backends) == 1


def test_probe_returns_empty_when_no_gpu_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    # Only a CPU device listed -> no GPU backend to pin -> empty.
    monkeypatch.setattr(dev_mod.subprocess, "run", _fake_run("  CPU0: host cpu (64000 MiB)\n"))
    assert probe_devices(Path("/bin/llama-server")) == []


def test_probe_returns_empty_on_subprocess_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: object, **_k: object) -> subprocess.CompletedProcess:
        raise OSError("no such binary")

    monkeypatch.setattr(dev_mod.subprocess, "run", _boom)
    assert probe_devices(Path("/bin/llama-server")) == []


def test_probe_env_sets_pci_bus_order() -> None:
    assert dev_mod._probe_env()["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_visible_env_cuda_pins_by_index_and_pins_order() -> None:
    env = visible_env((FleetDevice("CUDA", 2, "", 0, 0), FleetDevice("CUDA", 3, "", 0, 0)))
    assert env == {"CUDA_VISIBLE_DEVICES": "2,3", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


def test_visible_env_rocm_emits_single_var_on_clean_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ROCR and HIP filter sequentially, so the child pins with one var only;
    # with neither parent var set it defaults to HIP at the absolute index.
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    env = visible_env((FleetDevice("ROCm", 1, "", 0, 0),))
    assert env == {"HIP_VISIBLE_DEVICES": "1"}


def test_visible_env_vulkan_uses_ggml_var() -> None:
    assert visible_env((FleetDevice("Vulkan", 0, "", 0, 0),)) == {"GGML_VK_VISIBLE_DEVICES": "0"}


def test_visible_env_sycl_uses_oneapi_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ONEAPI_DEVICE_SELECTOR", raising=False)
    env = visible_env((FleetDevice("SYCL", 0, "", 0, 0), FleetDevice("SYCL", 1, "", 0, 0)))
    assert env == {"ONEAPI_DEVICE_SELECTOR": "level_zero:0,1"}


def test_visible_env_metal_and_empty_pin_nothing() -> None:
    assert visible_env(()) == {}
    assert visible_env((FleetDevice("Metal", 0, "", 0, 0),)) == {}


class TestPresetVisibleDeviceComposition:
    """A pod-preset visible-devices var makes probe indices RELATIVE; the child
    env must map them back through the parent list to the same physical devices."""

    def test_cuda_integer_parent_list_composes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
        env = visible_env((FleetDevice("CUDA", 0, "", 0, 0), FleetDevice("CUDA", 1, "", 0, 0)))
        assert env["CUDA_VISIBLE_DEVICES"] == "2,3"

    def test_cuda_integer_parent_subset_picks_the_right_physical_gpu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
        env = visible_env((FleetDevice("CUDA", 1, "", 0, 0),))
        assert env["CUDA_VISIBLE_DEVICES"] == "3"  # relative 1 -> physical 3

    def test_cuda_uuid_parent_list_composes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaa, GPU-bbb")
        env = visible_env((FleetDevice("CUDA", 1, "", 0, 0),))
        assert env["CUDA_VISIBLE_DEVICES"] == "GPU-bbb"

    def test_cuda_index_past_parent_list_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An index outside the parent restriction is an invariant violation; pinning
        a bare absolute index into a (possibly UUID) parent list would select the
        wrong GPU, so composition fails loudly instead (bb-7jg1.13)."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
        with pytest.raises(ValueError, match="outside the parent visible-devices list"):
            visible_env((FleetDevice("CUDA", 0, "", 0, 0), FleetDevice("CUDA", 1, "", 0, 0)))

    def test_cuda_clean_env_emits_relative_ids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
        env = visible_env((FleetDevice("CUDA", 0, "", 0, 0), FleetDevice("CUDA", 1, "", 0, 0)))
        assert env == {"CUDA_VISIBLE_DEVICES": "0,1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}

    def test_preset_cuda_device_order_is_respected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
        assert dev_mod._probe_env()["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"
        env = visible_env((FleetDevice("CUDA", 0, "", 0, 0),))
        assert env["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"  # same order the probe used

    def test_probe_env_defaults_order_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
        assert dev_mod._probe_env()["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"

    def test_rocm_rocr_only_parent_emits_rocr_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Parent restricted via ROCR only: emit ROCR composed against it, and do
        # NOT also emit HIP (which would re-index within the ROCR survivors).
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5")
        monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
        env = visible_env((FleetDevice("ROCm", 1, "", 0, 0),))
        assert env == {"ROCR_VISIBLE_DEVICES": "5"}  # relative 1 -> physical 5

    def test_rocm_hip_only_parent_emits_hip_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The finding's scenario: HIP-only parent. Emitting an absolute ROCR here
        # would select the wrong physical card, so only HIP is composed/emitted.
        monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1,3")
        env = visible_env((FleetDevice("HIP", 0, "", 0, 0),))
        assert env == {"HIP_VISIBLE_DEVICES": "1"}  # relative 0 -> physical 1, no ROCR

    def test_rocm_both_parents_set_emits_hip_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # When both are set, the inherited ROCR stays in force and only HIP is
        # re-pinned (within the ROCR survivors), so the cap is never doubled.
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2")
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "6,7")
        env = visible_env((FleetDevice("ROCm", 1, "", 0, 0),))
        assert env == {"HIP_VISIBLE_DEVICES": "7"}  # relative 1 -> physical 7

    def test_vulkan_parent_list_composes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GGML_VK_VISIBLE_DEVICES", "1,2")
        env = visible_env((FleetDevice("Vulkan", 1, "", 0, 0),))
        assert env == {"GGML_VK_VISIBLE_DEVICES": "2"}

    def test_sycl_level_zero_parent_list_composes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:2,3")
        env = visible_env((FleetDevice("SYCL", 1, "", 0, 0),))
        assert env == {"ONEAPI_DEVICE_SELECTOR": "level_zero:3"}  # relative 1 -> physical 3

    def test_sycl_non_level_zero_parent_emits_absolute(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Only the level_zero:i,j shape is composable; other shapes pass through.
        monkeypatch.setenv("ONEAPI_DEVICE_SELECTOR", "opencl:0,1")
        env = visible_env((FleetDevice("SYCL", 0, "", 0, 0), FleetDevice("SYCL", 1, "", 0, 0)))
        assert env == {"ONEAPI_DEVICE_SELECTOR": "level_zero:0,1"}
