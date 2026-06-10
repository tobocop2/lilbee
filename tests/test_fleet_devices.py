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


def test_visible_env_rocm_uses_rocr_and_hip() -> None:
    env = visible_env((FleetDevice("ROCm", 1, "", 0, 0),))
    assert env == {"ROCR_VISIBLE_DEVICES": "1", "HIP_VISIBLE_DEVICES": "1"}


def test_visible_env_vulkan_uses_ggml_var() -> None:
    assert visible_env((FleetDevice("Vulkan", 0, "", 0, 0),)) == {"GGML_VK_VISIBLE_DEVICES": "0"}


def test_visible_env_sycl_uses_oneapi_selector() -> None:
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

    def test_cuda_index_past_parent_list_falls_back_to_relative(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
        env = visible_env((FleetDevice("CUDA", 0, "", 0, 0), FleetDevice("CUDA", 1, "", 0, 0)))
        assert env["CUDA_VISIBLE_DEVICES"] == "2,1"

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

    def test_rocm_parent_lists_compose_per_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5")
        monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
        env = visible_env((FleetDevice("ROCm", 1, "", 0, 0),))
        assert env == {"ROCR_VISIBLE_DEVICES": "5", "HIP_VISIBLE_DEVICES": "1"}

    def test_hip_parent_list_composes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "6,7")
        env = visible_env((FleetDevice("HIP", 0, "", 0, 0),))
        assert env == {"ROCR_VISIBLE_DEVICES": "0", "HIP_VISIBLE_DEVICES": "6"}

    def test_vulkan_parent_list_composes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GGML_VK_VISIBLE_DEVICES", "1,2")
        env = visible_env((FleetDevice("Vulkan", 1, "", 0, 0),))
        assert env == {"GGML_VK_VISIBLE_DEVICES": "2"}
