"""Tests for the gpu_stats orchestrator (probe_gpu_stats).

Vendor-specific parsing lives in gpu_backends/; those are tested in
tests/providers/fleet/gpu_backends/. This file tests the orchestrator: grouping
by backend, merging live samples into GpuStat, and structural fallbacks.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from lilbee.providers.fleet import gpu_stats as stats_mod
from lilbee.providers.fleet.devices import MIB, FleetDevice
from lilbee.providers.fleet.gpu_backends import util_backend_name
from lilbee.providers.fleet.gpu_backends.base import UtilSample
from lilbee.providers.fleet.gpu_stats import GpuStat, intel_grant_binary, probe_gpu_stats

_CUDA = (
    FleetDevice("CUDA", 0, "NVIDIA A40", 48 * 1024 * MIB, 47 * 1024 * MIB),
    FleetDevice("CUDA", 1, "NVIDIA A40", 48 * 1024 * MIB, 47 * 1024 * MIB),
)
_ROCM = (FleetDevice("ROCm", 0, "AMD RX 7900", 16 * 1024 * MIB, 14 * 1024 * MIB),)
_SYCL = (FleetDevice("SYCL", 0, "Intel Arc A770", 16 * 1024 * MIB, 15 * 1024 * MIB),)
_METAL = (FleetDevice("MTL", 0, "Apple M1 Pro", 21845 * MIB, 21844 * MIB),)


def _sample(
    index: int,
    util: int | None = 50,
    temp: int | None = None,
    free: int = 0,
    total: int = 0,
) -> UtilSample:
    return UtilSample(
        index=index,
        utilization_pct=util,
        temperature_c=temp,
        free_bytes=free,
        total_bytes=total,
    )


# ---------------------------------------------------------------------------
# empty / no devices
# ---------------------------------------------------------------------------


def test_empty_device_list_returns_empty() -> None:
    assert probe_gpu_stats(()) == {}


# ---------------------------------------------------------------------------
# structural fallback when backend returns {}
# ---------------------------------------------------------------------------


def test_unknown_backend_uses_structural_fallback() -> None:
    vulkan = (FleetDevice("Vulkan", 0, "GPU", 24 * 1024 * MIB, 22 * 1024 * MIB),)
    result = probe_gpu_stats(vulkan)
    assert result[0] == GpuStat(0, None, 22 * 1024 * MIB, 24 * 1024 * MIB)


def test_backend_returning_empty_uses_structural_fallback() -> None:
    with patch("lilbee.providers.fleet.gpu_stats.resolve_backend") as mock_resolve:
        mock_resolve.return_value.sample.return_value = {}
        result = probe_gpu_stats(_CUDA[:1])
    assert result[0].utilization_pct is None
    assert result[0].free_bytes == 47 * 1024 * MIB


# ---------------------------------------------------------------------------
# live sample merging
# ---------------------------------------------------------------------------


def test_live_util_and_temp_merged() -> None:
    live = {0: _sample(0, util=72, temp=61)}
    with patch("lilbee.providers.fleet.gpu_stats.resolve_backend") as mock_resolve:
        mock_resolve.return_value.sample.return_value = live
        result = probe_gpu_stats(_CUDA[:1])
    assert result[0].utilization_pct == 72
    assert result[0].temperature_c == 61


def test_live_vram_preferred_when_backend_provides_it() -> None:
    live_free = 10 * MIB
    live_total = 100 * MIB
    live = {0: _sample(0, free=live_free, total=live_total)}
    with patch("lilbee.providers.fleet.gpu_stats.resolve_backend") as mock_resolve:
        mock_resolve.return_value.sample.return_value = live
        result = probe_gpu_stats(_SYCL)
    assert result[0].free_bytes == live_free
    assert result[0].total_bytes == live_total


def test_structural_vram_kept_when_backend_returns_zero_zero() -> None:
    """amd-smi metric mode returns free=0, total=0; keep structural VRAM."""
    live = {0: _sample(0, util=40, free=0, total=0)}
    with patch("lilbee.providers.fleet.gpu_stats.resolve_backend") as mock_resolve:
        mock_resolve.return_value.sample.return_value = live
        result = probe_gpu_stats(_ROCM)
    assert result[0].utilization_pct == 40
    assert result[0].free_bytes == 14 * 1024 * MIB
    assert result[0].total_bytes == 16 * 1024 * MIB


def test_result_sorted_by_index() -> None:
    devices = (
        FleetDevice("CUDA", 2, "GPU", 8 * MIB, 7 * MIB),
        FleetDevice("CUDA", 0, "GPU", 8 * MIB, 7 * MIB),
        FleetDevice("CUDA", 1, "GPU", 8 * MIB, 7 * MIB),
    )
    live = {0: _sample(0), 1: _sample(1), 2: _sample(2)}
    with patch("lilbee.providers.fleet.gpu_stats.resolve_backend") as mock_resolve:
        mock_resolve.return_value.sample.return_value = live
        result = probe_gpu_stats(devices)
    assert list(result.keys()) == [0, 1, 2]


def test_dispatch_once_per_backend() -> None:
    """Two CUDA devices dispatch one sample() call, not two."""
    live = {0: _sample(0, util=30), 1: _sample(1, util=50)}
    with patch("lilbee.providers.fleet.gpu_stats.resolve_backend") as mock_resolve:
        mock_resolve.return_value.sample.return_value = live
        probe_gpu_stats(_CUDA)
        assert mock_resolve.return_value.sample.call_count == 1


def test_index_from_backend_not_in_devices_is_ignored() -> None:
    """A live sample for index 99 (not in the device list) is silently dropped."""
    live = {0: _sample(0), 99: _sample(99)}
    with patch("lilbee.providers.fleet.gpu_stats.resolve_backend") as mock_resolve:
        mock_resolve.return_value.sample.return_value = live
        result = probe_gpu_stats(_CUDA[:1])
    assert 99 not in result
    assert result[0].utilization_pct == 50


# ---------------------------------------------------------------------------
# backend exception isolation
# ---------------------------------------------------------------------------


def test_backend_exception_does_not_propagate() -> None:
    with patch("lilbee.providers.fleet.gpu_stats.resolve_backend") as mock_resolve:
        mock_resolve.return_value.sample.side_effect = RuntimeError("tool crashed")
        result = probe_gpu_stats(_CUDA[:1])
    assert result[0].utilization_pct is None


# ---------------------------------------------------------------------------
# GpuStat fields
# ---------------------------------------------------------------------------


def test_gpustat_temperature_c_defaults_to_none() -> None:
    stat = GpuStat(0, 50, 100, 200)
    assert stat.temperature_c is None


def test_gpustat_temperature_c_set() -> None:
    stat = GpuStat(0, 50, 100, 200, temperature_c=75)
    assert stat.temperature_c == 75


# ---------------------------------------------------------------------------
# Metal device via orchestrator (structural VRAM fallback)
# ---------------------------------------------------------------------------


def test_metal_device_returns_structural_fallback() -> None:
    """With no ioreg util, the Metal device keeps structural VRAM and util None."""
    with patch("lilbee.providers.fleet.gpu_backends.apple._ioreg_output", return_value=""):
        result = probe_gpu_stats(_METAL)
    assert result[0].utilization_pct is None
    assert result[0].temperature_c is None
    assert result[0].free_bytes == 21844 * MIB
    assert result[0].total_bytes == 21845 * MIB


# ---------------------------------------------------------------------------
# _safe_sample
# ---------------------------------------------------------------------------


def test_safe_sample_returns_empty_for_unregistered_backend() -> None:
    assert stats_mod._safe_sample("Vulkan", frozenset({0})) == {}


def test_safe_sample_returns_empty_on_exception() -> None:
    with patch("lilbee.providers.fleet.gpu_stats.resolve_backend") as mock_resolve:
        mock_resolve.return_value.sample.side_effect = OSError("boom")
        result = stats_mod._safe_sample("CUDA", frozenset({0}))
    assert result == {}


# ---------------------------------------------------------------------------
# Vendor-based util routing (a Vulkan-exposed consumer GPU)
# ---------------------------------------------------------------------------

_VULKAN_INTEL = (
    FleetDevice("Vulkan", 0, "Intel(R) UHD Graphics (CML GT2)", 12 * 1024 * MIB, 11 * 1024 * MIB),
)


def test_vulkan_intel_device_routes_to_intel_util() -> None:
    """A Vulkan device named Intel dispatches to the Intel (SYCL) util backend."""
    live = {0: _sample(0, util=77)}
    with patch("lilbee.providers.fleet.gpu_stats.resolve_backend") as mock_resolve:
        mock_resolve.return_value.sample.return_value = live
        result = probe_gpu_stats(_VULKAN_INTEL)
    mock_resolve.assert_called_once_with("SYCL")
    assert result[0].utilization_pct == 77
    # structural VRAM is kept (the util backend reports 0/0 for an iGPU)
    assert result[0].total_bytes == 12 * 1024 * MIB


def test_util_backend_name_recognized_backend_unchanged() -> None:
    assert util_backend_name("CUDA", "NVIDIA A40") == "CUDA"
    assert util_backend_name("SYCL", "Intel Arc") == "SYCL"


def test_util_backend_name_vulkan_maps_by_vendor() -> None:
    assert util_backend_name("Vulkan", "Intel(R) UHD Graphics") == "SYCL"
    assert util_backend_name("Vulkan", "NVIDIA GeForce GTX 1650 Ti") == "CUDA"
    assert util_backend_name("Vulkan", "AMD Radeon RX 7900") == "ROCm"
    assert util_backend_name("Vulkan", "Radeon Graphics") == "ROCm"


def test_util_backend_name_vulkan_unknown_vendor_unchanged() -> None:
    assert util_backend_name("Vulkan", "Some Mystery Accelerator") == "Vulkan"


def test_util_backend_name_unknown_backend_unchanged() -> None:
    assert util_backend_name("OpenCL", "whatever") == "OpenCL"


# ---------------------------------------------------------------------------
# intel_grant_binary (the binary a grant would unblock, for the surfaces to format)
# ---------------------------------------------------------------------------


def _hint(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    monkeypatch.setattr(stats_mod, "intel_gpu_top_grant_binary", lambda: value)


def test_intel_grant_binary_intel_missing_util_returns_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hint(monkeypatch, "GRANT ME")
    dev = FleetDevice("SYCL", 0, "Intel Arc A770", 0, 0)
    assert intel_grant_binary([dev], {0: GpuStat(0, None, 0, 0)}) == "GRANT ME"


def test_intel_grant_binary_vulkan_intel_routes_to_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    _hint(monkeypatch, "GRANT ME")
    dev = FleetDevice("Vulkan", 0, "Intel(R) UHD Graphics", 0, 0)
    assert intel_grant_binary([dev], {0: GpuStat(0, None, 0, 0)}) == "GRANT ME"


def test_intel_grant_binary_silent_when_util_present(monkeypatch: pytest.MonkeyPatch) -> None:
    _hint(monkeypatch, "GRANT ME")
    dev = FleetDevice("SYCL", 0, "Intel Arc A770", 0, 0)
    assert intel_grant_binary([dev], {0: GpuStat(0, 50, 0, 0)}) is None


def test_intel_grant_binary_silent_for_non_intel(monkeypatch: pytest.MonkeyPatch) -> None:
    _hint(monkeypatch, "GRANT ME")
    dev = FleetDevice("CUDA", 0, "NVIDIA A40", 0, 0)
    assert intel_grant_binary([dev], {0: GpuStat(0, None, 0, 0)}) is None


def test_intel_grant_binary_silent_when_no_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    _hint(monkeypatch, None)
    dev = FleetDevice("SYCL", 0, "Intel Arc A770", 0, 0)
    assert intel_grant_binary([dev], {0: GpuStat(0, None, 0, 0)}) is None


def test_intel_grant_binary_skips_index_absent_from_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    _hint(monkeypatch, "GRANT ME")
    dev = FleetDevice("SYCL", 0, "Intel Arc A770", 0, 0)
    assert intel_grant_binary([dev], {}) is None
