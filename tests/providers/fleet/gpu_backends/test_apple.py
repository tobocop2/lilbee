"""Tests for the Apple Metal ioreg utilization backend."""

from __future__ import annotations

import pytest

from lilbee.providers.fleet import gpu_backends
from lilbee.providers.fleet.gpu_backends import apple as apple_mod
from lilbee.providers.fleet.gpu_backends.apple import BACKEND_KEY, AppleBackend
from lilbee.providers.fleet.gpu_backends.base import UtilSample

_INDICES = frozenset({0})

# A realistic single-GPU PerformanceStatistics line from `ioreg` on Apple Silicon.
_IOREG = (
    '    | |   "PerformanceStatistics" = {"In use system memory"=216498176,'
    '"Tiler Utilization %"=3,"Renderer Utilization %"=12,'
    '"Device Utilization %"=42,"Alloc system memory"=20317028352}'
)


def test_parse_reads_device_utilization() -> None:
    samples = apple_mod._parse_ioreg(_IOREG, _INDICES)
    assert samples[0] == UtilSample(
        index=0, utilization_pct=42, temperature_c=None, free_bytes=0, total_bytes=0
    )


def test_parse_keeps_structural_vram_sentinel() -> None:
    """free/total stay 0 so the orchestrator keeps the structural probe VRAM."""
    sample = apple_mod._parse_ioreg(_IOREG, _INDICES)[0]
    assert sample.free_bytes == 0
    assert sample.total_bytes == 0
    assert sample.temperature_c is None


def test_parse_no_field_returns_empty() -> None:
    assert apple_mod._parse_ioreg('"Renderer Utilization %"=5', _INDICES) == {}


def test_parse_empty_returns_empty() -> None:
    assert apple_mod._parse_ioreg("", _INDICES) == {}


def test_sample_runs_ioreg(monkeypatch: pytest.MonkeyPatch) -> None:
    """sample() reads util through run_smi (the ioreg boundary)."""
    monkeypatch.setattr(apple_mod, "run_smi", lambda *_a, **_k: _IOREG)
    assert AppleBackend().sample(_INDICES)[0].utilization_pct == 42


def test_sample_empty_when_ioreg_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(apple_mod, "run_smi", lambda *_a, **_k: "")
    assert AppleBackend().sample(_INDICES) == {}


def test_ioreg_command_roots_at_gpu_accelerator(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe must root at the GPU accelerator, not a depth-capped key match.

    Regression: '-d 1 -k PerformanceStatistics' never reaches the GPU node and read
    0% under load; the query must class-match IOAccelerator so utilization is live.
    """
    seen: dict[str, list[str]] = {}
    monkeypatch.setattr(
        apple_mod, "run_smi", lambda tool, args, timeout: seen.update(tool=tool, args=args) or ""
    )
    apple_mod._ioreg_output()
    assert seen["tool"] == "ioreg"
    assert "-c" in seen["args"]
    assert "IOAccelerator" in seen["args"]


def test_backend_key_is_mtl() -> None:
    assert BACKEND_KEY == "MTL"


def test_registry_maps_mtl_to_apple_backend() -> None:
    assert isinstance(gpu_backends.resolve_backend("MTL"), AppleBackend)


def test_registry_maps_metal_alias_to_apple_backend() -> None:
    """Build-dependent 'Metal' string is also registered defensively."""
    assert isinstance(gpu_backends.resolve_backend("Metal"), AppleBackend)


def test_registry_returns_none_for_unknown() -> None:
    assert gpu_backends.resolve_backend("Vulkan") is None
    assert gpu_backends.resolve_backend("BLAS") is None
    assert gpu_backends.resolve_backend("") is None
