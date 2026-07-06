"""Tests for the SYCL/Intel utilization backend (xpu-smi)."""

from __future__ import annotations

import pytest

from lilbee.providers.fleet import gpu_backends
from lilbee.providers.fleet.devices import MIB
from lilbee.providers.fleet.gpu_backends import intel as intel_mod
from lilbee.providers.fleet.gpu_backends.intel import IntelBackend, _parse_xpu_smi

_INDICES = frozenset({0})

_XPU_JSON = """\
[{
  "device_id": 0,
  "gpu_utilization": 88,
  "gpu_temperature": 73,
  "gpu_memory_used_in_mb": 1024,
  "gpu_memory_size_in_mb": 16384
}]
"""


# ---------------------------------------------------------------------------
# _parse_xpu_smi
# ---------------------------------------------------------------------------


def test_parse_xpu_smi_happy_path() -> None:
    result = _parse_xpu_smi(_XPU_JSON, _INDICES)
    assert result[0].utilization_pct == 88
    assert result[0].temperature_c == 73
    assert result[0].total_bytes == 16384 * MIB
    assert result[0].free_bytes == (16384 - 1024) * MIB


def test_parse_xpu_smi_device_list_wrapper() -> None:
    raw = '{"device_list": [{"device_id": 0, "gpu_utilization": 55}]}'
    result = _parse_xpu_smi(raw, _INDICES)
    assert result[0].utilization_pct == 55


def test_parse_xpu_smi_empty_string() -> None:
    assert _parse_xpu_smi("", _INDICES) == {}


def test_parse_xpu_smi_malformed_json() -> None:
    assert _parse_xpu_smi("{bad", _INDICES) == {}


def test_parse_xpu_smi_skips_out_of_range_indices() -> None:
    result = _parse_xpu_smi(_XPU_JSON, frozenset({5}))
    assert result == {}


def test_parse_xpu_smi_no_vram_gives_zero() -> None:
    raw = '[{"device_id": 0, "gpu_utilization": 10}]'
    result = _parse_xpu_smi(raw, _INDICES)
    assert result[0].free_bytes == 0
    assert result[0].total_bytes == 0


# ---------------------------------------------------------------------------
# IntelBackend.sample
# run_smi is imported by name into intel_mod; patch there, not in base.
# ---------------------------------------------------------------------------


def test_sample_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intel_mod, "run_smi", lambda *_a, **_k: _XPU_JSON)
    result = IntelBackend().sample(_INDICES)
    assert result[0].utilization_pct == 88
    assert result[0].temperature_c == 73


def test_sample_tool_absent_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intel_mod, "run_smi", lambda *_a, **_k: "")
    assert IntelBackend().sample(_INDICES) == {}


def test_sample_nonzero_exit_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intel_mod, "run_smi", lambda *_a, **_k: "")
    assert IntelBackend().sample(_INDICES) == {}


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_registry_maps_sycl_to_intel_backend() -> None:
    assert isinstance(gpu_backends.resolve_backend("SYCL"), IntelBackend)


# ---------------------------------------------------------------------------
# Defensive branches (malformed CLI output)
# ---------------------------------------------------------------------------


def test_parse_xpu_smi_skips_non_dict_items() -> None:
    """Non-dict list entries are skipped, not crashed on."""
    raw = '[1, "x", {"device_id": 0, "gpu_utilization": 22}]'
    result = _parse_xpu_smi(raw, frozenset({0}))
    assert list(result) == [0]
    assert result[0].utilization_pct == 22


def test_parse_xpu_smi_skips_non_int_device_id() -> None:
    """A non-integer device_id is skipped."""
    assert _parse_xpu_smi('[{"device_id": "abc", "gpu_utilization": 5}]', frozenset({0})) == {}
