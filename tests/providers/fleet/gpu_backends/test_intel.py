"""Tests for the SYCL/Intel utilization backend (xpu-smi).

The fixture below mirrors the real `xpu-smi stats -d <id> -j` shape derived from
the xpumanager CLI source (a ``device_level`` array of {metrics_type, value}
objects keyed by the xpum_stats_type_enum names). Live capture on Intel Data
Center GPU Max / Arc hardware confirms the numeric scaling and swaps this fixture
for real tool output.
"""

from __future__ import annotations

import pytest

from lilbee.providers.fleet import gpu_backends
from lilbee.providers.fleet.gpu_backends import intel as intel_mod
from lilbee.providers.fleet.gpu_backends.intel import IntelBackend, _parse_xpu_smi

# Real device_level shape: one {metrics_type, value} entry per metric.
_XPU_JSON = """\
{
  "device_id": 0,
  "device_level": [
    {"metrics_type": "XPUM_STATS_GPU_UTILIZATION", "value": 88},
    {"metrics_type": "XPUM_STATS_EU_ACTIVE", "value": 60},
    {"metrics_type": "XPUM_STATS_GPU_CORE_TEMPERATURE", "value": 73},
    {"metrics_type": "XPUM_STATS_MEMORY_USED", "value": 1073741824}
  ]
}
"""


# ---------------------------------------------------------------------------
# _parse_xpu_smi
# ---------------------------------------------------------------------------


def test_parse_xpu_smi_happy_path() -> None:
    sample = _parse_xpu_smi(_XPU_JSON, 0)
    assert sample is not None
    assert sample.utilization_pct == 88
    assert sample.temperature_c == 73


def test_parse_xpu_smi_keys_sample_by_requested_index() -> None:
    """The sample's index is the requested device, not any id parsed from output."""
    sample = _parse_xpu_smi(_XPU_JSON, 3)
    assert sample is not None
    assert sample.index == 3


def test_parse_xpu_smi_vram_is_sentinel() -> None:
    """stats reports memory-used but no total; VRAM stays the 0/0 structural sentinel."""
    sample = _parse_xpu_smi(_XPU_JSON, 0)
    assert sample is not None
    assert sample.free_bytes == 0
    assert sample.total_bytes == 0


def test_parse_xpu_smi_decimal_string_value() -> None:
    """A metric value emitted as a decimal string is coerced to int."""
    raw = '{"device_level": [{"metrics_type": "XPUM_STATS_GPU_UTILIZATION", "value": "55.0"}]}'
    sample = _parse_xpu_smi(raw, 0)
    assert sample is not None
    assert sample.utilization_pct == 55


def test_parse_xpu_smi_missing_util_metric_leaves_none() -> None:
    """A device_level without the utilization metric yields util None, not a crash."""
    raw = '{"device_level": [{"metrics_type": "XPUM_STATS_GPU_CORE_TEMPERATURE", "value": 50}]}'
    sample = _parse_xpu_smi(raw, 0)
    assert sample is not None
    assert sample.utilization_pct is None
    assert sample.temperature_c == 50


def test_parse_xpu_smi_device_list_wrapper() -> None:
    raw = (
        '{"device_list": [{"device_id": 0, "device_level": '
        '[{"metrics_type": "XPUM_STATS_GPU_UTILIZATION", "value": 42}]}]}'
    )
    sample = _parse_xpu_smi(raw, 0)
    assert sample is not None
    assert sample.utilization_pct == 42


def test_parse_xpu_smi_bare_list_wrapper() -> None:
    raw = '[{"device_level": [{"metrics_type": "XPUM_STATS_GPU_UTILIZATION", "value": 17}]}]'
    sample = _parse_xpu_smi(raw, 0)
    assert sample is not None
    assert sample.utilization_pct == 17


def test_parse_xpu_smi_empty_string() -> None:
    assert _parse_xpu_smi("", 0) is None


def test_parse_xpu_smi_malformed_json() -> None:
    assert _parse_xpu_smi("{bad", 0) is None


def test_parse_xpu_smi_no_device_level_returns_none() -> None:
    assert _parse_xpu_smi('{"device_id": 0}', 0) is None


def test_parse_xpu_smi_skips_non_dict_entries() -> None:
    """Non-dict device_level entries are skipped, not crashed on."""
    raw = '{"device_level": [1, "x", {"metrics_type": "XPUM_STATS_GPU_UTILIZATION", "value": 22}]}'
    sample = _parse_xpu_smi(raw, 0)
    assert sample is not None
    assert sample.utilization_pct == 22


# ---------------------------------------------------------------------------
# IntelBackend.sample
# run_smi is imported by name into intel_mod; patch there, not in base.
# ---------------------------------------------------------------------------


def test_sample_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intel_mod, "run_smi", lambda *_a, **_k: _XPU_JSON)
    result = IntelBackend().sample(frozenset({0}))
    assert result[0].utilization_pct == 88
    assert result[0].temperature_c == 73


def test_sample_runs_stats_per_index_with_device_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each requested index runs `xpu-smi stats -d <index> -j` once."""
    calls: list[list[str]] = []

    def _run_smi(_tool: str, args: list[str], *_a: object, **_k: object) -> str:
        calls.append(args)
        return _XPU_JSON

    monkeypatch.setattr(intel_mod, "run_smi", _run_smi)
    IntelBackend().sample(frozenset({0, 1}))
    assert calls == [["stats", "-d", "0", "-j"], ["stats", "-d", "1", "-j"]]


def test_sample_tool_absent_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intel_mod, "run_smi", lambda *_a, **_k: "")
    assert IntelBackend().sample(frozenset({0})) == {}


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_registry_maps_sycl_to_intel_backend() -> None:
    assert isinstance(gpu_backends.resolve_backend("SYCL"), IntelBackend)
