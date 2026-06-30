"""Tests for the ROCm/HIP utilization backend (amd-smi + rocm-smi fallback)."""

from __future__ import annotations

import pytest

from lilbee.providers.fleet import gpu_backends
from lilbee.providers.fleet.gpu_backends import amd as amd_mod
from lilbee.providers.fleet.gpu_backends.amd import AmdBackend, _parse_amd_smi, _parse_rocm_smi

_INDICES = frozenset({0, 1})

# Flat shape (older amd-smi versions).
_AMD_SMI_JSON_FLAT = (
    '[{"gpu": 0, "gfx_activity": 72, "temperature_c": 61},'
    ' {"gpu": 1, "gfx_activity": 45, "temperature_c": 58}]'
)

# Nested shape (newer amd-smi versions): util as {"value": N, "unit": "%"},
# temp as {"edge": N} under a "temperature" key.
_AMD_SMI_JSON_NESTED = (
    '[{"gpu": 0, "gfx_activity": {"value": 72, "unit": "%"}, "temperature": {"edge": 61}}]'
)

_ROCM_SMI_JSON = (
    '{"card0": {"GPU use (%)": "35", "Temperature (Sensor edge) (C)": "52"},'
    ' "card1": {"GPU use (%)": "80", "Temperature (Sensor edge) (C)": "67"}}'
)

# rocm-smi with VRAM keys present (byte values).
_ROCM_SMI_JSON_VRAM = (
    '{"card0": {'
    '"GPU use (%)": "40",'
    '"Temperature (Sensor edge) (C)": "55",'
    '"VRAM Total Memory (B)": "17179869184",'
    '"VRAM Total Used Memory (B)": "2147483648"'
    "}}"
)


# ---------------------------------------------------------------------------
# _parse_amd_smi -- flat shape
# ---------------------------------------------------------------------------


def test_parse_amd_smi_flat_happy_path() -> None:
    result = _parse_amd_smi(_AMD_SMI_JSON_FLAT, _INDICES)
    assert result[0].utilization_pct == 72
    assert result[0].temperature_c == 61
    assert result[1].utilization_pct == 45
    assert result[1].temperature_c == 58


def test_parse_amd_smi_skips_indices_not_requested() -> None:
    raw = '[{"gpu": 0, "gfx_activity": 10}, {"gpu": 5, "gfx_activity": 99}]'
    result = _parse_amd_smi(raw, frozenset({0}))
    assert 5 not in result
    assert result[0].utilization_pct == 10


def test_parse_amd_smi_empty_string() -> None:
    assert _parse_amd_smi("", _INDICES) == {}


def test_parse_amd_smi_malformed_json() -> None:
    assert _parse_amd_smi("not json", _INDICES) == {}


def test_parse_amd_smi_gpu_dict_wrapper() -> None:
    raw = '{"gpu": [{"gpu": 0, "gfx_activity": 55, "temperature_c": 40}]}'
    result = _parse_amd_smi(raw, frozenset({0}))
    assert result[0].utilization_pct == 55


def test_parse_amd_smi_vram_sentinel() -> None:
    """amd-smi metric mode omits VRAM; free_bytes/total_bytes stay 0."""
    result = _parse_amd_smi(_AMD_SMI_JSON_FLAT, frozenset({0}))
    assert result[0].free_bytes == 0
    assert result[0].total_bytes == 0


# ---------------------------------------------------------------------------
# _parse_amd_smi -- nested shape (newer amd-smi versions)
# ---------------------------------------------------------------------------


def test_parse_amd_smi_nested_value_shape() -> None:
    """Nested {"value": N} util and {"edge": N} temp are extracted correctly."""
    result = _parse_amd_smi(_AMD_SMI_JSON_NESTED, frozenset({0}))
    assert result[0].utilization_pct == 72
    assert result[0].temperature_c == 61


def test_parse_amd_smi_nested_does_not_break_flat() -> None:
    """Flat and nested shapes can both be parsed (regression guard)."""
    flat = _parse_amd_smi(_AMD_SMI_JSON_FLAT, frozenset({0}))
    nested = _parse_amd_smi(_AMD_SMI_JSON_NESTED, frozenset({0}))
    assert flat[0].utilization_pct == nested[0].utilization_pct
    assert flat[0].temperature_c == nested[0].temperature_c


# ---------------------------------------------------------------------------
# _parse_rocm_smi
# ---------------------------------------------------------------------------


def test_parse_rocm_smi_happy_path() -> None:
    result = _parse_rocm_smi(_ROCM_SMI_JSON, _INDICES)
    assert result[0].utilization_pct == 35
    assert result[0].temperature_c == 52
    assert result[1].utilization_pct == 80


def test_parse_rocm_smi_decimal_string_util() -> None:
    """rocm-smi may emit util as "35.0"; extract_int must handle it."""
    raw = '{"card0": {"GPU use (%)": "35.0", "Temperature (Sensor edge) (C)": "52.5"}}'
    result = _parse_rocm_smi(raw, frozenset({0}))
    assert result[0].utilization_pct == 35
    assert result[0].temperature_c == 52


def test_parse_rocm_smi_vram_keys_parsed() -> None:
    """When VRAM Total Memory/Used keys are present, live VRAM is reported."""
    result = _parse_rocm_smi(_ROCM_SMI_JSON_VRAM, frozenset({0}))
    total_b = 17_179_869_184
    used_b = 2_147_483_648
    assert result[0].total_bytes == total_b
    assert result[0].free_bytes == total_b - used_b


def test_parse_rocm_smi_vram_absent_leaves_sentinel() -> None:
    """Without VRAM keys the sentinel 0/0 triggers structural-VRAM fallback."""
    result = _parse_rocm_smi(_ROCM_SMI_JSON, frozenset({0}))
    assert result[0].free_bytes == 0
    assert result[0].total_bytes == 0


def test_parse_rocm_smi_empty_string() -> None:
    assert _parse_rocm_smi("", _INDICES) == {}


def test_parse_rocm_smi_malformed_json() -> None:
    assert _parse_rocm_smi("not json", _INDICES) == {}


def test_parse_rocm_smi_non_dict_top_level() -> None:
    assert _parse_rocm_smi("[1, 2, 3]", _INDICES) == {}


# ---------------------------------------------------------------------------
# AmdBackend.sample dispatch
# run_smi is imported by name into amd_mod; patch there, not in base.
# ---------------------------------------------------------------------------


def test_sample_uses_amd_smi_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        amd_mod,
        "run_smi",
        lambda tool, *_a, **_k: _AMD_SMI_JSON_FLAT if tool == amd_mod._TOOL_AMD_SMI else "",
    )
    result = AmdBackend().sample(frozenset({0}))
    assert result[0].utilization_pct == 72


def test_falls_back_to_rocm_smi_when_amd_smi_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    def _run_smi(tool: str, *_a: object, **_k: object) -> str:
        return "" if tool == amd_mod._TOOL_AMD_SMI else _ROCM_SMI_JSON

    monkeypatch.setattr(amd_mod, "run_smi", _run_smi)
    result = AmdBackend().sample(frozenset({0}))
    assert result[0].utilization_pct == 35


def test_sample_returns_empty_when_both_tools_return_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(amd_mod, "run_smi", lambda *_a, **_k: "")
    assert AmdBackend().sample(frozenset({0})) == {}


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_registry_maps_rocm_to_amd_backend() -> None:
    assert isinstance(gpu_backends.resolve_backend("ROCm"), AmdBackend)


def test_registry_maps_hip_to_amd_backend() -> None:
    assert isinstance(gpu_backends.resolve_backend("HIP"), AmdBackend)


# ---------------------------------------------------------------------------
# Defensive branches (malformed CLI output)
# ---------------------------------------------------------------------------


def test_parse_amd_smi_skips_non_dict_items() -> None:
    """Non-dict list entries are skipped, not crashed on."""
    raw = '[1, "x", {"gpu": 0, "gfx_activity": 10}]'
    result = _parse_amd_smi(raw, frozenset({0}))
    assert list(result) == [0]
    assert result[0].utilization_pct == 10


def test_parse_amd_smi_skips_non_int_index() -> None:
    """A non-integer gpu id is skipped."""
    assert _parse_amd_smi('[{"gpu": "abc", "gfx_activity": 10}]', frozenset({0})) == {}


def test_parse_amd_smi_util_none_when_no_flat_or_nested() -> None:
    """Util stays None when neither flat nor nested {"value": N} util is present."""
    result = _parse_amd_smi('[{"gpu": 0, "temperature_c": 50}]', frozenset({0}))
    assert result[0].utilization_pct is None
    assert result[0].temperature_c == 50


def test_parse_rocm_smi_skips_non_dict_card_value() -> None:
    """A card whose value is not a dict is skipped."""
    assert _parse_rocm_smi('{"card0": "x"}', frozenset({0})) == {}
