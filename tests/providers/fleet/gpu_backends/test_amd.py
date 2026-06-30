"""Tests for the ROCm/HIP utilization backend (amd-smi + rocm-smi fallback)."""

from __future__ import annotations

import subprocess

import pytest

from lilbee.providers.fleet import gpu_backends
from lilbee.providers.fleet.gpu_backends import amd as amd_mod
from lilbee.providers.fleet.gpu_backends.amd import AmdBackend, _parse_amd_smi, _parse_rocm_smi

_INDICES = frozenset({0, 1})

_AMD_SMI_JSON = '[{"gpu": 0, "gfx_activity": 72, "temperature_c": 61}, {"gpu": 1, "gfx_activity": 45, "temperature_c": 58}]'
_ROCM_SMI_JSON = '{"card0": {"GPU use (%)": "35", "Temperature (Sensor edge) (C)": "52"}, "card1": {"GPU use (%)": "80", "Temperature (Sensor edge) (C)": "67"}}'


def _fake_run(stdout: str, returncode: int = 0):
    def _run(*_a: object, **_k: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")

    return _run


# ---------------------------------------------------------------------------
# _parse_amd_smi
# ---------------------------------------------------------------------------


def test_parse_amd_smi_happy_path() -> None:
    result = _parse_amd_smi(_AMD_SMI_JSON, _INDICES)
    assert result[0].utilization_pct == 72
    assert result[0].temperature_c == 61
    assert result[1].utilization_pct == 45
    assert result[1].temperature_c == 58


def test_parse_amd_smi_skips_indices_not_requested() -> None:
    json = '[{"gpu": 0, "gfx_activity": 10}, {"gpu": 5, "gfx_activity": 99}]'
    result = _parse_amd_smi(json, frozenset({0}))
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
    result = _parse_amd_smi(_AMD_SMI_JSON, frozenset({0}))
    assert result[0].free_bytes == 0
    assert result[0].total_bytes == 0


# ---------------------------------------------------------------------------
# _parse_rocm_smi
# ---------------------------------------------------------------------------


def test_parse_rocm_smi_happy_path() -> None:
    result = _parse_rocm_smi(_ROCM_SMI_JSON, _INDICES)
    assert result[0].utilization_pct == 35
    assert result[0].temperature_c == 52
    assert result[1].utilization_pct == 80


def test_parse_rocm_smi_empty_string() -> None:
    assert _parse_rocm_smi("", _INDICES) == {}


def test_parse_rocm_smi_malformed_json() -> None:
    assert _parse_rocm_smi("not json", _INDICES) == {}


def test_parse_rocm_smi_non_dict_top_level() -> None:
    assert _parse_rocm_smi("[1, 2, 3]", _INDICES) == {}


# ---------------------------------------------------------------------------
# AmdBackend.sample dispatch
# ---------------------------------------------------------------------------


def test_sample_uses_amd_smi_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(amd_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(amd_mod.subprocess, "run", _fake_run(_AMD_SMI_JSON))
    result = AmdBackend().sample(frozenset({0}))
    assert result[0].utilization_pct == 72


def test_sample_falls_back_to_rocm_smi_when_amd_smi_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    def _which(name: str) -> str | None:
        return None if name == amd_mod._TOOL_AMD_SMI else f"/usr/bin/{name}"

    monkeypatch.setattr(amd_mod.shutil, "which", _which)
    monkeypatch.setattr(amd_mod.subprocess, "run", _fake_run(_ROCM_SMI_JSON))
    result = AmdBackend().sample(frozenset({0}))
    assert result[0].utilization_pct == 35


def test_sample_falls_back_to_rocm_smi_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(amd_mod.shutil, "which", lambda name: f"/usr/bin/{name}")

    def _run(*args: object, **_k: object) -> subprocess.CompletedProcess:
        cmd = list(args[0])  # type: ignore[arg-type]
        if amd_mod._TOOL_AMD_SMI in cmd[0]:
            return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=_ROCM_SMI_JSON, stderr="")

    monkeypatch.setattr(amd_mod.subprocess, "run", _run)
    result = AmdBackend().sample(frozenset({0}))
    assert result[0].utilization_pct == 35


def test_sample_returns_empty_when_both_tools_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(amd_mod.shutil, "which", lambda _: None)
    assert AmdBackend().sample(frozenset({0})) == {}


def test_sample_oserror_falls_back_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(amd_mod.shutil, "which", lambda name: f"/usr/bin/{name}")

    def _oserr(*_a: object, **_k: object) -> subprocess.CompletedProcess:
        raise OSError("tool not executable")

    monkeypatch.setattr(amd_mod.subprocess, "run", _oserr)
    assert AmdBackend().sample(frozenset({0})) == {}


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_registry_maps_rocm_to_amd_backend() -> None:
    assert isinstance(gpu_backends.resolve_backend("ROCm"), AmdBackend)


def test_registry_maps_hip_to_amd_backend() -> None:
    assert isinstance(gpu_backends.resolve_backend("HIP"), AmdBackend)
