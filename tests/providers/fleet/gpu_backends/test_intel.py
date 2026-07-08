"""Tests for the Intel utilization backend (xpu-smi -> fdinfo -> intel_gpu_top).

The xpu-smi and intel_gpu_top fixtures mirror the real tool output (xpu-smi from
the xpumanager CLI source; intel_gpu_top captured live from a CometLake iGPU).
Live capture confirms the numbers; these tests pin the parsing and source order.
"""

from __future__ import annotations

import subprocess

import pytest

from lilbee.providers.fleet import gpu_backends
from lilbee.providers.fleet.gpu_backends import intel as intel_mod
from lilbee.providers.fleet.gpu_backends.intel import IntelBackend, _parse_xpu_smi

# Real xpu-smi device_level shape.
_XPU_JSON = """\
{
  "device_id": 0,
  "device_level": [
    {"metrics_type": "XPUM_STATS_GPU_UTILIZATION", "value": 88},
    {"metrics_type": "XPUM_STATS_GPU_CORE_TEMPERATURE", "value": 73}
  ]
}
"""

# Real intel_gpu_top -J shape: an unclosed, comma-terminated array of samples,
# each with an "engines" map of {"busy": pct}. Last sample peaks Render/3D at 85.
_IGT_STREAM = (
    "[\n"
    '{"engines":{"Render/3D":{"busy":1.6,"unit":"%"},"Video":{"busy":0.0,"unit":"%"}}},\n'
    '{"engines":{"Render/3D":{"busy":85.0,"unit":"%"},"Video":{"busy":12.0,"unit":"%"}}},\n'
)


def _no_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralize the fdinfo + intel_gpu_top sources so xpu-smi tests are hermetic."""
    monkeypatch.setattr(intel_mod.fdinfo, "read_drm_util", lambda *_a, **_k: None)
    monkeypatch.setattr(intel_mod, "_intel_gpu_top_output", lambda: "")


# ---------------------------------------------------------------------------
# xpu-smi parsing
# ---------------------------------------------------------------------------


def test_parse_xpu_smi_happy_path() -> None:
    sample = _parse_xpu_smi(_XPU_JSON, 0)
    assert sample is not None
    assert sample.utilization_pct == 88
    assert sample.temperature_c == 73
    assert sample.free_bytes == 0
    assert sample.total_bytes == 0


def test_parse_xpu_smi_keys_sample_by_requested_index() -> None:
    sample = _parse_xpu_smi(_XPU_JSON, 3)
    assert sample is not None and sample.index == 3


def test_parse_xpu_smi_decimal_string_value() -> None:
    raw = '{"device_level": [{"metrics_type": "XPUM_STATS_GPU_UTILIZATION", "value": "55.0"}]}'
    sample = _parse_xpu_smi(raw, 0)
    assert sample is not None and sample.utilization_pct == 55


def test_parse_xpu_smi_missing_util_metric_leaves_none() -> None:
    raw = '{"device_level": [{"metrics_type": "XPUM_STATS_GPU_CORE_TEMPERATURE", "value": 50}]}'
    sample = _parse_xpu_smi(raw, 0)
    assert sample is not None and sample.utilization_pct is None and sample.temperature_c == 50


def test_parse_xpu_smi_device_list_wrapper() -> None:
    raw = (
        '{"device_list": [{"device_id": 0, "device_level": '
        '[{"metrics_type": "XPUM_STATS_GPU_UTILIZATION", "value": 42}]}]}'
    )
    sample = _parse_xpu_smi(raw, 0)
    assert sample is not None and sample.utilization_pct == 42


def test_parse_xpu_smi_bare_list_wrapper() -> None:
    raw = '[{"device_level": [{"metrics_type": "XPUM_STATS_GPU_UTILIZATION", "value": 17}]}]'
    sample = _parse_xpu_smi(raw, 0)
    assert sample is not None and sample.utilization_pct == 17


def test_parse_xpu_smi_empty_string() -> None:
    assert _parse_xpu_smi("", 0) is None


def test_parse_xpu_smi_malformed_json() -> None:
    assert _parse_xpu_smi("{bad", 0) is None


def test_parse_xpu_smi_no_device_level_returns_none() -> None:
    assert _parse_xpu_smi('{"device_id": 0}', 0) is None


def test_parse_xpu_smi_skips_non_dict_entries() -> None:
    raw = '{"device_level": [1, "x", {"metrics_type": "XPUM_STATS_GPU_UTILIZATION", "value": 22}]}'
    sample = _parse_xpu_smi(raw, 0)
    assert sample is not None and sample.utilization_pct == 22


# ---------------------------------------------------------------------------
# Source order: xpu-smi -> fdinfo -> intel_gpu_top
# ---------------------------------------------------------------------------


def test_sample_prefers_xpu_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intel_mod, "run_smi", lambda *_a, **_k: _XPU_JSON)
    monkeypatch.setattr(intel_mod.fdinfo, "read_drm_util", lambda *_a, **_k: 99)
    result = IntelBackend().sample(frozenset({0}))
    assert result[0].utilization_pct == 88  # xpu-smi won, not fdinfo


def test_sample_runs_xpu_smi_per_index_with_device_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def _run_smi(_tool: str, args: list[str], *_a: object, **_k: object) -> str:
        calls.append(args)
        return _XPU_JSON

    monkeypatch.setattr(intel_mod, "run_smi", _run_smi)
    IntelBackend().sample(frozenset({0, 1}))
    assert calls == [["stats", "-d", "0", "-j"], ["stats", "-d", "1", "-j"]]


def test_sample_falls_back_to_fdinfo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intel_mod, "run_smi", lambda *_a, **_k: "")
    monkeypatch.setattr(intel_mod.fdinfo, "read_drm_util", lambda *_a, **_k: 42)
    monkeypatch.setattr(intel_mod, "_intel_gpu_top_output", lambda: _IGT_STREAM)
    result = IntelBackend().sample(frozenset({0}))
    assert result[0].utilization_pct == 42  # fdinfo won, not intel_gpu_top


def test_sample_falls_back_to_intel_gpu_top(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intel_mod, "run_smi", lambda *_a, **_k: "")
    monkeypatch.setattr(intel_mod.fdinfo, "read_drm_util", lambda *_a, **_k: None)
    monkeypatch.setattr(intel_mod, "_intel_gpu_top_output", lambda: _IGT_STREAM)
    result = IntelBackend().sample(frozenset({0}))
    assert result[0].utilization_pct == 85  # busiest engine in the last sample


def test_sample_all_sources_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intel_mod, "run_smi", lambda *_a, **_k: "")
    _no_fallbacks(monkeypatch)
    assert IntelBackend().sample(frozenset({0})) == {}


def test_fdinfo_sample_empty_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intel_mod.fdinfo, "read_drm_util", lambda *_a, **_k: 50)
    assert intel_mod._fdinfo_samples(frozenset()) == {}


def test_intel_gpu_top_sample_empty_indices() -> None:
    assert intel_mod._intel_gpu_top_samples(frozenset()) == {}


# ---------------------------------------------------------------------------
# intel_gpu_top parsing
# ---------------------------------------------------------------------------


def test_igt_max_busy_from_stream() -> None:
    assert intel_mod._igt_max_busy(_IGT_STREAM) == 85


def test_igt_max_busy_closed_array() -> None:
    assert intel_mod._igt_max_busy('[{"engines":{"Render/3D":{"busy":30.0}}}]') == 30


def test_igt_max_busy_empty_string() -> None:
    assert intel_mod._igt_max_busy("") is None


def test_igt_max_busy_malformed() -> None:
    assert intel_mod._igt_max_busy("{bad") is None


def test_igt_max_busy_no_engines() -> None:
    assert intel_mod._igt_max_busy('{"frequency": {"actual": 300}}') is None


def test_igt_max_busy_engines_not_dict() -> None:
    assert intel_mod._igt_max_busy('{"engines": []}') is None


def test_igt_max_busy_no_numeric_busy() -> None:
    assert intel_mod._igt_max_busy('{"engines": {"Render/3D": {"busy": "n/a"}}}') is None


def test_last_json_object_empty_list() -> None:
    assert intel_mod._last_json_object("[]") is None


def test_last_json_object_non_dict_last() -> None:
    assert intel_mod._last_json_object("[1, 2]") is None


def test_last_json_object_bare_dict() -> None:
    assert intel_mod._last_json_object('{"engines": {}}') == {"engines": {}}


def test_last_json_object_non_container() -> None:
    assert intel_mod._last_json_object("5") is None


# ---------------------------------------------------------------------------
# intel_gpu_top invocation (subprocess)
# ---------------------------------------------------------------------------


def test_intel_gpu_top_output_tool_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intel_mod.shutil, "which", lambda _t: None)
    assert intel_mod._intel_gpu_top_output() == ""


def test_intel_gpu_top_output_clean_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intel_mod.shutil, "which", lambda _t: "/usr/bin/intel_gpu_top")

    class _Proc:
        stdout = "err-exit-output"

    monkeypatch.setattr(intel_mod.subprocess, "run", lambda *_a, **_k: _Proc())
    assert intel_mod._intel_gpu_top_output() == "err-exit-output"


def test_intel_gpu_top_output_timeout_str_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intel_mod.shutil, "which", lambda _t: "/usr/bin/intel_gpu_top")

    def _raise(*_a: object, **_k: object) -> None:
        raise subprocess.TimeoutExpired(cmd="intel_gpu_top", timeout=1.3, output=_IGT_STREAM)

    monkeypatch.setattr(intel_mod.subprocess, "run", _raise)
    assert intel_mod._intel_gpu_top_output() == _IGT_STREAM


def test_intel_gpu_top_output_timeout_bytes_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intel_mod.shutil, "which", lambda _t: "/usr/bin/intel_gpu_top")

    def _raise(*_a: object, **_k: object) -> None:
        raise subprocess.TimeoutExpired(cmd="x", timeout=1.3, output=b"[\n{}")

    monkeypatch.setattr(intel_mod.subprocess, "run", _raise)
    assert intel_mod._intel_gpu_top_output() == "[\n{}"


def test_intel_gpu_top_output_timeout_no_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intel_mod.shutil, "which", lambda _t: "/usr/bin/intel_gpu_top")

    def _raise(*_a: object, **_k: object) -> None:
        raise subprocess.TimeoutExpired(cmd="x", timeout=1.3)

    monkeypatch.setattr(intel_mod.subprocess, "run", _raise)
    assert intel_mod._intel_gpu_top_output() == ""


def test_intel_gpu_top_output_os_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intel_mod.shutil, "which", lambda _t: "/usr/bin/intel_gpu_top")

    def _raise(*_a: object, **_k: object) -> None:
        raise OSError("boom")

    monkeypatch.setattr(intel_mod.subprocess, "run", _raise)
    assert intel_mod._intel_gpu_top_output() == ""


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_registry_maps_sycl_to_intel_backend() -> None:
    assert isinstance(gpu_backends.resolve_backend("SYCL"), IntelBackend)
