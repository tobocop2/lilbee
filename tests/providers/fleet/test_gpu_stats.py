"""Tests for live per-GPU stats (utilization + free memory) probing."""

from __future__ import annotations

import subprocess
import threading

import pytest

from lilbee.providers.fleet import gpu_stats as stats_mod
from lilbee.providers.fleet.devices import _MIB, FleetDevice
from lilbee.providers.fleet.gpu_stats import GpuStat, probe_gpu_stats

_CUDA = (
    FleetDevice("CUDA", 0, "NVIDIA A40", 48 * 1024 * _MIB, 47 * 1024 * _MIB),
    FleetDevice("CUDA", 1, "NVIDIA A40", 48 * 1024 * _MIB, 47 * 1024 * _MIB),
)
_ROCM = (
    FleetDevice("ROCm", 0, "AMD RX 7900", 16 * 1024 * _MIB, 14 * 1024 * _MIB),
    FleetDevice("ROCm", 1, "AMD RX 7900", 16 * 1024 * _MIB, 14 * 1024 * _MIB),
)
_HIP = (FleetDevice("HIP", 0, "AMD MI300X", 192 * 1024 * _MIB, 180 * 1024 * _MIB),)
_SYCL = (FleetDevice("SYCL", 0, "Intel Arc A770", 16 * 1024 * _MIB, 15 * 1024 * _MIB),)
_METAL = (FleetDevice("MTL", 0, "Apple M1 Pro", 21845 * _MIB, 21844 * _MIB),)


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    """The nvidia-smi probe cache is process-wide; reset it so each test re-probes."""
    stats_mod._smi_cache.reset()


def _fake_run(stdout: str, returncode: int = 0):
    def _run(*_a: object, **_k: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")

    return _run


# =============================================================================
# CUDA / nvidia-smi
# =============================================================================


def test_parses_nvidia_smi_utilization_and_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        stats_mod.subprocess, "run", _fake_run("0, 37, 3536, 46068\n1, 12, 3352, 46068\n")
    )
    result = probe_gpu_stats(_CUDA)
    assert result[0] == GpuStat(0, 37, (46068 - 3536) * _MIB, 46068 * _MIB)
    assert result[1] == GpuStat(1, 12, (46068 - 3352) * _MIB, 46068 * _MIB)


def test_non_cuda_devices_skip_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: object, **_k: object) -> subprocess.CompletedProcess:
        raise AssertionError("nvidia-smi should not run for non-CUDA backends")

    monkeypatch.setattr(stats_mod.subprocess, "run", _boom)
    metal = (FleetDevice("metal", 0, "Apple M3", 32 * 1024 * _MIB, 20 * 1024 * _MIB),)
    (stat,) = probe_gpu_stats(metal).values()
    assert stat.utilization_pct is None
    assert stat.free_bytes == 20 * 1024 * _MIB
    assert stat.total_bytes == 32 * 1024 * _MIB


def test_falls_back_to_structural_totals_on_smi_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: object, **_k: object) -> subprocess.CompletedProcess:
        raise OSError("nvidia-smi missing")

    monkeypatch.setattr(stats_mod.subprocess, "run", _boom)
    result = probe_gpu_stats(_CUDA)
    assert result[0] == GpuStat(0, None, 47 * 1024 * _MIB, 48 * 1024 * _MIB)


def test_nonzero_returncode_yields_no_smi_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stats_mod.subprocess, "run", _fake_run("0, 50, 100, 200\n", returncode=9))
    result = probe_gpu_stats(_CUDA)
    assert result[0].utilization_pct is None


def test_malformed_lines_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        stats_mod.subprocess, "run", _fake_run("garbage\n0, x, y, z\n1, 20, 1024, 46068\n")
    )
    result = probe_gpu_stats(_CUDA)
    assert result[0].utilization_pct is None  # device 0 had no valid smi row
    assert result[1].utilization_pct == 20


def test_empty_device_list_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stats_mod.subprocess, "run", _fake_run(""))
    assert probe_gpu_stats(()) == {}


def test_concurrent_probes_coalesce_to_one_nvidia_smi_call(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def _counting_run(*_a: object, **_k: object) -> subprocess.CompletedProcess:
        calls["n"] += 1
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout="0, 30, 100, 46068\n", stderr=""
        )

    monkeypatch.setattr(stats_mod.subprocess, "run", _counting_run)
    # Two probes inside the TTL window share one nvidia-smi spawn.
    probe_gpu_stats(_CUDA)
    probe_gpu_stats(_CUDA)
    assert calls["n"] == 1
    # After a reset (or once the window lapses) the next probe spawns again.
    stats_mod._smi_cache.reset()
    probe_gpu_stats(_CUDA)
    assert calls["n"] == 2


def test_ttl_expiry_triggers_new_smi_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cache re-probes after the TTL window elapses."""
    calls = {"n": 0}
    fake_now = [0.0]

    def _counting_run(*_a: object, **_k: object) -> subprocess.CompletedProcess:
        calls["n"] += 1
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout="0, 10, 100, 46068\n", stderr=""
        )

    monkeypatch.setattr(stats_mod.subprocess, "run", _counting_run)
    monkeypatch.setattr(stats_mod.time, "monotonic", lambda: fake_now[0])

    probe_gpu_stats(_CUDA)
    assert calls["n"] == 1

    # Still inside TTL: no new probe.
    fake_now[0] = stats_mod._SMI_CACHE_TTL_S - 0.01
    probe_gpu_stats(_CUDA)
    assert calls["n"] == 1

    # Past TTL: new probe.
    fake_now[0] = stats_mod._SMI_CACHE_TTL_S + 0.01
    probe_gpu_stats(_CUDA)
    assert calls["n"] == 2


def test_concurrent_threads_do_not_double_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lock prevents concurrent threads from spawning nvidia-smi simultaneously.

    Two threads call probe_gpu_stats on a fresh cache. The first acquires the
    lock and calls _nvidia_smi_stats; the second blocks on the lock and then
    reuses the cached value without calling it again. The probe count must be 1.
    """
    calls = {"n": 0}
    # Let the first thread hold the lock briefly so the second arrives while it
    # is still inside _SmiCache.stats().
    ready = threading.Event()
    proceed = threading.Event()

    original_smi_stats = stats_mod._nvidia_smi_stats

    def _gated_smi_stats() -> dict[int, GpuStat]:
        calls["n"] += 1
        ready.set()       # signal: first thread is inside the probe
        proceed.wait()    # wait until the second thread has queued on the lock
        return original_smi_stats()

    monkeypatch.setattr(stats_mod, "_nvidia_smi_stats", _gated_smi_stats)
    monkeypatch.setattr(stats_mod.subprocess, "run", _fake_run("0, 20, 100, 46068\n"))

    results: list[dict] = [{}, {}]
    errors: list[Exception] = []

    def _probe(idx: int) -> None:
        try:
            results[idx] = probe_gpu_stats(_CUDA)
        except Exception as exc:
            errors.append(exc)

    t0 = threading.Thread(target=_probe, args=(0,))
    t0.start()
    ready.wait(timeout=5)  # wait until t0 is holding the lock inside _gated_smi_stats

    t1 = threading.Thread(target=_probe, args=(1,))
    t1.start()
    # Give t1 time to block on the lock; then let t0 finish.
    import time as _time
    _time.sleep(0.05)
    proceed.set()

    t0.join(timeout=10)
    t1.join(timeout=10)

    assert not errors
    # Both threads get valid results; only one spawned nvidia-smi.
    assert results[0] and results[1]
    assert calls["n"] == 1


def test_nvidia_smi_resolved_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """_nvidia_smi_output resolves the binary path on each call, not at import time."""
    resolved = []

    def _fake_which(name: str) -> str | None:
        resolved.append(name)
        return "/custom/nvidia-smi"

    monkeypatch.setattr(stats_mod.shutil, "which", _fake_which)
    monkeypatch.setattr(stats_mod.subprocess, "run", _fake_run("0, 10, 100, 46068\n"))

    stats_mod._nvidia_smi_output()
    assert "nvidia-smi" in resolved, "shutil.which should be called at call time"


def test_macos_no_cuda_skips_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    """On macOS (no CUDA devices), nvidia-smi is never invoked."""
    def _boom(*_a: object, **_k: object) -> subprocess.CompletedProcess:
        raise AssertionError("nvidia-smi must not run on macOS/non-CUDA")

    monkeypatch.setattr(stats_mod.subprocess, "run", _boom)
    result = probe_gpu_stats(_METAL)
    assert result[0].utilization_pct is None


def test_windows_no_cuda_skips_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Windows without CUDA, nvidia-smi is never invoked."""
    def _boom(*_a: object, **_k: object) -> subprocess.CompletedProcess:
        raise AssertionError("nvidia-smi must not run when no CUDA devices present")

    monkeypatch.setattr(stats_mod.subprocess, "run", _boom)
    # Non-CUDA device simulates a CPU-only or Vulkan-only Windows build.
    win_device = (FleetDevice("Vulkan", 0, "NVIDIA RTX 4090", 24 * 1024 * _MIB, 22 * 1024 * _MIB),)
    result = probe_gpu_stats(win_device)
    assert result[0] == GpuStat(0, None, 22 * 1024 * _MIB, 24 * 1024 * _MIB)


def test_cuda_temperature_is_none_from_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    """nvidia-smi query doesn't include temperature; temperature_c stays None."""
    monkeypatch.setattr(
        stats_mod.subprocess, "run", _fake_run("0, 55, 2000, 46068\n")
    )
    result = probe_gpu_stats(_CUDA[:1])
    assert result[0].utilization_pct == 55
    assert result[0].temperature_c is None


# =============================================================================
# ROCm / HIP (amd-smi + rocm-smi fallback)
# =============================================================================


_AMD_SMI_JSON = """\
[
  {"gpu": 0, "gfx_activity": 72, "temperature_c": 61},
  {"gpu": 1, "gfx_activity": 45, "temperature_c": 58}
]
"""

_ROCM_SMI_JSON = """\
{
  "card0": {
    "GPU use (%)": "35",
    "Temperature (Sensor edge) (C)": "52"
  },
  "card1": {
    "GPU use (%)": "80",
    "Temperature (Sensor edge) (C)": "67"
  }
}
"""


def test_rocm_amd_smi_parses_util_and_temp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stats_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(stats_mod.subprocess, "run", _fake_run(_AMD_SMI_JSON))
    result = probe_gpu_stats(_ROCM)
    assert result[0].utilization_pct == 72
    assert result[0].temperature_c == 61
    assert result[1].utilization_pct == 45
    assert result[1].temperature_c == 58


def test_rocm_falls_back_to_rocm_smi_when_amd_smi_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    def _which(name: str) -> str | None:
        return None if name == stats_mod._TOOL_AMD_SMI else f"/usr/bin/{name}"

    def _fake_run_rocm(*_a: object, **_k: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=_ROCM_SMI_JSON, stderr="")

    monkeypatch.setattr(stats_mod.shutil, "which", _which)
    monkeypatch.setattr(stats_mod.subprocess, "run", _fake_run_rocm)
    result = probe_gpu_stats(_ROCM)
    assert result[0].utilization_pct == 35
    assert result[0].temperature_c == 52


def test_hip_uses_rocm_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stats_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(stats_mod.subprocess, "run", _fake_run(_AMD_SMI_JSON))
    result = probe_gpu_stats(_HIP)
    assert result[0].utilization_pct == 72


def test_rocm_amd_smi_failure_falls_back_to_structural(monkeypatch: pytest.MonkeyPatch) -> None:
    """When both amd-smi and rocm-smi fail, structural fallback is used."""
    monkeypatch.setattr(stats_mod.shutil, "which", lambda _: None)
    result = probe_gpu_stats(_ROCM)
    assert result[0].utilization_pct is None
    assert result[0].free_bytes == 14 * 1024 * _MIB


def test_amd_smi_malformed_json_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stats_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(stats_mod.subprocess, "run", _fake_run("not json"))
    result = probe_gpu_stats(_ROCM[:1])
    assert result[0].utilization_pct is None


def test_rocm_smi_malformed_json_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def _which(name: str) -> str | None:
        return None if name == stats_mod._TOOL_AMD_SMI else f"/usr/bin/{name}"

    monkeypatch.setattr(stats_mod.shutil, "which", _which)
    monkeypatch.setattr(stats_mod.subprocess, "run", _fake_run("not json"))
    result = probe_gpu_stats(_ROCM[:1])
    assert result[0].utilization_pct is None


def test_amd_smi_nonzero_exit_falls_back_to_rocm_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stats_mod.shutil, "which", lambda name: f"/usr/bin/{name}")

    def _run(*args: object, **_k: object) -> subprocess.CompletedProcess:
        cmd = list(args[0])  # type: ignore[arg-type]
        if stats_mod._TOOL_AMD_SMI in cmd[0]:
            return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=_ROCM_SMI_JSON, stderr="")

    monkeypatch.setattr(stats_mod.subprocess, "run", _run)
    result = probe_gpu_stats(_ROCM[:1])
    assert result[0].utilization_pct == 35


def test_rocm_indices_not_in_result_are_filtered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only indices belonging to the probed devices are included."""
    extra_json = '[{"gpu": 0, "gfx_activity": 10}, {"gpu": 5, "gfx_activity": 99}]'
    monkeypatch.setattr(stats_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(stats_mod.subprocess, "run", _fake_run(extra_json))
    result = probe_gpu_stats(_ROCM[:1])  # only index 0
    assert 5 not in result
    assert result[0].utilization_pct == 10


# =============================================================================
# SYCL / Intel (xpu-smi)
# =============================================================================

_XPU_SMI_JSON = """\
[
  {
    "device_id": 0,
    "gpu_utilization": 88,
    "gpu_temperature": 73,
    "gpu_memory_used_in_mb": 1024,
    "gpu_memory_size_in_mb": 16384
  }
]
"""


def test_sycl_xpu_smi_parses_util_temp_and_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stats_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(stats_mod.subprocess, "run", _fake_run(_XPU_SMI_JSON))
    result = probe_gpu_stats(_SYCL)
    assert result[0].utilization_pct == 88
    assert result[0].temperature_c == 73
    assert result[0].total_bytes == 16384 * _MIB
    assert result[0].free_bytes == (16384 - 1024) * _MIB


def test_sycl_xpu_smi_absent_falls_back_to_structural(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stats_mod.shutil, "which", lambda _: None)
    result = probe_gpu_stats(_SYCL)
    assert result[0].utilization_pct is None
    assert result[0].free_bytes == 15 * 1024 * _MIB


def test_sycl_xpu_smi_malformed_json_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stats_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(stats_mod.subprocess, "run", _fake_run("{bad"))
    result = probe_gpu_stats(_SYCL)
    assert result[0].utilization_pct is None


def test_sycl_xpu_smi_nonzero_exit_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stats_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(stats_mod.subprocess, "run", _fake_run(_XPU_SMI_JSON, returncode=1))
    result = probe_gpu_stats(_SYCL)
    assert result[0].utilization_pct is None


def test_sycl_subprocess_error_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stats_mod.shutil, "which", lambda name: f"/usr/bin/{name}")

    def _oserr(*_a: object, **_k: object) -> subprocess.CompletedProcess:
        raise OSError("xpu-smi not executable")

    monkeypatch.setattr(stats_mod.subprocess, "run", _oserr)
    result = probe_gpu_stats(_SYCL)
    assert result[0].utilization_pct is None


# =============================================================================
# Apple Metal (MTL) -- stub backend
# =============================================================================


def test_metal_backend_returns_structural_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """MTL backend is a stub; util/temp stay None, VRAM comes from structural probe."""
    def _boom(*_a: object, **_k: object) -> subprocess.CompletedProcess:
        raise AssertionError("No CLI tool should be invoked for Metal")

    monkeypatch.setattr(stats_mod.subprocess, "run", _boom)
    result = probe_gpu_stats(_METAL)
    assert result[0].utilization_pct is None
    assert result[0].temperature_c is None
    assert result[0].free_bytes == 21844 * _MIB
    assert result[0].total_bytes == 21845 * _MIB


def test_metal_backend_does_not_invoke_any_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Metal stub must not attempt to call any external tool."""
    invocations: list[str] = []

    def _which(name: str) -> str | None:
        invocations.append(name)
        return None

    monkeypatch.setattr(stats_mod.shutil, "which", _which)
    probe_gpu_stats(_METAL)
    assert not invocations, f"unexpected which() calls: {invocations}"


# =============================================================================
# Unknown / unregistered backends
# =============================================================================


def test_unknown_backend_falls_back_to_structural(monkeypatch: pytest.MonkeyPatch) -> None:
    """A backend string not in the registry (e.g. 'Vulkan') uses structural fallback."""
    def _boom(*_a: object, **_k: object) -> subprocess.CompletedProcess:
        raise AssertionError("no CLI should run for unknown backend")

    monkeypatch.setattr(stats_mod.subprocess, "run", _boom)
    vulkan = (FleetDevice("Vulkan", 0, "NVIDIA RTX 4090", 24 * 1024 * _MIB, 22 * 1024 * _MIB),)
    result = probe_gpu_stats(vulkan)
    assert result[0] == GpuStat(0, None, 22 * 1024 * _MIB, 24 * 1024 * _MIB)


# =============================================================================
# GpuStat field coverage
# =============================================================================


def test_gpustat_temperature_c_defaults_to_none() -> None:
    stat = GpuStat(0, 50, 100, 200)
    assert stat.temperature_c is None


def test_gpustat_temperature_c_set() -> None:
    stat = GpuStat(0, 50, 100, 200, temperature_c=75)
    assert stat.temperature_c == 75


# =============================================================================
# parse helpers
# =============================================================================


def test_extract_int_finds_first_matching_key() -> None:
    obj = {"a": 10, "b": 20}
    assert stats_mod._extract_int(obj, ("b", "a")) == 20


def test_extract_int_returns_none_for_missing_keys() -> None:
    assert stats_mod._extract_int({}, ("x", "y")) is None


def test_extract_int_skips_non_int_values() -> None:
    obj = {"a": "bad", "b": 7}
    assert stats_mod._extract_int(obj, ("a", "b")) == 7


def test_extract_int_returns_none_for_non_dict() -> None:
    assert stats_mod._extract_int("string", ("a",)) is None


def test_parse_device_index_card_prefix() -> None:
    assert stats_mod._parse_device_index("card0") == 0
    assert stats_mod._parse_device_index("card12") == 12


def test_parse_device_index_bracket_format() -> None:
    assert stats_mod._parse_device_index("GPU[3]") == 3


def test_parse_device_index_no_digit_returns_none() -> None:
    assert stats_mod._parse_device_index("GPU") is None
