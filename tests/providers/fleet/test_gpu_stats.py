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


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    """The nvidia-smi probe cache is process-wide; reset it so each test re-probes."""
    stats_mod._smi_cache.reset()


def _fake_run(stdout: str, returncode: int = 0):
    def _run(*_a: object, **_k: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")

    return _run


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
    assert stat == GpuStat(0, None, 20 * 1024 * _MIB, 32 * 1024 * _MIB)


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
    mac_device = (FleetDevice("Metal", 0, "Apple M3 Pro", 32 * 1024 * _MIB, 20 * 1024 * _MIB),)
    result = probe_gpu_stats(mac_device)
    assert result[0] == GpuStat(0, None, 20 * 1024 * _MIB, 32 * 1024 * _MIB)


def test_windows_no_cuda_skips_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Windows without CUDA, nvidia-smi is never invoked."""
    def _boom(*_a: object, **_k: object) -> subprocess.CompletedProcess:
        raise AssertionError("nvidia-smi must not run when no CUDA devices present")

    monkeypatch.setattr(stats_mod.subprocess, "run", _boom)
    # Non-CUDA device simulates a CPU-only or Vulkan-only Windows build.
    win_device = (FleetDevice("Vulkan", 0, "NVIDIA RTX 4090", 24 * 1024 * _MIB, 22 * 1024 * _MIB),)
    result = probe_gpu_stats(win_device)
    assert result[0] == GpuStat(0, None, 22 * 1024 * _MIB, 24 * 1024 * _MIB)
