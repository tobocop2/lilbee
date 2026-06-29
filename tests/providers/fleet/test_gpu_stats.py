"""Tests for live per-GPU stats (utilization + free memory) probing."""

from __future__ import annotations

import subprocess

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
