"""Tests for the NVIDIA/CUDA utilization backend."""

from __future__ import annotations

import subprocess
import threading

import pytest

from lilbee.providers.fleet import gpu_backends
from lilbee.providers.fleet.devices import MIB
from lilbee.providers.fleet.gpu_backends import base as base_mod
from lilbee.providers.fleet.gpu_backends import nvidia as nvidia_mod
from lilbee.providers.fleet.gpu_backends.nvidia import NvidiaBackend, SmiCache, _parse_smi_output


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    nvidia_mod._cache.reset()


def _fake_run(stdout: str, returncode: int = 0):
    def _run(*_a: object, **_k: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")

    return _run


# ---------------------------------------------------------------------------
# parse helper
# ---------------------------------------------------------------------------


def test_parse_smi_output_happy_path() -> None:
    out = "0, 37, 3536, 46068\n1, 12, 3352, 46068\n"
    result = _parse_smi_output(out)
    assert result[0].utilization_pct == 37
    assert result[0].free_bytes == (46068 - 3536) * MIB
    assert result[0].total_bytes == 46068 * MIB
    assert result[0].temperature_c is None
    assert result[1].utilization_pct == 12


def test_parse_smi_output_malformed_lines_skipped() -> None:
    out = "garbage\n0, x, y, z\n1, 20, 1024, 46068\n"
    result = _parse_smi_output(out)
    assert 0 not in result
    assert result[1].utilization_pct == 20


def test_parse_smi_output_empty() -> None:
    assert _parse_smi_output("") == {}


# ---------------------------------------------------------------------------
# NvidiaBackend.sample
# ---------------------------------------------------------------------------


def test_sample_returns_indexed_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    # run_smi is imported by name into nvidia_mod; patch there, not in base.
    monkeypatch.setattr(nvidia_mod, "run_smi", lambda *_a, **_k: "0, 37, 100, 46068\n1, 12, 100, 46068\n")
    backend = NvidiaBackend()
    result = backend.sample(frozenset({0}))
    assert 0 in result
    assert 1 not in result


def test_sample_falls_back_to_empty_on_smi_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nvidia_mod, "run_smi", lambda *_a, **_k: "")
    backend = NvidiaBackend()
    assert backend.sample(frozenset({0})) == {}


def test_sample_empty_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nvidia_mod, "run_smi", lambda *_a, **_k: "")
    backend = NvidiaBackend()
    assert backend.sample(frozenset({0})) == {}


# ---------------------------------------------------------------------------
# SmiCache TTL + thread safety
# ---------------------------------------------------------------------------


def test_cache_coalesces_concurrent_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def _count(*_a: object, **_k: object) -> str:
        calls["n"] += 1
        return "0, 30, 100, 46068\n"

    monkeypatch.setattr(nvidia_mod, "run_smi", _count)
    backend = NvidiaBackend()
    backend.sample(frozenset({0}))
    backend.sample(frozenset({0}))
    assert calls["n"] == 1
    nvidia_mod._cache.reset()
    backend.sample(frozenset({0}))
    assert calls["n"] == 2


def test_cache_ttl_triggers_new_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    fake_now = [0.0]

    def _count(*_a: object, **_k: object) -> str:
        calls["n"] += 1
        return "0, 10, 100, 46068\n"

    monkeypatch.setattr(nvidia_mod, "run_smi", _count)
    monkeypatch.setattr(nvidia_mod.time, "monotonic", lambda: fake_now[0])

    cache = SmiCache()
    cache.stats()
    assert calls["n"] == 1

    fake_now[0] = nvidia_mod._CACHE_TTL_S - 0.01
    cache.stats()
    assert calls["n"] == 1

    fake_now[0] = nvidia_mod._CACHE_TTL_S + 0.01
    cache.stats()
    assert calls["n"] == 2


def test_concurrent_threads_probe_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    ready = threading.Event()
    proceed = threading.Event()

    original = nvidia_mod._parse_smi_output

    def _gated(out: str) -> dict:
        calls["n"] += 1
        ready.set()
        proceed.wait()
        return original(out)

    monkeypatch.setattr(nvidia_mod, "_parse_smi_output", _gated)
    monkeypatch.setattr(nvidia_mod, "run_smi", lambda *_a, **_k: "0, 20, 100, 46068\n")

    cache = SmiCache()
    results: list[dict] = [{}, {}]
    errors: list[Exception] = []

    def _probe(idx: int) -> None:
        try:
            results[idx] = cache.stats()
        except Exception as exc:
            errors.append(exc)

    t0 = threading.Thread(target=_probe, args=(0,))
    t0.start()
    ready.wait(timeout=5)

    t1 = threading.Thread(target=_probe, args=(1,))
    t1.start()

    import time as _time
    _time.sleep(0.05)
    proceed.set()

    t0.join(timeout=10)
    t1.join(timeout=10)

    assert not errors
    assert results[0] and results[1]
    assert calls["n"] == 1


def test_which_resolved_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_smi resolves the binary via shutil.which on each call (tested in base)."""
    resolved: list[str] = []

    def _fake_which(name: str) -> str | None:
        resolved.append(name)
        return "/custom/nvidia-smi"

    monkeypatch.setattr(base_mod.shutil, "which", _fake_which)
    monkeypatch.setattr(
        base_mod.subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess(args=[], returncode=0, stdout="0, 10, 100, 46068\n", stderr=""),
    )
    nvidia_mod._smi_output()
    assert "nvidia-smi" in resolved


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_registry_maps_cuda_to_nvidia_backend() -> None:
    assert isinstance(gpu_backends.resolve_backend("CUDA"), NvidiaBackend)
