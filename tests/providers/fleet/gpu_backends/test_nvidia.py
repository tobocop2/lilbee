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
    out = "0, 37, 61, 3536, 46068\n1, 12, 61, 3352, 46068\n"
    result = _parse_smi_output(out)
    assert result[0].utilization_pct == 37
    assert result[0].free_bytes == (46068 - 3536) * MIB
    assert result[0].total_bytes == 46068 * MIB
    assert result[0].temperature_c == 61
    assert result[1].utilization_pct == 12


def test_parse_smi_output_reads_each_cards_temperature() -> None:
    result = _parse_smi_output("0, 37, 70, 3536, 46068\n1, 12, 86, 3352, 46068\n")
    assert [result[0].temperature_c, result[1].temperature_c] == [70, 86]


def test_parse_smi_output_unavailable_temperature_keeps_the_rest() -> None:
    result = _parse_smi_output("0, 37, [N/A], 3536, 46068\n")
    assert result[0].temperature_c is None
    assert result[0].utilization_pct == 37
    assert result[0].total_bytes == 46068 * MIB


def test_parse_smi_output_four_column_row_is_skipped() -> None:
    """A row without the temperature column is not read as shifted columns."""
    assert _parse_smi_output("0, 37, 3536, 46068\n") == {}


def test_card_count_counts_rows_whose_memory_did_not_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv = "0, 37, 61, 100, 46068\n1, 12, 61, [N/A], [N/A]\nbad row\n"
    monkeypatch.setattr(nvidia_mod, "run_smi", lambda *_a, **_k: csv)
    assert nvidia_mod.card_count() == 2
    assert list(nvidia_mod._cache.stats()) == [0]


def test_card_count_is_zero_when_smi_cannot_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nvidia_mod, "run_smi", lambda *_a, **_k: "")
    assert nvidia_mod.card_count() == 0


def test_card_count_and_stats_share_one_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def _count(*_a: object, **_k: object) -> str:
        calls["n"] += 1
        return "0, 30, 61, 100, 46068\n"

    monkeypatch.setattr(nvidia_mod, "run_smi", _count)
    nvidia_mod.card_count()
    NvidiaBackend().sample(frozenset({0}))
    assert calls["n"] == 1


def test_a_slow_probe_does_not_age_its_own_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """The entry is stamped when the probe ends, so a probe longer than the TTL is still fresh."""
    now = [0.0]
    calls = {"n": 0}

    def _slow(*_a: object, **_k: object) -> str:
        calls["n"] += 1
        now[0] += 5.0
        return "0, 30, 61, 100, 46068\n"

    monkeypatch.setattr(nvidia_mod, "run_smi", _slow)
    monkeypatch.setattr(nvidia_mod.time, "monotonic", lambda: now[0])
    cache = SmiCache()
    cache.stats()
    cache.stats()
    assert calls["n"] == 1


def test_parse_smi_output_malformed_lines_skipped() -> None:
    out = "garbage\n0, x, 61, y, z\n1, 20, 61, 1024, 46068\n"
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
    csv = "0, 37, 61, 100, 46068\n1, 12, 61, 100, 46068\n"
    monkeypatch.setattr(nvidia_mod, "run_smi", lambda *_a, **_k: csv)
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
        return "0, 30, 61, 100, 46068\n"

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
        return "0, 10, 61, 100, 46068\n"

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
    monkeypatch.setattr(nvidia_mod, "run_smi", lambda *_a, **_k: "0, 20, 61, 100, 46068\n")

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
        lambda *_a, **_k: subprocess.CompletedProcess(
            args=[], returncode=0, stdout="0, 10, 61, 100, 46068\n", stderr=""
        ),
    )
    nvidia_mod._smi_output()
    assert "nvidia-smi" in resolved


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_registry_maps_cuda_to_nvidia_backend() -> None:
    assert isinstance(gpu_backends.resolve_backend("CUDA"), NvidiaBackend)
