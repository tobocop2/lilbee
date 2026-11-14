"""Tests for the NVIDIA/CUDA utilization backend."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

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
    nvidia_mod._cache.stats()
    assert "nvidia-smi" in resolved


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_registry_maps_cuda_to_nvidia_backend() -> None:
    assert isinstance(gpu_backends.resolve_backend("CUDA"), NvidiaBackend)


# ---------------------------------------------------------------------------
# Card facts and host facts
# ---------------------------------------------------------------------------

_CARDS_CSV = (
    "0, GPU-a, 24576, 23552, NVIDIA GeForce RTX 3090\n"
    "1, GPU-b, 12288, 10240, NVIDIA GeForce RTX 3060\n"
)
_NA_FIRST_CSV = "0, GPU-a, [N/A], [N/A], n0\n1, GPU-b, 8192, 8000, n1\n"
_COMPUTE_CSV = "0, 8.6\n1, 8.6\n"
_CUDA_XML = (
    "<nvidia_smi_log><driver_version>550.1</driver_version>"
    "<cuda_version>12.4</cuda_version></nvidia_smi_log>"
)


class _Smi:
    """A fake nvidia-smi that answers each query from a table and counts the runs."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, **answers: str) -> None:
        self.answers = {"cards": _CARDS_CSV, "compute": _COMPUTE_CSV, "cuda": _CUDA_XML} | answers
        self.runs: list[str] = []
        self.now = 0.0
        self.cost = 0.0
        monkeypatch.setattr(nvidia_mod, "run_smi", self)
        monkeypatch.setattr(nvidia_mod.time, "monotonic", lambda: self.now)

    def __call__(self, _tool: str, args: list[str], _timeout: float = 5.0) -> str:
        kind = "cuda" if "-x" in args else "compute" if "compute_cap" in args[0] else "cards"
        self.runs.append(kind)
        self.now += self.cost
        return self.answers[kind]


def test_cards_read_uuid_name_and_memory_in_driver_order(monkeypatch: pytest.MonkeyPatch) -> None:
    _Smi(monkeypatch)
    assert nvidia_mod.cards() == [
        nvidia_mod.NvidiaCard(0, "GPU-a", "NVIDIA GeForce RTX 3090", 24576 * MIB, 23552 * MIB),
        nvidia_mod.NvidiaCard(1, "GPU-b", "NVIDIA GeForce RTX 3060", 12288 * MIB, 10240 * MIB),
    ]


def test_cards_are_empty_when_nvidia_smi_cannot_run(monkeypatch: pytest.MonkeyPatch) -> None:
    _Smi(monkeypatch, cards="")
    assert nvidia_mod.cards() == []


def test_cards_are_unknown_when_one_memory_figure_does_not_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keeping the other card would number it as if the unreadable one did not exist."""
    _Smi(monkeypatch, cards=_NA_FIRST_CSV)
    assert nvidia_mod.cards() == []


def test_cards_skip_a_row_without_all_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    _Smi(monkeypatch, cards="garbage\n1, GPU-b, 8192, n1\n0, GPU-a, 8192, 8000, n0\n")
    assert [c.uuid for c in nvidia_mod.cards()] == ["GPU-a"]


def test_host_facts_read_compute_capability_and_driver_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Smi(monkeypatch)
    assert nvidia_mod.host_facts() == {
        "cuda_driver": 12040,
        "gpus": [
            {"name": "NVIDIA GeForce RTX 3090", "cc": [8, 6]},
            {"name": "NVIDIA GeForce RTX 3060", "cc": [8, 6]},
        ],
    }


def test_host_facts_are_none_when_nvidia_smi_fails_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Smi(monkeypatch, cards="")
    assert nvidia_mod.host_facts() is None


def test_host_facts_are_unreadable_when_the_probe_hits_its_time_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smi = _Smi(monkeypatch, cards="")
    smi.cost = nvidia_mod._TIMEOUT_S
    assert nvidia_mod.host_facts() == {"cuda_driver": None, "gpus": None}
    assert smi.runs == ["cards"]


def test_a_card_that_prints_no_compute_capability_leaves_the_gpus_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Smi(monkeypatch, compute="0, 8.6\n1, [N/A]\n")
    facts = nvidia_mod.host_facts()
    assert facts is not None
    assert (facts["gpus"], facts["cuda_driver"]) == (None, 12040)


@pytest.mark.parametrize("xml", ["", "<cuda_version>N/A</cuda_version>", "<x>12.4</x>"])
def test_a_driver_with_no_cuda_version_leaves_it_unknown(
    monkeypatch: pytest.MonkeyPatch, xml: str
) -> None:
    _Smi(monkeypatch, cuda=xml)
    facts = nvidia_mod.host_facts()
    assert facts is not None
    assert facts["cuda_driver"] is None
    assert [g["name"] for g in facts["gpus"]] == [
        "NVIDIA GeForce RTX 3090",
        "NVIDIA GeForce RTX 3060",
    ]


def test_the_compute_capability_and_cuda_version_are_read_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smi = _Smi(monkeypatch)
    nvidia_mod.host_facts()
    smi.now += 3600
    nvidia_mod.host_facts()
    assert smi.runs.count("compute") == 1
    assert smi.runs.count("cuda") == 1
    assert smi.runs.count("cards") == 2


def test_a_probe_that_hit_its_time_limit_is_not_run_again_for_thirty_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smi = _Smi(monkeypatch, cards="")
    smi.cost = nvidia_mod._TIMEOUT_S
    nvidia_mod.cards()
    smi.now += nvidia_mod._WEDGED_TTL_S - 1
    nvidia_mod.cards()
    assert smi.runs == ["cards"]
    smi.now += 2
    nvidia_mod.cards()
    assert smi.runs == ["cards", "cards"]


def test_a_probe_that_fails_at_once_is_run_again_after_the_short_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smi = _Smi(monkeypatch, cards="")
    nvidia_mod.cards()
    smi.now += nvidia_mod._CACHE_TTL_S + 0.1
    smi.answers["cards"] = _CARDS_CSV
    assert len(nvidia_mod.cards()) == 2
    assert smi.runs == ["cards", "cards"]


def test_readable_cards_leave_out_a_card_whose_memory_does_not_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Smi(monkeypatch, cards=_NA_FIRST_CSV)
    assert [c.uuid for c in nvidia_mod.readable_cards()] == ["GPU-b"]


def test_host_facts_keep_a_card_whose_memory_does_not_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Smi(monkeypatch, cards=_NA_FIRST_CSV)
    assert nvidia_mod.host_facts() == {
        "cuda_driver": 12040,
        "gpus": [{"name": "n0", "cc": [8, 6]}, {"name": "n1", "cc": [8, 6]}],
    }


def test_a_gpu_name_with_a_comma_keeps_its_card(monkeypatch: pytest.MonkeyPatch) -> None:
    csv = "0, GPU-a, 24576, 23552, Acme, Inc. Card\n1, GPU-b, 8192, 8000, n1\n"
    _Smi(monkeypatch, cards=csv)
    assert [(c.uuid, c.name, c.total_bytes) for c in nvidia_mod.cards()] == [
        ("GPU-a", "Acme, Inc. Card", 24576 * MIB),
        ("GPU-b", "n1", 8192 * MIB),
    ]


@pytest.mark.parametrize("kind", ["compute", "cuda"])
def test_a_fast_failure_of_a_host_query_is_run_again_after_the_short_cache(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    smi = _Smi(monkeypatch, **{kind: ""})
    nvidia_mod.host_facts()
    smi.now += nvidia_mod._CACHE_TTL_S + 0.1
    smi.answers[kind] = {"compute": _COMPUTE_CSV, "cuda": _CUDA_XML}[kind]
    facts = nvidia_mod.host_facts()
    assert facts is not None
    assert facts["cuda_driver"] == 12040
    assert [g["cc"] for g in facts["gpus"]] == [[8, 6], [8, 6]]
    assert smi.runs.count(kind) == 2


@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in tool is a shell script")
def test_a_tool_that_never_answers_is_stopped_at_the_time_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool = tmp_path / "nvidia-smi"
    tool.write_text("#!/bin/sh\nexec sleep 30\n", encoding="utf-8")
    tool.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(nvidia_mod, "_TIMEOUT_S", 0.5)
    started = time.monotonic()
    reading = nvidia_mod._cards_probe.read()
    assert time.monotonic() - started < 3.0
    assert reading == nvidia_mod.Reading("", True)
