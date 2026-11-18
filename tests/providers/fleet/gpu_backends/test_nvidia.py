"""Tests for the NVIDIA/CUDA utilization backend."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from lilbee.providers.fleet import devices, gpu_backends
from lilbee.providers.fleet.devices import MIB
from lilbee.providers.fleet.gpu_backends import base as base_mod
from lilbee.providers.fleet.gpu_backends import nvidia as nvidia_mod
from lilbee.providers.fleet.gpu_backends.nvidia import NvidiaBackend, SmiCache, _parse_smi_output
from tests._nvidia_smi import prints


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
    monkeypatch.setattr(nvidia_mod, "run_tool", prints(lambda *_a, **_k: csv))
    assert nvidia_mod.card_count() == 2
    assert list(nvidia_mod._cache.stats()) == [0]


def test_a_stats_row_whose_index_does_not_read_names_no_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv = "0, 37, 61, 100, 46068\n[N/A], 12, 61, 100, 46068\n"
    monkeypatch.setattr(nvidia_mod, "run_tool", prints(lambda *_a, **_k: csv))
    assert nvidia_mod.card_count() == 1
    assert list(nvidia_mod._cache.stats()) == [0]


def test_card_count_is_zero_when_smi_cannot_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nvidia_mod, "run_tool", prints(lambda *_a, **_k: ""))
    assert nvidia_mod.card_count() == 0


def test_card_count_and_stats_share_one_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def _count(*_a: object, **_k: object) -> str:
        calls["n"] += 1
        return "0, 30, 61, 100, 46068\n"

    monkeypatch.setattr(nvidia_mod, "run_tool", prints(_count))
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

    monkeypatch.setattr(nvidia_mod, "run_tool", prints(_slow))
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
    # run_tool is imported by name into nvidia_mod; patch there, not in base.
    csv = "0, 37, 61, 100, 46068\n1, 12, 61, 100, 46068\n"
    monkeypatch.setattr(nvidia_mod, "run_tool", prints(lambda *_a, **_k: csv))
    backend = NvidiaBackend()
    result = backend.sample(frozenset({0}))
    assert 0 in result
    assert 1 not in result


def test_sample_falls_back_to_empty_on_smi_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nvidia_mod, "run_tool", prints(lambda *_a, **_k: ""))
    backend = NvidiaBackend()
    assert backend.sample(frozenset({0})) == {}


def test_sample_empty_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nvidia_mod, "run_tool", prints(lambda *_a, **_k: ""))
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

    monkeypatch.setattr(nvidia_mod, "run_tool", prints(_count))
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

    monkeypatch.setattr(nvidia_mod, "run_tool", prints(_count))
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
    monkeypatch.setattr(nvidia_mod, "run_tool", prints(lambda *_a, **_k: "0, 20, 61, 100, 46068\n"))

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
    """run_tool resolves the binary via shutil.which on each call (tested in base)."""
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
    """A fake nvidia-smi that answers each query from a table and counts the runs.

    An empty answer is a tool that cannot start, unless *codes* gives the query an exit code.
    """

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, codes: dict[str, int] | None = None, **answers: str
    ) -> None:
        self.answers = {"cards": _CARDS_CSV, "compute": _COMPUTE_CSV, "cuda": _CUDA_XML} | answers
        self.codes = codes or {}
        self.runs: list[str] = []
        self.now = 0.0
        self.cost = 0.0
        monkeypatch.setattr(nvidia_mod, "run_tool", self)
        monkeypatch.setattr(nvidia_mod.time, "monotonic", lambda: self.now)

    def __call__(
        self, _tool: str, args: list[str], _timeout: float = 5.0
    ) -> tuple[str, int | None]:
        kind = "cuda" if "-x" in args else "compute" if "compute_cap" in args[0] else "cards"
        self.runs.append(kind)
        self.now += self.cost
        answer = self.answers[kind]
        return answer, self.codes.get(kind, 0 if answer else None)


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


_LOST_CARD0_CSV = (
    "Unable to determine the device handle for GPU0000:01:00.0: Unknown Error\n"
    "1, GPU-b, 12288, 10240, NVIDIA GeForce RTX 3060\n"
)


@pytest.mark.parametrize(
    "csv",
    [
        _LOST_CARD0_CSV,
        "0, GPU-a, 24576, 23552, n0\n1, GPU-b, 8192, n1\n",
        "0, GPU-a, 24576, 23552, n0\nx1, GPU-b, 8192, 8000, n1\n",
    ],
)
def test_cards_are_unknown_when_a_row_does_not_parse(
    monkeypatch: pytest.MonkeyPatch, csv: str
) -> None:
    """Keeping the survivor would list it under the lost card's number."""
    _Smi(monkeypatch, cards=csv)
    assert nvidia_mod.cards() == []


@pytest.mark.parametrize(
    ("csv", "rows"),
    [
        (_NA_FIRST_CSV, 2),
        ("0, GPU-a, 24576, 23552, n0\n1, GPU-b, [N/A], [N/A], n1\n", 2),
        (_LOST_CARD0_CSV, 2),
        ("", 0),
    ],
    ids=["unread-first", "unread-second", "unparsed-row-is-a-card", "cannot-run"],
)
def test_card_rows_count_every_row_the_driver_prints(
    monkeypatch: pytest.MonkeyPatch, csv: str, rows: int
) -> None:
    _Smi(monkeypatch, cards=csv)
    assert nvidia_mod.card_rows() == rows


def test_card_totals_keep_a_row_that_does_not_parse_as_an_unread_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The row takes its position as index, so a mask can still name it."""
    _Smi(monkeypatch, cards=_LOST_CARD0_CSV)
    assert nvidia_mod.card_totals() == [
        nvidia_mod.CardTotal(0, "", None),
        nvidia_mod.CardTotal(1, "GPU-b", 12288 * MIB),
    ]


def test_host_facts_leave_the_gpus_unknown_when_a_row_does_not_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Smi(monkeypatch, cards=_LOST_CARD0_CSV)
    assert nvidia_mod.host_facts() == {"cuda_driver": 12040, "gpus": None}


@pytest.mark.parametrize("field", ["uuid", "name"])
def test_cards_are_unknown_when_a_uuid_or_name_does_not_read(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    row = {"uuid": "GPU-b", "name": "n1"} | {field: "[N/A]"}
    second = f"1, {row['uuid']}, 8192, 8000, {row['name']}\n"
    _Smi(monkeypatch, cards=f"0, GPU-a, 24576, 23552, n0\n{second}")
    assert nvidia_mod.cards() == []


def test_the_na_marker_is_not_a_gpu_name(monkeypatch: pytest.MonkeyPatch) -> None:
    _Smi(monkeypatch, cards="0, GPU-a, 24576, 23552, n0\n1, GPU-b, 8192, 8000, [N/A]\n")
    assert nvidia_mod.host_facts() == {"cuda_driver": 12040, "gpus": None}


def test_a_name_that_holds_the_na_marker_and_a_comma_is_a_gpu_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Smi(monkeypatch, cards="0, GPU-a, 24576, 23552, Card [N/A]x, rev 2\n", compute="0, 8.6\n")
    assert nvidia_mod.host_facts() == {
        "cuda_driver": 12040,
        "gpus": [{"name": "Card [N/A]x, rev 2", "cc": [8, 6]}],
    }


def test_the_card_list_keeps_a_name_that_holds_the_na_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Smi(monkeypatch, cards="0, GPU-a, 24576, 23552, Card [N/A]x, rev 2\n")
    assert nvidia_mod.cards() == [
        nvidia_mod.NvidiaCard(0, "GPU-a", "Card [N/A]x, rev 2", 24576 * MIB, 23552 * MIB)
    ]


def test_the_card_list_keeps_a_uuid_that_holds_the_na_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Smi(monkeypatch, cards="0, GPU-[N/A]-a, 24576, 23552, n0\n")
    assert nvidia_mod.cards() == [
        nvidia_mod.NvidiaCard(0, "GPU-[N/A]-a", "n0", 24576 * MIB, 23552 * MIB)
    ]


def test_the_card_list_is_unknown_when_a_name_is_exactly_the_na_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Smi(monkeypatch, cards="0, GPU-a, 24576, 23552, [N/A]\n")
    assert nvidia_mod.cards() == []


def test_a_card_whose_index_does_not_read_takes_its_row_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Smi(
        monkeypatch,
        cards="0, GPU-a, 24576, 23552, n0\n[N/A], GPU-b, 8192, 8000, n1\n",
        compute="0, 8.6\n[N/A], 7.5\n",
    )
    assert [(c.index, c.uuid) for c in nvidia_mod.cards()] == [(0, "GPU-a"), (1, "GPU-b")]
    assert nvidia_mod.card_totals() == [
        nvidia_mod.CardTotal(0, "GPU-a", 24576 * MIB),
        nvidia_mod.CardTotal(1, "GPU-b", 8192 * MIB),
    ]
    assert nvidia_mod.host_facts() == {
        "cuda_driver": 12040,
        "gpus": [{"name": "n0", "cc": [8, 6]}, {"name": "n1", "cc": [7, 5]}],
    }


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


@pytest.mark.parametrize("code", [6, 9, 12])
def test_an_exit_that_names_no_driver_or_no_card_reads_as_no_driver(
    monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    _Smi(monkeypatch, cards="", codes={"cards": code})
    assert nvidia_mod.host_facts() is None


def test_rows_printed_by_a_failed_card_probe_are_still_cards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Smi(monkeypatch, codes={"cards": 255})
    assert [c.index for c in nvidia_mod.cards()] == [0, 1]
    assert [t.index for t in nvidia_mod.card_totals()] == [0, 1]


@pytest.mark.parametrize("code", [1, 15, 18])
def test_any_other_fast_failure_leaves_the_driver_unknown(
    monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    """A version mismatch or a lost card is a driver in a bad state, not a host without one."""
    smi = _Smi(monkeypatch, cards="", codes={"cards": code})
    assert nvidia_mod.host_facts() == {"cuda_driver": None, "gpus": None}
    assert smi.runs == ["cards"]


def test_a_driver_that_lists_no_card_still_reads_its_cuda_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smi = _Smi(monkeypatch, cards="", codes={"cards": 0})
    assert nvidia_mod.host_facts() == {"cuda_driver": 12040, "gpus": []}
    assert "compute" not in smi.runs


def test_a_compute_row_that_does_not_parse_leaves_the_gpus_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Smi(monkeypatch, compute="0, 8.6\nUnable to determine the device handle for GPU1\n")
    assert nvidia_mod.host_facts() == {"cuda_driver": 12040, "gpus": None}


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


def test_card_totals_keep_a_card_whose_total_does_not_read_with_its_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Smi(monkeypatch, cards=_NA_FIRST_CSV)
    assert nvidia_mod.card_totals() == [
        nvidia_mod.CardTotal(0, "GPU-a", None),
        nvidia_mod.CardTotal(1, "GPU-b", 8192 * MIB),
    ]


def test_card_totals_keep_a_card_whose_free_memory_does_not_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Smi(monkeypatch, cards="0, GPU-a, 24576, 20000, n0\n1, GPU-b, 12288, [N/A], n1\n")
    assert nvidia_mod.card_totals() == [
        nvidia_mod.CardTotal(0, "GPU-a", 24576 * MIB),
        nvidia_mod.CardTotal(1, "GPU-b", 12288 * MIB),
    ]


_CARD_TABLE = (
    {
        "index": "0",
        "uuid": "GPU-a",
        "name": "Acme, Inc. Card",
        "memory.total": "24576",
        "memory.free": "23552",
    },
    {
        "index": "1",
        "uuid": "GPU-b",
        "name": "NVIDIA GeForce RTX 3060",
        "memory.total": "12288",
        "memory.free": "10240",
    },
)


def _print_asked_columns(_tool: str, args: list[str], _timeout: float = 5.0) -> str:
    """A fake nvidia-smi that prints the card columns in the order the query names them."""
    if "-x" in args:
        return _CUDA_XML
    columns = args[0].removeprefix("--query-gpu=").split(",")
    if columns == ["index", "compute_cap"]:
        return _COMPUTE_CSV
    return "".join(", ".join(card[c] for c in columns) + "\n" for card in _CARD_TABLE)


def test_the_card_query_and_its_parser_name_the_same_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nvidia_mod, "run_tool", prints(_print_asked_columns))
    assert nvidia_mod.cards() == [
        nvidia_mod.NvidiaCard(0, "GPU-a", "Acme, Inc. Card", 24576 * MIB, 23552 * MIB),
        nvidia_mod.NvidiaCard(1, "GPU-b", "NVIDIA GeForce RTX 3060", 12288 * MIB, 10240 * MIB),
    ]
    assert nvidia_mod.card_totals() == [
        nvidia_mod.CardTotal(0, "GPU-a", 24576 * MIB),
        nvidia_mod.CardTotal(1, "GPU-b", 12288 * MIB),
    ]
    facts = nvidia_mod.host_facts()
    assert facts is not None
    assert [g["name"] for g in facts["gpus"]] == ["Acme, Inc. Card", "NVIDIA GeForce RTX 3060"]


@pytest.mark.parametrize("kind", ["cards", "compute", "cuda"])
def test_a_fast_failure_is_not_run_again_inside_the_short_cache(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    smi = _Smi(monkeypatch, **{kind: ""})
    nvidia_mod.host_facts()
    smi.now += nvidia_mod._CACHE_TTL_S - 0.1
    nvidia_mod.host_facts()
    assert smi.runs.count(kind) == 1


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
    assert reading == nvidia_mod.Reading("", True, None)


_TWO_CARDS = (
    "0, GPU-aaaa, 8192, 8000, NVIDIA GeForce GTX 1070 Ti\n"
    "1, GPU-bbbb, 8192, 7000, NVIDIA GeForce GTX 1070 Ti\n"
)
_TWO_STATS = "0, 37, 61, 3536, 8192\n1, 12, 61, 3352, 8192\n"
_DRIVER_ERROR = "Unable to determine the device handle for GPU 0000:01:00.0: Unknown Error\n"


def _exits(code: int | None, printed: str):
    def run(*_a: object, **_k: object) -> tuple[str, int | None]:
        return printed, code

    return run


class TestNonZeroExit:
    """nvidia-smi can print every card row and still exit non-zero."""

    def test_the_cards_it_printed_are_kept(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(nvidia_mod, "run_tool", _exits(255, _TWO_CARDS))
        found = nvidia_mod.cards()
        assert [(c.index, c.uuid, c.total_bytes) for c in found] == [
            (0, "GPU-aaaa", 8192 * MIB),
            (1, "GPU-bbbb", 8192 * MIB),
        ]
        assert nvidia_mod.card_rows() == 2
        assert [t.index for t in nvidia_mod.card_totals()] == [0, 1]

    def test_the_stats_probe_keeps_its_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(nvidia_mod, "run_tool", _exits(255, _TWO_STATS))
        assert nvidia_mod.card_count() == 2
        assert sorted(nvidia_mod._cache.stats()) == [0, 1]

    def test_a_new_stats_cache_keeps_the_rows_of_a_failed_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(nvidia_mod, "run_tool", _exits(255, _TWO_STATS))
        fresh = SmiCache()
        assert fresh.cards() == 2
        assert sorted(fresh.stats()) == [0, 1]

    def test_one_warning_names_the_exit_code(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(nvidia_mod, "run_tool", _exits(255, _TWO_CARDS))
        with caplog.at_level("WARNING", logger=nvidia_mod.__name__):
            nvidia_mod.cards()
            nvidia_mod.card_rows()
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert "exit code 255" in warnings[0].getMessage()

    def test_a_repeated_exit_code_warns_once_and_a_new_one_warns_again(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(nvidia_mod, "_CACHE_TTL_S", 0.0)
        probe = nvidia_mod._Probe(nvidia_mod._CARD_ARGS, 0.0, len(nvidia_mod._CARD_COLUMNS))
        with caplog.at_level("WARNING", logger=nvidia_mod.__name__):
            for code in (255, 255, 10):
                monkeypatch.setattr(nvidia_mod, "run_tool", _exits(code, _TWO_CARDS))
                probe.read()
        assert [r.getMessage().split()[3] for r in caplog.records] == ["255", "10"]

    def test_an_exit_zero_warns_of_nothing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(nvidia_mod, "run_tool", _exits(0, _TWO_CARDS))
        with caplog.at_level("WARNING", logger=nvidia_mod.__name__):
            assert len(nvidia_mod.cards()) == 2
        assert caplog.records == []

    def test_text_with_no_card_row_reports_no_cards(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(nvidia_mod, "run_tool", _exits(255, _DRIVER_ERROR))
        assert nvidia_mod.cards() == []
        assert nvidia_mod.card_rows() == 0
        assert nvidia_mod.card_count() == 0

    def test_a_timeout_reports_no_cards(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(nvidia_mod, "run_tool", _exits(None, _TWO_CARDS))
        assert nvidia_mod.cards() == []
        assert nvidia_mod.card_rows() == 0

    def test_a_missing_binary_reports_no_cards(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(nvidia_mod, "run_tool", _exits(None, ""))
        assert nvidia_mod.cards() == []
        assert nvidia_mod.host_facts() is None

    def test_host_facts_still_say_the_driver_state_is_unknown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(nvidia_mod, "run_tool", _exits(255, _TWO_CARDS))
        assert nvidia_mod.host_facts() == {"cuda_driver": None, "gpus": None}

    def test_a_failed_answer_is_read_again(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reads: list[int] = []

        def run(*_a: object, **_k: object) -> tuple[str, int | None]:
            reads.append(1)
            return _TWO_CARDS, 255

        monkeypatch.setattr(nvidia_mod, "run_tool", run)
        monkeypatch.setattr(nvidia_mod, "_CACHE_TTL_S", 0.0)
        probe = nvidia_mod._Probe(nvidia_mod._CARD_ARGS, math.inf, nvidia_mod._FIELDS)
        probe.read()
        probe.read()
        assert len(reads) == 2

    def test_a_gap_in_the_printed_indices_pins_no_child(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rows 0 and 2 mean card 1 is lost; position numbering would pin a child to it."""
        gap = _TWO_CARDS.replace("1, GPU-bbbb", "2, GPU-bbbb")
        monkeypatch.setattr(nvidia_mod, "run_tool", _exits(255, gap))
        assert nvidia_mod.cards() == []
        assert nvidia_mod.card_rows() == 0
        assert devices.visible_nvidia_cards() == []
        assert devices.visible_env(tuple(devices.visible_nvidia_cards())) == {}

    def test_a_gap_in_the_stats_rows_counts_no_card(self, monkeypatch: pytest.MonkeyPatch) -> None:
        gap = _TWO_STATS.replace("1, 12", "2, 12")
        monkeypatch.setattr(nvidia_mod, "run_tool", _exits(255, gap))
        assert nvidia_mod.card_count() == 0
        assert nvidia_mod._cache.stats() == {}

    @pytest.mark.parametrize(
        "printed",
        [
            _DRIVER_ERROR + "1, GPU-bbbb, 8192, 7000, NVIDIA GeForce GTX 1070 Ti\n"
            "2, GPU-cccc, 8192, 7000, NVIDIA GeForce GTX 1070 Ti\n",
            _DRIVER_ERROR + _TWO_CARDS,
            _TWO_CARDS + _DRIVER_ERROR,
        ],
        ids=["error-then-rows-1-2", "error-then-rows-0-1", "rows-0-1-then-error"],
    )
    def test_an_unparsed_line_in_a_failed_run_keeps_no_row(
        self, monkeypatch: pytest.MonkeyPatch, printed: str
    ) -> None:
        """An error line takes a position and hides a gap, so the run is not trusted."""
        monkeypatch.setattr(nvidia_mod, "run_tool", _exits(255, printed))
        assert nvidia_mod.cards() == []
        assert nvidia_mod.card_rows() == 0
        assert devices.visible_nvidia_cards() == []

    @pytest.mark.parametrize(
        "printed",
        [
            _DRIVER_ERROR + "1, 12, 61, 3352, 8192\n2, 40, 61, 3352, 8192\n",
            _DRIVER_ERROR + _TWO_STATS,
            _TWO_STATS + _DRIVER_ERROR,
        ],
        ids=["error-then-rows-1-2", "error-then-rows-0-1", "rows-0-1-then-error"],
    )
    def test_an_unparsed_line_in_failed_stats_counts_no_card(
        self, monkeypatch: pytest.MonkeyPatch, printed: str
    ) -> None:
        monkeypatch.setattr(nvidia_mod, "run_tool", _exits(255, printed))
        assert nvidia_mod.card_count() == 0
        assert nvidia_mod._cache.stats() == {}

    def test_a_gap_after_an_exit_zero_is_not_this_probes_rule(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gap = _TWO_CARDS.replace("1, GPU-bbbb", "2, GPU-bbbb")
        monkeypatch.setattr(nvidia_mod, "run_tool", _exits(0, gap))
        assert [c.index for c in nvidia_mod.cards()] == [0, 2]

    @pytest.mark.parametrize("code", [8, 10, 14, 255])
    def test_a_card_state_exit_keeps_the_printed_rows(
        self, monkeypatch: pytest.MonkeyPatch, code: int
    ) -> None:
        monkeypatch.setattr(nvidia_mod, "run_tool", _exits(code, _TWO_CARDS))
        assert [c.index for c in nvidia_mod.cards()] == [0, 1]

    @pytest.mark.parametrize("code", [2, 3, 4, 13, 15])
    def test_an_exit_that_says_the_answer_is_not_a_card_list_keeps_nothing(
        self, monkeypatch: pytest.MonkeyPatch, code: int
    ) -> None:
        """A bad argument, a refused operation or a card off the bus makes the rows unknown."""
        monkeypatch.setattr(nvidia_mod, "run_tool", _exits(code, _TWO_CARDS))
        assert nvidia_mod.cards() == []
        assert nvidia_mod.card_rows() == 0
        assert nvidia_mod.host_facts() == {"cuda_driver": None, "gpus": None}

    @pytest.mark.parametrize("code", [6, 9, 12])
    def test_a_no_driver_exit_keeps_nothing_and_agrees_with_the_host_facts(
        self, monkeypatch: pytest.MonkeyPatch, code: int
    ) -> None:
        monkeypatch.setattr(nvidia_mod, "run_tool", _exits(code, _TWO_CARDS))
        assert (nvidia_mod.cards(), nvidia_mod.card_rows()) == ([], 0)
        assert nvidia_mod.host_facts() is None
        monkeypatch.setattr(nvidia_mod, "run_tool", _exits(code, _TWO_STATS))
        assert nvidia_mod.card_count() == 0
