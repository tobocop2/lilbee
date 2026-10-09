"""Every limit of thresholds.toml changes what is reported when its value changes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from tools.qa.crawl_parity import parity
from tools.qa.crawl_parity.leftover import LeftBehind, ProcessLeft
from tools.qa.crawl_parity.model import Kind, Layer, Mode, Side
from tools.qa.crawl_parity.pipeline import left_behind_differences
from tools.qa.crawl_parity.sides import CrawlResult
from tools.qa.crawl_parity.tests._support import STRICT, pages, truth
from tools.qa.crawl_parity.thresholds import LeftBehindLimits, ParityLimits, TruthLimits

TRUTHS = {
    "/a": truth("/a", [("p", True, "alpha beta gamma delta")]),
    "/b": truth("/b", [("p", True, "epsilon zeta")]),
    "/gone": truth("/gone", [("p", True, "not found")], status=404),
}


def kinds(
    oracle: Mapping[str, str | None], candidate: Mapping[str, str | None], limits: ParityLimits
) -> list[Kind]:
    comparison = parity.compare(
        pages(**oracle), pages(**candidate), TRUTHS, Layer.CRAWLER, Mode.HTTP, limits
    )
    return [difference.kind for difference in comparison.differences]


def test_pages_lost_max() -> None:
    oracle = {"a": "alpha beta gamma delta", "b": "epsilon zeta"}
    assert kinds(oracle, {"a": "alpha beta gamma delta"}, STRICT) == [Kind.PAGE_LOST]
    assert kinds(oracle, {"a": "alpha beta gamma delta"}, replace(STRICT, pages_lost_max=1)) == []
    assert kinds(oracle, {}, replace(STRICT, pages_lost_max=1)) == [Kind.PAGE_LOST, Kind.PAGE_LOST]


def test_pages_extra_unreadable_max() -> None:
    candidate = {"gone": "not found"}
    assert kinds({}, candidate, STRICT) == [Kind.PAGE_EXTRA]
    assert kinds({}, candidate, replace(STRICT, pages_extra_unreadable_max=1)) == []


def test_words_added_per_page_max() -> None:
    oracle, candidate = {"a": "alpha"}, {"a": "alpha stray"}
    assert kinds(oracle, candidate, STRICT) == [Kind.TEXT_ADDED]
    assert kinds(oracle, candidate, replace(STRICT, words_added_per_page_max=1)) == []


def test_symbols_changed_per_page_max() -> None:
    oracle, candidate = {"a": "alpha beta gamma delta"}, {"a": "alpha beta gamma delta [ ]"}
    assert kinds(oracle, candidate, STRICT) == [Kind.TEXT_ADDED]
    assert kinds(oracle, candidate, replace(STRICT, symbols_changed_per_page_max=2)) == []


def test_structure_changed_per_page_max() -> None:
    oracle, candidate = {"a": "# alpha\n\nbeta gamma delta"}, {"a": "alpha\n\nbeta gamma delta"}
    assert kinds(oracle, candidate, STRICT) == [Kind.STRUCTURE]
    assert kinds(oracle, candidate, replace(STRICT, structure_changed_per_page_max=1)) == []


def test_truth_limits() -> None:
    strict = TruthLimits(visible_recall_min=1.0, unexplained_words_per_page_max=0)
    loose = TruthLimits(visible_recall_min=0.5, unexplained_words_per_page_max=1)
    found = pages(a="alpha beta gamma stray", b="epsilon zeta")
    truths = {path: record for path, record in TRUTHS.items() if path != "/gone"}

    def against(limits: TruthLimits) -> list[Kind]:
        differences = parity.against_truth(found, truths, Layer.CRAWLER, Mode.HTTP, limits)
        return [difference.kind for difference in differences]

    assert against(strict) == [Kind.VISIBLE_MISSED, Kind.TEXT_ADDED]
    assert against(loose) == []


def _crawl(processes: int, temp_entries: int, threads: int) -> CrawlResult:
    left = LeftBehind(
        tuple(ProcessLeft(index, "chrome", ()) for index in range(processes)),
        tuple(f"profile-{index}" for index in range(temp_entries)),
    )
    return CrawlResult(
        Side.CANDIDATE,
        Layer.CRAWLER,
        Mode.BROWSER,
        {},
        0,
        1.0,
        1.0,
        0.1,
        {},
        threads,
        left,
        0.1,
        "",
    )


def test_left_behind_limits() -> None:
    result = _crawl(processes=1, temp_entries=1, threads=1)
    strict = LeftBehindLimits(processes_max=0, temp_entries_max=0, threads_max=0)
    assert len(left_behind_differences(result, None, strict)) == 3
    for field in ("processes_max", "temp_entries_max", "threads_max"):
        assert len(left_behind_differences(result, None, replace(strict, **{field: 1}))) == 2
    assert left_behind_differences(_crawl(0, 0, 0), None, strict) == []


def test_every_parity_truth_and_left_behind_limit_is_named_by_a_test() -> None:
    """A limit added to these three groups without a test that names it fails this census."""
    tests = Path(__file__).read_text(encoding="utf-8")
    tests += (Path(__file__).parent / "test_parity.py").read_text(encoding="utf-8")
    limits = [
        *ParityLimits.__dataclass_fields__,
        *TruthLimits.__dataclass_fields__,
        *LeftBehindLimits.__dataclass_fields__,
    ]
    assert len(limits) == 12
    assert [name for name in limits if name not in tests] == []
