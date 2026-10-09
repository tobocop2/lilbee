"""A whole run with stand-in crawlers: real replay, real driver processes, real verdict."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from tools.qa.crawl_parity import cli, leftover, pipeline, report
from tools.qa.crawl_parity.corpus import Corpus, Record, Response, Seed, load_synthetic
from tools.qa.crawl_parity.leftover import LeftBehind, ProcessLeft
from tools.qa.crawl_parity.model import Kind, Layer, Mode, Side, Status
from tools.qa.crawl_parity.pipeline import Plan, Yardstick, left_behind_differences, temp_pattern
from tools.qa.crawl_parity.sides import CrawlResult, SideConfig
from tools.qa.crawl_parity.tests._support import truth
from tools.qa.crawl_parity.thresholds import LeftBehindLimits, load_thresholds
from tools.qa.crawl_parity.truth import BrowserName, Target, Truth, TruthCache
from tools.qa.crawl_parity.verdict import Expected

DRIVERS = Path(__file__).parent / "drivers"
HTML = (("Content-Type", "text/html; charset=utf-8"),)
FAST = {Yardstick.PARITY, Yardstick.LEFT_BEHIND}


def _page(body: str) -> Response:
    return Response(200, HTML, f"<html><body>{body}</body></html>".encode())


CORPUS = Corpus(
    "tiny",
    {
        "/t/": Record("/t/", _page('<a href="/t/one">one</a> <a href="/t/two">two</a>')),
        "/t/one": Record("/t/one", _page("<h1>First</h1><p>alpha beta text delta</p>")),
        "/t/two": Record("/t/two", _page("<h1>Second</h1><p>epsilon zeta</p>")),
    },
    (Seed("f", "/t/", (Mode.HTTP,)),),
)


def side(which: Side, driver: str) -> SideConfig:
    return SideConfig(
        which,
        f"stand-in {driver}",
        {Layer.CRAWLER: Path(sys.executable)},
        str(DRIVERS / driver),
        "",
    )


def fake_capture(
    corpus: Corpus,
) -> Callable[[dict[str, Target], BrowserName, TruthCache], dict[str, Truth]]:
    """Ground truth without a browser: every word of a page's body is visible text."""

    def capture(
        targets: dict[str, Target], _browser: BrowserName, _cache: TruthCache
    ) -> dict[str, Truth]:
        truths: dict[str, Truth] = {}
        for path in targets:
            body = corpus.records[path].response.body.decode(errors="replace")
            text = " ".join(body.replace("<", " <").replace(">", "> ").split())
            words = " ".join(
                part for part in text.split() if not part.startswith(("<", "href", "/"))
            )
            truths[path] = truth(path, [("p", True, words)])
        return truths

    return capture


def plan(tmp_path: Path, candidate: str, corpus: Corpus = CORPUS, **changes: object) -> Plan:
    base = Plan(
        corpus=corpus,
        sides={
            Side.ORACLE: side(Side.ORACLE, "fake_good.py"),
            Side.CANDIDATE: side(Side.CANDIDATE, candidate),
        },
        thresholds=load_thresholds(),
        expected=[],
        work=tmp_path / "work",
        modes=(Mode.HTTP,),
        yardsticks=frozenset(FAST),
        seeds=("f",),
    )
    return replace(base, **changes)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def no_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "capture", fake_capture(CORPUS))
    monkeypatch.setattr(leftover, "GRACE_SECONDS", 0.1)


def test_two_equal_sides_pass_with_no_finding(tmp_path: Path) -> None:
    outcome = pipeline.run(plan(tmp_path, "fake_good.py"))
    assert outcome.verdict is not None and outcome.verdict.passed
    assert outcome.differences == []
    assert len(outcome.pages[(Side.CANDIDATE, Layer.CRAWLER, Mode.HTTP)]) == 3
    assert [c.oracle_saved for c in outcome.comparisons] == [3]
    assert sum(outcome.unrecorded.values()) == 0


def test_a_lossy_candidate_fails_with_each_defect_named(tmp_path: Path) -> None:
    outcome = pipeline.run(plan(tmp_path, "fake_lossy.py"))
    assert outcome.verdict is not None and not outcome.verdict.passed
    found = {(d.kind, d.page, d.feature, d.counts) for d in outcome.differences}
    assert found == {
        (Kind.TEXT_LOST, "/t/one", "p", True),
        (Kind.PAGE_LOST, "/t/two", "fake-failure-n", True),
    }
    assert {f.status for f in outcome.verdict.findings} == {Status.NEW}


def test_known_differences_still_fail_and_accepted_ones_pass(tmp_path: Path) -> None:
    known = [Expected("text-lost/http/crawler/p", "x#1"), Expected("page-lost/*", "x#2")]
    failing = pipeline.run(plan(tmp_path, "fake_lossy.py", expected=known))
    assert failing.verdict is not None and not failing.verdict.passed
    assert {f.status for f in failing.verdict.findings} == {Status.KNOWN}
    accepted = [replace(entry, accepted=True) for entry in known]
    passing = pipeline.run(plan(tmp_path, "fake_lossy.py", expected=accepted, work=tmp_path / "w2"))
    assert passing.verdict is not None and passing.verdict.passed


def test_an_expected_entry_is_fixed_after_a_whole_run_that_does_not_show_it(tmp_path: Path) -> None:
    entry = Expected("text-lost/http/*/p", "x#1")
    outcome = pipeline.run(plan(tmp_path, "fake_good.py", expected=[entry]))
    assert outcome.verdict is not None
    assert outcome.verdict.fixed == (entry,) and outcome.verdict.not_run == ()


def test_retrieval_without_a_lilbee_to_index_with_fails_as_not_measured(tmp_path: Path) -> None:
    yardsticks = frozenset({*FAST, Yardstick.RETRIEVAL})
    outcome = pipeline.run(plan(tmp_path, "fake_good.py", yardsticks=yardsticks))
    assert outcome.verdict is not None and not outcome.verdict.passed
    assert outcome.verdict.not_measured == ("retrieval: no [retrieval] section in the sides file",)


def test_speed_is_measured_with_its_load_and_refused_above_the_limit(tmp_path: Path) -> None:
    limits = load_thresholds()
    calm = replace(limits, speed=replace(limits.speed, repeats=1, load_max=1000.0))
    measured = pipeline.run(
        plan(tmp_path, "fake_good.py", yardsticks=frozenset({Yardstick.SPEED}), thresholds=calm)
    )
    (speed,) = measured.speeds
    assert speed.comparison is not None and speed.comparison.compared
    assert all(
        "(load " in sample.text() for samples in speed.samples.values() for sample in samples
    )
    busy = replace(limits, speed=replace(limits.speed, repeats=1, load_max=-1.0))
    refused = pipeline.run(
        plan(
            tmp_path,
            "fake_good.py",
            yardsticks=frozenset({Yardstick.SPEED}),
            thresholds=busy,
            work=tmp_path / "w2",
        )
    )
    assert refused.verdict is not None and not refused.verdict.passed
    assert "above the limit" in refused.verdict.not_measured[0]


def test_with_no_oracle_the_candidate_is_judged_against_the_page(tmp_path: Path) -> None:
    alone = {Side.CANDIDATE: side(Side.CANDIDATE, "fake_lossy.py")}
    outcome = pipeline.run(plan(tmp_path, "fake_lossy.py", sides=alone))
    kinds = {(d.kind, d.page) for d in outcome.differences}
    assert (Kind.PAGE_LOST, "/t/two") in kinds
    assert (Kind.VISIBLE_MISSED, "/t/one") in kinds
    assert outcome.comparisons == []


def test_a_run_of_part_of_the_corpus_calls_no_entry_fixed(tmp_path: Path) -> None:
    two_seeds = replace(CORPUS, seeds=(*CORPUS.seeds, Seed("other", "/t/one", (Mode.HTTP,))))
    entry = Expected("text-lost/http/*/p", "x#1")
    outcome = pipeline.run(plan(tmp_path, "fake_good.py", corpus=two_seeds, expected=[entry]))
    assert outcome.verdict is not None
    assert outcome.verdict.fixed == () and outcome.verdict.not_run == (entry,)


def test_the_report_names_the_verdict_the_normalisations_and_each_finding(tmp_path: Path) -> None:
    lossy = plan(tmp_path, "fake_lossy.py")
    outcome = pipeline.run(lossy)
    path = report.write(lossy, outcome, tmp_path / "report")
    text = path.read_text(encoding="utf-8")
    assert "**FAIL: 2 NEW, 0 KNOWN, 0 ACCEPTED, 0 NOT-COUNTED, 0 FIXED, 0 NOT-RUN**" in text
    assert "`page-lost/http/crawler/fake-failure-n`" in text
    assert "markdown-syntax" in text and "unicode-nfc" in text and "visible-case" in text
    lines = (tmp_path / "report" / report.DIFFERENCES_FILE).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    pages_table = (tmp_path / "report" / report.PAGES_FILE).read_text(encoding="utf-8").splitlines()
    assert len(pages_table) == 1 + 2


def test_the_command_exits_one_on_fail_and_zero_on_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(pipeline, "capture", fake_capture(load_synthetic()))
    python = Path(sys.executable).as_posix()

    def sides_file(name: str, candidate: str) -> Path:
        path = tmp_path / name
        path.write_text(
            f'[oracle]\ncrawler = "{python}"\n'
            f'crawler_driver = "{(DRIVERS / "fake_good.py").as_posix()}"\n'
            f'[candidate]\ncrawler = "{python}"\n'
            f'crawler_driver = "{(DRIVERS / candidate).as_posix()}"\n',
            encoding="utf-8",
        )
        return path

    def run(name: str, candidate: str) -> int:
        return cli.main(
            [
                "run",
                "--sides",
                str(sides_file(f"{name}.toml", candidate)),
                "--work",
                str(tmp_path / name),
                "--report",
                str(tmp_path / name / "report"),
                "--modes",
                "http",
                "--seeds",
                "w",
                "--skip",
                "speed,retrieval",
                "--expected",
                str(tmp_path / "none.toml"),
            ]
        )

    (tmp_path / "none.toml").write_text("", encoding="utf-8")
    assert run("same", "fake_good.py") == cli.EXIT_PASS
    assert capsys.readouterr().out.startswith("PASS: 0 NEW")
    assert run("lossy", "fake_lossy.py") == cli.EXIT_FAIL
    assert capsys.readouterr().out.startswith("FAIL: ")


def _result(**left: object) -> CrawlResult:
    return CrawlResult(
        side=Side.CANDIDATE,
        layer=Layer.CRAWLER,
        mode=Mode.BROWSER,
        pages={},
        return_code=0,
        wall_seconds=1.0,
        crawl_seconds=1.0,
        first_page_seconds=0.1,
        versions={},
        threads_started=left.pop("threads", 0),  # type: ignore[arg-type]
        left=LeftBehind(**left),  # type: ignore[arg-type]
        load_before=0.1,
        stderr_tail="",
    )


LIMITS = LeftBehindLimits(processes_max=0, temp_entries_max=0, threads_max=0)


def test_a_clean_run_has_no_left_behind_difference() -> None:
    assert left_behind_differences(_result(), None, LIMITS) == []


def test_each_thing_left_is_one_named_difference() -> None:
    chrome = (ProcessLeft(1, "chrome", (9222,)), ProcessLeft(2, "chrome", ()))
    result = _result(processes=chrome, temp_entries=("crawlberg-chrome-eLqtoA",), threads=4)
    found = {d.signature: d.amount for d in left_behind_differences(result, None, LIMITS)}
    assert found == {
        "left-behind/browser/crawler/process:chrome": 2,
        "left-behind/browser/crawler/listening-socket": 1,
        "left-behind/browser/crawler/temp:crawlberg-chrome-*": 1,
        "left-behind/browser/crawler/threads": 4,
    }


def test_threads_count_only_beyond_the_oracles() -> None:
    assert left_behind_differences(_result(threads=4), _result(threads=4), LIMITS) == []
    (difference,) = left_behind_differences(_result(threads=6), _result(threads=4), LIMITS)
    assert difference.amount == 2


def test_temp_pattern_replaces_a_random_ending_only() -> None:
    assert temp_pattern("crawlberg-chrome-eLqtoA") == "crawlberg-chrome-*"
    assert temp_pattern(".tmpAb12Cd") == ".*"
    assert temp_pattern("cache") == "cache"


def test_the_verdict_layer_is_the_highest_both_sides_have(tmp_path: Path) -> None:
    both = plan(tmp_path, "fake_good.py")
    assert both.top_layer() is Layer.CRAWLER
    with_lilbee = replace(
        both.sides[Side.CANDIDATE],
        interpreters={Layer.CRAWLER: Path(sys.executable), Layer.LILBEE: Path(sys.executable)},
    )
    mixed = replace(both, sides={Side.ORACLE: both.sides[Side.ORACLE], Side.CANDIDATE: with_lilbee})
    assert mixed.top_layer() is Layer.CRAWLER
    assert mixed.layers(Side.CANDIDATE) == [Layer.CRAWLER, Layer.LILBEE]
