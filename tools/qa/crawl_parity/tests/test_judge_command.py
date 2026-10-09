"""The judge command: a saved run set against a changed expected.toml, with no new crawl."""

from __future__ import annotations

from pathlib import Path

import pytest
from tools.qa.crawl_parity import cli, leftover, pipeline, report
from tools.qa.crawl_parity.model import Status
from tools.qa.crawl_parity.tests.test_pipeline import CORPUS, fake_capture, plan
from tools.qa.crawl_parity.verdict import Expected


@pytest.fixture
def saved_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The report directory of a run whose candidate drops a word and loses a page."""
    monkeypatch.setattr(pipeline, "capture", fake_capture(CORPUS))
    monkeypatch.setattr(leftover, "GRACE_SECONDS", 0.1)
    lossy = plan(tmp_path, "fake_lossy.py")
    report.write(lossy, pipeline.run(lossy), tmp_path / "report")
    return tmp_path / "report"


def test_a_saved_run_gives_its_differences_back_unchanged(saved_run: Path) -> None:
    differences = report.read_differences(saved_run)
    assert sorted(d.signature for d in differences) == [
        "page-lost/http/crawler/fake-failure-n",
        "text-lost/http/crawler/p",
    ]
    assert all(d.counts and d.page.startswith("/t/") for d in differences)


def test_a_saved_run_is_judged_again_under_new_entries(saved_run: Path) -> None:
    assert {f.status for f in report.rejudge(saved_run, []).findings} == {Status.NEW}
    entries = [Expected("text-lost/*", "x#1"), Expected("page-lost/*", "x#2", accepted=True)]
    verdict = report.rejudge(saved_run, entries)
    assert sorted(f.status for f in verdict.findings) == [Status.ACCEPTED, Status.KNOWN]
    written = (saved_run / report.VERDICT_FILE).read_text(encoding="utf-8")
    assert "**FAIL: 0 NEW, 1 KNOWN, 1 ACCEPTED" in written and "x#1" in written


def test_an_entry_is_fixed_on_a_saved_run_only_for_what_that_run_measured(saved_run: Path) -> None:
    gone = Expected("text-lost/http/*/no-such-place", "x#3")
    other_mode = Expected("text-lost/browser/*/p", "x#4")
    verdict = report.rejudge(saved_run, [gone, other_mode])
    assert verdict.fixed == (gone,) and verdict.not_run == (other_mode,)


def test_the_judge_command_exits_by_the_new_verdict(
    saved_run: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    expected = tmp_path / "expected.toml"
    expected.write_text("", encoding="utf-8")
    command = ["judge", "--report", str(saved_run), "--expected", str(expected)]
    assert cli.main(command) == cli.EXIT_FAIL
    assert capsys.readouterr().out.startswith("FAIL: 2 NEW")
    expected.write_text(
        '[[known]]\nsignature = "*"\nissue = "x#1"\naccepted = true\n', encoding="utf-8"
    )
    assert cli.main(command) == cli.EXIT_PASS
    assert capsys.readouterr().out.startswith("PASS: 0 NEW, 0 KNOWN, 2 ACCEPTED")
