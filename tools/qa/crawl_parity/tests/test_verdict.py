"""The verdict: KNOWN still fails, ACCEPTED does not, FIXED needs a run that could have shown it."""

from __future__ import annotations

from pathlib import Path

from tools.qa.crawl_parity.model import Difference, Kind, Layer, Mode, Status
from tools.qa.crawl_parity.verdict import EXPECTED_FILE, Expected, judge, load_expected

LOST = Difference(Kind.TEXT_LOST, Mode.HTTP, Layer.CONVERTER, "svg>text", "/a", 2)
LOST_AGAIN = Difference(Kind.TEXT_LOST, Mode.HTTP, Layer.CONVERTER, "svg>text", "/b", 3)
HIDDEN = Difference(Kind.TEXT_LOST, Mode.HTTP, Layer.CONVERTER, "hidden:p", "/a", 1, counts=False)
RAN = {(Kind.TEXT_LOST, Mode.HTTP)}


def test_no_difference_passes() -> None:
    verdict = judge([], [], RAN)
    assert verdict.passed
    assert verdict.findings == ()


def test_an_unknown_difference_is_new_and_fails() -> None:
    verdict = judge([LOST], [], RAN)
    assert [(f.signature, f.status) for f in verdict.findings] == [
        ("text-lost/http/converter/svg>text", Status.NEW)
    ]
    assert not verdict.passed


def test_differences_with_one_signature_are_one_finding_over_their_pages() -> None:
    (finding,) = judge([LOST, LOST_AGAIN], [], RAN).findings
    assert finding.pages == 2
    assert len(finding.differences) == 2


def test_a_filed_difference_is_known_carries_its_issue_and_still_fails() -> None:
    verdict = judge([LOST], [Expected("text-lost/*/converter/svg*", "h2m#750")], RAN)
    assert [(f.status, f.issue) for f in verdict.findings] == [(Status.KNOWN, "h2m#750")]
    assert not verdict.passed
    assert verdict.fixed == ()


def test_an_accepted_difference_passes() -> None:
    verdict = judge([LOST], [Expected("text-lost/*/converter/svg*", "h2m#750", accepted=True)], RAN)
    assert [f.status for f in verdict.findings] == [Status.ACCEPTED]
    assert verdict.passed


def test_the_first_matching_entry_wins() -> None:
    entries = [Expected("text-lost/*", "first#1"), Expected("text-lost/http/*", "second#2")]
    verdict = judge([LOST], entries, RAN)
    assert verdict.findings[0].issue == "first#1"
    assert [entry.issue for entry in verdict.fixed] == ["second#2"]


def test_a_difference_that_does_not_count_never_fails_and_keeps_its_issue() -> None:
    verdict = judge([HIDDEN], [Expected("text-lost/*/*/hidden:*", "h2m#753")], RAN)
    assert [(f.status, f.issue) for f in verdict.findings] == [(Status.NOT_COUNTED, "h2m#753")]
    assert verdict.passed
    assert judge([HIDDEN], [], RAN).passed


def test_an_entry_nothing_matches_is_fixed_only_when_its_kind_and_mode_ran() -> None:
    http = Expected("text-lost/http/*/gone", "x#1")
    browser = Expected("text-lost/browser/*/gone", "x#2")
    speed = Expected("slower/*/*/pages-per-second", "x#3")
    verdict = judge([], [http, browser, speed], RAN)
    assert verdict.fixed == (http,)
    assert verdict.not_run == (browser, speed)
    assert verdict.passed


def test_a_yardstick_that_gave_no_measurement_fails_the_run() -> None:
    verdict = judge(
        [], [], RAN, not_measured=("speed, http mode: load 2.10 is above the limit 1.00",)
    )
    assert not verdict.passed


def test_expected_file_loads_and_every_entry_names_an_issue(tmp_path: Path) -> None:
    path = tmp_path / "expected.toml"
    path.write_text(
        '[[known]]\nsignature = "page-lost/*/*/x"\nissue = "a#1"\n\n'
        '[[known]]\nsignature = "slower/*/*/*"\nissue = "a#2"\naccepted = true\nnote = "n"\n',
        encoding="utf-8",
    )
    assert load_expected(path) == [
        Expected("page-lost/*/*/x", "a#1"),
        Expected("slower/*/*/*", "a#2", "n", True),
    ]
    shipped = load_expected(EXPECTED_FILE)
    assert len(shipped) > 20
    assert all(entry.issue and entry.signature.count("/") == 3 for entry in shipped)
    assert not any(entry.accepted for entry in shipped)
