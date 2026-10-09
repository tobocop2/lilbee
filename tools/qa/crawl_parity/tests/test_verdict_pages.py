"""An expected entry can name the pages it covers; a signature is known only if every page is."""

from __future__ import annotations

from pathlib import Path

from tools.qa.crawl_parity.model import Difference, Kind, Layer, Mode, Status
from tools.qa.crawl_parity.verdict import Expected, judge, load_expected

RAN = {(Kind.TEXT_LOST, Mode.HTTP)}


def lost(page: str, counts: bool = True) -> Difference:
    return Difference(Kind.TEXT_LOST, Mode.HTTP, Layer.CRAWLER, "p", page, counts=counts)


def test_an_entry_for_one_page_does_not_cover_the_same_signature_on_another() -> None:
    deflate = Expected("text-lost/http/crawler/*", "cb#614", page="/n/deflate")
    known = judge([lost("/n/deflate")], [deflate], RAN)
    assert [(f.status, f.issue) for f in known.findings] == [(Status.KNOWN, "cb#614")]
    mixed = judge([lost("/n/deflate"), lost("/n/other")], [deflate], RAN)
    assert [(f.status, f.issue) for f in mixed.findings] == [(Status.NEW, "cb#614")]
    assert mixed.fixed == ()


def test_a_page_glob_and_two_issues_for_one_signature() -> None:
    entries = [
        Expected("text-lost/*", "cb#604", page="/n/latin1-*"),
        Expected("text-lost/*", "cb#614", page="/n/deflate"),
    ]
    verdict = judge([lost("/n/latin1-none"), lost("/n/deflate")], entries, RAN)
    assert [(f.status, f.issue) for f in verdict.findings] == [(Status.KNOWN, "cb#604, cb#614")]


def test_a_signature_is_accepted_only_when_every_counting_page_is() -> None:
    accepted = Expected("text-lost/*", "a#1", accepted=True, page="/a")
    filed = Expected("text-lost/*", "b#2", page="/b")
    assert judge([lost("/a")], [accepted, filed], RAN).findings[0].status is Status.ACCEPTED
    both = judge([lost("/a"), lost("/b")], [accepted, filed], RAN)
    assert both.findings[0].status is Status.KNOWN and not both.passed
    assert both.fixed == ()


def test_a_difference_that_does_not_count_needs_no_entry_beside_one_that_does() -> None:
    filed = Expected("text-lost/*", "b#2", page="/b")
    verdict = judge([lost("/a", counts=False), lost("/b")], [filed], RAN)
    assert verdict.findings[0].status is Status.KNOWN


def test_the_page_of_an_entry_is_read_from_the_file(tmp_path: Path) -> None:
    path = tmp_path / "expected.toml"
    path.write_text(
        '[[known]]\nsignature = "text-lost/*"\nissue = "x#1"\npage = "/n/*"\n', encoding="utf-8"
    )
    (entry,) = load_expected(path)
    assert entry.page == "/n/*"
    assert entry.matches(lost("/n/deflate")) and not entry.matches(lost("/f/x"))
