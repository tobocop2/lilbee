"""Yardsticks A and B: what is reported, what counts, and which layer a difference belongs to."""

from __future__ import annotations

from dataclasses import replace

from tools.qa.crawl_parity import parity
from tools.qa.crawl_parity.model import Difference, Kind, Layer, Mode
from tools.qa.crawl_parity.tests._support import STRICT, pages, truth
from tools.qa.crawl_parity.thresholds import ParityLimits, TruthLimits
from tools.qa.crawl_parity.tokens import markdown_tokens
from tools.qa.crawl_parity.truth import Truth

VISIBLE = truth(
    "/p",
    [
        ("p", True, "alpha beta gamma delta"),
        ("svg>text", True, "chart label"),
        ("div>span", False, "secret hidden"),
    ],
    attributes=["tooltip words"],
)
TRUTHS = {"/p": VISIBLE}
COUNT_INVISIBLE = replace(STRICT, invisible_text_lost_counts=True)
ONE_WORD = replace(STRICT, words_lost_per_page_max=1)


def compare(
    oracle: str | None,
    candidate: str | None,
    truths: dict[str, Truth] | None = None,
    limits: ParityLimits = STRICT,
) -> parity.Comparison:
    return parity.compare(
        pages(p=oracle),
        pages(p=candidate),
        TRUTHS if truths is None else truths,
        Layer.CRAWLER,
        Mode.HTTP,
        limits,
    )


def facts(comparison: parity.Comparison) -> list[tuple[Kind, str, float, bool]]:
    return [(d.kind, d.feature, d.amount, d.counts) for d in comparison.differences]


def test_identical_pages_give_no_difference_and_one_measure() -> None:
    comparison = compare("alpha beta gamma delta", "alpha beta gamma delta")
    assert comparison.differences == []
    assert [m.path for m in comparison.measures] == ["/p"]
    assert comparison.oracle_saved == comparison.candidate_saved == 1


def test_a_lost_visible_word_counts_and_names_its_place_in_the_page() -> None:
    comparison = compare("alpha beta chart label", "alpha beta")
    assert facts(comparison) == [(Kind.TEXT_LOST, "svg>text", 2.0, True)]
    assert comparison.differences[0].signature == "text-lost/http/crawler/svg>text"
    assert comparison.words_lost == 2


def test_a_lost_hidden_word_is_reported_and_does_not_count_by_default() -> None:
    assert facts(compare("alpha secret hidden", "alpha")) == [
        (Kind.TEXT_LOST, "hidden:div>span", 2.0, False)
    ]


def test_a_lost_hidden_word_counts_when_the_limit_says_so() -> None:
    comparison = compare("alpha secret hidden", "alpha", limits=COUNT_INVISIBLE)
    assert facts(comparison) == [(Kind.TEXT_LOST, "hidden:div>span", 2.0, True)]


def test_a_word_the_oracle_repeats_beyond_the_page_does_not_count() -> None:
    assert facts(compare("alpha alpha beta", "alpha beta")) == [
        (Kind.TEXT_LOST, "oracle-repeat:p", 1.0, False)
    ]


def test_a_word_the_candidate_repeats_beyond_the_page_counts() -> None:
    assert facts(compare("alpha beta", "alpha beta beta")) == [
        (Kind.TEXT_ADDED, "repeat:p", 1.0, True)
    ]


def test_visible_text_only_the_candidate_holds_does_not_count() -> None:
    assert facts(compare("alpha", "alpha beta")) == [(Kind.TEXT_ADDED, "p", 1.0, False)]


def test_added_words_the_page_does_not_hold_count_and_are_named() -> None:
    comparison = compare("alpha", "alpha SVG Image")
    assert facts(comparison) == [(Kind.TEXT_ADDED, "not-in-page:image+svg", 2.0, True)]


def test_added_attribute_and_hidden_text_counts() -> None:
    comparison = compare("alpha", "alpha tooltip secret")
    assert facts(comparison) == [
        (Kind.TEXT_ADDED, "attribute", 1.0, True),
        (Kind.TEXT_ADDED, "hidden:div>span", 1.0, True),
    ]


def test_without_truth_every_lost_and_added_word_counts() -> None:
    comparison = compare("alpha beta", "alpha gamma", truths={})
    assert facts(comparison) == [
        (Kind.TEXT_LOST, "no-truth", 1.0, True),
        (Kind.TEXT_ADDED, "no-truth", 1.0, True),
    ]


def test_a_failed_capture_is_no_truth_and_hides_no_loss() -> None:
    failed = {"/p": Truth("/p", 0, False, error="navigation failed")}
    assert facts(compare("alpha beta", "alpha", truths=failed)) == [
        (Kind.TEXT_LOST, "no-truth", 1.0, True)
    ]


def test_losses_within_the_per_page_limit_are_not_differences() -> None:
    assert compare("alpha beta", "alpha", limits=ONE_WORD).differences == []
    assert len(compare("alpha beta gamma", "alpha", limits=ONE_WORD).differences) == 1


def test_symbols_are_compared_apart_from_words() -> None:
    comparison = compare("alpha beta", "alpha beta [ ]")
    assert facts(comparison) == [(Kind.TEXT_ADDED, "symbols", 2.0, True)]
    assert comparison.words_added == 0


def test_a_flattened_table_is_a_structure_difference_with_no_word_lost() -> None:
    table = "| alpha | beta |\n|---|---|\n| gamma | delta |\n"
    comparison = compare(table, "alpha beta\n\ngamma delta\n")
    assert [(d.kind, d.feature) for d in comparison.differences] == [(Kind.STRUCTURE, "table")]


def test_a_page_only_the_oracle_saved_is_lost_with_the_candidate_reason() -> None:
    comparison = compare("alpha", None)
    assert facts(comparison) == [(Kind.PAGE_LOST, "boom-n", 1.0, True)]
    assert comparison.measures == []


def test_a_lost_page_a_reader_does_not_get_does_not_count() -> None:
    not_found = {"/p": truth("/p", [("p", True, "not found")], status=404)}
    assert facts(compare("alpha", None, truths=not_found)) == [
        (Kind.PAGE_LOST, "boom-n", 1.0, False)
    ]


def test_a_page_only_the_candidate_saved_counts_only_when_a_reader_does_not_get_it() -> None:
    assert facts(compare(None, "alpha")) == [(Kind.PAGE_EXTRA, "boom-n", 1.0, False)]
    not_found = {"/p": truth("/p", [("p", True, "not found")], status=404)}
    assert facts(compare(None, "alpha", truths=not_found)) == [
        (Kind.PAGE_EXTRA, "boom-n", 1.0, True)
    ]


def test_a_page_one_side_never_returned_has_a_reason() -> None:
    comparison = parity.compare(pages(p="alpha"), {}, TRUTHS, Layer.LILBEE, Mode.HTTP, STRICT)
    assert [d.feature for d in comparison.differences] == ["not-returned"]


def test_reason_feature_takes_out_addresses_and_numbers() -> None:
    reason = "browser_timeout: http://127.0.0.1:5123/b/alert timed out after 30s"
    assert parity.reason_feature(reason) == "browser-timeout-url-timed-out-after-ns"
    assert parity.reason_feature(None) == parity.NO_REASON
    assert parity.reason_feature("!!!") == parity.NO_REASON


def test_score_counts_visible_words_held_and_words_not_shown() -> None:
    result = parity.score(markdown_tokens("alpha beta secret tooltip stray"), VISIBLE)
    assert result.recall == 2 / 6
    assert dict(result.missed) == {"gamma": 1, "delta": 1, "chart": 1, "label": 1}
    assert (result.hidden_held, result.attribute_held) == (1, 1)
    assert dict(result.unexplained) == {"stray": 1}


def test_score_holds_a_visible_word_in_another_case() -> None:
    result = parity.score(markdown_tokens("ALPHA Beta gamma delta chart"), VISIBLE)
    assert dict(result.missed) == {"label": 1}
    assert not result.unexplained


def test_against_truth_reports_a_missing_page_low_recall_and_unexplained_words() -> None:
    limits = TruthLimits(visible_recall_min=0.9, unexplained_words_per_page_max=0)
    lost = parity.against_truth({}, TRUTHS, Layer.CRAWLER, Mode.HTTP, limits)
    assert [(d.kind, d.feature) for d in lost] == [(Kind.PAGE_LOST, "not-returned")]
    poor = parity.against_truth(pages(p="alpha stray"), TRUTHS, Layer.CRAWLER, Mode.HTTP, limits)
    assert {d.kind for d in poor} == {Kind.VISIBLE_MISSED, Kind.TEXT_ADDED}
    whole = "alpha beta gamma delta chart label"
    assert parity.against_truth(pages(p=whole), TRUTHS, Layer.CRAWLER, Mode.HTTP, limits) == []


def _difference(layer: Layer, feature: str, page: str = "/p") -> Difference:
    return Difference(Kind.TEXT_LOST, Mode.HTTP, layer, feature, page)


def test_a_difference_moves_to_the_lowest_layer_that_shows_it() -> None:
    by_layer = {
        Layer.CONVERTER: [_difference(Layer.CONVERTER, "svg>text")],
        Layer.CRAWLER: [_difference(Layer.CRAWLER, "svg>text"), _difference(Layer.CRAWLER, "p")],
        Layer.LILBEE: [
            _difference(Layer.LILBEE, "svg>text"),
            _difference(Layer.LILBEE, "p"),
            _difference(Layer.LILBEE, "li"),
            _difference(Layer.LILBEE, "svg>text", page="/other"),
        ],
    }
    attributed = parity.attribute_to_lowest_layer(by_layer, Layer.LILBEE)
    assert [(d.feature, d.page, d.layer) for d in attributed] == [
        ("svg>text", "/p", Layer.CONVERTER),
        ("p", "/p", Layer.CRAWLER),
        ("li", "/p", Layer.LILBEE),
        ("svg>text", "/other", Layer.LILBEE),
    ]
