"""The human-audit sample: blindness, stratification, and honest agreement.

The audit exists to make the judge's error rate a published number. These pin
the three ways that could quietly stop being true: the annotator seeing the
judge's score, the sample landing entirely on the judge's mode, and an
agreement figure reported off too few rows.
"""

import json

import pytest
from evals.benchmark.human_audit import (
    MIN_AUDIT_ROWS,
    AuditRow,
    InsufficientAuditError,
    agreement,
    read_audit_sheet,
    stratified_sample,
    write_audit_sheet,
)
from evals.retrieval.judging import DIMENSIONS, SCORE_MAX, SCORE_MIN

pytest.importorskip("sklearn")


def _row(gid):
    return AuditRow(gid=gid, question="q?", source="a.txt", ground="g", answer="a")


def _grades(spec):
    """``{gid: score}`` on every dimension, from a gid-to-score mapping."""
    return {gid: dict.fromkeys(DIMENSIONS, score) for gid, score in spec.items()}


def test_the_sheet_never_shows_the_annotator_the_judges_score():
    # An agreement number computed against a score the annotator was looking at
    # measures suggestibility, not agreement.
    payload = _row("g1").to_dict()
    assert "score" not in payload
    assert all(payload[dimension] is None for dimension in DIMENSIONS)
    assert set(payload) == {"gid", "question", "source", "ground", "answer", *DIMENSIONS}


def test_the_sample_spreads_across_scores_rather_than_the_judges_mode():
    # A judge that mostly says 5 would otherwise produce an audit of mostly 5s,
    # which measures agreement where it is easiest.
    grades = _grades({f"g{i}": 5 for i in range(90)} | {f"h{i}": 1 for i in range(10)})
    chosen = stratified_sample(grades, 10, dimension="faithfulness", seed=1)
    low = sum(1 for gid in chosen if gid.startswith("h"))
    # Round-robin over the two score buckets, so roughly half despite 9:1 skew.
    assert low == 5


def test_sampling_is_reproducible_under_a_seed():
    grades = _grades({f"g{i}": (i % 5) + 1 for i in range(60)})
    first = stratified_sample(grades, 20, dimension="relevance", seed=7)
    second = stratified_sample(grades, 20, dimension="relevance", seed=7)
    assert first == second


def test_sampling_a_thin_bucket_does_not_repeat_rows():
    grades = _grades({f"g{i}": 5 for i in range(30)} | {"h0": 1})
    chosen = stratified_sample(grades, 20, dimension="citation", seed=3)
    assert len(chosen) == len(set(chosen))


def test_an_unknown_dimension_is_refused():
    with pytest.raises(ValueError, match="unknown dimension"):
        stratified_sample(_grades({"g1": 3}), 1, dimension="helpfulness", seed=1)


def test_round_trip_through_the_sheet(tmp_path):
    path = tmp_path / "audit.jsonl"
    write_audit_sheet(path, [_row("g1"), _row("g2")])
    # An unfilled sheet yields nothing rather than defaulting to a score.
    assert read_audit_sheet(path) == {}
    filled = [
        {**json.loads(line), **dict.fromkeys(DIMENSIONS, 4)}
        for line in path.read_text().splitlines()
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in filled))
    assert read_audit_sheet(path) == {
        "g1": _grades({"g1": 4})["g1"],
        "g2": _grades({"g2": 4})["g2"],
    }


def test_a_partly_filled_sheet_reports_only_what_was_filled(tmp_path):
    # A half-finished audit should shrink the sample it covers, not silently
    # score the untouched rows.
    path = tmp_path / "audit.jsonl"
    path.write_text(
        json.dumps({"gid": "g1", **dict.fromkeys(DIMENSIONS, 3)})
        + "\n"
        + json.dumps({"gid": "g2", **dict.fromkeys(DIMENSIONS, None)})
        + "\n"
    )
    assert set(read_audit_sheet(path)) == {"g1"}


@pytest.mark.parametrize("bad", [SCORE_MAX + 1, SCORE_MIN - 1])
def test_a_score_outside_the_rubric_scale_is_refused(tmp_path, bad):
    path = tmp_path / "audit.jsonl"
    path.write_text(json.dumps({"gid": "g1", **dict.fromkeys(DIMENSIONS, bad)}) + "\n")
    with pytest.raises(ValueError, match="outside the rubric"):
        read_audit_sheet(path)


def test_a_non_integer_score_is_refused_rather_than_coerced(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_text(json.dumps({"gid": "g1", **dict.fromkeys(DIMENSIONS, 3.5)}) + "\n")
    with pytest.raises(ValueError, match="non-integer"):
        read_audit_sheet(path)


def test_agreement_refuses_to_report_on_too_few_rows():
    # The whole point of the audit is a number a reader can trust; a kappa on a
    # handful of rows is noise wearing a statistic's name.
    small = _grades({f"g{i}": 3 for i in range(MIN_AUDIT_ROWS - 1)})
    with pytest.raises(InsufficientAuditError, match="not reported below"):
        agreement(small, small)


def test_a_judge_that_matches_the_human_scores_perfect_agreement():
    grades = _grades({f"g{i}": (i % 5) + 1 for i in range(MIN_AUDIT_ROWS)})
    results = agreement(grades, grades)
    assert {result.dimension for result in results} == set(DIMENSIONS)
    for result in results:
        assert result.kappa == pytest.approx(1.0)
        assert result.exact_match == pytest.approx(1.0)
        assert result.mean_absolute_error == pytest.approx(0.0)


def test_a_judge_that_disagrees_is_reported_as_disagreeing():
    judge = _grades({f"g{i}": (i % 5) + 1 for i in range(MIN_AUDIT_ROWS)})
    # Inverted: same range, opposite order, so kappa must go sharply negative
    # rather than merely below one.
    human = _grades({f"g{i}": SCORE_MAX - (i % 5) for i in range(MIN_AUDIT_ROWS)})
    for result in agreement(judge, human):
        assert result.kappa < 0.0
        assert result.spearman < 0.0


def test_agreement_covers_only_the_rows_both_sides_scored():
    judge = _grades({f"g{i}": 3 for i in range(MIN_AUDIT_ROWS + 20)})
    human = _grades({f"g{i}": 3 for i in range(MIN_AUDIT_ROWS)})
    for result in agreement(judge, human):
        assert result.n == MIN_AUDIT_ROWS


def test_a_rare_score_is_taken_exhaustively_not_capped_at_a_quota():
    # Round-robin, not an even split. The docstring's own worked example: 108
    # fives and 12 ones, asking for 60, gives every one of the ones. Capping the
    # rare bucket at a quota would leave the judge least tested exactly where it
    # is least tested already.
    grades = _grades({f"g{i}": 5 for i in range(108)} | {f"h{i}": 1 for i in range(12)})
    chosen = stratified_sample(grades, 60, dimension="faithfulness", seed=1)
    rare = [gid for gid in chosen if gid.startswith("h")]
    assert len(rare) == 12
    assert len(chosen) == 60


def test_asking_for_more_than_exists_returns_everything_once():
    grades = _grades({f"g{i}": 3 for i in range(5)})
    chosen = stratified_sample(grades, 50, dimension="relevance", seed=1)
    assert sorted(chosen) == sorted(grades)
