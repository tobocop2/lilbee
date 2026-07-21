"""The PyTerrier comparison layer: same numbers, and the constraints it carries.

The point of adopting ``pt.Experiment`` is that it changes no number while
taking over the multi-arm table and the paired test. The first test here is
therefore the one that matters: PyTerrier and the ir_measures path must agree
exactly, or the harness has two scorers that can drift.
"""

import pytest

pytest.importorskip("pyterrier")

from evals.benchmark.experiment import (
    compare_arms,
    qrels_to_frame,
    randomization_test,
    run_to_frame,
    topics_frame,
)
from evals.benchmark.metrics import score_run

METRICS = ["nDCG@10", "Recall@20", "MRR@10"]


def _qrels(n=30):
    return {f"q{i}": {f"d{i}": 1} for i in range(n)}


def _run(hit_every, n=30):
    """An arm that finds the relevant document on every ``hit_every``-th query."""
    run = {}
    for i in range(n):
        docs = {f"d{i}": 10.0} if i % hit_every == 0 else {}
        docs.update({f"x{j}": 9.0 - j for j in range(12)})
        run[f"q{i}"] = docs
    return run


def test_pyterrier_reproduces_the_ir_measures_numbers_exactly():
    # The whole justification for the swap. A disagreement here means the
    # published table and the per-query vectors behind the CIs come from two
    # different computations.
    qrels = _qrels()
    runs = {"dense": _run(3), "hybrid": _run(2)}
    aggregated = compare_arms(runs, qrels, METRICS, baseline="dense", resamples=200, seed=1)
    for arm, run in runs.items():
        direct = score_run(qrels, run, METRICS)["aggregated"]
        row = aggregated[aggregated["name"] == arm].iloc[0]
        for metric in METRICS:
            assert float(row[metric]) == pytest.approx(direct[metric], abs=5e-5)


def test_columns_use_the_display_metric_names_not_the_measure_strings():
    # PyTerrier names columns after the measure ("R@20"). The manifest, the
    # metric module and the report all say "Recall@20"; two conventions in the
    # published artifacts is a reconciliation job for the reader.
    aggregated = compare_arms(
        {"a": _run(2), "b": _run(3)}, _qrels(), METRICS, baseline="a", resamples=200, seed=1
    )
    assert set(METRICS) <= set(aggregated.columns)
    assert "R@20" not in aggregated.columns


def test_the_baseline_arm_is_the_first_row_and_has_no_p_value():
    # The arms are passed in with the baseline second, so this also pins that
    # compare_arms reorders rather than trusting the caller's dict order: a
    # baseline that landed in row 1 would silently make PyTerrier's baseline=0
    # test a different arm than the one the manifest declares.
    import math

    aggregated = compare_arms(
        {"hybrid": _run(2), "dense": _run(3)},
        _qrels(),
        METRICS,
        baseline="dense",
        resamples=200,
        seed=1,
    )
    assert aggregated.iloc[0]["name"] == "dense"
    # Nothing is tested against itself, so the baseline row carries no p-value.
    for metric in METRICS:
        assert math.isnan(float(aggregated.iloc[0][f"{metric} p-value"]))
    assert not math.isnan(float(aggregated.iloc[1]["nDCG@10 p-value"]))


def test_an_undeclared_baseline_is_refused():
    with pytest.raises(ValueError, match="baseline 'missing' is not among the arms"):
        compare_arms({"a": _run(2)}, _qrels(), METRICS, baseline="missing")


def test_unknown_metrics_are_refused():
    with pytest.raises(ValueError, match="unknown metrics"):
        compare_arms({"a": _run(2)}, _qrels(), ["nDCG@5"], baseline="a")


def test_the_randomization_test_returns_a_statistic_and_a_p_value():
    # PyTerrier's contract for a custom test is (statistic, p_value) from two
    # per-query vectors; anything else fails inside the rendering pass.
    test = randomization_test(resamples=200, seed=1)
    statistic, p_value = test([0.0] * 10, [1.0] * 10)
    assert isinstance(statistic, float)
    assert 0.0 <= p_value <= 1.0


def test_identical_arms_are_not_reported_as_a_difference():
    # An all-zero difference vector has nothing to permute. Returning p=1 is the
    # honest answer; letting scipy raise would abort the whole study on an A/A
    # null control, which is a comparison the manifest deliberately declares.
    test = randomization_test(resamples=200, seed=1)
    statistic, p_value = test([0.5] * 10, [0.5] * 10)
    assert (statistic, p_value) == (0.0, 1.0)


def test_topics_come_from_the_qrels_so_an_unanswered_query_still_counts():
    # The denominator is the judged topic set. Taking it from the runs would let
    # an arm improve its mean by returning nothing on its hard queries.
    qrels = _qrels(5)
    assert list(topics_frame(qrels)["qid"]) == [f"q{i}" for i in range(5)]


def test_run_frames_carry_the_columns_pyterrier_requires():
    frame = run_to_frame({"q1": {"d1": 2.0, "d2": 5.0}})
    assert list(frame.columns) == ["qid", "docno", "score", "rank"]
    # Ranked by descending score, so rank 0 is the best-scoring document.
    assert list(frame.sort_values("rank")["docno"]) == ["d2", "d1"]


def test_qrels_frames_carry_the_columns_pyterrier_requires():
    frame = qrels_to_frame({"q1": {"d1": 2}})
    assert list(frame.columns) == ["qid", "docno", "label"]
    assert int(frame.iloc[0]["label"]) == 2
