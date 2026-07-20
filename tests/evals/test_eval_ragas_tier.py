"""Tier-2 RAGAS aggregation: real per-arm scores, honest denominators.

The evaluator seam returns per-sample scores (RAGAS emits NaN when a metric
cannot be computed), so the aggregation that decides the published mean and the
n behind it is pure and testable without ragas installed.
"""

import math

import pytest
from evals.benchmark.ragas_tier import (
    MIN_COVERAGE,
    RagasCoverageError,
    Sample,
    score_ragas,
)


def _samples(n):
    return [
        Sample(question=f"q{i}", answer=f"a{i}", contexts=[f"c{i}"], ground_truth=f"g{i}")
        for i in range(n)
    ]


def _evaluator(per_metric):
    def evaluate(rows, metrics):
        assert len(rows) == len(next(iter(per_metric.values())))
        return {metric: per_metric[metric] for metric in metrics}

    return evaluate


def test_means_are_averaged_over_the_scored_samples():
    scores = score_ragas(
        _samples(3),
        ["faithfulness"],
        evaluate_fn=_evaluator({"faithfulness": [0.0, 0.5, 1.0]}),
    )
    assert scores.means["faithfulness"] == pytest.approx(0.5)
    assert scores.scored["faithfulness"] == 3
    assert scores.total == 3


def test_uncomputable_samples_are_counted_not_silently_dropped():
    # One NaN out of four: the mean is over the three that scored, but the
    # count records that only three of four samples produced a score.
    scores = score_ragas(
        _samples(4),
        ["faithfulness"],
        evaluate_fn=_evaluator({"faithfulness": [1.0, float("nan"), 1.0, 1.0]}),
        min_coverage=0.5,
    )
    assert scores.means["faithfulness"] == pytest.approx(1.0)
    assert scores.scored["faithfulness"] == 3
    assert scores.total == 4
    assert scores.coverage("faithfulness") == pytest.approx(0.75)


def test_coverage_below_the_floor_fails_rather_than_publishing_a_flattering_mean():
    # A system whose answers fail RAGAS more often would otherwise score higher.
    nan = float("nan")
    with pytest.raises(RagasCoverageError, match="faithfulness"):
        score_ragas(
            _samples(10),
            ["faithfulness"],
            evaluate_fn=_evaluator({"faithfulness": [1.0] * 3 + [nan] * 7}),
        )


def test_a_metric_that_scored_nothing_is_not_reported_as_zero():
    with pytest.raises(RagasCoverageError):
        score_ragas(
            _samples(3),
            ["faithfulness"],
            evaluate_fn=_evaluator({"faithfulness": [float("nan")] * 3}),
        )


def test_full_coverage_is_accepted_at_the_floor():
    n = 10
    kept = math.ceil(MIN_COVERAGE * n)
    values = [1.0] * kept + [float("nan")] * (n - kept)
    scores = score_ragas(
        _samples(n), ["faithfulness"], evaluate_fn=_evaluator({"faithfulness": values})
    )
    assert scores.coverage("faithfulness") >= MIN_COVERAGE


def test_several_metrics_are_aggregated_independently():
    scores = score_ragas(
        _samples(2),
        ["faithfulness", "context_recall"],
        evaluate_fn=_evaluator({"faithfulness": [1.0, 0.0], "context_recall": [0.25, 0.75]}),
    )
    assert scores.means["faithfulness"] == pytest.approx(0.5)
    assert scores.means["context_recall"] == pytest.approx(0.5)


def test_scores_round_trip_to_a_serializable_dict():
    scores = score_ragas(
        _samples(2), ["faithfulness"], evaluate_fn=_evaluator({"faithfulness": [1.0, 0.0]})
    )
    payload = scores.to_dict()
    assert payload["means"]["faithfulness"] == pytest.approx(0.5)
    assert payload["scored"]["faithfulness"] == 2
    assert payload["total"] == 2
