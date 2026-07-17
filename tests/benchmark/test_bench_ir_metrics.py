"""Per-query shaping and aggregation of IR metrics with a fake evaluator.

pytrec_eval is a C extension; these tests inject a fake evaluator factory so the
shaping and mean-aggregation logic is covered without it installed.
"""

import pytest

from evals.benchmark.ir_metrics import METRIC_MEASURES, MetricScores, score_run


class _FakeEvaluator:
    def __init__(self, per_query):
        self._per_query = per_query

    def evaluate(self, run):
        # The run is accepted but the fake returns pre-baked measure values.
        assert run  # sanity: a run was passed through
        return self._per_query


def _factory(per_query):
    def make(qrels, measures):
        assert measures <= set(METRIC_MEASURES.values())
        return _FakeEvaluator(per_query)

    return make


def test_score_run_shapes_per_query_and_aggregates_by_mean():
    per_query = {
        "q1": {"ndcg_cut_10": 1.0, "recall_20": 0.5},
        "q2": {"ndcg_cut_10": 0.0, "recall_20": 1.0},
    }
    scores = score_run(
        {"q1": {"d": 1}},
        {"q1": {"d": 0.9}},
        ["nDCG@10", "Recall@20"],
        evaluator_factory=_factory(per_query),
    )
    assert isinstance(scores, MetricScores)
    assert scores.per_query["nDCG@10"] == {"q1": 1.0, "q2": 0.0}
    assert scores.aggregated["nDCG@10"] == 0.5
    assert scores.aggregated["Recall@20"] == 0.75


def test_score_run_rejects_unknown_metrics():
    with pytest.raises(ValueError, match="unknown metrics"):
        score_run({}, {}, ["nDCG@5"], evaluator_factory=_factory({}))


def test_score_run_empty_run_aggregates_to_zero():
    scores = score_run(
        {"q1": {"d": 1}}, {"q1": {"d": 0.1}}, ["MRR@10"], evaluator_factory=_factory({})
    )
    assert scores.aggregated["MRR@10"] == 0.0
