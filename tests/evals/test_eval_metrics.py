"""Tier-1 metric scoring: the choices this harness makes around ir_measures.

Whether RR@10 cuts at ten is ir_measures' business and is tested there. What is
tested here is what the harness decides on top of it: which topics form the
denominator, which are excluded, and which metric names it will accept.
"""

import pytest
from evals.benchmark.metrics import METRIC_MEASURES, score_run

pytest.importorskip("ir_measures")


def _run_with_relevant_at(rank: int) -> tuple[dict, dict]:
    """A single-query run whose only relevant document sits at ``rank``."""
    qrels = {"q1": {"REL": 1}}
    # Decreasing scores so the ordering is unambiguous and tie-free.
    docs = {f"d{i}": 100.0 - i for i in range(1, rank)}
    docs["REL"] = 100.0 - rank
    docs.update({f"d{i}": 100.0 - i for i in range(rank + 1, rank + 30)})
    return qrels, {"q1": docs}


def test_a_topic_the_run_missed_scores_zero_rather_than_vanishing():
    # The denominator is the qrels topic set. Dropping an unanswered topic would
    # reward an arm for returning nothing on its hard queries.
    qrels = {"q1": {"REL": 1}, "q2": {"REL": 1}}
    run = {"q1": {"REL": 1.0}}
    scores = score_run(qrels, run, ["nDCG@10"])
    assert scores["per_query"]["nDCG@10"]["q2"] == 0.0
    assert scores["aggregated"]["nDCG@10"] == pytest.approx(0.5)


def test_a_run_topic_absent_from_qrels_does_not_enter_the_mean():
    qrels = {"q1": {"REL": 1}}
    run = {"q1": {"REL": 1.0}, "q_unjudged": {"REL": 1.0}}
    scores = score_run(qrels, run, ["nDCG@10"])
    assert set(scores["per_query"]["nDCG@10"]) == {"q1"}


def test_unknown_metric_names_are_rejected():
    with pytest.raises(ValueError, match="unknown metrics"):
        score_run({"q1": {"d1": 1}}, {"q1": {"d1": 1.0}}, ["nDCG@5"])


def test_every_declared_metric_carries_its_depth_in_its_measure_string():
    # The measure string is the whole depth contract; there is no second place
    # a depth can be declared and drift out of step with it.
    assert all("@" in measure for measure in METRIC_MEASURES.values())
