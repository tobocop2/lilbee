"""Tier-1 metric scoring: the choices this harness makes around ir_measures.

Whether RR@10 cuts at ten is ir_measures' business and is tested there. What is
tested here is what the harness decides on top of it: which topics form the
denominator, which are excluded, and which metric names it will accept.
"""

import pytest
from evals.benchmark.metrics import METRIC_MEASURES, judged_at_k, score_run

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


def test_judged_at_k_reports_the_share_of_results_the_labels_cover():
    # Two of the four retrieved documents carry a judgment, relevant or not:
    # coverage is about whether the labels speak to the document at all.
    qrels = {"q1": {"a": 1, "b": 0}}
    run = {"q1": {"a": 4.0, "b": 3.0, "unjudged1": 2.0, "unjudged2": 1.0}}
    assert judged_at_k(qrels, run, k=4) == pytest.approx(0.5)


def test_judged_at_k_averages_over_the_qrels_topics_including_unanswered_ones():
    # A topic the run never answered contributes zero coverage, matching how
    # every other figure here is averaged over the qrels topic set.
    qrels = {"q1": {"a": 1}, "q2": {"z": 1}}
    run = {"q1": {"a": 1.0}}
    assert judged_at_k(qrels, run, k=10) == pytest.approx(0.5)


def test_judged_at_k_cuts_ties_the_same_way_the_scorer_ranks():
    # Three documents tied on score, depth 1. The harness' tie rule is doc_id
    # descending, so "d3" is the document at rank 1; an ascending tie-break here
    # would report coverage over a document the scorer never put at that rank.
    qrels = {"q1": {"d3": 1}}
    run = {"q1": {"d1": 1.0, "d2": 1.0, "d3": 1.0}}
    assert judged_at_k(qrels, run, k=1) == pytest.approx(1.0)


def test_judged_at_k_is_zero_when_the_run_and_qrels_name_documents_differently():
    # The document-id namespace mismatch signature: every metric scores zero and
    # looks like a terrible system, while coverage shows the labels and the run
    # are not discussing the same documents at all.
    qrels = {"q1": {"beir-doc-1": 1}}
    run = {"q1": {"1f3c-uuid-a": 9.0, "1f3c-uuid-b": 8.0}}
    assert judged_at_k(qrels, run) == 0.0
    assert score_run(qrels, run, ["nDCG@10"])["aggregated"]["nDCG@10"] == 0.0
