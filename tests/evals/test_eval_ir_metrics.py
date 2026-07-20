"""Tier-1 metric shaping: depth truncation, the scored denominator, aggregation.

The evaluator seam is filled with a fake that mirrors pytrec_eval's semantics
(it scores only query ids present in both the run and the qrels) and records
the run it was handed, so truncation is observable without the C extension.
"""

import pytest
from evals.benchmark.ir_metrics import METRIC_MEASURES, score_run


class _RecordingEvaluator:
    """Scores a run the way pytrec_eval does, remembering what it received."""

    def __init__(self, qrels, measures, calls):
        self._qrels = qrels
        self._measures = measures
        self._calls = calls

    def evaluate(self, run):
        self._calls.append({"measures": set(self._measures), "run": run})
        scored = {}
        for query_id, docs in run.items():
            judged = self._qrels.get(query_id)
            if judged is None:
                continue
            # Score descending, ties on doc_id descending: trec_eval's rule, which
            # is what the real scorer applies. A fake that broke ties the other
            # way would hide a truncation that hands pytrec_eval the wrong ten.
            ranked = sorted(docs.items(), key=lambda item: (item[1], item[0]), reverse=True)
            values = {}
            if "recip_rank" in self._measures:
                hit = next(
                    (rank for rank, (doc, _) in enumerate(ranked, 1) if judged.get(doc, 0) > 0),
                    None,
                )
                values["recip_rank"] = 1.0 / hit if hit else 0.0
            if "recall_20" in self._measures:
                positives = {doc for doc, grade in judged.items() if grade > 0}
                found = {doc for doc, _ in ranked[:20]} & positives
                values["recall_20"] = len(found) / len(positives) if positives else 0.0
            if "ndcg_cut_10" in self._measures:
                top10 = ranked[:10]
                values["ndcg_cut_10"] = 1.0 if any(judged.get(d, 0) > 0 for d, _ in top10) else 0.0
            scored[query_id] = values
        return scored


def _factory(calls):
    def make(qrels, measures):
        return _RecordingEvaluator(qrels, measures, calls)

    return make


def _run_of(query_id, doc_ids):
    """A single-query run whose scores descend in the order doc_ids is given."""
    return {query_id: {doc: float(len(doc_ids) - index) for index, doc in enumerate(doc_ids)}}


def test_mrr_at_10_ignores_a_relevant_document_below_rank_10():
    # First relevant document sits at rank 11; MRR@10 must score this query 0.
    docs = [f"d{i}" for i in range(1, 21)]
    qrels = {"q1": {"d11": 1}}
    scores = score_run(qrels, _run_of("q1", docs), ["MRR@10"], evaluator_factory=_factory([]))
    assert scores.per_query["MRR@10"]["q1"] == 0.0
    assert scores.aggregated["MRR@10"] == 0.0


def test_mrr_at_10_counts_a_relevant_document_at_rank_10():
    docs = [f"d{i}" for i in range(1, 21)]
    qrels = {"q1": {"d10": 1}}
    scores = score_run(qrels, _run_of("q1", docs), ["MRR@10"], evaluator_factory=_factory([]))
    assert scores.per_query["MRR@10"]["q1"] == pytest.approx(0.1)


def test_mrr_at_10_truncates_the_run_handed_to_the_evaluator():
    calls = []
    docs = [f"d{i}" for i in range(1, 21)]
    score_run({"q1": {"d1": 1}}, _run_of("q1", docs), ["MRR@10"], evaluator_factory=_factory(calls))
    handed = calls[0]["run"]["q1"]
    assert len(handed) == 10
    assert set(handed) == {f"d{i}" for i in range(1, 11)}


def test_truncation_breaks_ties_the_way_pytrec_eval_does():
    calls = []
    # Eleven documents all tied on score. pytrec_eval ignores the run file's rank
    # column and breaks score ties on doc_id DESCENDING, so the ten it would keep
    # are d02..d12 and the one it drops is d01. Truncating the other way would
    # hand the scorer a different ten than it would have picked itself.
    run = {"q1": {f"d{i:02d}": 1.0 for i in range(1, 13)}}
    score_run({"q1": {"d12": 1}}, run, ["MRR@10"], evaluator_factory=_factory(calls))
    assert set(calls[0]["run"]["q1"]) == {f"d{i:02d}" for i in range(3, 13)}


def test_metrics_that_cut_internally_are_not_truncated():
    calls = []
    docs = [f"d{i}" for i in range(1, 21)]
    score_run(
        {"q1": {"d1": 1}}, _run_of("q1", docs), ["Recall@20"], evaluator_factory=_factory(calls)
    )
    assert len(calls[0]["run"]["q1"]) == 20


def test_a_qrels_topic_the_run_missed_scores_zero_rather_than_vanishing():
    # q2 is judged but the arm returned nothing for it: it must score 0, not be dropped.
    qrels = {"q1": {"d1": 1}, "q2": {"d9": 1}}
    scores = score_run(qrels, _run_of("q1", ["d1"]), ["MRR@10"], evaluator_factory=_factory([]))
    assert scores.per_query["MRR@10"] == {"q1": 1.0, "q2": 0.0}
    assert scores.aggregated["MRR@10"] == pytest.approx(0.5)


def test_the_denominator_is_the_qrels_topic_set_not_the_answered_set():
    qrels = {f"q{i}": {"d1": 1} for i in range(1, 5)}
    scores = score_run(qrels, _run_of("q1", ["d1"]), ["MRR@10"], evaluator_factory=_factory([]))
    # One of four topics answered perfectly: 1.0/4, not 1.0/1.
    assert scores.aggregated["MRR@10"] == pytest.approx(0.25)
    assert len(scores.per_query["MRR@10"]) == 4


def test_an_arm_that_answers_nothing_scores_zero_rather_than_zero_topics():
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    scores = score_run(qrels, {}, ["MRR@10"], evaluator_factory=_factory([]))
    assert scores.aggregated["MRR@10"] == 0.0
    assert scores.per_query["MRR@10"] == {"q1": 0.0, "q2": 0.0}


def test_run_queries_absent_from_the_qrels_are_ignored():
    qrels = {"q1": {"d1": 1}}
    run = {**_run_of("q1", ["d1"]), "unjudged": {"d1": 1.0}}
    scores = score_run(qrels, run, ["MRR@10"], evaluator_factory=_factory([]))
    assert set(scores.per_query["MRR@10"]) == {"q1"}
    assert scores.aggregated["MRR@10"] == 1.0


def test_empty_qrels_aggregate_to_zero():
    scores = score_run({}, {}, ["MRR@10"], evaluator_factory=_factory([]))
    assert scores.aggregated["MRR@10"] == 0.0
    assert scores.per_query["MRR@10"] == {}


def test_several_metrics_are_scored_together():
    qrels = {"q1": {"d1": 1}, "q2": {"d3": 1}}
    run = {**_run_of("q1", ["d1", "d2"]), **_run_of("q2", ["d2", "d3"])}
    scores = score_run(
        qrels, run, ["nDCG@10", "Recall@20", "MRR@10"], evaluator_factory=_factory([])
    )
    assert set(scores.aggregated) == {"nDCG@10", "Recall@20", "MRR@10"}
    assert scores.aggregated["MRR@10"] == pytest.approx(0.75)


def test_unknown_metric_is_rejected():
    with pytest.raises(ValueError, match="unknown metrics: P@5"):
        score_run({}, {}, ["P@5"], evaluator_factory=_factory([]))


def test_aggregated_scores_are_rounded_to_four_places():
    qrels = {f"q{i}": {"d1": 1} for i in range(3)}
    scores = score_run(qrels, _run_of("q0", ["d1"]), ["MRR@10"], evaluator_factory=_factory([]))
    assert scores.aggregated["MRR@10"] == 0.3333


def test_to_dict_round_trips_both_layers():
    qrels = {"q1": {"d1": 1}}
    scores = score_run(qrels, _run_of("q1", ["d1"]), ["MRR@10"], evaluator_factory=_factory([]))
    assert scores.to_dict() == {
        "per_query": {"MRR@10": {"q1": 1.0}},
        "aggregated": {"MRR@10": 1.0},
    }


def test_every_display_metric_maps_to_a_measure():
    assert set(METRIC_MEASURES) == {"nDCG@10", "Recall@20", "MRR@10"}
