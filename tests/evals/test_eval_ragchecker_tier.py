"""The RAGChecker tier: the payload shape and the retriever/generator split.

RAGChecker itself is an unmaintained package reached over the network, so the
evaluator is injected and what these exercise is the harness' side: that samples
arrive in the schema RAGChecker declares, that a missing metric is refused
rather than read as zero, and that the attribution split reports the two halves
of the pipeline separately, which is the whole reason this tier exists.
"""

import math

import pytest
from evals.benchmark.ragas_tier import Sample
from evals.benchmark.ragchecker_tier import (
    GENERATOR_METRICS,
    OPENAI_COMPATIBLE_PREFIX,
    RAGCHECKER_METRICS,
    RETRIEVER_METRICS,
    RagCheckerJudge,
    RagCheckerScoreError,
    RagCheckerScores,
    attribution,
    score_ragchecker,
    to_ragchecker_payload,
)


def _samples(n=2):
    return [
        Sample(
            question=f"q{i}?",
            answer=f"answer {i}",
            contexts=[f"context {i}a", f"context {i}b"],
            ground_truth=f"truth {i}",
        )
        for i in range(n)
    ]


def _ids(n=2):
    return [f"q{i}" for i in range(n)]


def _all_scored(value=0.5):
    return dict.fromkeys(RAGCHECKER_METRICS, value)


def test_the_payload_uses_ragcheckers_own_field_names():
    # Its schema names the same things under different keys; a mismatch here is
    # silently-empty input rather than an error, since every field is optional
    # in its dataclass.
    payload = to_ragchecker_payload(_samples(1), _ids(1))
    row = payload["results"][0]
    assert set(row) == {"query_id", "query", "gt_answer", "response", "retrieved_context"}
    assert row["query"] == "q0?"
    assert row["gt_answer"] == "truth 0"
    assert row["response"] == "answer 0"
    assert [context["text"] for context in row["retrieved_context"]] == ["context 0a", "context 0b"]


def test_the_payload_keeps_the_id_each_answer_was_collected_under():
    # A second numbering invented here would make a RAGChecker row impossible to
    # trace back to the run file and the qrels it came from.
    payload = to_ragchecker_payload(_samples(2), ["custom-a", "custom-b"])
    assert [row["query_id"] for row in payload["results"]] == ["custom-a", "custom-b"]


def test_mismatched_samples_and_ids_are_refused():
    with pytest.raises(ValueError, match="every scored answer must keep the id"):
        to_ragchecker_payload(_samples(2), _ids(1))


def test_scores_are_split_into_retriever_and_generator_groups():
    # The split is the point of the tier: RAGAS' faithfulness moves for either
    # cause, so a study using it alone cannot say which half changed.
    scores = score_ragchecker(_samples(), _ids(), evaluate_fn=lambda _rows: _all_scored())
    assert set(scores.retriever) == set(RETRIEVER_METRICS)
    assert set(scores.generator) == set(GENERATOR_METRICS)
    assert not set(scores.retriever) & set(scores.generator)


def test_a_nested_result_shape_is_flattened():
    # RAGChecker groups its metrics under overall/retriever/generator keys. The
    # grouping this module publishes comes from its own lists, so an upstream
    # regrouping cannot silently move a metric between halves.
    nested = {
        "overall_metrics": {"precision": 0.5, "recall": 0.5, "f1": 0.5},
        "retriever_metrics": dict.fromkeys(RETRIEVER_METRICS, 0.5),
        "generator_metrics": dict.fromkeys(GENERATOR_METRICS, 0.5),
    }
    scores = score_ragchecker(_samples(), _ids(), evaluate_fn=lambda _rows: nested)
    assert scores.retriever == dict.fromkeys(RETRIEVER_METRICS, 0.5)


@pytest.mark.parametrize("absent", ["claim_recall", "hallucination", "f1"])
def test_a_missing_metric_is_refused_rather_than_read_as_zero(absent):
    # Zero is a real score for every one of these, so a metric that vanished
    # would be indistinguishable from a perfect or catastrophic result.
    partial = {key: value for key, value in _all_scored().items() if key != absent}
    with pytest.raises(RagCheckerScoreError, match=absent):
        score_ragchecker(_samples(), _ids(), evaluate_fn=lambda _rows: partial)


def test_a_nan_metric_is_refused_too():
    with pytest.raises(RagCheckerScoreError, match="claim_recall"):
        score_ragchecker(
            _samples(),
            _ids(),
            evaluate_fn=lambda _rows: {**_all_scored(), "claim_recall": math.nan},
        )


def test_attribution_reports_the_two_halves_separately():
    # A retrieval-only improvement must show on the retriever side and not the
    # generator side; that separation is what RAGAS cannot provide.
    baseline = RagCheckerScores(
        overall=dict.fromkeys(("precision", "recall", "f1"), 0.5),
        retriever=dict.fromkeys(RETRIEVER_METRICS, 0.4),
        generator=dict.fromkeys(GENERATOR_METRICS, 0.6),
    )
    better_retrieval = RagCheckerScores(
        overall=baseline.overall,
        retriever=dict.fromkeys(RETRIEVER_METRICS, 0.7),
        generator=dict.fromkeys(GENERATOR_METRICS, 0.6),
    )
    deltas = attribution(baseline, better_retrieval)
    assert deltas["retriever_delta"] == pytest.approx(0.3)
    assert deltas["generator_delta"] == pytest.approx(0.0)


def test_generator_delta_orients_lower_is_better_metrics():
    # More hallucination is worse, more faithfulness is better. Averaging raw
    # deltas would cancel the two; the aggregate must sign-flip hallucination so
    # a rise in it counts as a regression, not as offsetting the faithfulness win.
    baseline = RagCheckerScores(
        overall=dict.fromkeys(("precision", "recall", "f1"), 0.5),
        retriever=dict.fromkeys(RETRIEVER_METRICS, 0.4),
        generator=dict.fromkeys(GENERATOR_METRICS, 0.5),
    )
    # Only hallucination moves, and it moves up (worse).
    worse_generator = RagCheckerScores(
        overall=baseline.overall,
        retriever=baseline.retriever,
        generator={**dict.fromkeys(GENERATOR_METRICS, 0.5), "hallucination": 0.8},
    )
    deltas = attribution(baseline, worse_generator)
    # Raw delta on hallucination is +0.3; oriented it is -0.3, so the mean over
    # the six generator metrics is negative, not the +0.05 a naive mean gives.
    assert deltas["generator_delta"] < 0
    assert deltas["generator_delta"] == pytest.approx(-0.3 / len(GENERATOR_METRICS))


def test_the_judge_model_is_addressed_as_an_openai_compatible_route():
    # litellm routes "openai/<name>" to api_base verbatim; without the prefix it
    # resolves a provider from the environment and the manifest's judge is not
    # what scores the answers.
    judge = RagCheckerJudge(model="my-local-model", base_url="http://pod:8081/v1")
    assert not judge.model.startswith(OPENAI_COMPATIBLE_PREFIX)
    assert judge.base_url.endswith("/v1")
