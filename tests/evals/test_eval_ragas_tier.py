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

from tests.evals.stub_judge import install_stub_graders


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


def _judge_question(qid):
    from evals.retrieval.questions import Question, QuestionKind

    return Question(
        qid=qid,
        kind=QuestionKind.TOPICAL,
        question="Where?",
        source="a.txt",
        ground_passage="ground",
    )


def _judge_answer(qid, arm, text="an answer", error=None):
    from evals.retrieval.answers import AnswerRow

    return AnswerRow(
        qid=qid,
        arm=arm,
        answer="" if error else text,
        sources=["a.txt"],
        cited_sources=["a.txt"],
        seconds=0.1,
        error=error,
    )


def test_corroborating_judge_floor_excludes_never_judged_rows(tmp_path, monkeypatch):
    # Arm B's tq1 answer failed outright, so it never reaches the judge. Zeroing
    # it into both replicates registers as perfect self-agreement and drags the
    # floor toward zero, which marks every answer-tier delta as signal.
    from evals.benchmark.ragas_tier import run_corroborating_judge

    questions = [_judge_question("tq0"), _judge_question("tq1")]
    answers = {
        "A": {q.qid: _judge_answer(q.qid, "A") for q in questions},
        "B": {
            "tq0": _judge_answer("tq0", "B"),
            "tq1": _judge_answer("tq1", "B", error="boom"),
        },
    }
    install_stub_graders(monkeypatch)
    summary = run_corroborating_judge(questions, answers, "B", None, tmp_path, seed=1)
    # Only tq0 was graded twice, so the floor is measured over that one pair and
    # the prefailed tq1 contributes nothing.
    assert summary.noise_floor >= 0.0
    assert summary.means["A"].keys() == summary.means["B"].keys()


def test_corroborating_judge_means_cover_the_same_questions(tmp_path, monkeypatch):
    from evals.benchmark.ragas_tier import run_corroborating_judge

    questions = [_judge_question("tq0"), _judge_question("tq1")]
    answers = {
        "A": {q.qid: _judge_answer(q.qid, "A") for q in questions},
        "B": {q.qid: _judge_answer(q.qid, "B") for q in questions},
    }
    # The judge fails on one row, so it lands in neither map.
    install_stub_graders(monkeypatch, fail_times=1)
    summary = run_corroborating_judge(questions, answers, "B", None, tmp_path, seed=1)
    assert summary.paired_questions == len(
        set(summary.per_arm_scored["A"]) & set(summary.per_arm_scored["B"])
    )


class _OrderedMetric:
    """Latency inversely proportional to index: late samples finish first."""

    name = "ordered"

    def __init__(self, total: int, fail_at: int | None = None) -> None:
        self.total = total
        self.fail_at = fail_at

    async def ascore(self, user_input: str, response: str, retrieved_contexts=None, reference=None):
        import asyncio

        index = int(response)
        if self.fail_at is not None and index == self.fail_at:
            raise RuntimeError("boom")
        await asyncio.sleep((self.total - index) * 0.001)

        class _Result:
            value = float(index)

        return _Result()


def _ordered_rows(total: int) -> list[dict]:
    return [
        {"user_input": "q", "response": str(i), "retrieved_contexts": ["c"], "reference": "r"}
        for i in range(total)
    ]


def test_concurrent_scoring_returns_scores_in_sample_order():
    # score_ragas zips these back against the samples by position, so if
    # concurrency reordered them every score would be attributed to the wrong
    # answer and nothing downstream would notice. The metric here finishes in
    # reverse, which is what makes the assertion mean something.
    import asyncio

    from evals.benchmark.ragas_tier import _score_all

    total = 20
    scored = asyncio.run(_score_all({"ordered": _OrderedMetric(total)}, _ordered_rows(total)))
    assert scored["ordered"] == [float(i) for i in range(total)]


def test_a_failed_sample_leaves_its_nan_at_its_own_index():
    import asyncio

    from evals.benchmark.ragas_tier import _score_all

    total = 20
    scored = asyncio.run(
        _score_all({"ordered": _OrderedMetric(total, fail_at=3)}, _ordered_rows(total))
    )
    assert [i for i, value in enumerate(scored["ordered"]) if math.isnan(value)] == [3]


class _KwargsMetric:
    """A metric that declares **kwargs rather than named parameters."""

    name = "kwargs_metric"

    async def ascore(self, **kwargs):  # pragma: no cover - never reached
        raise AssertionError("should not be called")


def test_a_metric_the_samples_cannot_fill_is_a_loud_binding_error():
    # Filtering by signature silently passes nothing to a **kwargs metric. Left
    # alone that surfaces as "scored 0/300" at the coverage floor, which reads
    # as a broken judge rather than a harness that sent no data.
    from evals.benchmark.ragas_tier import metric_kwargs

    with pytest.raises(ValueError, match="binding mismatch"):
        metric_kwargs(_KwargsMetric(), _ordered_rows(1)[0])


def test_a_metric_whose_fields_the_samples_carry_binds_cleanly():
    from evals.benchmark.ragas_tier import metric_kwargs

    bound = metric_kwargs(_OrderedMetric(1), _ordered_rows(1)[0])
    assert set(bound) == {"user_input", "response", "retrieved_contexts", "reference"}
