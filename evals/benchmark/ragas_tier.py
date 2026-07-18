"""Tier-2 answer quality: RAGAS metrics plus the reused blind judge.

RAGAS scores the generated answers (faithfulness, answer relevancy, context
precision/recall). The blind duplicate-arm judge from ``evals.retrieval`` is
reused unchanged as a corroborating signal, with its own noise floor measured
by grading one arm twice. ragas is an optional heavy dependency, imported
lazily; the wiring is injectable so tests need neither ragas nor a real model.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.retrieval.answers import AnswerRow
from evals.retrieval.blinding import build_blind_rows, unblind
from evals.retrieval.judging import DIMENSIONS, judge_rows
from evals.retrieval.llm import ChatFn
from evals.retrieval.questions import Question
from evals.retrieval.scoring import ARM_REPLICATE, NOISE_REPLICATE, dimension_means, noise_floor

RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

RAGAS_INSTALL_HINT = (
    "ragas is required for the answer tier; install the benchmark deps: "
    "uv pip install -r evals/benchmark/requirements.txt"
)

# question, answer, contexts, ground_truth -> per-metric score.
RagasEvaluateFn = Callable[[list[dict[str, Any]], list[str]], dict[str, float]]


@dataclass(frozen=True)
class Sample:
    """One generated answer with the context it was grounded on."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str

    def to_ragas_row(self) -> dict[str, Any]:
        return {
            "user_input": self.question,
            "response": self.answer,
            "retrieved_contexts": list(self.contexts),
            "reference": self.ground_truth,
        }


def _default_ragas_evaluate(rows: list[dict[str, Any]], metrics: list[str]) -> dict[str, float]:
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas import metrics as ragas_metrics
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(RAGAS_INSTALL_HINT) from exc
    selected = [getattr(ragas_metrics, name) for name in metrics]
    dataset = EvaluationDataset.from_list(rows)
    result = evaluate(dataset=dataset, metrics=selected)
    scores = result.to_pandas()[metrics].mean().to_dict()
    return {metric: float(scores[metric]) for metric in metrics}


def score_ragas(
    samples: list[Sample],
    metrics: list[str] | None = None,
    *,
    evaluate_fn: RagasEvaluateFn | None = None,
) -> dict[str, float]:
    """Mean RAGAS score per metric over the samples."""
    evaluator = evaluate_fn or _default_ragas_evaluate
    rows = [sample.to_ragas_row() for sample in samples]
    return evaluator(rows, list(metrics if metrics is not None else RAGAS_METRICS))


@dataclass(frozen=True)
class JudgeSummary:
    """The corroborating judge's per-arm means and its measured noise floor."""

    noise_floor: float
    means: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {"noise_floor": self.noise_floor, "means": self.means}


def run_corroborating_judge(
    questions: list[Question],
    answers_by_arm: dict[str, dict[str, AnswerRow]],
    noise_arm: str,
    chat: ChatFn,
    work_dir: Path,
    *,
    seed: int,
) -> JudgeSummary:
    """Run the reused blind judge and summarize per-arm means and its noise floor.

    The noise arm is graded twice under fresh opaque ids; the disagreement
    between those two passes is the judge's noise floor, so a small answer-tier
    gap is never oversold.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    blind = build_blind_rows(questions, answers_by_arm, noise_arm, random.Random(seed))
    grades = judge_rows(blind.rows, chat, work_dir / "grades.jsonl")
    for gid in blind.prefailed:
        grades[gid] = dict.fromkeys(DIMENSIONS, 0)
    unblinded = unblind(blind.assignments, grades)
    means = {
        arm: dimension_means(replicates.get(ARM_REPLICATE, {}))
        for arm, replicates in unblinded.items()
    }
    noise_replicates = unblinded.get(noise_arm, {})
    floor = noise_floor(
        noise_replicates.get(ARM_REPLICATE, {}), noise_replicates.get(NOISE_REPLICATE, {})
    )
    return JudgeSummary(noise_floor=floor, means=means)
