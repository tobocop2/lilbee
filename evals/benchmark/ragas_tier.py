"""Tier-2 answer quality: RAGAS metrics plus the reused blind judge.

RAGAS scores the generated answers (faithfulness, answer relevancy, context
precision/recall). The blind duplicate-arm judge from ``evals.retrieval`` is
reused unchanged as a corroborating signal, with its own noise floor measured
by grading one arm twice. ragas is an optional heavy dependency, imported
lazily; the wiring is injectable so tests need neither ragas nor a real model.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evals.retrieval.answers import AnswerRow
from evals.retrieval.blinding import build_blind_rows, unblind
from evals.retrieval.judging import DIMENSIONS, judge_rows
from evals.retrieval.llm import ChatFn
from evals.retrieval.questions import Question
from evals.retrieval.scoring import (
    ARM_REPLICATE,
    NOISE_REPLICATE,
    noise_floor,
    paired_dimension_means,
    paired_qids,
)

RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

RAGAS_INSTALL_HINT = (
    "ragas is required for the answer tier; install the benchmark deps: "
    "uv pip install -r evals/benchmark/requirements.txt"
)

# Fraction of samples a metric must actually score before its mean is publishable.
MIN_COVERAGE = 0.9

# rows, metrics -> per-metric list of per-sample scores (NaN where uncomputable).
RagasEvaluateFn = Callable[[list[dict[str, Any]], list[str]], dict[str, list[float]]]


class RagasCoverageError(RuntimeError):
    """A metric scored too few samples for its mean to mean anything."""


@dataclass(frozen=True)
class RagasScores:
    """Per-metric means and the sample counts they were computed over.

    RAGAS emits NaN when a metric cannot be computed (parse failure, empty
    retrieved contexts, a refusal). Averaging with those silently dropped makes a
    metric over 12 of 300 answers indistinguishable from one over all 300, and
    rewards the arm whose answers fail more often. The count travels with the
    mean so the denominator is never implicit.
    """

    means: dict[str, float]
    scored: dict[str, int]
    total: int

    def coverage(self, metric: str) -> float:
        return self.scored[metric] / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"means": self.means, "scored": self.scored, "total": self.total}


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


@dataclass(frozen=True)
class RagasJudge:
    """The model RAGAS grades with, taken from the frozen manifest.

    Without this, ragas' ``evaluate`` falls back to a process-global default
    model resolved from the environment, so the manifest's pinned judge and
    temperature would not be what actually scored the answers.
    """

    model: str
    base_url: str
    api_key: str = "not-needed"
    temperature: float = 0.0


def _build_ragas_judge(judge: RagasJudge) -> Any:  # pragma: no cover - needs ragas + a model
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(
        ChatOpenAI(
            model=judge.model,
            base_url=judge.base_url,
            api_key=judge.api_key,
            temperature=judge.temperature,
        )
    )


def make_ragas_evaluator(judge: RagasJudge) -> RagasEvaluateFn:
    """A RAGAS evaluator bound to the manifest's judge, returning per-sample scores."""

    def evaluate_fn(  # pragma: no cover - exercised only with ragas installed
        rows: list[dict[str, Any]], metrics: list[str]
    ) -> dict[str, list[float]]:
        try:
            from ragas import EvaluationDataset, evaluate
            from ragas import metrics as ragas_metrics
        except ImportError as exc:
            raise RuntimeError(RAGAS_INSTALL_HINT) from exc
        selected = [getattr(ragas_metrics, name) for name in metrics]
        dataset = EvaluationDataset.from_list(rows)
        result = evaluate(dataset=dataset, metrics=selected, llm=_build_ragas_judge(judge))
        frame = result.to_pandas()
        return {metric: [float(value) for value in frame[metric]] for metric in metrics}

    return evaluate_fn


def score_ragas(
    samples: list[Sample],
    metrics: list[str] | None = None,
    *,
    evaluate_fn: RagasEvaluateFn,
    min_coverage: float = MIN_COVERAGE,
) -> RagasScores:
    """Mean RAGAS score per metric, with the sample count each mean was computed over.

    Samples a metric could not score (NaN) are excluded from the mean but kept in
    the denominator record; a metric that scored less than ``min_coverage`` of the
    samples raises rather than publishing a mean over a flattering subset.
    """
    selected = list(metrics if metrics is not None else RAGAS_METRICS)
    rows = [sample.to_ragas_row() for sample in samples]
    raw = evaluate_fn(rows, selected)
    total = len(samples)
    means: dict[str, float] = {}
    scored: dict[str, int] = {}
    for metric in selected:
        values = [value for value in raw[metric] if not math.isnan(value)]
        scored[metric] = len(values)
        means[metric] = statistics.fmean(values) if values else 0.0
    result = RagasScores(means=means, scored=scored, total=total)
    thin = [
        f"{metric} scored {scored[metric]}/{total}"
        for metric in selected
        if result.coverage(metric) < min_coverage
    ]
    if thin:
        raise RagasCoverageError(
            f"RAGAS coverage below {min_coverage:.0%}: {'; '.join(thin)}. "
            "A mean over this subset would reward the arm whose answers fail more often."
        )
    return result


@dataclass(frozen=True)
class JudgeSummary:
    """The corroborating judge's per-arm means and its measured noise floor."""

    noise_floor: float
    means: dict[str, dict[str, float]]
    paired_questions: int = 0
    per_arm_scored: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "noise_floor": self.noise_floor,
            "means": self.means,
            "paired_questions": self.paired_questions,
            "per_arm_scored": self.per_arm_scored,
        }


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

    The noise arm is graded twice under two equivalent phrasings of the grading
    prompt; the disagreement between those passes is the judge's noise floor.

    Keeps the judge-returned grades separate from the scored grades for the same
    reasons the retrieval scorer does. Prefailed answers never reached a judge,
    so counting their mechanical zeros as agreement would deflate the floor
    toward zero, and per-arm means must cover the questions both arms have an
    outcome for rather than whatever each arm's judge happened to parse.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    blind = build_blind_rows(questions, answers_by_arm, noise_arm, random.Random(seed))
    graded = judge_rows(blind.rows, chat, work_dir / "grades.jsonl")
    judged = unblind(blind.assignments, graded)
    scored_grades = dict(graded)
    for gid in blind.prefailed:
        scored_grades[gid] = dict.fromkeys(DIMENSIONS, 0)
    unblinded = unblind(blind.assignments, scored_grades)
    arms = list(answers_by_arm)
    means = paired_dimension_means(unblinded, arms)
    noise_replicates = judged.get(noise_arm, {})
    floor = noise_floor(
        noise_replicates.get(ARM_REPLICATE, {}), noise_replicates.get(NOISE_REPLICATE, {})
    )
    return JudgeSummary(
        noise_floor=floor,
        means=means,
        paired_questions=len(paired_qids(unblinded, arms)),
        per_arm_scored={arm: sorted(unblinded.get(arm, {}).get(ARM_REPLICATE, {})) for arm in arms},
    )
