"""Tier-2 answer quality: RAGAS metrics plus the reused blind judge.

RAGAS scores the generated answers (faithfulness, answer relevancy, context
precision/recall). The blind duplicate-arm judge from ``evals.retrieval`` is
reused unchanged as a corroborating signal, with its own noise floor measured
by grading one arm twice. ragas is an optional heavy dependency, imported
lazily; the wiring is injectable so tests need neither ragas nor a real model.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import random
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evals.deps import install_hint
from evals.retrieval.answers import AnswerRow
from evals.retrieval.blinding import build_blind_rows, unblind
from evals.retrieval.judging import DIMENSIONS, SCORE_MIN, judge_rows
from evals.retrieval.questions import Question
from evals.retrieval.scoring import (
    ARM_REPLICATE,
    NOISE_REPLICATE,
    noise_floor,
    paired_dimension_means,
    paired_qids,
)

RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

RAGAS_INSTALL_HINT = install_hint("ragas", "for the answer tier")

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
    """The models RAGAS grades with, taken from the frozen manifest.

    Without this, ragas resolves a process-global default model from the
    environment, so the manifest's pinned judge and temperature would not be
    what actually scored the answers.

    ``embedding_model`` and ``embedding_base_url`` serve answer relevancy, which
    scores by embedding questions generated back from the answer. It is the one
    tier-2 metric that needs an embedder as well as a judge; leave them empty
    and that metric cannot be requested.
    """

    model: str
    base_url: str
    api_key: str = "not-needed"
    temperature: float = 0.0
    embedding_model: str = ""
    embedding_base_url: str = ""


# metric name -> the ragas collections class that computes it. Each class owns
# its own prompt, its structured output model, and its retry behaviour; the
# arguments each one wants are read off its own ``ascore`` signature below
# rather than tabulated here, so a signature change surfaces as a TypeError at
# the call rather than as a silently unfilled field.
COLLECTION_METRICS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "AnswerRelevancy",
    "context_precision": "ContextPrecisionWithReference",
    "context_recall": "ContextRecall",
}

# Concurrent grading calls in flight. The judge is one llama-server with a fixed
# slot count, so an unbounded gather over a few hundred answers just queues.
MAX_CONCURRENT_GRADES = 8


def _build_metric(name: str, judge: RagasJudge) -> Any:  # pragma: no cover - needs ragas
    from openai import AsyncOpenAI
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics import collections

    llm = llm_factory(
        judge.model,
        provider="openai",
        client=AsyncOpenAI(base_url=judge.base_url, api_key=judge.api_key),
        temperature=judge.temperature,
    )
    metric_cls = getattr(collections, COLLECTION_METRICS[name])
    if name != "answer_relevancy":
        return metric_cls(llm=llm)
    if not judge.embedding_model or not judge.embedding_base_url:
        raise ValueError(
            "answer_relevancy scores by embedding questions regenerated from the "
            "answer, so it needs an embedder as well as a judge; set the manifest's "
            "embedder and pass --embedding-base-url, or drop it from --metrics"
        )
    embeddings = embedding_factory(
        provider="openai",
        model=judge.embedding_model,
        client=AsyncOpenAI(base_url=judge.embedding_base_url, api_key=judge.api_key),
    )
    return metric_cls(llm=llm, embeddings=embeddings)


def make_ragas_evaluator(judge: RagasJudge) -> RagasEvaluateFn:
    """A RAGAS evaluator bound to the manifest's judge, returning per-sample scores.

    Each metric is scored through its own ``ascore``, which is what the ragas
    collections API exposes per sample. The older ``evaluate`` path returned a
    dataframe that had to be unpacked back into per-sample lists, and it does not
    accept these metric classes at all.
    """

    def evaluate_fn(  # pragma: no cover - exercised only with ragas installed
        rows: list[dict[str, Any]], metrics: list[str]
    ) -> dict[str, list[float]]:
        try:
            unknown = [name for name in metrics if name not in COLLECTION_METRICS]
            if unknown:
                raise ValueError(f"unknown ragas metrics: {', '.join(sorted(unknown))}")
            built = {name: _build_metric(name, judge) for name in metrics}
        except ImportError as exc:
            raise RuntimeError(RAGAS_INSTALL_HINT) from exc
        return asyncio.run(_score_all(built, rows))

    return evaluate_fn


async def _score_one(metric: Any, row: dict[str, Any]) -> float:  # pragma: no cover - needs ragas
    """One metric on one sample, NaN when it could not be computed.

    Each collections metric takes a different subset of the sample's fields, so
    the arguments are selected from the metric's own signature. A metric that
    raises (a refusal, empty contexts, an unparseable response) yields NaN,
    which ``score_ragas`` counts against that metric's coverage rather than
    silently averaging over whatever survived.
    """
    accepted = inspect.signature(metric.ascore).parameters
    kwargs = {key: value for key, value in row.items() if key in accepted}
    try:
        result = await metric.ascore(**kwargs)
    except Exception as exc:
        # Reported, not swallowed. The NaN is already accounted for by the
        # coverage floor, but a systematic cause (wrong endpoint, a judge that
        # refuses the whole batch) is only diagnosable if the reason is on
        # stderr rather than inferred from a coverage number at the end.
        print(f"{metric.name} could not score a sample: {exc}", file=sys.stderr)
        return math.nan
    return float(result.value)


async def _score_all(  # pragma: no cover - needs ragas
    metrics: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, list[float]]:
    """Every metric on every sample, bounded concurrency, in row order."""
    limit = asyncio.Semaphore(MAX_CONCURRENT_GRADES)

    async def guarded(metric: Any, row: dict[str, Any]) -> float:
        async with limit:
            return await _score_one(metric, row)

    scored: dict[str, list[float]] = {}
    for name, metric in metrics.items():
        scored[name] = list(await asyncio.gather(*(guarded(metric, row) for row in rows)))
    return scored


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
    llm: Any,
    work_dir: Path,
    *,
    seed: int,
) -> JudgeSummary:
    """Run the reused blind judge and summarize per-arm means and its noise floor.

    The noise arm is graded twice under two equivalent presentations of the
    rubric; the disagreement between those passes is the judge's noise floor.

    Keeps the judge-returned grades separate from the scored grades for the same
    reasons the retrieval scorer does. Prefailed answers never reached a judge,
    so counting their mechanical zeros as agreement would deflate the floor
    toward zero, and per-arm means must cover the questions both arms have an
    outcome for rather than whatever each arm's judge happened to parse.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    blind = build_blind_rows(questions, answers_by_arm, noise_arm, random.Random(seed))
    graded = judge_rows(blind.rows, llm, work_dir / "grades.jsonl")
    judged = unblind(blind.assignments, graded)
    scored_grades = dict(graded)
    for gid in blind.prefailed:
        # The rubric's bottom level already describes a missing answer ("misses
        # the question entirely"), so a prefailed row scores there rather than
        # at an off-scale 0 that no reader can place against the other means.
        scored_grades[gid] = dict.fromkeys(DIMENSIONS, SCORE_MIN)
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
