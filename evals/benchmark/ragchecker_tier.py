"""Claim-level answer scoring, computed by RAGChecker.

RAGAS is the harness' primary answer-quality scorer, and it has a specific hole:
its faithfulness moves when retrieval changes and when generation changes, by
similar amounts, so it cannot say which half of the pipeline failed. For an A/B
whose entire subject is retrieval, that is the diagnosis the study most needs.

RAGChecker decomposes answers and ground truth into claims and checks each one
by entailment, which splits the result into retriever-side and generator-side
metrics. It is also the stronger citation: NeurIPS 2024 Datasets and Benchmarks
(a full track), meta-evaluated against 280 human-labelled instances, where
RAGAS' own correlation figures are self-reported on a dataset its authors built.

Two things about it are worth knowing before reading a number it produced:

- Its last release is 0.1.9 from September 2024 and its last commit December
  2024. It works, but it is not maintained, so it is pinned and treated as a
  cross-check on RAGAS rather than as the primary scorer.
- It reaches its judge through litellm. Both the claim extractor and the
  entailment checker take an ``api_base``, so a self-hosted OpenAI-compatible
  endpoint works, which is what the manifest's judge is.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from evals.benchmark.ragas_tier import Sample
from evals.deps import install_hint

RAGCHECKER_INSTALL_HINT = install_hint("ragchecker", "for claim-level answer scoring")

# Metrics RAGChecker reports, split by what a change in them implicates. The
# split is the reason this tier exists: a drop in claim_recall is a retrieval
# problem and a drop in hallucination is a generation problem, and RAGAS'
# faithfulness cannot tell those apart.
RETRIEVER_METRICS = ("claim_recall", "context_precision")
GENERATOR_METRICS = (
    "context_utilization",
    "noise_sensitivity_in_relevant",
    "noise_sensitivity_in_irrelevant",
    "hallucination",
    "self_knowledge",
    "faithfulness",
)
OVERALL_METRICS = ("precision", "recall", "f1")
RAGCHECKER_METRICS = OVERALL_METRICS + RETRIEVER_METRICS + GENERATOR_METRICS

# litellm routes any "openai/<name>" model string to the api_base verbatim, which
# is how a self-hosted server is addressed. Naming the prefix once keeps the
# convention out of the call sites.
OPENAI_COMPATIBLE_PREFIX = "openai/"

# rows -> per-metric aggregate. Injected so tests need neither ragchecker nor a
# judge, matching how the RAGAS tier is wired.
RagCheckerEvaluateFn = Callable[[list[dict[str, Any]]], dict[str, float]]


@dataclass(frozen=True)
class RagCheckerJudge:
    """The model RAGChecker extracts and checks claims with.

    ``base_url`` is threaded to litellm's ``api_base`` for both the extractor and
    the checker. Without it litellm resolves a provider from the environment and
    the manifest's pinned judge would not be what actually scored the answers.
    """

    model: str
    base_url: str
    api_key: str = "not-needed"
    batch_size: int = 32


def to_ragchecker_payload(samples: list[Sample], query_ids: list[str]) -> dict[str, Any]:
    """Shape samples as RAGChecker's ``RAGResults`` JSON.

    Its schema names the same four things the harness already carries, under
    different keys: query, response, gt_answer, retrieved_context. The ids are
    required by the schema and carry no meaning to the scorer, so the caller's
    own query ids are used rather than inventing a second numbering.
    """
    if len(samples) != len(query_ids):
        raise ValueError(
            f"{len(samples)} samples but {len(query_ids)} query ids; every scored "
            "answer must keep the id it was collected under"
        )
    return {
        "results": [
            {
                "query_id": query_id,
                "query": sample.question,
                "gt_answer": sample.ground_truth,
                "response": sample.answer,
                "retrieved_context": [
                    {"doc_id": str(index), "text": context}
                    for index, context in enumerate(sample.contexts)
                ],
            }
            for query_id, sample in zip(query_ids, samples, strict=True)
        ]
    }


def make_ragchecker_evaluator(judge: RagCheckerJudge) -> RagCheckerEvaluateFn:
    """A RAGChecker evaluator bound to the manifest's judge."""

    def evaluate_fn(rows: list[dict[str, Any]]) -> dict[str, float]:  # pragma: no cover - needs it
        try:
            from ragchecker import RAGChecker, RAGResults
            from ragchecker.metrics import all_metrics
        except ImportError as exc:
            raise RuntimeError(RAGCHECKER_INSTALL_HINT) from exc
        import json

        model = judge.model
        if not model.startswith(OPENAI_COMPATIBLE_PREFIX):
            model = f"{OPENAI_COMPATIBLE_PREFIX}{model}"
        results = RAGResults.from_json(json.dumps({"results": rows}))
        evaluator = RAGChecker(
            extractor_name=model,
            checker_name=model,
            extractor_api_base=judge.base_url,
            checker_api_base=judge.base_url,
            batch_size_extractor=judge.batch_size,
            batch_size_checker=judge.batch_size,
            # litellm requires a non-empty key even where the server ignores it.
            openai_api_key=judge.api_key,
        )
        evaluator.evaluate(results, all_metrics)
        return dict(results.metrics)

    return evaluate_fn


@dataclass(frozen=True)
class RagCheckerScores:
    """Per-metric scores, split by which half of the pipeline they implicate."""

    overall: dict[str, float]
    retriever: dict[str, float]
    generator: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {"overall": self.overall, "retriever": self.retriever, "generator": self.generator}


def _flatten(metrics: dict[str, Any]) -> dict[str, float]:
    """RAGChecker nests its metrics one level; read them as a flat map.

    ``evaluate`` returns ``{"overall_metrics": {...}, "retriever_metrics": {...},
    "generator_metrics": {...}}``. Flattening once here means the grouping this
    module publishes comes from its own declared lists rather than from whichever
    key an unmaintained package happened to nest a metric under.
    """
    flat: dict[str, float] = {}
    for value in metrics.values():
        if isinstance(value, dict):
            flat.update({key: float(inner) for key, inner in value.items()})
    flat.update(
        {key: float(value) for key, value in metrics.items() if not isinstance(value, dict)}
    )
    return flat


class RagCheckerScoreError(RuntimeError):
    """RAGChecker returned nothing usable for a metric the study reports."""


def score_ragchecker(
    samples: list[Sample], query_ids: list[str], *, evaluate_fn: RagCheckerEvaluateFn
) -> RagCheckerScores:
    """Score answers with RAGChecker, grouped by retriever versus generator.

    A metric that came back absent or NaN raises rather than being reported as
    zero. Zero is a meaningful score here (no claims recalled, no hallucination),
    so a missing metric silently rendered as 0.0 would read as a perfect or a
    catastrophic result depending on which one vanished.
    """
    payload = to_ragchecker_payload(samples, query_ids)
    flat = _flatten(evaluate_fn(payload["results"]))
    missing = sorted(
        name for name in RAGCHECKER_METRICS if name not in flat or math.isnan(flat[name])
    )
    if missing:
        raise RagCheckerScoreError(
            f"RAGChecker returned no usable value for {missing}; reporting those as "
            "zero would be indistinguishable from a real score, since zero is "
            "meaningful for every one of them"
        )
    return RagCheckerScores(
        overall={name: flat[name] for name in OVERALL_METRICS},
        retriever={name: flat[name] for name in RETRIEVER_METRICS},
        generator={name: flat[name] for name in GENERATOR_METRICS},
    )


def attribution(baseline: RagCheckerScores, arm: RagCheckerScores) -> dict[str, float]:
    """Mean movement on each side of the pipeline, arm minus baseline.

    The number this tier exists to produce. RAGAS reports one faithfulness that
    moves for either cause; this reports the two separately, so a study can say
    whether a change helped because retrieval improved or because the generator
    made better use of what it was given.
    """
    return {
        "retriever_delta": statistics.fmean(
            arm.retriever[name] - baseline.retriever[name] for name in RETRIEVER_METRICS
        ),
        "generator_delta": statistics.fmean(
            arm.generator[name] - baseline.generator[name] for name in GENERATOR_METRICS
        ),
    }
