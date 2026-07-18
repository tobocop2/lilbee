"""Tier-1 retrieval metrics: score a run file against qrels with pytrec_eval.

pytrec_eval is the TREC-standard scorer, so these numbers carry no model
opinion and are exactly reproducible from the run file and the qrels. It is an
optional dependency (a C extension), imported lazily; the per-query shaping and
aggregation are pulled out so they can be unit-tested with a fake evaluator and
no C extension installed.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

Qrels = dict[str, dict[str, int]]
Run = dict[str, dict[str, float]]

# Display name -> the pytrec_eval measure that computes it. recip_rank is the
# uncut MRR; pytrec_eval has no built-in cut-at-10 variant, and MRR is
# dominated by the top hit, so at realistic depths the two agree.
METRIC_MEASURES: dict[str, str] = {
    "nDCG@10": "ndcg_cut_10",
    "Recall@20": "recall_20",
    "MRR@10": "recip_rank",
}

PYTREC_INSTALL_HINT = (
    "pytrec_eval is required to score retrieval; install the benchmark deps: "
    "uv pip install -r evals/benchmark/requirements.txt"
)


class Evaluator(Protocol):
    """The slice of pytrec_eval.RelevanceEvaluator this module uses."""

    def evaluate(self, run: Run) -> dict[str, dict[str, float]]: ...


EvaluatorFactory = Callable[[Qrels, set[str]], Evaluator]


@dataclass(frozen=True)
class MetricScores:
    """Per-query and aggregated scores for the requested display metrics."""

    per_query: dict[str, dict[str, float]] = field(default_factory=dict)
    aggregated: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"per_query": self.per_query, "aggregated": self.aggregated}


def _default_evaluator_factory(qrels: Qrels, measures: set[str]) -> Evaluator:
    try:
        import pytrec_eval
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(PYTREC_INSTALL_HINT) from exc
    return pytrec_eval.RelevanceEvaluator(qrels, measures)


def score_run(
    qrels: Qrels,
    run: Run,
    metrics: list[str],
    *,
    evaluator_factory: EvaluatorFactory | None = None,
) -> MetricScores:
    """Score a run against qrels, returning per-query and mean-aggregated metrics."""
    unknown = [metric for metric in metrics if metric not in METRIC_MEASURES]
    if unknown:
        raise ValueError(f"unknown metrics: {', '.join(unknown)}")
    factory = evaluator_factory or _default_evaluator_factory
    measures = {METRIC_MEASURES[metric] for metric in metrics}
    raw = factory(qrels, measures).evaluate(run)
    per_query: dict[str, dict[str, float]] = {}
    aggregated: dict[str, float] = {}
    for metric in metrics:
        measure = METRIC_MEASURES[metric]
        scores = {qid: float(values[measure]) for qid, values in raw.items()}
        per_query[metric] = scores
        aggregated[metric] = round(statistics.fmean(scores.values()), 4) if scores else 0.0
    return MetricScores(per_query=per_query, aggregated=aggregated)
