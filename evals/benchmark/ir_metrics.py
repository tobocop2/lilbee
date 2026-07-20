"""Tier-1 retrieval metrics: score a run file against qrels with pytrec_eval.

pytrec_eval is the TREC-standard scorer, so these numbers carry no model
opinion and are exactly reproducible from the run file and the qrels. It is an
optional dependency (a C extension), imported lazily; the per-query shaping and
aggregation are pulled out so they can be unit-tested with a fake evaluator and
no C extension installed.

Two properties this module is responsible for, both of which decide whether the
published numbers mean anything:

Depth. pytrec_eval's ``recip_rank`` is uncut -- it searches the whole run, so a
first relevant document at rank 11 contributes 1/11 to something labelled
"MRR@10". Runs are collected deeper than 10, so this is not hypothetical: on the
committed BEIR runs the uncut and cut-at-10 values differ by up to 0.004, the
same order as the deltas the study reports as findings. Metrics that do not cut
internally therefore declare an explicit ``depth`` and the run is truncated to
it before scoring.

Denominator. pytrec_eval only returns query ids present in the run, so a query
an arm returned nothing for disappears from the mean instead of scoring zero --
which rewards an arm for failing, and makes two arms that failed on different
queries non-comparable. Aggregation here is over the qrels topic set: an
unanswered topic scores 0.0 and is kept in ``per_query`` so downstream paired
statistics align both arms on the full topic set.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

Qrels = dict[str, dict[str, int]]
Run = dict[str, dict[str, float]]


@dataclass(frozen=True)
class MetricSpec:
    """A display metric: the pytrec_eval measure, and the depth to cut at.

    ``depth`` is None when the measure already cuts internally (ndcg_cut_10,
    recall_20) and an integer when the run must be truncated first.
    """

    measure: str
    depth: int | None = None


METRIC_SPECS: dict[str, MetricSpec] = {
    "nDCG@10": MetricSpec("ndcg_cut_10"),
    "Recall@20": MetricSpec("recall_20"),
    "MRR@10": MetricSpec("recip_rank", depth=10),
}

# Display name -> measure, kept for callers that only need the mapping.
METRIC_MEASURES: dict[str, str] = {
    name: spec.measure for name, spec in METRIC_SPECS.items()
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


def truncate_run(run: Run, depth: int) -> Run:
    """Keep each query's top ``depth`` documents, ties broken on doc_id ascending.

    The ordering key matches the one collapse_hits used to rank the run file, so
    truncating here selects the same documents the run file already ranked first.
    """
    return {
        query_id: dict(
            sorted(docs.items(), key=lambda item: (-item[1], item[0]))[:depth]
        )
        for query_id, docs in run.items()
    }


def score_run(
    qrels: Qrels,
    run: Run,
    metrics: list[str],
    *,
    evaluator_factory: EvaluatorFactory | None = None,
) -> MetricScores:
    """Score a run against qrels, returning per-query and mean-aggregated metrics.

    Metrics are grouped by the depth they need, so each distinct truncation costs
    one evaluator pass. Aggregation is over the qrels topic set: a topic the run
    has no entry for scores 0.0 rather than being dropped from the denominator.
    """
    unknown = [metric for metric in metrics if metric not in METRIC_SPECS]
    if unknown:
        raise ValueError(f"unknown metrics: {', '.join(unknown)}")
    factory = evaluator_factory or _default_evaluator_factory

    measures_by_depth: dict[int | None, set[str]] = defaultdict(set)
    for metric in metrics:
        spec = METRIC_SPECS[metric]
        measures_by_depth[spec.depth].add(spec.measure)

    raw: dict[str, dict[str, float]] = {}
    for depth, measures in measures_by_depth.items():
        scoped = run if depth is None else truncate_run(run, depth)
        for query_id, values in factory(qrels, measures).evaluate(scoped).items():
            raw.setdefault(query_id, {}).update(values)

    topics = sorted(qrels)
    per_query: dict[str, dict[str, float]] = {}
    aggregated: dict[str, float] = {}
    for metric in metrics:
        measure = METRIC_SPECS[metric].measure
        scores = {qid: float(raw.get(qid, {}).get(measure, 0.0)) for qid in topics}
        per_query[metric] = scores
        aggregated[metric] = round(statistics.fmean(scores.values()), 4) if scores else 0.0
    return MetricScores(per_query=per_query, aggregated=aggregated)
