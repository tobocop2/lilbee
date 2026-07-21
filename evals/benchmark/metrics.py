"""Tier-1 retrieval metrics, computed by ir_measures.

``ir_measures`` is the standard interface over trec_eval/pytrec_eval. It owns
everything that used to live here:

- **Cut depth.** ``RR@10`` is reciprocal rank cut at 10. Nothing truncates a run
  or reasons about tie-breaking. The hand-rolled layer that did selected each
  query's top N and rescored the survivors, which is a different computation
  from scoring the full run under a cutoff; it disagreed with the reference on
  99 of FiQA's 648 topics and published MRR@10 0.4729 against the true 0.4700.
- **Denominator.** A topic in the qrels that the run never answered scores 0
  rather than vanishing from the mean, so an arm cannot be rewarded for
  returning nothing on its hard queries.

trec_eval semantics specifically, not merely "some IR library": published BEIR
numbers are trec_eval numbers, and a benchmark scored under another convention
is not comparable to the baselines a reader will check it against. ranx, for
instance, cuts depth correctly but ships its own nDCG variant and scores the
same FiQA run at 0.4060 where trec_eval gives 0.4033.
"""

from __future__ import annotations

from typing import Any

from evals.deps import install_hint

Qrels = dict[str, dict[str, int]]
Run = dict[str, dict[str, float]]

# Display name -> ir_measures measure string. The right-hand side is the entire
# depth contract; there is no second place a depth can drift out of step.
METRIC_MEASURES: dict[str, str] = {
    "nDCG@10": "nDCG@10",
    "Recall@20": "R@20",
    "MRR@10": "RR@10",
}

IR_MEASURES_INSTALL_HINT = install_hint("ir_measures", "to score retrieval")


def score_run(qrels: Qrels, run: Run, metrics: list[str]) -> dict[str, Any]:
    """Per-query and aggregated scores for the requested display metrics."""
    unknown = [metric for metric in metrics if metric not in METRIC_MEASURES]
    if unknown:
        raise ValueError(f"unknown metrics: {', '.join(unknown)}")
    try:
        import ir_measures
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(IR_MEASURES_INSTALL_HINT) from exc

    parsed = {name: ir_measures.parse_measure(METRIC_MEASURES[name]) for name in metrics}
    measures = list(parsed.values())
    aggregate = ir_measures.calc_aggregate(measures, qrels, run)

    per_measure: dict[str, dict[str, float]] = {str(m): {} for m in measures}
    for result in ir_measures.iter_calc(measures, qrels, run):
        per_measure[str(result.measure)][result.query_id] = result.value

    topics = sorted(qrels)
    return {
        "per_query": {
            name: {qid: float(per_measure[str(m)].get(qid, 0.0)) for qid in topics}
            for name, m in parsed.items()
        },
        "aggregated": {name: round(float(aggregate[m]), 4) for name, m in parsed.items()},
    }
