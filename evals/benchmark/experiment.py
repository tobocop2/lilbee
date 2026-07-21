"""Multi-arm retrieval comparison, computed by PyTerrier.

``pt.Experiment`` scores every arm against the qrels, runs a paired test of each
arm against the baseline, and returns one row per arm. It takes the same
``ir_measures`` measure objects the metric layer already uses, so adopting it
does not move a single number: the cut depth is still part of the measure and
the scorer is still trec_eval's.

Three things about it were verified against the source rather than the docs,
because each one would otherwise be a wrong assumption baked into published
numbers:

- It never starts a JVM on this path. ``pyterrier/_evaluation/_experiment.py``
  contains no Java reference, and ``pt.java.started()`` is False after a full
  Experiment call. PyTerrier is a Terrier binding, so the opposite would have
  put a JVM on the benchmark pod.
- It ships no confidence intervals. The only "bootstrap" in the package is
  Terrier's Java ``bootstrapInitialisation``, unrelated. The CI is the effect
  size this study reports, so it stays here on scipy.
- Its ``correction=`` corrects across *systems within one metric*
  (``_rendering.py`` loops per p-value column). This study's family is every
  comparison it publishes, across metrics and datasets both, which is strictly
  larger. Correction is therefore left off here and applied by the caller over
  the whole family.

The default test is a paired t-test. ``test=`` accepts any callable matching
``TEST_FN_TYPE``, so the randomization test Smucker, Allan and Carterette
recommend for IR is passed in instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.deps import install_hint

PYTERRIER_INSTALL_HINT = install_hint("python-terrier", "to compare retrieval arms")

# Randomization-test resamples. Distinct from the bootstrap's, since the two
# procedures answer different questions and are reported side by side.
DEFAULT_RESAMPLES = 10000


def randomization_test(resamples: int, seed: int) -> Any:
    """A paired sign-flip randomization test shaped as PyTerrier's ``test=`` callable.

    PyTerrier types the hook as ``Callable[[Sequence, Sequence], Tuple[Any, float]]``:
    it hands over the baseline's and the arm's per-query scores and wants
    ``(statistic, p_value)``. That is the same paired vector ``stats.compare``
    works on, so both report the same test rather than two different ones.
    """

    def test(baseline_scores: Any, arm_scores: Any) -> tuple[float, float]:
        import numpy as np
        from scipy import stats as scipy_stats

        diffs = np.asarray(arm_scores, dtype=float) - np.asarray(baseline_scores, dtype=float)
        # An all-zero difference vector has nothing to permute, and scipy
        # declines the degenerate case; the arms are identical on these queries.
        if not diffs.any():
            return 0.0, 1.0
        result = scipy_stats.permutation_test(
            (diffs,),
            lambda x, axis: np.mean(x, axis=axis),
            permutation_type="samples",
            n_resamples=resamples,
            alternative="two-sided",
            random_state=seed,
        )
        return float(result.statistic), float(result.pvalue)

    return test


def run_to_frame(run: dict[str, dict[str, float]]) -> Any:
    """Shape a run map as the ``qid``/``docno``/``score``/``rank`` frame PyTerrier takes."""
    import pandas as pd

    rows = [
        {"qid": query_id, "docno": doc_id, "score": float(score), "rank": rank}
        for query_id, scored in run.items()
        for rank, (doc_id, score) in enumerate(
            sorted(scored.items(), key=lambda item: (item[1], item[0]), reverse=True)
        )
    ]
    return pd.DataFrame(rows, columns=["qid", "docno", "score", "rank"])


def qrels_to_frame(qrels: dict[str, dict[str, int]]) -> Any:
    """Shape a qrels map as the ``qid``/``docno``/``label`` frame PyTerrier takes."""
    import pandas as pd

    rows = [
        {"qid": query_id, "docno": doc_id, "label": int(grade)}
        for query_id, judged in qrels.items()
        for doc_id, grade in judged.items()
    ]
    return pd.DataFrame(rows, columns=["qid", "docno", "label"])


def topics_frame(qrels: dict[str, dict[str, int]]) -> Any:
    """The topic set every arm is scored over: the qrels' queries, not the runs'.

    Taking it from the qrels is what makes an unanswered query score zero rather
    than leave the denominator, which is the same rule the metric layer applies.
    """
    import pandas as pd

    return pd.DataFrame([{"qid": query_id, "query": ""} for query_id in sorted(qrels)])


def compare_arms(
    runs: dict[str, dict[str, dict[str, float]]],
    qrels: dict[str, dict[str, int]],
    metrics: list[str],
    *,
    baseline: str,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 20260714,
) -> Any:
    """Score every arm against ``baseline``; return the aggregated frame.

    Aggregated only, because PyTerrier asserts ``not perquery`` whenever a
    baseline is given: one call yields the significance table or the per-query
    vectors, never both. The per-query vectors this study also needs (for the
    bootstrap CI and the family-wide correction) come from ``metrics.score_run``,
    which is the same ir_measures computation over the same measures, so the two
    cannot disagree.

    Correction is deliberately not requested here. See the module docstring: this
    study's family spans metrics and datasets, and PyTerrier's spans only the
    systems in one call.
    """
    try:
        import pyterrier as pt
    except ImportError as exc:
        raise RuntimeError(PYTERRIER_INSTALL_HINT) from exc
    from evals.benchmark.metrics import METRIC_MEASURES

    if baseline not in runs:
        raise ValueError(
            f"baseline '{baseline}' is not among the arms {sorted(runs)}; every "
            "comparison in this study is against a declared baseline"
        )
    unknown = [metric for metric in metrics if metric not in METRIC_MEASURES]
    if unknown:
        raise ValueError(f"unknown metrics: {', '.join(sorted(unknown))}")

    import ir_measures

    measures = [ir_measures.parse_measure(METRIC_MEASURES[name]) for name in metrics]
    # Baseline first, so PyTerrier's baseline=0 names the arm the caller meant.
    names = [baseline] + sorted(name for name in runs if name != baseline)
    frames = [run_to_frame(runs[name]) for name in names]
    frame = pt.Experiment(
        frames,
        topics_frame(qrels),
        qrels_to_frame(qrels),
        eval_metrics=measures,
        names=names,
        baseline=0,
        test=randomization_test(resamples, seed),
        correction=None,
    )
    # PyTerrier names its columns after the measure ("R@20"); the manifest,
    # the metrics module and the report all speak the display name
    # ("Recall@20"). Renaming here keeps one convention in the artifacts rather
    # than two that a reader has to reconcile.
    display = {str(measure): name for name, measure in zip(metrics, measures, strict=True)}
    return frame.rename(
        columns={
            column: column.replace(measure, display[measure], 1)
            for column in frame.columns
            for measure in display
            if column.startswith(measure)
        }
    )


def load_runs(run_paths: dict[str, Path]) -> dict[str, dict[str, dict[str, float]]]:
    """Read one TREC run file per arm."""
    from evals.benchmark.runfile import read_run

    return {name: read_run(path) for name, path in run_paths.items()}
