"""Paired statistics on per-query metric vectors, computed by scipy.

Two arms are scored on the same queries, so every metric difference is paired.
scipy supplies all three procedures and none of them is reimplemented here:

- ``scipy.stats.bootstrap(method="percentile")`` for the 95% CI on the mean
  difference
- ``scipy.stats.permutation_test(permutation_type="samples")`` for the paired
  sign-flip randomization test, which is the test Smucker, Allan and Carterette
  (CIKM 2007) recommend for IR evaluation
- ``scipy.stats.false_discovery_control(method="bh")`` for Benjamini-Hochberg

Everything is seeded, so a result is reproducible run to run.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import Any

from evals.deps import install_hint

DEFAULT_RESAMPLES = 10000
DEFAULT_SEED = 20260714
DEFAULT_ALPHA = 0.05

SCIPY_INSTALL_HINT = install_hint("scipy", "for benchmark statistics")


@dataclass(frozen=True)
class PairedResult:
    """The paired comparison of arm B minus arm A on one metric.

    ``significant`` here is the single-test verdict from the bootstrap CI. It is
    NOT the verdict to publish when more than one comparison was run: a study
    that tests four arms on three datasets across three metrics runs 36 tests,
    and selecting the best of them inflates the type-I rate well past alpha. Feed
    the whole family through ``benjamini_hochberg`` and decide on the adjusted p.
    """

    metric: str
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool
    resamples: int = DEFAULT_RESAMPLES
    bootstrap_seed: int = DEFAULT_SEED
    permutation_seed: int = DEFAULT_SEED + 1

    @property
    def p_floor(self) -> float:
        """The smallest p-value this many resamples can express, two-sided.

        Two-sided, so both tails count and the floor is 2/(resamples+1) rather
        than 1/(resamples+1). The old hand-rolled test only ever counted the
        tail it observed, which understated the floor by a factor of two and so
        reported a bound as though it were a measurement.
        """
        return 2.0 / (self.resamples + 1)

    @property
    def p_at_floor(self) -> bool:
        """True when p sits at the smallest value the resampling can express.

        At the floor the p-value is a bound, not a measurement: it should be
        rendered as "< p_floor" rather than quoted as a point estimate.
        """
        return self.p_value <= self.p_floor

    def to_dict(self) -> dict[str, Any]:
        # p_floor travels with the row so renderers quote the bound rather than
        # recomputing the formula and drifting from it.
        return {**asdict(self), "p_at_floor": self.p_at_floor, "p_floor": self.p_floor}


def _scipy_stats() -> Any:
    try:
        from scipy import stats
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(SCIPY_INSTALL_HINT) from exc
    return stats


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, in the input's order.

    Controls the false-discovery rate across a family of comparisons. Without
    it, reporting the best of N correlated arms at its raw p-value claims a
    confidence the study did not earn.
    """
    if not p_values:
        return []
    adjusted = _scipy_stats().false_discovery_control(p_values, method="bh")
    return [float(value) for value in adjusted]


def paired_bootstrap_ci(
    diffs: list[float], resamples: int, seed: int, alpha: float = DEFAULT_ALPHA
) -> tuple[float, float]:
    """Percentile CI of the mean paired difference from a seeded bootstrap."""
    if not diffs:
        return 0.0, 0.0
    # A constant difference vector has no spread to resample, and scipy declines
    # the degenerate estimate; the interval is the point itself.
    if len(set(diffs)) == 1:
        return diffs[0], diffs[0]
    import numpy as np

    interval = (
        _scipy_stats()
        .bootstrap(
            (np.asarray(diffs, dtype=float),),
            np.mean,
            n_resamples=resamples,
            method="percentile",
            confidence_level=1.0 - alpha,
            random_state=seed,
        )
        .confidence_interval
    )
    return float(interval.low), float(interval.high)


def permutation_test(diffs: list[float], resamples: int, seed: int) -> float:
    """Two-sided paired sign-flip randomization p-value for a nonzero mean."""
    if not diffs or all(diff == 0.0 for diff in diffs):
        return 1.0
    import numpy as np

    result = _scipy_stats().permutation_test(
        (np.asarray(diffs, dtype=float),),
        lambda x, axis: np.mean(x, axis=axis),
        permutation_type="samples",
        n_resamples=resamples,
        alternative="two-sided",
        random_state=seed,
    )
    return float(result.pvalue)


def _paired_diffs(
    a_scores: dict[str, float], b_scores: dict[str, float]
) -> tuple[list[float], list[float], list[float]]:
    """Align two per-query score maps on shared query ids, sorted for determinism."""
    shared = sorted(qid for qid in a_scores if qid in b_scores)
    a_vec = [a_scores[qid] for qid in shared]
    b_vec = [b_scores[qid] for qid in shared]
    diffs = [b - a for a, b in zip(a_vec, b_vec, strict=True)]
    return a_vec, b_vec, diffs


def compare(
    metric: str,
    a_scores: dict[str, float],
    b_scores: dict[str, float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> PairedResult:
    """Full paired comparison of arm B minus arm A on one metric's per-query scores."""
    a_vec, b_vec, diffs = _paired_diffs(a_scores, b_scores)
    # Distinct sub-seeds: driving both procedures off one stream makes the CI and
    # the p-value the same draws twice, which defeats reporting them as
    # corroborating evidence. Both stay reproducible.
    bootstrap_seed, permutation_seed = seed, seed + 1
    ci_low, ci_high = paired_bootstrap_ci(diffs, resamples, bootstrap_seed, alpha)
    p_value = permutation_test(diffs, resamples, permutation_seed)
    mean_diff = statistics.fmean(diffs) if diffs else 0.0
    return PairedResult(
        metric=metric,
        n=len(diffs),
        mean_a=statistics.fmean(a_vec) if a_vec else 0.0,
        mean_b=statistics.fmean(b_vec) if b_vec else 0.0,
        mean_diff=mean_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        significant=not (ci_low <= 0.0 <= ci_high),
        resamples=resamples,
        bootstrap_seed=bootstrap_seed,
        permutation_seed=permutation_seed,
    )
