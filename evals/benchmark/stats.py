"""Paired statistics on per-query metric vectors.

Two arms are scored on the same queries, so every metric difference is paired.
A paired bootstrap gives a 95% confidence interval for the mean difference and
a sign-flip randomization test gives a p-value. Both are seeded and pure, so a
result is bit-for-bit reproducible. A difference whose CI crosses zero is
flagged not significant.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import asdict, dataclass
from typing import Any

DEFAULT_RESAMPLES = 10000
DEFAULT_SEED = 20260714
DEFAULT_ALPHA = 0.05
# Probability of flipping a paired difference's sign in the randomization test.
SIGN_FLIP_PROBABILITY = 0.5


@dataclass(frozen=True)
class PairedResult:
    """The paired comparison of arm B minus arm A on one metric."""

    metric: str
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Linear-interpolated percentile of an already-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def paired_bootstrap_ci(
    diffs: list[float], resamples: int, seed: int, alpha: float
) -> tuple[float, float]:
    """Percentile CI of the mean paired difference from a seeded bootstrap."""
    if not diffs:
        return 0.0, 0.0
    rng = random.Random(seed)
    count = len(diffs)
    means: list[float] = []
    for _ in range(resamples):
        resample = [diffs[rng.randrange(count)] for _ in range(count)]
        means.append(statistics.fmean(resample))
    means.sort()
    return _percentile(means, alpha / 2), _percentile(means, 1 - alpha / 2)


def permutation_test(diffs: list[float], resamples: int, seed: int) -> float:
    """Two-sided sign-flip randomization p-value for a nonzero mean difference."""
    if not diffs:
        return 1.0
    rng = random.Random(seed)
    observed = abs(statistics.fmean(diffs))
    at_least_as_extreme = 0
    for _ in range(resamples):
        flipped = [d if rng.random() < SIGN_FLIP_PROBABILITY else -d for d in diffs]
        if abs(statistics.fmean(flipped)) >= observed:
            at_least_as_extreme += 1
    return (at_least_as_extreme + 1) / (resamples + 1)


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
    ci_low, ci_high = paired_bootstrap_ci(diffs, resamples, seed, alpha)
    p_value = permutation_test(diffs, resamples, seed)
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
    )
