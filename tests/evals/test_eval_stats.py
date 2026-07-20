"""Paired statistics: bootstrap CI, randomization p, and their determinism.

These exercise the seeded pure functions directly, so every number here is
reproducible and the significance rule is pinned to observable behaviour rather
than to the implementation.
"""

import pytest
from evals.benchmark import stats


def _const_diffs(value: float, n: int) -> tuple[dict[str, float], dict[str, float]]:
    """Two per-query maps whose differences are all exactly ``value``."""
    a = {f"q{i}": 0.0 for i in range(n)}
    b = {f"q{i}": value for i in range(n)}
    return a, b


def test_percentile_of_empty_is_zero():
    assert stats._percentile([], 0.5) == 0.0


def test_percentile_of_singleton_ignores_fraction():
    assert stats._percentile([7.0], 0.0) == 7.0
    assert stats._percentile([7.0], 1.0) == 7.0


def test_percentile_interpolates_linearly():
    assert stats._percentile([0.0, 10.0], 0.5) == pytest.approx(5.0)
    assert stats._percentile([0.0, 10.0, 20.0], 0.25) == pytest.approx(5.0)


def test_bootstrap_ci_of_a_constant_difference_is_that_constant():
    low, high = stats.paired_bootstrap_ci([0.3, 0.3, 0.3], resamples=200, seed=1, alpha=0.05)
    assert low == pytest.approx(0.3)
    assert high == pytest.approx(0.3)


def test_bootstrap_ci_of_no_diffs_is_zero_zero():
    assert stats.paired_bootstrap_ci([], resamples=100, seed=1, alpha=0.05) == (0.0, 0.0)


def test_bootstrap_ci_is_seed_reproducible():
    diffs = [0.1, -0.2, 0.4, 0.0, 0.3]
    first = stats.paired_bootstrap_ci(diffs, resamples=500, seed=7, alpha=0.05)
    again = stats.paired_bootstrap_ci(diffs, resamples=500, seed=7, alpha=0.05)
    assert first == again


def test_permutation_p_of_no_diffs_is_one():
    assert stats.permutation_test([], resamples=100, seed=1) == 1.0


def test_permutation_p_of_all_zero_diffs_is_one():
    # Every sign-flip leaves the mean at zero, so all resamples are as extreme.
    assert stats.permutation_test([0.0, 0.0, 0.0], resamples=100, seed=1) == 1.0


def test_permutation_p_is_small_for_a_large_consistent_effect():
    diffs = [0.5] * 20
    p = stats.permutation_test(diffs, resamples=2000, seed=3)
    assert p < 0.01


def test_permutation_p_never_undercuts_the_resample_floor():
    # The smallest reportable p is 1/(resamples+1); it can never be 0.
    p = stats.permutation_test([1.0] * 10, resamples=50, seed=1)
    assert p == pytest.approx(1 / 51)


def test_compare_reports_a_significant_positive_effect():
    a, b = _const_diffs(0.4, 30)
    result = stats.compare("nDCG@10", a, b, resamples=500, seed=2)
    assert result.n == 30
    assert result.mean_a == pytest.approx(0.0)
    assert result.mean_b == pytest.approx(0.4)
    assert result.mean_diff == pytest.approx(0.4)
    assert result.ci_low > 0.0
    assert result.significant is True


def test_compare_flags_a_ci_that_crosses_zero_as_not_significant():
    a = {"q1": 0.0, "q2": 1.0, "q3": 0.0, "q4": 1.0}
    b = {"q1": 1.0, "q2": 0.0, "q3": 1.0, "q4": 0.0}
    result = stats.compare("MRR@10", a, b, resamples=500, seed=5)
    assert result.mean_diff == pytest.approx(0.0)
    assert result.ci_low <= 0.0 <= result.ci_high
    assert result.significant is False


def test_compare_aligns_on_shared_query_ids_only():
    a = {"q1": 0.2, "q2": 0.5, "only_a": 0.9}
    b = {"q1": 0.4, "q2": 0.5, "only_b": 0.1}
    result = stats.compare("Recall@20", a, b, resamples=100, seed=1)
    assert result.n == 2


def test_compare_with_zero_overlap_reports_a_degenerate_null():
    # No shared queries: the current behaviour is an n=0 null. This documents
    # the shape the CLI must guard against, not a desirable result.
    result = stats.compare("MRR@10", {"a": 1.0}, {"b": 1.0}, resamples=100, seed=1)
    assert result.n == 0
    assert result.mean_diff == 0.0
    assert (result.ci_low, result.ci_high) == (0.0, 0.0)
    assert result.p_value == 1.0
    assert result.significant is False


def test_compare_result_round_trips_through_to_dict():
    a, b = _const_diffs(0.1, 5)
    result = stats.compare("nDCG@10", a, b, resamples=100, seed=1)
    payload = result.to_dict()
    assert payload["metric"] == "nDCG@10"
    assert set(payload) == {
        "metric",
        "n",
        "mean_a",
        "mean_b",
        "mean_diff",
        "ci_low",
        "ci_high",
        "p_value",
        "significant",
    }
