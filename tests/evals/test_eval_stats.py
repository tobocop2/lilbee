"""Paired statistics: bootstrap CI, randomization p, and their determinism.

scipy owns the three procedures, so these do not re-verify its arithmetic. What
they pin is this module's contract on top of it: query alignment, the two
independent seed streams, the resampling floor, and the significance rule.
"""

import pytest
from evals.benchmark import stats


def _const_diffs(value: float, n: int) -> tuple[dict[str, float], dict[str, float]]:
    """Two per-query maps whose differences are all exactly ``value``."""
    a = {f"q{i}": 0.0 for i in range(n)}
    b = {f"q{i}": value for i in range(n)}
    return a, b


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
    # Two-sided: both tails count, so the floor is 2/(resamples+1).
    assert p == pytest.approx(2 / 51)


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
        "resamples",
        "bootstrap_seed",
        "permutation_seed",
        "p_at_floor",
        "p_floor",
    }


def test_bootstrap_and_permutation_do_not_share_a_random_stream():
    # Both procedures are reported as corroborating evidence, so they must not
    # be driven off the same underlying draws.
    a, b = _const_diffs(0.2, 12)
    result = stats.compare("nDCG@10", a, b, resamples=200, seed=11)
    assert result.bootstrap_seed != result.permutation_seed


def test_p_value_floor_is_exposed_rather_than_read_as_an_exact_value():
    # 1/(resamples+1) is the smallest attainable p; reporting it as a point
    # estimate overstates precision.
    a, b = _const_diffs(1.0, 10)
    result = stats.compare("MRR@10", a, b, resamples=50, seed=1)
    assert result.p_value == pytest.approx(2 / 51)
    assert result.p_floor == pytest.approx(2 / 51)
    assert result.p_at_floor is True
    assert result.resamples == 50


def test_a_middling_p_is_not_flagged_as_the_floor():
    a = {"q1": 0.0, "q2": 1.0, "q3": 0.0, "q4": 1.0}
    b = {"q1": 1.0, "q2": 0.0, "q3": 1.0, "q4": 0.0}
    result = stats.compare("MRR@10", a, b, resamples=200, seed=3)
    assert result.p_at_floor is False


def test_benjamini_hochberg_leaves_a_single_test_alone():
    assert stats.benjamini_hochberg([0.04]) == [pytest.approx(0.04)]


def test_benjamini_hochberg_preserves_input_order():
    adjusted = stats.benjamini_hochberg([0.5, 0.01, 0.2])
    assert adjusted[1] < adjusted[2] < adjusted[0]


def test_benjamini_hochberg_raises_borderline_p_above_alpha():
    # The committed best-of-four claim: p=0.012 selected from a family of 36.
    family = [0.012] + [0.4] * 35
    adjusted = stats.benjamini_hochberg(family)
    assert adjusted[0] > 0.05


def test_benjamini_hochberg_keeps_a_strong_effect_significant():
    family = [0.00001] + [0.4] * 35
    assert stats.benjamini_hochberg(family)[0] < 0.05


def test_benjamini_hochberg_is_monotone():
    adjusted = stats.benjamini_hochberg([0.01, 0.02, 0.03, 0.04])
    assert adjusted == sorted(adjusted)


def test_benjamini_hochberg_caps_at_one():
    assert all(value <= 1.0 for value in stats.benjamini_hochberg([0.9, 0.95, 0.99]))


def test_empty_family_adjusts_to_nothing():
    assert stats.benjamini_hochberg([]) == []
