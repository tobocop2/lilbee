"""Deterministic paired statistics on known small inputs."""

from evals.benchmark.stats import _percentile, compare, paired_bootstrap_ci, permutation_test


def test_percentile_interpolates_between_neighbors():
    values = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert _percentile(values, 0.0) == 0.0
    assert _percentile(values, 1.0) == 4.0
    assert _percentile(values, 0.5) == 2.0


def test_percentile_handles_short_lists():
    assert _percentile([], 0.5) == 0.0
    assert _percentile([7.0], 0.9) == 7.0


def test_bootstrap_ci_is_seed_reproducible_and_positive_for_a_clear_gap():
    diffs = [1.0] * 20
    first = paired_bootstrap_ci(diffs, resamples=200, seed=1, alpha=0.05)
    second = paired_bootstrap_ci(diffs, resamples=200, seed=1, alpha=0.05)
    assert first == second  # identical seed -> identical CI
    assert first == (1.0, 1.0)  # every resample of constant 1.0 has mean 1.0


def test_permutation_p_is_one_when_there_is_no_difference():
    assert permutation_test([0.0, 0.0, 0.0], resamples=100, seed=3) == 1.0


def test_permutation_p_is_small_for_a_consistent_gap():
    p = permutation_test([1.0] * 12, resamples=500, seed=3)
    assert p < 0.05


def test_compare_flags_a_clear_gap_significant():
    a = {f"q{i}": 0.0 for i in range(15)}
    b = {f"q{i}": 1.0 for i in range(15)}
    result = compare("nDCG@10", a, b, resamples=300, seed=7)
    assert result.n == 15
    assert result.mean_diff == 1.0
    assert result.mean_a == 0.0
    assert result.mean_b == 1.0
    assert result.significant is True
    assert result.p_value < 0.05


def test_compare_flags_a_crossing_ci_not_significant():
    a = {"q0": 0.0, "q1": 1.0, "q2": 0.0, "q3": 1.0}
    b = {"q0": 1.0, "q1": 0.0, "q2": 1.0, "q3": 0.0}  # symmetric, mean diff 0
    result = compare("nDCG@10", a, b, resamples=300, seed=7)
    assert result.mean_diff == 0.0
    assert result.significant is False
    assert result.ci_low <= 0.0 <= result.ci_high


def test_compare_pairs_only_shared_query_ids():
    a = {"q0": 0.2, "q1": 0.4, "only_a": 1.0}
    b = {"q0": 0.5, "q1": 0.9, "only_b": 0.0}
    result = compare("Recall@20", a, b, resamples=100, seed=1)
    assert result.n == 2
