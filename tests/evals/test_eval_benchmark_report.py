"""The benchmark report titles itself from the arms that actually ran."""

from evals.benchmark.report import render_report


def _rows(arm_a, arm_b):
    return [
        {
            "row_type": "meta",
            "run_id": "r",
            "fingerprint": "abc123def456",
            "arm_a": arm_a,
            "arm_b": arm_b,
        },
        {
            "row_type": "ir",
            "dataset": "scifact",
            "metric": "nDCG@10",
            "mean_a": 0.5,
            "mean_b": 0.6,
            "mean_diff": 0.1,
            "ci_low": 0.05,
            "ci_high": 0.15,
            "p_value": 0.01,
            "significant": True,
            "resamples": 10000,
            "p_at_floor": False,
        },
    ]


def test_title_names_the_arms_that_ran():
    # A single-system ablation must not be titled as a cross-system comparison.
    assert "# Retrieval benchmark: dense vs w1.0" in render_report(_rows("dense", "w1.0"))


def test_title_follows_a_cross_system_pairing_too():
    report = render_report(_rows("lilbee-parity", "ragflow-default"))
    assert "# Retrieval benchmark: lilbee-parity vs ragflow-default" in report
