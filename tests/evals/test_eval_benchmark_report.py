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


def _audit_row(**over):
    row = {
        "row_type": "calibration",
        "dimension": "faithfulness",
        "n": 1600,
        "spearman": 0.61,
        "kendall": 0.52,
        "expert_ceiling": 0.798,
        "fraction_of_ceiling": 0.7644,
    }
    return {**row, **over}


def _versions_row(**over):
    row = {
        "row_type": "versions",
        "judge_model": "fable-5",
        "judge_base_url": "http://judge:8081/v1",
        "scorers": {"ragas": "0.4.3", "ir_measures": "0.4.3"},
    }
    return {**row, **over}


def test_the_report_states_the_judges_agreement_with_humans():
    # The answer tier otherwise rests entirely on one model's opinion of
    # another model's output, with no stated error rate.
    report = render_report([*_rows("a", "b"), _audit_row()])
    assert "Judge agreement with human raters" in report
    assert "+0.610" in report
    # The ceiling has to travel with it: the same correlation means different
    # things against a 0.80 ceiling and a 0.40 one.
    assert "0.798" in report
    assert "76%" in report


def test_the_report_names_the_judge_and_the_scorer_versions():
    # ragas metric prompts move between releases, so a number is only
    # reproducible if the report says which release produced it.
    report = render_report([*_rows("a", "b"), _versions_row()])
    assert "fable-5" in report
    assert "| ragas | 0.4.3 |" in report


def test_a_run_without_an_audit_omits_the_section_rather_than_faking_it():
    # An absent audit must read as absent, not as an agreement of zero.
    report = render_report(_rows("a", "b"))
    assert "Judge agreement" not in report
    assert "What produced these numbers" not in report
