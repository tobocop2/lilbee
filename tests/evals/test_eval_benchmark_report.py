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
        _ir_row("scifact", "nDCG@10", arm_a, arm_b),
    ]


def _ir_row(dataset, metric, arm_a, arm_b, mean_a=0.5, mean_b=0.6):
    return {
        "row_type": "ir",
        "dataset": dataset,
        "metric": metric,
        "arm_a": arm_a,
        "arm_b": arm_b,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_diff": round(mean_b - mean_a, 4),
        "ci_low": 0.05,
        "ci_high": 0.15,
        "p_value": 0.01,
        "ci_excludes_zero": True,
        "resamples": 10000,
        "p_at_floor": False,
    }


def test_title_names_the_arms_that_ran():
    # The title must name the two configs that ran, not a fixed label.
    report = render_report(_rows("dense", "w1.0"))
    assert "# Retrieval benchmark: dense vs w1.0" in report
    # Single comparison: the arms are the value-column headers.
    assert "| dense" in report and "| w1.0" in report


def test_title_follows_a_different_pairing_too():
    report = render_report(_rows("lilbee-full", "lilbee-baseline"))
    assert "# Retrieval benchmark: lilbee-full vs lilbee-baseline" in report


def test_a_multi_arm_study_labels_each_comparison_by_its_own_arms():
    # An ablation file holds several comparisons that do not share one arm pair.
    # Every row must be labelled with the arms that produced it, never the first
    # meta row's arms, and the title cannot claim a single pairing.
    rows = [
        {
            "row_type": "meta",
            "run_id": "ablation",
            "fingerprint": "abc123def456",
            "arm_a": "dense",
            "arm_b": "w0.5",
        },
        _ir_row("scifact", "nDCG@10", "dense", "w0.5", mean_a=0.50, mean_b=0.55),
        _ir_row("scifact", "nDCG@10", "dense", "w1.0", mean_a=0.50, mean_b=0.61),
    ]
    report = render_report(rows)
    assert "# Retrieval benchmark: ablation (2 comparisons)" in report
    # Both comparisons are named in the table; neither is silently relabelled.
    assert "dense vs w0.5" in report
    assert "dense vs w1.0" in report
    # The w1.0 row's own mean (0.61) must appear, not be dropped under w0.5.
    assert "0.6100" in report


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
    # pandas pads the cells, so assert on content rather than spacing.
    assert "ragas" in report and "0.4.3" in report


def test_a_run_without_an_audit_omits_the_section_rather_than_faking_it():
    # An absent audit must read as absent, not as an agreement of zero.
    report = render_report(_rows("a", "b"))
    assert "Judge agreement" not in report
    assert "What produced these numbers" not in report
