"""Markdown rendering of Tier-1, Tier-2, and the coverage matrix."""

import pytest

from evals.benchmark.report import render_report


def _ir_row(metric, mean_a, mean_b, ci_low, ci_high, significant, p=0.01):
    return {
        "row_type": "ir",
        "dataset": "scifact",
        "metric": metric,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_diff": round(mean_b - mean_a, 4),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p,
        "significant": significant,
    }


def test_report_leads_with_tier1_table_and_marks_significance():
    rows = [
        {
            "row_type": "meta",
            "run_id": "r1",
            "fingerprint": "abcdef123456",
            "arm_a": "lilbee",
            "arm_b": "ragflow",
        },
        _ir_row("nDCG@10", 0.60, 0.70, 0.05, 0.15, True),
        _ir_row("Recall@20", 0.80, 0.79, -0.05, 0.03, False),
    ]
    report = render_report(rows)
    assert report.index("Tier 1") < report.index("Tier 2") if "Tier 2" in report else True
    assert "| scifact | nDCG@10 | 0.6000 | 0.7000 | +0.1000 |" in report
    assert "(n.s.)" in report  # the crossing-CI row is flagged
    assert "Arm A = lilbee, arm B = ragflow" in report


def test_report_renders_ragas_and_judge_noise():
    rows = [
        _ir_row("nDCG@10", 0.6, 0.7, 0.05, 0.15, True),
        {"row_type": "ragas", "metric": "faithfulness", "arm_a": 0.9, "arm_b": 0.85},
        {"row_type": "judge", "noise_floor": 0.12, "means": {}},
    ]
    report = render_report(rows)
    assert "Tier 2 - answer quality (RAGAS)" in report
    assert "| faithfulness | 0.9000 | 0.8500 |" in report
    assert "0.12 per dimension" in report


def test_report_marks_derived_qrel_datasets_in_coverage():
    rows = [
        _ir_row("nDCG@10", 0.6, 0.7, 0.05, 0.15, True),
        {
            "row_type": "coverage",
            "feature": "table extraction",
            "dataset": "tat-dqa",
            "metric": "nDCG@10",
            "derived": True,
        },
    ]
    report = render_report(rows)
    assert "| table extraction | tat-dqa | nDCG@10 | derived |" in report


def test_report_requires_scored_rows():
    with pytest.raises(ValueError, match="no ir or ragas rows"):
        render_report([{"row_type": "meta", "arm_a": "a", "arm_b": "b"}])
