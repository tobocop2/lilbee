"""Markdown report rendering from results rows."""

from evals.retrieval.report import render_report
from evals.retrieval.scoring import ResultRowType


def _summary(noise: float = 0.1) -> dict:
    return {
        "row_type": ResultRowType.SUMMARY,
        "noise_floor": noise,
        "judged": 4,
        "arms": {
            "A": {
                "means": {"faithfulness": 1.5, "relevance": 1.8, "citation": 1.0},
                "count_pass": [1, 2],
                "known_item_pass": [2, 2],
                "errors": 0,
            },
            "B": {
                "means": {"faithfulness": 1.55, "relevance": 1.2, "citation": 1.0},
                "count_pass": [2, 2],
                "known_item_pass": [1, 2],
                "errors": 1,
            },
        },
    }


def _test_row(metric, p_value, ci=(-0.1, 0.1)):
    return {
        "metric": metric,
        "n": 40,
        "mean_a": 0.0,
        "mean_b": 0.0,
        "mean_diff": 0.0,
        "ci_low": ci[0],
        "ci_high": ci[1],
        "p_value": p_value,
        "significant": False,
        "resamples": 10000,
        "p_at_floor": False,
    }


def test_render_report_shows_the_delta_and_its_paired_verdict():
    report = render_report([_summary()])
    # No paired test supplied: the report says so rather than implying a verdict.
    assert "| faithfulness (0-2) | 1.5 | 1.55 | +0.05 | - | - | not tested |" in report
    assert "| relevance (0-2) | 1.8 | 1.2 | -0.6 | - | - | not tested |" in report


def test_render_report_decides_significance_on_the_adjusted_p():
    summary = _summary()
    summary["dimension_tests"] = [
        _test_row("faithfulness", 0.9),
        _test_row("relevance", 0.0001, ci=(-0.9, -0.3)),
        _test_row("citation", 0.8),
    ]
    report = render_report([summary])
    relevance = next(line for line in report.splitlines() if line.startswith("| relevance"))
    assert "significant" in relevance
    faithfulness = next(line for line in report.splitlines() if line.startswith("| faithfulness"))
    assert "n.s." in faithfulness


def test_a_lone_borderline_p_does_not_survive_the_family_adjustment():
    summary = _summary()
    # p=0.04 alongside two nulls: adjusted across three dimensions it fails.
    summary["dimension_tests"] = [
        _test_row("faithfulness", 0.04),
        _test_row("relevance", 0.7),
        _test_row("citation", 0.9),
    ]
    report = render_report([summary])
    faithfulness = next(line for line in report.splitlines() if line.startswith("| faithfulness"))
    assert "n.s." in faithfulness


def test_report_states_the_noise_floor_is_not_a_significance_threshold():
    report = render_report([_summary()])
    assert "not a threshold for a difference of means" in report


def test_render_report_includes_exact_truth_and_failures():
    report = render_report([_summary()])
    assert "count questions | 1/2 | 2/2" in report
    assert "known-item citation | 2/2 | 1/2" in report
    assert "hard failures | 0 | 1" in report
    assert "0.1" in report


def test_render_report_requires_a_summary_row():
    try:
        render_report([{"row_type": ResultRowType.QUESTION}])
    except ValueError as exc:
        assert "summary" in str(exc)
    else:
        raise AssertionError("expected ValueError")
