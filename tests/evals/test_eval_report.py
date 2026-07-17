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


def test_render_report_flags_deltas_within_the_noise_floor():
    report = render_report([_summary()])
    assert "| faithfulness (0-2) | 1.5 | 1.55 | +0.05 (within noise) |" in report
    assert "| relevance (0-2) | 1.8 | 1.2 | -0.6 |" in report


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
