"""Judge calibration against SummEval's existing expert ratings.

Whether Spearman is a correlation is scipy's business. What is tested here is
the harness' side: that the dataset is flattened to one row per rated summary,
that the two dimensions map to the ones a human actually rated, that a capped
run still covers every system, and that a correlation is refused when too few
pairs were graded to support one.
"""

import pytest
from evals.benchmark.calibration import (
    DIMENSION_MAP,
    EXPERT_AGREEMENT,
    MIN_CALIBRATION_PAIRS,
    CalibrationPair,
    InsufficientCalibrationError,
    calibrate,
)


def _pairs(n=MIN_CALIBRATION_PAIRS):
    return [
        CalibrationPair(
            pair_id=f"p{i}",
            ground="article",
            response=f"summary {i}",
            human={"faithfulness": float(i % 5 + 1), "relevance": float(i % 5 + 1)},
        )
        for i in range(n)
    ]


def test_only_dimensions_a_human_actually_rated_are_calibrated():
    # A summary cites nothing, so there is no human citation label here.
    # Inventing a mapping for it would be worse than reporting the gap.
    assert set(DIMENSION_MAP.values()) == {"faithfulness", "relevance"}
    assert "citation" not in DIMENSION_MAP.values()


def test_every_calibrated_dimension_has_a_published_ceiling():
    # A correlation with no ceiling beside it cannot be read as good or bad.
    assert set(EXPERT_AGREEMENT) == set(DIMENSION_MAP.values())


def test_a_judge_tracking_the_humans_is_reported_near_the_ceiling():
    pairs = _pairs()
    judge = {p.pair_id: {d: int(v) for d, v in p.human.items()} for p in pairs}
    for result in calibrate(judge, pairs):
        assert result.spearman > 0.9
        assert result.n == len(pairs)


def test_the_result_is_expressed_against_the_expert_ceiling():
    # The same raw correlation means different things against a 0.80 ceiling and
    # a 0.40 one, which is why the fraction travels with it.
    pairs = _pairs()
    judge = {p.pair_id: {d: int(v) for d, v in p.human.items()} for p in pairs}
    results = {r.dimension: r for r in calibrate(judge, pairs)}
    faith, rel = results["faithfulness"], results["relevance"]
    assert faith.spearman == pytest.approx(rel.spearman, abs=1e-9)
    # Identical correlation, different ceilings, so different standing.
    assert faith.fraction_of_ceiling < rel.fraction_of_ceiling


def test_too_few_graded_pairs_is_refused():
    pairs = _pairs(MIN_CALIBRATION_PAIRS - 1)
    judge = {p.pair_id: {d: int(v) for d, v in p.human.items()} for p in pairs}
    with pytest.raises(InsufficientCalibrationError, match="not reported below"):
        calibrate(judge, pairs)


def test_pairs_the_judge_never_graded_are_left_out_rather_than_defaulted():
    pairs = _pairs(MIN_CALIBRATION_PAIRS + 40)
    judge = {p.pair_id: {d: int(v) for d, v in p.human.items()} for p in pairs[:-40]}
    for result in calibrate(judge, pairs):
        assert result.n == MIN_CALIBRATION_PAIRS


def test_flattening_produces_one_row_per_rated_summary(monkeypatch):
    # The dataset nests sixteen summaries and sixteen score lists per article;
    # the judge grades one response at a time.
    from evals.benchmark import calibration

    fake = [
        {
            "id": "a1",
            "text": "article one",
            "machine_summaries": ["s0", "s1", "s2"],
            "consistency": [1.0, 3.0, 5.0],
            "relevance": [2.0, 4.0, 5.0],
        }
    ]

    class _Rows(list):
        def select(self, indices):
            return _Rows(self[i] for i in indices)

    monkeypatch.setattr(calibration, "load_dataset", lambda *a, **k: _Rows(fake), raising=False)
    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        type("m", (), {"load_dataset": lambda *a, **k: _Rows(fake)}),
    )
    flattened = calibration.load_summeval()
    assert len(flattened) == 3
    assert [p.response for p in flattened] == ["s0", "s1", "s2"]
    assert flattened[0].human == {"faithfulness": 1.0, "relevance": 2.0}
    assert flattened[2].human == {"faithfulness": 5.0, "relevance": 5.0}
