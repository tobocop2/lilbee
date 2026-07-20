"""Markdown rendering of a scored eval run."""

from __future__ import annotations

from typing import Any

from evals.benchmark.stats import DEFAULT_ALPHA, benjamini_hochberg
from evals.retrieval.judging import DIMENSIONS
from evals.retrieval.scoring import ResultRowType


def _adjusted_p(tests: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Family-adjusted p per dimension, across the dimensions actually tested."""
    names = [name for name in DIMENSIONS if name in tests]
    adjusted = benjamini_hochberg([float(tests[name]["p_value"]) for name in names])
    return dict(zip(names, adjusted, strict=True))


def _verdict(adjusted: float | None) -> str:
    if adjusted is None:
        return "not tested"
    return "significant" if adjusted <= DEFAULT_ALPHA else "n.s."


def render_report(rows: list[dict[str, Any]]) -> str:
    """The human-readable report for a results.jsonl produced by score."""
    summary = next((row for row in rows if row["row_type"] == ResultRowType.SUMMARY), None)
    if summary is None:
        raise ValueError("results contain no summary row; run score first")
    arms: dict[str, dict[str, Any]] = summary["arms"]
    first, second = list(arms)
    noise = summary["noise_floor"]

    graded = summary.get("judge_graded", {})
    scored = summary.get("scored", {})
    counts = " and ".join(f"{graded.get(arm, '?')} for {arm}" for arm in (first, second))
    mean_counts = " and ".join(f"{scored.get(arm, '?')} for {arm}" for arm in (first, second))
    judge_model = summary.get("judge_model") or "an unrecorded model"
    lines = [
        "# Retrieval eval report",
        "",
        f"Graded by {judge_model}. Of {summary.get('judgeable', '?')} judgeable "
        f"questions the judge returned a usable grade for {counts}; answers that "
        "failed outright and grades that came back unparseable are not in those "
        f"counts. Answers scored per arm: {mean_counts}, since a failed answer "
        "scores zero rather than being dropped. The per-dimension means below "
        f"are over the {summary.get('paired_questions', '?')} questions both arms "
        "have an outcome for, so the two means cover the same set rather than "
        "each arm averaging over whatever its judge happened to parse. "
        "Judges saw only question + ground truth + one answer; no arm labels.",
        f"Judge noise floor: plus or minus {noise} per dimension, measured over "
        f"{summary.get('noise_pairs', '?')} questions from {summary.get('noise_arm', '?')} "
        "graded twice under two equivalent phrasings of the grading prompt. That is a "
        "per-question disagreement: it describes how steady the judge is on one answer, "
        "and is not a threshold for a difference of means. Significance below comes "
        "from a paired test on the per-question grades, family-adjusted across the "
        "dimensions tested.",
        "",
        f"| dimension | {first} | {second} | delta | 95% CI | adj. p | verdict |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    tests = {test["metric"]: test for test in summary.get("dimension_tests", [])}
    adjusted_by_dimension = _adjusted_p(tests)
    for dimension in DIMENSIONS:
        first_mean = arms[first]["means"][dimension]
        second_mean = arms[second]["means"][dimension]
        test = tests.get(dimension)
        adjusted = adjusted_by_dimension.get(dimension)
        interval = f"[{test['ci_low']:+.3f}, {test['ci_high']:+.3f}]" if test is not None else "-"
        shown = f"{adjusted:.3f}" if adjusted is not None else "-"
        lines.append(
            f"| {dimension} (0-2) | {first_mean} | {second_mean} "
            f"| {round(second_mean - first_mean, 3):+g} | {interval} | {shown} "
            f"| {_verdict(adjusted)} |"
        )
    lines += [
        "",
        f"| exact-truth check | {first} | {second} |",
        "| --- | --- | --- |",
    ]
    for label, key in (
        ("count questions", "count_pass"),
        ("known-item citation", "known_item_pass"),
    ):
        cells = [f"{arms[arm][key][0]}/{arms[arm][key][1]}" for arm in (first, second)]
        lines.append(f"| {label} | {cells[0]} | {cells[1]} |")
    lines.append(f"| hard failures | {arms[first]['errors']} | {arms[second]['errors']} |")
    return "\n".join(lines) + "\n"
