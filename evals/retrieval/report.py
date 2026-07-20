"""Markdown rendering of a scored eval run."""

from __future__ import annotations

from typing import Any

from evals.retrieval.judging import DIMENSIONS
from evals.retrieval.scoring import ResultRowType


def _delta_cell(first: float, second: float, noise: float) -> str:
    delta = round(second - first, 3)
    flag = "" if abs(delta) > noise else " (within noise)"
    return f"{delta:+g}{flag}"


def render_report(rows: list[dict[str, Any]]) -> str:
    """The human-readable report for a results.jsonl produced by score."""
    summary = next((row for row in rows if row["row_type"] == ResultRowType.SUMMARY), None)
    if summary is None:
        raise ValueError("results contain no summary row; run score first")
    arms: dict[str, dict[str, Any]] = summary["arms"]
    first, second = list(arms)
    noise = summary["noise_floor"]

    graded = summary.get("judge_graded", {})
    counts = " and ".join(f"{graded.get(arm, '?')} for {arm}" for arm in (first, second))
    judge_model = summary.get("judge_model") or "an unrecorded model"
    lines = [
        "# Retrieval eval report",
        "",
        f"Graded by {judge_model}. The judge returned grades for {counts}, out of "
        f"{summary.get('judgeable', '?')} judgeable questions; answers that failed "
        "outright and grades the judge returned unparseable are not in those counts. "
        "Judges saw only question + ground truth + one answer; no arm labels.",
        f"Judge noise floor: plus or minus {noise} per dimension, measured over "
        f"{summary.get('noise_pairs', '?')} questions from {summary.get('noise_arm', '?')} "
        "graded twice under two equivalent phrasings of the grading prompt. Deltas at "
        "or below it are labeled within noise.",
        "",
        f"| dimension | {first} | {second} | delta |",
        "| --- | --- | --- | --- |",
    ]
    for dimension in DIMENSIONS:
        first_mean = arms[first]["means"][dimension]
        second_mean = arms[second]["means"][dimension]
        lines.append(
            f"| {dimension} (0-2) | {first_mean} | {second_mean} "
            f"| {_delta_cell(first_mean, second_mean, noise)} |"
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
