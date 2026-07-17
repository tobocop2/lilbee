"""Markdown rendering of a scored benchmark run.

Leads with the Tier-1 label-scored retrieval numbers (the reproducible core),
then the Tier-2 RAGAS answer scores, then the feature-to-dataset coverage
matrix with derived-qrel datasets clearly marked.
"""

from __future__ import annotations

from typing import Any


def _rows_of(rows: list[dict[str, Any]], row_type: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("row_type") == row_type]


def _arm_labels(rows: list[dict[str, Any]]) -> tuple[str, str]:
    meta = _rows_of(rows, "meta")
    if meta:
        return meta[0]["arm_a"], meta[0]["arm_b"]
    return "arm A", "arm B"


def _sig_cell(row: dict[str, Any]) -> str:
    ci = f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
    flag = "" if row["significant"] else " (n.s.)"
    return f"{ci}{flag}"


def _ir_section(rows: list[dict[str, Any]], arm_a: str, arm_b: str) -> list[str]:
    ir_rows = _rows_of(rows, "ir")
    if not ir_rows:
        return []
    lines = [
        "## Tier 1 - retrieval, scored against human labels",
        "",
        "Computed by pytrec_eval against each dataset's published relevance "
        "labels. No model judges anything, so these numbers are exactly "
        "reproducible from the run files and qrels.",
        "",
        f"| dataset | metric | {arm_a} | {arm_b} | delta | 95% CI (B - A) | p |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in ir_rows:
        lines.append(
            f"| {row['dataset']} | {row['metric']} | {row['mean_a']:.4f} | "
            f"{row['mean_b']:.4f} | {row['mean_diff']:+.4f} | {_sig_cell(row)} | "
            f"{row['p_value']:.3f} |"
        )
    lines.append("")
    return lines


def _ragas_section(rows: list[dict[str, Any]], arm_a: str, arm_b: str) -> list[str]:
    ragas_rows = _rows_of(rows, "ragas")
    if not ragas_rows:
        return []
    lines = [
        "## Tier 2 - answer quality (RAGAS)",
        "",
        f"| metric | {arm_a} | {arm_b} |",
        "| --- | --- | --- |",
    ]
    for row in ragas_rows:
        lines.append(f"| {row['metric']} | {row['arm_a']:.4f} | {row['arm_b']:.4f} |")
    for judge in _rows_of(rows, "judge"):
        lines += [
            "",
            f"Corroborating blind judge noise floor: plus or minus "
            f"{judge['noise_floor']} per dimension (one arm graded twice). "
            "Answer-tier gaps at or below it are not oversold.",
        ]
    lines.append("")
    return lines


def _coverage_section(rows: list[dict[str, Any]]) -> list[str]:
    coverage_rows = _rows_of(rows, "coverage")
    if not coverage_rows:
        return []
    lines = [
        "## Feature coverage",
        "",
        "Each feature under test maps to a dataset that stresses it. Datasets "
        "whose retrieval labels are derived from human gold evidence are marked.",
        "",
        "| feature under test | proven by | tier-1 metric | qrels |",
        "| --- | --- | --- | --- |",
    ]
    for row in coverage_rows:
        qrels = "derived" if row.get("derived") else "native"
        lines.append(f"| {row['feature']} | {row['dataset']} | {row['metric']} | {qrels} |")
    lines.append("")
    return lines


def render_report(rows: list[dict[str, Any]]) -> str:
    """Render results rows to the human-readable benchmark markdown report."""
    arm_a, arm_b = _arm_labels(rows)
    meta = _rows_of(rows, "meta")
    header = ["# lilbee vs RAGFlow retrieval benchmark", ""]
    if meta:
        header.append(
            f"Run `{meta[0].get('run_id', '?')}` "
            f"(manifest `{meta[0].get('fingerprint', '?')[:12]}`). "
            f"Arm A = {arm_a}, arm B = {arm_b}."
        )
        header.append("")
    lines = (
        header
        + _ir_section(rows, arm_a, arm_b)
        + _ragas_section(rows, arm_a, arm_b)
        + _coverage_section(rows)
    )
    if not _rows_of(rows, "ir") and not _rows_of(rows, "ragas"):
        raise ValueError("results contain no ir or ragas rows; run score-ir first")
    return "\n".join(lines).rstrip() + "\n"
