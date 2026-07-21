"""Markdown rendering of a scored benchmark run.

Leads with the Tier-1 label-scored retrieval numbers (the reproducible core),
then the Tier-2 RAGAS answer scores, then the feature-to-dataset coverage
matrix with derived-qrel datasets clearly marked.
"""

from __future__ import annotations

from typing import Any

from evals.benchmark.stats import DEFAULT_ALPHA, benjamini_hochberg


def _rows_of(rows: list[dict[str, Any]], row_type: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("row_type") == row_type]


def _arm_labels(rows: list[dict[str, Any]]) -> tuple[str, str]:
    meta = _rows_of(rows, "meta")
    if meta:
        return meta[0]["arm_a"], meta[0]["arm_b"]
    return "arm A", "arm B"


def _p_cell(row: dict[str, Any]) -> str:
    """A p-value at the resampling floor is a bound, not a point estimate."""
    if row.get("p_at_floor") and row.get("p_floor"):
        return f"< {float(row['p_floor']):.1e}"
    return f"{row['p_value']:.3f}"


def _ir_section(rows: list[dict[str, Any]], arm_a: str, arm_b: str) -> list[str]:
    ir_rows = _rows_of(rows, "ir")
    if not ir_rows:
        return []
    # Significance is decided on the family-adjusted p across every comparison in
    # the study, not per row. Deciding per row from its own CI while printing a
    # raw p gives two verdicts that can contradict each other, and quoting the
    # best of N raw p-values claims a confidence the study did not earn.
    adjusted = benjamini_hochberg([float(row["p_value"]) for row in ir_rows])
    family = len(ir_rows)
    lines = [
        "## Tier 1 - retrieval, scored against human labels",
        "",
        "Computed by ir_measures, on its pytrec_eval backend, against each "
        "dataset's published relevance "
        "labels. No model judges anything, so these numbers are exactly "
        "reproducible from the run files and qrels.",
        "",
        f"Significance is decided on the Benjamini-Hochberg adjusted p across all "
        f"{family} comparisons in this study, at alpha {DEFAULT_ALPHA}. The CI is "
        "the effect size, not a second verdict. A raw p from the best of several "
        "arms is not evidence at its face value.",
        "",
        f"| dataset | metric | {arm_a} | {arm_b} | delta | 95% CI (B - A) | p | "
        "adj. p | significant |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row, adj in zip(ir_rows, adjusted, strict=True):
        verdict = "yes" if adj <= DEFAULT_ALPHA else "no"
        lines.append(
            f"| {row['dataset']} | {row['metric']} | {row['mean_a']:.4f} | "
            f"{row['mean_b']:.4f} | {row['mean_diff']:+.4f} | "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | {_p_cell(row)} | "
            f"{adj:.3f} | {verdict} |"
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
        "Each mean carries the number of answers that actually scored; RAGAS "
        "cannot score every answer, and the two arms need not fail equally often.",
        "",
        f"| metric | {arm_a} | n | {arm_b} | n |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in ragas_rows:
        lines.append(
            f"| {row['metric']} | {row['arm_a']:.4f} | {row.get('n_a', '?')} | "
            f"{row['arm_b']:.4f} | {row.get('n_b', '?')} |"
        )
    for judge in _rows_of(rows, "judge"):
        lines += [
            "",
            f"Corroborating blind judge noise floor: plus or minus "
            f"{judge['noise_floor']} per dimension (one arm graded twice). "
            "Answer-tier gaps at or below it are not oversold.",
        ]
    lines.append("")
    return lines


def _provenance_section(rows: list[dict[str, Any]]) -> list[str]:
    """Which judge and which scorer builds produced the numbers above.

    A model-graded number is only reproducible if the reader knows what graded
    it. The judge model is frozen in the manifest; the scorer versions are
    recorded at score time, since a pinned requirements file states an intention
    and this states what actually ran.
    """
    version_rows = _rows_of(rows, "versions")
    if not version_rows:
        return []
    record = version_rows[0]
    lines = [
        "## What produced these numbers",
        "",
        f"Answers graded by `{record.get('judge_model', '?')}`, served at "
        f"`{record.get('judge_base_url', '?')}`. The judge is held constant across "
        "arms and differs from the model that generated the answers.",
        "",
        "| scorer | version |",
        "| --- | --- |",
    ]
    for package, version in sorted(record.get("scorers", {}).items()):
        lines.append(f"| {package} | {version} |")
    lines.append("")
    return lines


def _audit_section(rows: list[dict[str, Any]]) -> list[str]:
    """Judge-versus-human agreement on the audited sample.

    Without this the answer-tier numbers rest entirely on a model's opinion of
    another model. The kappa is chance-corrected and quadratically weighted, so
    a near miss counts far less than a gross one, which suits an ordinal rubric.
    """
    audit_rows = _rows_of(rows, "calibration")
    if not audit_rows:
        return []
    lines = [
        "## Judge agreement with human raters",
        "",
        "The same rubric that grades this study's answers was run over SummEval, "
        "whose summaries were rated 1-5 by three experts years before this "
        "harness existed. The ceiling is how well those experts agreed with each "
        "other: a judge cannot track people more closely than they track "
        "themselves, so the raw correlation is reported against it rather than "
        "on its own.",
        "",
        "| dimension | n | Spearman | Kendall | expert ceiling | share of ceiling |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in audit_rows:
        lines.append(
            f"| {row['dimension']} | {row['n']} | {row['spearman']:+.3f} "
            f"| {row['kendall']:+.3f} | {row['expert_ceiling']:.3f} "
            f"| {row['fraction_of_ceiling']:.0%} |"
        )
    lines += [
        "",
        "Citation is not calibrated here: a summary cites nothing, so SummEval "
        "carries no human label for it.",
        "",
    ]
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
    if not _rows_of(rows, "ir") and not _rows_of(rows, "ragas"):
        raise ValueError("results contain no ir or ragas rows; run score-ir first")
    arm_a, arm_b = _arm_labels(rows)
    meta = _rows_of(rows, "meta")
    # Titled from the arms that actually ran. Hardcoding "lilbee vs RAGFlow"
    # labelled every single-system ablation as a cross-system comparison, which
    # is the same wrong-label problem the metric layer had.
    header = [f"# Retrieval benchmark: {arm_a} vs {arm_b}", ""]
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
        + _audit_section(rows)
        + _provenance_section(rows)
        + _coverage_section(rows)
    )
    return "\n".join(lines).rstrip() + "\n"
