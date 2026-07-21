"""Markdown rendering of a scored benchmark run.

Leads with the Tier-1 label-scored retrieval numbers (the reproducible core),
then the Tier-2 answer scores, then how well the judge tracks human raters, then
what produced the numbers, then the coverage matrix.

Tables are rendered by pandas rather than assembled from f-strings. Every
section previously built its own header row, its own ``| --- |`` separator and
its own per-row format string, which is five chances to misalign a column count
and no way to notice except by reading the output. ``to_markdown`` derives all
three from the records, so a section declares what its columns are called and
what goes in them, and nothing else.
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


def _section(title: str, prose: list[str], records: list[dict[str, Any]]) -> list[str]:
    """One heading, its explanation, and a table pandas lays out.

    Records arrive pre-formatted as strings. Formatting stays with the section
    that knows what a number means -- a p-value at the resampling floor is a
    bound and prints as ``< 2.0e-04``, a share of a ceiling prints as a
    percentage -- and pandas is left to do only the alignment.
    """
    if not records:
        return []
    import pandas as pd

    # disable_numparse is not optional here. to_markdown hands the cells to
    # tabulate, which re-parses anything that looks numeric: "+0.610" comes back
    # as 0.61, dropping the sign that separates a negative correlation from a
    # positive one and the precision the section chose on purpose. dtype=str
    # alone does not stop it, because the re-parse happens downstream of the
    # frame. Formatting belongs to whoever knows what the number means; tabulate
    # is here for the alignment only.
    table = pd.DataFrame(records, dtype=str).to_markdown(index=False, disable_numparse=True)
    return [f"## {title}", "", *prose, "", table, ""]


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
    return _section(
        "Tier 1 - retrieval, scored against human labels",
        [
            "Computed by ir_measures, on its pytrec_eval backend, against each "
            "dataset's published relevance labels. No model judges anything, so "
            "these numbers are exactly reproducible from the run files and qrels.",
            "",
            f"Significance is decided on the Benjamini-Hochberg adjusted p across "
            f"all {len(ir_rows)} comparisons in this study, at alpha "
            f"{DEFAULT_ALPHA}. The CI is the effect size, not a second verdict. A "
            "raw p from the best of several arms is not evidence at its face value.",
        ],
        [
            {
                "dataset": row["dataset"],
                "metric": row["metric"],
                arm_a: f"{row['mean_a']:.4f}",
                arm_b: f"{row['mean_b']:.4f}",
                "delta": f"{row['mean_diff']:+.4f}",
                "95% CI (B - A)": f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]",
                "p": _p_cell(row),
                "adj. p": f"{adj:.3f}",
                "significant": "yes" if adj <= DEFAULT_ALPHA else "no",
            }
            for row, adj in zip(ir_rows, adjusted, strict=True)
        ],
    )


def _ragas_section(rows: list[dict[str, Any]], arm_a: str, arm_b: str) -> list[str]:
    lines = _section(
        "Tier 2 - answer quality (RAGAS)",
        [
            "Each mean carries the number of answers that actually scored; RAGAS "
            "cannot score every answer, and the two arms need not fail equally often."
        ],
        [
            {
                "metric": row["metric"],
                arm_a: f"{row['arm_a']:.4f}",
                f"n ({arm_a})": row.get("n_a", "?"),
                arm_b: f"{row['arm_b']:.4f}",
                f"n ({arm_b})": row.get("n_b", "?"),
            }
            for row in _rows_of(rows, "ragas")
        ],
    )
    for judge in _rows_of(rows, "judge"):
        lines += [
            f"Corroborating blind judge noise floor: plus or minus "
            f"{judge['noise_floor']} per dimension (one arm graded twice). "
            "Answer-tier gaps at or below it are not oversold.",
            "",
        ]
    return lines


def _attribution_section(rows: list[dict[str, Any]]) -> list[str]:
    """Which half of the pipeline moved, which RAGAS alone cannot say."""
    return _section(
        "Tier 2 - where the change landed (RAGChecker)",
        [
            "RAGAS' faithfulness moves when retrieval changes and when generation "
            "changes, by similar amounts, so it cannot say which half failed. "
            "RAGChecker decomposes answers into claims and checks each by "
            "entailment, which separates the two.",
        ],
        [
            {
                "side": "retriever",
                "delta (B - A)": f"{row['retriever_delta']:+.4f}",
            }
            for row in _rows_of(rows, "ragchecker")
        ]
        + [
            {
                "side": "generator",
                "delta (B - A)": f"{row['generator_delta']:+.4f}",
            }
            for row in _rows_of(rows, "ragchecker")
        ],
    )


def _calibration_section(rows: list[dict[str, Any]]) -> list[str]:
    """How closely the judge tracks people who rated the same kind of thing."""
    lines = _section(
        "Judge agreement with human raters",
        [
            "The same rubric that grades this study's answers was run over "
            "SummEval, whose summaries were rated 1-5 by three experts years "
            "before this harness existed. The ceiling is how well those experts "
            "agreed with each other: a judge cannot track people more closely "
            "than they track themselves, so the raw correlation is reported "
            "against it rather than on its own.",
        ],
        [
            {
                "dimension": row["dimension"],
                "n": row["n"],
                "Spearman": f"{row['spearman']:+.3f}",
                "Kendall": f"{row['kendall']:+.3f}",
                "expert ceiling": f"{row['expert_ceiling']:.3f}",
                "share of ceiling": f"{row['fraction_of_ceiling']:.0%}",
            }
            for row in _rows_of(rows, "calibration")
        ],
    )
    if lines:
        lines += [
            "Citation is not calibrated here: a summary cites nothing, so "
            "SummEval carries no human label for it.",
            "",
        ]
    return lines


def _provenance_section(rows: list[dict[str, Any]]) -> list[str]:
    """Which judge and which scorer builds produced the numbers above.

    A model-graded number is only reproducible if the reader knows what graded
    it. The judge is frozen in the manifest; the scorer versions are recorded at
    score time, since a pin states an intention and this states what ran.
    """
    version_rows = _rows_of(rows, "versions")
    if not version_rows:
        return []
    record = version_rows[0]
    return _section(
        "What produced these numbers",
        [
            f"Answers graded by `{record.get('judge_model', '?')}`, served at "
            f"`{record.get('judge_base_url', '?')}`. The judge is held constant "
            "across arms and differs from the model that generated the answers.",
        ],
        [
            {"scorer": package, "version": version}
            for package, version in sorted(record.get("scorers", {}).items())
        ],
    )


def _machine_section(rows: list[dict[str, Any]]) -> list[str]:
    """The boxes, the timings and what they cost.

    A benchmark that publishes numbers without publishing the machine behind
    them is asking to be taken on trust. Throughput and GPU utilisation travel
    with the duration because four hours of compute and four hours of waiting on
    a network volume are the same number otherwise.
    """
    records: list[dict[str, Any]] = []
    for run in _rows_of(rows, "provenance"):
        for stage in run.get("stages", []):
            records.append(
                {
                    "stage": f"{run['stage_group']}: {stage['name']}",
                    "machine": run["gpu_summary"],
                    "wall": f"{stage['wall_seconds'] / 60:.1f} min",
                    "docs/s": f"{stage['documents_per_second']:,.0f}",
                    "GPU %": f"{stage['gpu_utilisation_mean']:.0f}",
                    "cost": f"${stage['cost_usd']:.2f}",
                }
            )
    return _section(
        "Machines, timings and cost",
        ["Read from the pods while the work ran, not reconstructed afterwards."],
        records,
    )


def _coverage_section(rows: list[dict[str, Any]]) -> list[str]:
    return _section(
        "Feature coverage",
        [
            "Each feature under test maps to a dataset that stresses it. Datasets "
            "whose retrieval labels are derived from human gold evidence are marked.",
        ],
        [
            {
                "feature under test": row["feature"],
                "proven by": row["dataset"],
                "tier-1 metric": row["metric"],
                "qrels": "derived" if row.get("derived") else "native",
            }
            for row in _rows_of(rows, "coverage")
        ],
    )


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
        header += [
            f"Run `{meta[0].get('run_id', '?')}` "
            f"(manifest `{meta[0].get('fingerprint', '?')[:12]}`). "
            f"Arm A = {arm_a}, arm B = {arm_b}.",
            "",
        ]
    lines = (
        header
        + _ir_section(rows, arm_a, arm_b)
        + _ragas_section(rows, arm_a, arm_b)
        + _attribution_section(rows)
        + _calibration_section(rows)
        + _provenance_section(rows)
        + _machine_section(rows)
        + _coverage_section(rows)
    )
    return "\n".join(lines).rstrip() + "\n"
