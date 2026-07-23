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


def _ir_row_arms(row: dict[str, Any]) -> tuple[str, str]:
    return row.get("arm_a", "arm A"), row.get("arm_b", "arm B")


def _ir_section(rows: list[dict[str, Any]]) -> list[str]:
    ir_rows = _rows_of(rows, "ir")
    if not ir_rows:
        return []
    # Significance is decided on the family-adjusted p across every comparison in
    # the study, not per row. Deciding per row from its own CI while printing a
    # raw p gives two verdicts that can contradict each other, and quoting the
    # best of N raw p-values claims a confidence the study did not earn.
    adjusted = benjamini_hochberg([float(row["p_value"]) for row in ir_rows])
    # Each row is labelled with its own arm pair. When the whole file is one
    # comparison the pair becomes the two value-column headers, as before; when a
    # file holds several comparisons (an ablation), a comparison column names the
    # arms per row so no comparison's scores print under another's arm names.
    pairs = {_ir_row_arms(row) for row in ir_rows}
    single = len(pairs) == 1
    records: list[dict[str, Any]] = []
    for row, adj in zip(ir_rows, adjusted, strict=True):
        arm_a, arm_b = _ir_row_arms(row)
        record: dict[str, Any] = {"dataset": row["dataset"], "metric": row["metric"]}
        if single:
            record[arm_a] = f"{row['mean_a']:.4f}"
            record[arm_b] = f"{row['mean_b']:.4f}"
        else:
            record["comparison"] = f"{arm_a} vs {arm_b}"
            record["arm A"] = f"{row['mean_a']:.4f}"
            record["arm B"] = f"{row['mean_b']:.4f}"
        record["delta"] = f"{row['mean_diff']:+.4f}"
        record["95% CI (B - A)"] = f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
        record["p"] = _p_cell(row)
        record["adj. p"] = f"{adj:.3f}"
        record["significant"] = "yes" if adj <= DEFAULT_ALPHA else "no"
        if row.get("judged_a") is not None and row.get("judged_b") is not None:
            record["judged"] = f"{float(row['judged_a']):.0%} / {float(row['judged_b']):.0%}"
        records.append(record)
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
            "",
            "`judged` is the share of each arm's top 10 that carries a human "
            "judgment (arm A / arm B). These sets are pooled from the systems that "
            "existed when they were built, so anything outside the pool counts as "
            "non-relevant: the lower this is, the more of each run the labels "
            "cannot speak to, and the more a small delta may be a pool artefact.",
        ],
        records,
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
            f"Corroborating blind judge self-consistency: grading one arm twice "
            f"under two rubric presentations disagrees by {judge['noise_floor']} "
            "per dimension on average. This is a per-question spread, not an error "
            "bar on a difference of means, so it is context on how noisy a single "
            "grade is, not a significance threshold for the arm gap above.",
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
            "before this harness existed. The reference is the published "
            "expert-to-expert agreement: a correlation of 0.6 means something "
            "different against a 0.80 reference than against a 0.40 one, so the "
            "raw correlation is reported beside it rather than on its own. The "
            "ratio can exceed 100 percent, because the judge is correlated "
            "against the mean of three experts, which is more reliable than any "
            "single expert, so it is a reference point rather than a hard ceiling.",
        ],
        [
            {
                "dimension": row["dimension"],
                "n": row["n"],
                "Spearman": f"{row['spearman']:+.3f}",
                "Kendall": f"{row['kendall']:+.3f}",
                "expert agreement": f"{row['expert_ceiling']:.3f}",
                "vs expert agreement": f"{row['fraction_of_ceiling']:.0%}",
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
        ["Per-stage wall time, throughput and cost recorded for this run's stages."],
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
    # A file that holds one comparison is titled by its two arms; one that holds
    # several (an ablation) cannot honestly claim a single arm pair in its title,
    # so it is named for the run instead and each comparison is labelled in the
    # table. Hardcoding one pair was the wrong-label problem the metric layer had.
    ir_pairs = {_ir_row_arms(row) for row in _rows_of(rows, "ir")}
    multi_arm = len(ir_pairs) > 1
    if multi_arm:
        run_id = meta[0].get("run_id", "?") if meta else "?"
        header = [f"# Retrieval benchmark: {run_id} ({len(ir_pairs)} comparisons)", ""]
    else:
        header = [f"# Retrieval benchmark: {arm_a} vs {arm_b}", ""]
    if meta:
        provenance = (
            f"Run `{meta[0].get('run_id', '?')}` "
            f"(manifest `{meta[0].get('fingerprint', '?')[:12]}`)."
        )
        if not multi_arm:
            provenance += f" Arm A = {arm_a}, arm B = {arm_b}."
        header += [provenance, ""]
    lines = (
        header
        + _ir_section(rows)
        + _ragas_section(rows, arm_a, arm_b)
        + _attribution_section(rows)
        + _calibration_section(rows)
        + _provenance_section(rows)
        + _machine_section(rows)
        + _coverage_section(rows)
    )
    return "\n".join(lines).rstrip() + "\n"
