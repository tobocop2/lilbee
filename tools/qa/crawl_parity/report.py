"""The report of one run: a markdown file, the differences as JSON lines, and one verdict line."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict
from pathlib import Path

from tools.qa.crawl_parity.model import Difference, Kind, Layer, Mode, Side, Status
from tools.qa.crawl_parity.parity import Comparison, PageMeasure
from tools.qa.crawl_parity.pipeline import Outcome, Plan
from tools.qa.crawl_parity.tokens import NORMALISATION_NOTES
from tools.qa.crawl_parity.verdict import Expected, Finding, Verdict, judge

REPORT_FILE = "report.md"
DIFFERENCES_FILE = "differences.jsonl"
PAGES_FILE = "pages.tsv"
COVERAGE_FILE = "coverage.json"
VERDICT_FILE = "verdict.md"
DETAIL_CHARS = 90
STATUS_ORDER = (Status.NEW, Status.KNOWN, Status.ACCEPTED, Status.NOT_COUNTED)
VISIBLE_CASE_NOTE = (
    "visible-case: yardstick B counts a visible word held in another letter case as held"
)


def verdict_line(verdict: Verdict) -> str:
    """PASS or FAIL with the count of findings at each status."""
    counts = ", ".join(f"{len(verdict.with_status(status))} {status}" for status in STATUS_ORDER)
    word = "PASS" if verdict.passed else "FAIL"
    tail = f", {len(verdict.fixed)} {Status.FIXED}, {len(verdict.not_run)} {Status.NOT_RUN}"
    unmeasured = (
        f"; not measured: {'; '.join(verdict.not_measured)}" if verdict.not_measured else ""
    )
    return f"{word}: {counts}{tail}{unmeasured}"


def _finding_row(finding: Finding) -> str:
    first = finding.differences[0]
    amount = sum(difference.amount for difference in finding.differences)
    detail = first.detail[:DETAIL_CHARS].replace("|", "/").replace("\n", " ")
    return (
        f"| {finding.status} | `{finding.signature}` | {finding.issue or '-'} | {finding.pages} "
        f"| {amount:g} | {first.page} | {detail} |"
    )


def _findings_section(verdict: Verdict) -> list[str]:
    lines = [
        "## Findings",
        "",
        "| status | signature | issue | pages | amount | first page | detail |",
        "|---|---|---|---|---|---|---|",
    ]
    for status in STATUS_ORDER:
        lines.extend(_finding_row(finding) for finding in verdict.with_status(status))
    lines += ["", "## Expected entries that no difference matched", ""]
    for status, entries in ((Status.FIXED, verdict.fixed), (Status.NOT_RUN, verdict.not_run)):
        lines.extend(f"- {status} `{entry.signature}` ({entry.issue})" for entry in entries)
    return lines


def _mean_recall(measures: list[PageMeasure], side: Side) -> str:
    scores = [
        score.recall
        for measure in measures
        if (score := measure.oracle_truth if side is Side.ORACLE else measure.candidate_truth)
    ]
    return f"{statistics.mean(scores):.3f} ({len(scores)} pages)" if scores else "not measured"


def _unexplained(measures: list[PageMeasure], side: Side) -> int:
    return sum(
        sum(score.unexplained.values())
        for measure in measures
        if (score := measure.oracle_truth if side is Side.ORACLE else measure.candidate_truth)
    )


def _comparison_row(comparison: Comparison) -> str:
    return (
        f"| {comparison.layer} | {comparison.mode} | {comparison.oracle_saved} "
        f"| {comparison.candidate_saved} | {len(comparison.measures)} "
        f"| {comparison.words_lost} | {comparison.words_added} "
        f"| {comparison.structure_changed} "
        f"| {_mean_recall(comparison.measures, Side.ORACLE)} "
        f"| {_mean_recall(comparison.measures, Side.CANDIDATE)} "
        f"| {_unexplained(comparison.measures, Side.ORACLE)} "
        f"| {_unexplained(comparison.measures, Side.CANDIDATE)} |"
    )


def _yardstick_sections(outcome: Outcome) -> list[str]:
    lines = [
        "## Parity (A) and ground truth (B), totals for each layer and mode",
        "",
        "| layer | mode | oracle saved | candidate saved | both | words lost | words added "
        "| structure changed | oracle visible recall | candidate visible recall "
        "| oracle unexplained words | candidate unexplained words |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
        *(_comparison_row(comparison) for comparison in outcome.comparisons),
        "",
        "## Retrieval (C)",
        "",
    ]
    for item in outcome.retrievals:
        for side, result in item.results.items():
            lines.append(
                f"- {item.mode}, {side}: recall {result.recall:.3f} ({len(result.answered)} of "
                f"{result.asked}), {result.indexed_pages} pages indexed, {result.failed_pages} not "
                f"indexed, index {result.index_seconds:.1f} s, search {result.search_seconds:.1f} s"
            )
    lines += ["", "## Speed", ""]
    for speed in outcome.speeds:
        for side, samples in speed.samples.items():
            lines.append(f"- {speed.mode}, {side}: " + "; ".join(s.text() for s in samples))
        if speed.comparison is not None:
            state = "compared" if speed.comparison.compared else "NOT COMPARED"
            lines.append(f"- {speed.mode}: {state}, {speed.comparison.reason}")
    return lines


def _left_section(outcome: Outcome) -> list[str]:
    lines = [
        "",
        "## Left behind, for each crawl",
        "",
        "| side | layer | mode | processes | temp entries | threads started | return code |",
        "|---|---|---|---|---|---|---|",
    ]
    for crawl in outcome.crawls:
        lines.append(
            f"| {crawl.side} | {crawl.layer} | {crawl.mode} | {len(crawl.left.processes)} "
            f"| {len(crawl.left.temp_entries)} | {crawl.threads_started} | {crawl.return_code} |"
        )
    return lines


def render(plan: Plan, outcome: Outcome) -> str:
    """The markdown report of one run."""
    verdict = outcome.verdict
    assert verdict is not None  # set by pipeline.run before a report is made
    versions = {
        f"{crawl.side} {crawl.layer}": crawl.versions for crawl in outcome.crawls if crawl.versions
    }
    lines = [
        "# Crawl parity run",
        "",
        f"**{verdict_line(verdict)}**",
        "",
        f"Corpus `{plan.corpus.name}`: {len(plan.corpus.records)} records, "
        f"seeds {', '.join(plan.seeds)}; "
        f"modes {', '.join(plan.modes)}; yardsticks {', '.join(sorted(plan.yardsticks))}.",
        f"Verdict layer: {plan.top_layer()}. Ground truth browser: {plan.browser}; "
        f"{sum(1 for truth in outcome.truths.values() if truth.scorable())} pages with "
        "visible text.",
        *(f"- {side}: {config.label}" for side, config in plan.sides.items()),
        *(f"- versions, {name}: {found}" for name, found in sorted(versions.items())),
        f"- requests for an address that is not recorded: {dict(outcome.unrecorded) or 'none'}",
        "",
        "Normalisations applied before any comparison:",
        *(f"- {name}: {note}" for name, note in NORMALISATION_NOTES.items()),
        f"- {VISIBLE_CASE_NOTE}",
        "",
        *_findings_section(verdict),
        "",
        *_yardstick_sections(outcome),
        *_left_section(outcome),
        "",
    ]
    return "\n".join(lines)


def write(plan: Plan, outcome: Outcome, directory: Path) -> Path:
    """Write the report files into *directory* and return the report's path."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / REPORT_FILE).write_text(render(plan, outcome), encoding="utf-8")
    coverage = {
        "ran": sorted([kind.value, mode.value] for kind, mode in outcome.ran),
        "not_measured": outcome.not_measured,
    }
    (directory / COVERAGE_FILE).write_text(json.dumps(coverage), encoding="utf-8")
    with (directory / DIFFERENCES_FILE).open("w", encoding="utf-8") as handle:
        for difference in outcome.differences:
            record = {"signature": difference.signature, **asdict(difference)}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (directory / PAGES_FILE).open("w", encoding="utf-8") as handle:
        handle.write(
            "layer\tmode\tpage\twords_lost\twords_added\tstructure_changed\toracle_recall\tcandidate_recall\n"
        )
        for comparison in outcome.comparisons:
            for measure in comparison.measures:
                oracle = f"{measure.oracle_truth.recall:.3f}" if measure.oracle_truth else ""
                candidate = (
                    f"{measure.candidate_truth.recall:.3f}" if measure.candidate_truth else ""
                )
                handle.write(
                    f"{comparison.layer}\t{comparison.mode}\t{measure.path}\t{measure.words_lost}\t"
                    f"{measure.words_added}\t{measure.structure_changed}\t{oracle}\t{candidate}\n"
                )
    return directory / REPORT_FILE


def read_differences(directory: Path) -> list[Difference]:
    """The differences a run saved."""
    found: list[Difference] = []
    for line in (directory / DIFFERENCES_FILE).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        found.append(
            Difference(
                Kind(record["kind"]),
                Mode(record["mode"]),
                Layer(record["layer"]),
                record["feature"],
                record["page"],
                record["amount"],
                record["detail"],
                record["counts"],
            )
        )
    return found


def rejudge(directory: Path, expected: list[Expected]) -> Verdict:
    """The verdict on a saved run under *expected*, written beside the run's report."""
    coverage = json.loads((directory / COVERAGE_FILE).read_text(encoding="utf-8"))
    ran = {(Kind(kind), Mode(mode)) for kind, mode in coverage["ran"]}
    verdict = judge(read_differences(directory), expected, ran, tuple(coverage["not_measured"]))
    lines = [f"**{verdict_line(verdict)}**", "", *_findings_section(verdict), ""]
    (directory / VERDICT_FILE).write_text("\n".join(lines), encoding="utf-8")
    return verdict
