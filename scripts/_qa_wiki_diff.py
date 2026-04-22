"""Diff two wiki builds and emit a markdown report.

Companion to ``qa-wiki-rerank.sh``: takes the baseline (reranker off)
and reranked (reranker on) wiki roots, compares concept/entity page
sets by Jaccard overlap, compares citation sets per page, and surfaces
faithfulness-score deltas. Exits non-zero when a pass/fail assertion
about reranker impact fails.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_CITATION_RE = re.compile(r"^\[\^(src\d+)\]:\s*(.+)$")
_FAITHFULNESS_RE = re.compile(r"^faithfulness_score:\s*([0-9.]+)$", re.MULTILINE)


@dataclass
class PageStats:
    """Per-page snapshot: slug, citations, faithfulness score."""

    slug: str
    citations: frozenset[str]
    faithfulness: float | None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Wiki root from reranker-off run")
    parser.add_argument("reranked", type=Path, help="Wiki root from reranker-on run")
    parser.add_argument(
        "--min-jaccard",
        type=float,
        default=0.5,
        help="Minimum page-set overlap (bail out if reranker reshuffled too much)",
    )
    parser.add_argument(
        "--require-churn",
        action="store_true",
        default=True,
        help="At least one page must show < 0.9 citation Jaccard (proves rerank effect)",
    )
    args = parser.parse_args()

    baseline = _collect(args.baseline)
    reranked = _collect(args.reranked)
    report, failed = _render(baseline, reranked, args.min_jaccard, args.require_churn)
    print(report)
    return 1 if failed else 0


def _collect(wiki_root: Path) -> dict[str, PageStats]:
    out: dict[str, PageStats] = {}
    for subdir in ("concepts", "entities"):
        subdir_path = wiki_root / subdir
        if not subdir_path.is_dir():
            continue
        for md_path in sorted(subdir_path.rglob("*.md")):
            text = md_path.read_text(encoding="utf-8", errors="replace")
            slug = f"{subdir}/{md_path.relative_to(subdir_path).with_suffix('').as_posix()}"
            out[slug] = PageStats(
                slug=slug,
                citations=_parse_citations(text),
                faithfulness=_parse_faithfulness(text),
            )
    return out


def _parse_citations(text: str) -> frozenset[str]:
    citations: set[str] = set()
    for line in text.splitlines():
        match = _CITATION_RE.match(line)
        if match is not None:
            citations.add(match.group(2).strip())
    return frozenset(citations)


def _parse_faithfulness(text: str) -> float | None:
    match = _FAITHFULNESS_RE.search(text)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _render(
    baseline: dict[str, PageStats],
    reranked: dict[str, PageStats],
    min_jaccard: float,
    require_churn: bool,
) -> tuple[str, bool]:
    lines: list[str] = ["# Wiki reranker QA report", ""]
    failed = False

    baseline_slugs = set(baseline)
    reranked_slugs = set(reranked)
    page_jaccard = _jaccard(frozenset(baseline_slugs), frozenset(reranked_slugs))
    lines.append(f"- Baseline pages: {len(baseline_slugs)}")
    lines.append(f"- Reranked pages: {len(reranked_slugs)}")
    lines.append(f"- Page-set Jaccard: {page_jaccard:.2f}")
    if len(baseline_slugs) < 3 or len(reranked_slugs) < 3:
        lines.append("")
        lines.append("  ! corpus too thin (< 3 pages per run); skipping assertions")
        return "\n".join(lines), False
    if page_jaccard < min_jaccard:
        lines.append(f"  !! page Jaccard {page_jaccard:.2f} < threshold {min_jaccard:.2f}")
        failed = True
    if page_jaccard == 1.0:
        lines.append("  !! reranker changed neither clustering nor pages (suspicious)")
        failed = True
    lines.append("")

    common = baseline_slugs & reranked_slugs
    churn: list[tuple[float, str]] = []
    faithfulness_deltas: list[float] = []
    for slug in sorted(common):
        b = baseline[slug]
        r = reranked[slug]
        score = _jaccard(b.citations, r.citations)
        churn.append((score, slug))
        if b.faithfulness is not None and r.faithfulness is not None:
            faithfulness_deltas.append(r.faithfulness - b.faithfulness)

    lines.append("## Citation churn (top 3)")
    churn.sort()
    for score, slug in churn[:3]:
        lines.append(
            f"- `{slug}` - Jaccard {score:.2f} "
            f"({len(baseline[slug].citations)} -> {len(reranked[slug].citations)} citations)"
        )
    if require_churn and churn and churn[0][0] >= 0.9:
        lines.append("")
        lines.append("  !! no page dropped below 0.9 Jaccard -- reranker appears to be a no-op")
        failed = True
    lines.append("")

    if faithfulness_deltas:
        mean = sum(faithfulness_deltas) / len(faithfulness_deltas)
        lines.append(
            f"## Faithfulness delta (reranked - baseline): mean {mean:+.2f} "
            f"across {len(faithfulness_deltas)} pages"
        )
        lines.append("")

    return "\n".join(lines), failed


if __name__ == "__main__":
    sys.exit(main())
