"""A human-audited sample of the judge's grades, and the agreement it shows.

Every credible benchmark anchors to human judgement somewhere: BEIR's numbers
rest on TREC assessors, RAGChecker's meta-evaluation on 280 human-labelled
instances, TRUE on eleven annotated datasets. A pipeline whose answer-quality
numbers come entirely from a model, with no human-checked sample anywhere, is
asking a reader to accept the judge on faith, and the standard answer to "how do
you know your faithfulness score tracks a human" is a small audited sample
reported alongside.

``agreement`` is the part that matters and it does not care where the human
scores came from: a sheet somebody filled in, or a published dataset that was
annotated long before this harness existed. The second source is the one to
prefer, because it needs no annotator on this project and because a label
somebody else produced cannot be accused of being tuned to make these numbers
look good.

``stratified_sample`` and the sheet functions serve the first source only. They
are the fallback for a corpus with no public analogue, not the default path.

The statistics are sklearn's and scipy's. Quadratic-weighted Cohen's kappa is
the standard instrument for an ordinal scale, because it counts a 4-vs-5
disagreement as much less serious than a 1-vs-5, which unweighted kappa does not.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.deps import install_hint
from evals.retrieval.judging import DIMENSIONS, SCORE_MAX, SCORE_MIN

SKLEARN_INSTALL_HINT = install_hint("scikit-learn", "to measure judge agreement")

# Below this many audited rows a kappa is too noisy to publish. Not a magic
# number: RAGChecker's meta-evaluation used 280 instances, and the published
# guidance for kappa stability is a few dozen per category at minimum. Fifty is
# the floor at which the interval is narrow enough to say anything at all.
MIN_AUDIT_ROWS = 50


@dataclass(frozen=True)
class AuditRow:
    """One graded answer, as a human sees it: no judge score attached.

    The judge's grade is deliberately absent. Showing it would anchor the
    annotator to it, and an agreement number computed against a score the
    annotator was already looking at measures suggestibility, not agreement.
    """

    gid: str
    question: str
    source: str
    ground: str
    answer: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gid": self.gid,
            "question": self.question,
            "source": self.source,
            "ground": self.ground,
            "answer": self.answer,
            # Left blank for the annotator, one key per dimension.
            **{dimension: None for dimension in DIMENSIONS},
        }


def stratified_sample(
    grades: dict[str, dict[str, int]], size: int, *, dimension: str, seed: int
) -> list[str]:
    """Sample gids spread across the judge's score range, not across its mode.

    A uniform draw from a judge that mostly says 5 produces an audit of mostly
    5s, which measures agreement where it is easiest and says nothing about the
    boundary cases the score range exists to separate.

    Round-robin over the score buckets, which is not the same as an even split
    and should not be described as one. A rare score is taken *exhaustively* --
    every row of it, up to the sample size -- and the remainder is filled from
    whatever scores still have rows. Asking for 60 from 108 fives and 12 ones
    yields all 12 ones and 48 fives, not 30 and 30. That is the intended shape:
    the rare scores are where the judge is least tested, so the audit takes all
    of them it can get rather than capping them at a quota.
    """
    if dimension not in DIMENSIONS:
        raise ValueError(f"unknown dimension '{dimension}'; expected one of {list(DIMENSIONS)}")
    by_score: dict[int, list[str]] = {}
    for gid, scores in grades.items():
        by_score.setdefault(int(scores[dimension]), []).append(gid)
    rng = random.Random(seed)
    for bucket in by_score.values():
        bucket.sort()
        rng.shuffle(bucket)
    # Round-robin across the score buckets so the sample stays balanced even
    # when one score dominates and another has only a handful of rows.
    chosen: list[str] = []
    buckets = [by_score[score] for score in sorted(by_score)]
    while len(chosen) < size and any(buckets):
        for bucket in buckets:
            if not bucket:
                continue
            chosen.append(bucket.pop())
            if len(chosen) == size:
                break
    return chosen


def write_audit_sheet(path: Path, rows: list[AuditRow]) -> None:
    """Write the blind sheet a person fills in, one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row.to_dict()) + "\n" for row in rows))


def read_audit_sheet(path: Path) -> dict[str, dict[str, int]]:
    """Read back the filled sheet as ``gid -> dimension -> score``.

    Rows left blank are skipped rather than defaulted: a half-finished audit
    should shrink the sample it reports on, not silently score the unfilled
    rows at whatever a default happens to be.
    """
    audited: dict[str, dict[str, int]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        scores = {}
        for dimension in DIMENSIONS:
            value = record.get(dimension)
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"{path}: {record.get('gid')} has a non-integer {dimension} "
                    f"({value!r}); the sheet takes whole numbers "
                    f"{SCORE_MIN}-{SCORE_MAX}"
                )
            if not SCORE_MIN <= value <= SCORE_MAX:
                raise ValueError(
                    f"{path}: {record.get('gid')} scores {dimension} at {value}, "
                    f"outside the rubric's {SCORE_MIN}-{SCORE_MAX} scale"
                )
            scores[dimension] = value
        if scores:
            audited[str(record["gid"])] = scores
    return audited


@dataclass(frozen=True)
class Agreement:
    """How closely the judge tracked the human, on one dimension."""

    dimension: str
    n: int
    kappa: float
    spearman: float
    exact_match: float
    mean_absolute_error: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "n": self.n,
            "quadratic_weighted_kappa": round(self.kappa, 4),
            "spearman": round(self.spearman, 4),
            "exact_match": round(self.exact_match, 4),
            "mean_absolute_error": round(self.mean_absolute_error, 4),
        }


class InsufficientAuditError(RuntimeError):
    """Too few audited rows for an agreement number to mean anything."""


def agreement(
    judge: dict[str, dict[str, int]],
    human: dict[str, dict[str, int]],
    *,
    minimum: int = MIN_AUDIT_ROWS,
) -> list[Agreement]:
    """Per-dimension agreement between the judge and the human sample.

    Quadratic-weighted kappa is the headline: it is chance-corrected, which raw
    agreement is not, and it weights a near miss far below a gross one, which
    suits an ordinal rubric. Spearman travels with it because kappa alone cannot
    distinguish a judge that is biased but correctly ordered from one that is
    unordered, and those call for different responses.
    """
    try:
        from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error
    except ImportError as exc:
        raise RuntimeError(SKLEARN_INSTALL_HINT) from exc
    from scipy import stats as scipy_stats

    shared = sorted(gid for gid in judge if gid in human)
    if len(shared) < minimum:
        raise InsufficientAuditError(
            f"only {len(shared)} rows were audited by both the judge and a human; "
            f"agreement is not reported below {minimum}, since a kappa on fewer "
            "rows is too noisy to publish"
        )
    results: list[Agreement] = []
    for dimension in DIMENSIONS:
        pairs = [
            (judge[gid][dimension], human[gid][dimension])
            for gid in shared
            if dimension in human[gid] and dimension in judge[gid]
        ]
        if len(pairs) < minimum:
            continue
        judged = [pair[0] for pair in pairs]
        annotated = [pair[1] for pair in pairs]
        # A dimension where one side never varies has no rank correlation to
        # report; scipy returns NaN and kappa is undefined rather than zero.
        correlation = (
            float(scipy_stats.spearmanr(judged, annotated).statistic)
            if len(set(judged)) > 1 and len(set(annotated)) > 1
            else float("nan")
        )
        results.append(
            Agreement(
                dimension=dimension,
                n=len(pairs),
                kappa=float(
                    cohen_kappa_score(
                        judged,
                        annotated,
                        weights="quadratic",
                        labels=list(range(SCORE_MIN, SCORE_MAX + 1)),
                    )
                ),
                spearman=correlation,
                exact_match=float(accuracy_score(annotated, judged)),
                mean_absolute_error=float(mean_absolute_error(annotated, judged)),
            )
        )
    return results
