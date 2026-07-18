"""Blind row construction and mechanical unblinding.

Every gradable (question, answer) pair becomes a row under an opaque gid.
The noise arm's answers appear twice (replicates 0 and 1) so the judge's
per-question disagreement with itself is measurable; all rows shuffle
together, so a judge cannot tell arms, replicates, or that a comparison is
happening at all.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, NamedTuple

from evals.retrieval.answers import AnswerRow
from evals.retrieval.questions import Question, QuestionKind

GROUND_CHARS = 2400
ANSWER_CHARS = 2400
GID_SPACE = 10**9
NOISE_REPLICATES = 2
JUDGED_KINDS = (QuestionKind.TOPICAL, QuestionKind.KNOWN_ITEM)


@dataclass(frozen=True)
class BlindRow:
    """What a judge sees: no qid, no arm, no replicate."""

    gid: str
    question: str
    source: str
    ground: str
    answer: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BlindAssignment:
    """The secret side of a gid; never shown to a judge."""

    qid: str
    arm: str
    replicate: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BlindAssignment:
        return cls(qid=data["qid"], arm=data["arm"], replicate=data["replicate"])


class BlindSet(NamedTuple):
    rows: list[BlindRow]
    assignments: dict[str, BlindAssignment]
    prefailed: list[str]


def _new_gid(rng: random.Random, taken: set[str]) -> str:
    gid = f"g{rng.randrange(GID_SPACE):09d}"
    while gid in taken:
        gid = f"g{rng.randrange(GID_SPACE):09d}"
    return gid


def build_blind_rows(
    questions: list[Question],
    answers_by_arm: dict[str, dict[str, AnswerRow]],
    noise_arm: str,
    rng: random.Random,
) -> BlindSet:
    """Blind rows for every judged question, with the noise arm duplicated.

    Missing, errored, and empty answers are prefailed: they score zero without
    wasting a judge call, and their gids never reach a judge.
    """
    rows: list[BlindRow] = []
    assignments: dict[str, BlindAssignment] = {}
    prefailed: list[str] = []
    for question in questions:
        if question.kind not in JUDGED_KINDS:
            continue
        for arm, answers in answers_by_arm.items():
            replicates = NOISE_REPLICATES if arm == noise_arm else 1
            for replicate in range(replicates):
                gid = _new_gid(rng, set(assignments))
                assignments[gid] = BlindAssignment(qid=question.qid, arm=arm, replicate=replicate)
                answer = answers.get(question.qid)
                if answer is None or answer.error or not answer.answer.strip():
                    prefailed.append(gid)
                    continue
                rows.append(
                    BlindRow(
                        gid=gid,
                        question=question.question,
                        source=question.source,
                        ground=question.ground_passage[:GROUND_CHARS],
                        answer=answer.answer[:ANSWER_CHARS],
                    )
                )
    rng.shuffle(rows)
    return BlindSet(rows=rows, assignments=assignments, prefailed=prefailed)


def unblind(
    assignments: dict[str, BlindAssignment], grades: dict[str, dict[str, int]]
) -> dict[str, dict[int, dict[str, dict[str, int]]]]:
    """Regroup blind grades as arm -> replicate -> qid -> scores."""
    regrouped: dict[str, dict[int, dict[str, dict[str, int]]]] = {}
    for gid, assignment in assignments.items():
        by_replicate = regrouped.setdefault(assignment.arm, {})
        by_qid = by_replicate.setdefault(assignment.replicate, {})
        scores = grades.get(gid)
        if scores is not None:
            by_qid[assignment.qid] = scores
    return regrouped
