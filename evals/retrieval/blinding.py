"""Blind row construction and mechanical unblinding.

Every gradable (question, answer) pair becomes a row under an opaque gid.
The noise arm's answers appear twice (replicates 0 and 1) so the judge's
per-question disagreement with itself is measurable; all rows shuffle
together, so a judge cannot tell arms, replicates, or that a comparison is
happening at all.

The two replicates are graded under different but equivalent presentations of
the same rubric (see ``judging.RUBRICS``). An identical rubric produces an
identical prompt, and to a greedy decoder that returns an identical grade, which
would make the measured "noise" zero by construction.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from typing import Any, NamedTuple

from evals.retrieval.answers import AnswerRow
from evals.retrieval.questions import Question, QuestionKind

GROUND_CHARS = 2400
ANSWER_CHARS = 2400
GID_HEX_CHARS = 16
NOISE_REPLICATES = 2
JUDGED_KINDS = (QuestionKind.TOPICAL, QuestionKind.KNOWN_ITEM)


@dataclass(frozen=True)
class BlindRow:
    """What a judge sees: no qid, no arm, no replicate.

    ``variant`` selects which equivalent presentation of the grading rubric this
    row is graded under. It is not a hint about arm or replicate: it carries no
    identity, and a judge seeing one row cannot tell which variant it is or that
    a second pass exists.
    """

    gid: str
    question: str
    source: str
    ground: str
    answer: str
    variant: int = 0

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


def _gid_for(qid: str, arm: str, replicate: int, answer: str) -> str:
    """An opaque id derived from the row's content, not from its draw position.

    The judge checkpoint is keyed on this and skips any gid it has already
    graded. A positional id makes that skip unsound: change the question set,
    re-run an answer, or pass the two arms in the other order, and the k-th id
    is unchanged while the k-th row is now a different answer, so a stale grade
    is silently attributed to it. Deriving the id from the content means a
    genuine resume still hits the checkpoint and a changed row simply misses it.

    Hashed rather than concatenated so the id carries no readable arm or qid,
    and salted with the replicate so the noise arm's two copies stay distinct.
    """
    material = "\x00".join([qid, arm, str(replicate), answer])
    return "g" + hashlib.sha256(material.encode()).hexdigest()[:GID_HEX_CHARS]


def build_blind_rows(
    questions: list[Question],
    answers_by_arm: dict[str, dict[str, AnswerRow]],
    noise_arm: str,
    rng: random.Random,
) -> BlindSet:
    """Blind rows for every judged question, with the noise arm duplicated.

    Missing, errored, and empty answers are prefailed: the scorer puts them at
    the rubric's bottom level without wasting a judge call, and their gids never
    reach a judge.
    """
    rows: list[BlindRow] = []
    assignments: dict[str, BlindAssignment] = {}
    prefailed: list[str] = []
    for question in questions:
        if question.kind not in JUDGED_KINDS:
            continue
        for arm, answers in answers_by_arm.items():
            replicates = NOISE_REPLICATES if arm == noise_arm else 1
            answer = answers.get(question.qid)
            failed = answer is None or answer.error or not answer.answer.strip()
            text = "" if answer is None else answer.answer
            for replicate in range(replicates):
                gid = _gid_for(question.qid, arm, replicate, text)
                assignments[gid] = BlindAssignment(qid=question.qid, arm=arm, replicate=replicate)
                if failed:
                    prefailed.append(gid)
                    continue
                rows.append(
                    BlindRow(
                        gid=gid,
                        question=question.question,
                        source=question.source,
                        ground=question.ground_passage[:GROUND_CHARS],
                        answer=answer.answer[:ANSWER_CHARS],
                        variant=replicate,
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
