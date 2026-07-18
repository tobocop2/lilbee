"""Scoring: noise-floor math, per-dimension means, and exact-truth checks."""

from __future__ import annotations

import re
import statistics
from enum import StrEnum
from typing import Any

from evals.retrieval.answers import AnswerRow
from evals.retrieval.judging import DIMENSIONS
from evals.retrieval.questions import CountOracle, Question, QuestionKind

_NUMBER_RE = re.compile(r"\d+")

Grades = dict[str, dict[str, int]]
Unblinded = dict[str, dict[int, Grades]]

ARM_REPLICATE = 0
NOISE_REPLICATE = 1


class ResultRowType(StrEnum):
    QUESTION = "question"
    SUMMARY = "summary"


def noise_floor(rep0: Grades, rep1: Grades) -> float:
    """Mean absolute per-question, per-dimension disagreement between replicates."""
    shared = [qid for qid in rep0 if qid in rep1]
    if not shared:
        return 0.0
    per_question = [
        sum(abs(rep0[qid][d] - rep1[qid][d]) for d in DIMENSIONS) / len(DIMENSIONS)
        for qid in shared
    ]
    return round(statistics.mean(per_question), 3)


def dimension_means(grades: Grades) -> dict[str, float]:
    """Per-dimension mean scores over every graded question."""
    if not grades:
        return dict.fromkeys(DIMENSIONS, 0.0)
    return {
        d: round(statistics.mean(scores[d] for scores in grades.values()), 3) for d in DIMENSIONS
    }


def count_question_pass(oracle: CountOracle, answer: str) -> bool:
    """Both oracle numbers must appear in the answer."""
    numbers = set(_NUMBER_RE.findall(answer))
    return {str(oracle.chunks), str(oracle.sources)} <= numbers


def known_item_pass(source: str, row: AnswerRow) -> bool:
    """The expected document must be among the answer's cited sources."""
    return source in row.cited_sources


def _question_row(
    question: Question, arm: str, answer: AnswerRow | None, grades: Grades
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "row_type": ResultRowType.QUESTION,
        "qid": question.qid,
        "kind": question.kind,
        "arm": arm,
        "error": answer.error if answer is not None else "missing",
    }
    if question.kind in (QuestionKind.TOPICAL, QuestionKind.KNOWN_ITEM):
        row["grades"] = grades.get(question.qid)
    answered = answer is not None and not answer.error
    if question.kind is QuestionKind.COUNT and question.oracle is not None:
        row["exact_pass"] = answered and count_question_pass(question.oracle, answer.answer)
    elif question.kind is QuestionKind.KNOWN_ITEM:
        row["exact_pass"] = answered and known_item_pass(question.source, answer)
    return row


def _arm_summary(
    questions: list[Question], answers: dict[str, AnswerRow], grades: Grades
) -> dict[str, Any]:
    count_hits = count_total = known_hits = known_total = errors = 0
    for question in questions:
        answer = answers.get(question.qid)
        if answer is None or answer.error:
            errors += 1
            answer = None
        if question.kind is QuestionKind.COUNT and question.oracle is not None:
            count_total += 1
            count_hits += answer is not None and count_question_pass(question.oracle, answer.answer)
        elif question.kind is QuestionKind.KNOWN_ITEM:
            known_total += 1
            known_hits += answer is not None and known_item_pass(question.source, answer)
    return {
        "means": dimension_means(grades),
        "count_pass": [count_hits, count_total],
        "known_item_pass": [known_hits, known_total],
        "errors": errors,
    }


def build_results(
    questions: list[Question],
    answers_by_arm: dict[str, dict[str, AnswerRow]],
    unblinded: Unblinded,
    noise_arm: str,
) -> list[dict[str, Any]]:
    """Per-question rows for every arm, then one summary row, results.jsonl shaped."""
    rows: list[dict[str, Any]] = []
    for question in questions:
        for arm, answers in answers_by_arm.items():
            arm_grades = unblinded.get(arm, {}).get(ARM_REPLICATE, {})
            rows.append(_question_row(question, arm, answers.get(question.qid), arm_grades))
    noise_replicates = unblinded.get(noise_arm, {})
    summary: dict[str, Any] = {
        "row_type": ResultRowType.SUMMARY,
        "noise_floor": noise_floor(
            noise_replicates.get(ARM_REPLICATE, {}), noise_replicates.get(NOISE_REPLICATE, {})
        ),
        "judged": sum(
            1 for q in questions if q.kind in (QuestionKind.TOPICAL, QuestionKind.KNOWN_ITEM)
        ),
        "arms": {
            arm: _arm_summary(questions, answers, unblinded.get(arm, {}).get(ARM_REPLICATE, {}))
            for arm, answers in answers_by_arm.items()
        },
    }
    rows.append(summary)
    return rows
