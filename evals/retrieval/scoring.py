"""Scoring: noise-floor math, per-dimension means, and exact-truth checks."""

from __future__ import annotations

import re
import statistics
from enum import StrEnum
from typing import Any

from evals.benchmark.stats import DEFAULT_SEED, compare
from evals.retrieval.answers import AnswerRow
from evals.retrieval.judging import DIMENSIONS
from evals.retrieval.questions import CountOracle, Question, QuestionKind

_NUMBER_RE = re.compile(r"\d+")

Grades = dict[str, dict[str, int]]
Unblinded = dict[str, dict[int, Grades]]

ARM_REPLICATE = 0
NOISE_REPLICATE = 1
# A paired test needs exactly two arms.
PAIRED_ARMS = 2


class ResultRowType(StrEnum):
    QUESTION = "question"
    SUMMARY = "summary"


class MissingNoiseReplicate(RuntimeError):
    """The second judging pass produced nothing, so no noise floor was measured."""


def paired_dimension_tests(
    unblinded: Unblinded, arms: list[str], *, seed: int = DEFAULT_SEED
) -> list[dict[str, Any]]:
    """Paired per-dimension comparison of the two arms, over shared questions.

    The judge grades both arms on the same questions, so every dimension gap is
    paired and can be tested properly. The noise floor cannot do this job: it is
    a per-question, per-dimension disagreement, while the reported gap is a
    difference of means over many questions whose standard error shrinks with
    the question count, so comparing the two is a scale error in both directions.

    Reuses the benchmark's paired bootstrap and randomization test rather than
    hand-rolling a second one; p-values are family-adjusted by the caller.
    """
    if len(arms) != PAIRED_ARMS:
        return []
    first, second = arms
    tests: list[dict[str, Any]] = []
    for dimension in DIMENSIONS:
        a_scores = {
            qid: float(scores[dimension])
            for qid, scores in unblinded.get(first, {}).get(ARM_REPLICATE, {}).items()
        }
        b_scores = {
            qid: float(scores[dimension])
            for qid, scores in unblinded.get(second, {}).get(ARM_REPLICATE, {}).items()
        }
        tests.append(compare(dimension, a_scores, b_scores, seed=seed).to_dict())
    return tests


def noise_floor(rep0: Grades, rep1: Grades) -> float:
    """Mean absolute per-question, per-dimension disagreement between replicates.

    Raises when the two replicates share no question. A zero here would be
    indistinguishable from a perfectly self-consistent judge, and because the
    floor is the report's only significance threshold, a silent 0.0 marks every
    delta as outside the noise. A run that measured nothing must say so.
    """
    shared = [qid for qid in rep0 if qid in rep1]
    if not shared:
        raise MissingNoiseReplicate(
            "no question was graded in both judging passes, so the judge noise "
            "floor was never measured; check that the noise arm matches the one "
            "the judge pass used and that replicate 1 completed"
        )
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
    *,
    noise_grades: dict[int, Grades] | None = None,
    judge_model: str = "",
) -> list[dict[str, Any]]:
    """Per-question rows for every arm, then one summary row, results.jsonl shaped.

    ``noise_grades`` supplies the noise arm's two replicates as the judge
    actually returned them. It is passed separately because ``unblinded`` has
    prefailed rows mechanically zeroed in both replicates, which reads as perfect
    self-agreement and would deflate the floor toward zero.
    """
    rows: list[dict[str, Any]] = []
    for question in questions:
        for arm, answers in answers_by_arm.items():
            arm_grades = unblinded.get(arm, {}).get(ARM_REPLICATE, {})
            rows.append(_question_row(question, arm, answers.get(question.qid), arm_grades))
    replicates = noise_grades if noise_grades is not None else unblinded.get(noise_arm, {})
    summary: dict[str, Any] = {
        "row_type": ResultRowType.SUMMARY,
        "noise_floor": noise_floor(
            replicates.get(ARM_REPLICATE, {}), replicates.get(NOISE_REPLICATE, {})
        ),
        "noise_arm": noise_arm,
        "noise_pairs": len(
            set(replicates.get(ARM_REPLICATE, {})) & set(replicates.get(NOISE_REPLICATE, {}))
        ),
        "judge_model": judge_model,
        # Questions a judge actually returned a grade for, per arm. The count of
        # judgeable questions is not the same number: answers can prefail and
        # judges can return junk, and reporting the former as the latter
        # overstates every mean's n.
        "judge_graded": {
            arm: len(unblinded.get(arm, {}).get(ARM_REPLICATE, {}))
            for arm in answers_by_arm
        },
        "dimension_tests": paired_dimension_tests(unblinded, list(answers_by_arm)),
        "judgeable": sum(
            1 for q in questions if q.kind in (QuestionKind.TOPICAL, QuestionKind.KNOWN_ITEM)
        ),
        "arms": {
            arm: _arm_summary(questions, answers, unblinded.get(arm, {}).get(ARM_REPLICATE, {}))
            for arm, answers in answers_by_arm.items()
        },
    }
    rows.append(summary)
    return rows
