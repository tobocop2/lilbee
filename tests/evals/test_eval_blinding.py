"""Blinding: opaque ids, a duplicated noise arm, and mechanical unblinding."""

import random

from evals.retrieval.answers import AnswerRow
from evals.retrieval.blinding import build_blind_rows, unblind
from evals.retrieval.questions import Question, QuestionKind


def _question(qid: str, kind: QuestionKind = QuestionKind.TOPICAL) -> Question:
    return Question(
        qid=qid,
        kind=kind,
        question="Where did it happen?",
        source="a.txt",
        ground_passage="ground text",
    )


def _answer(qid: str, arm: str, answer: str = "an answer", error: str | None = None) -> AnswerRow:
    return AnswerRow(
        qid=qid, arm=arm, answer=answer, sources=[], cited_sources=[], seconds=0.1, error=error
    )


def _answers(questions, arm):
    return {q.qid: _answer(q.qid, arm) for q in questions}


def test_noise_arm_is_judged_twice_and_other_arm_once():
    questions = [_question("tq000"), _question("tq001")]
    blind = build_blind_rows(
        questions,
        {"A": _answers(questions, "A"), "B": _answers(questions, "B")},
        noise_arm="B",
        rng=random.Random(5),
    )
    assert len(blind.rows) == 6
    replicates = [(a.arm, a.replicate) for a in blind.assignments.values()]
    assert replicates.count(("A", 0)) == 2
    assert replicates.count(("B", 0)) == 2
    assert replicates.count(("B", 1)) == 2


def test_blind_rows_never_leak_arm_or_qid():
    # Multi-character arm names: a one-letter label matches any hex gid by
    # chance and would make the arm half of this assertion vacuous.
    questions = [_question("tq000")]
    arms = {
        "old-build": _answers(questions, "old-build"),
        "new-build": _answers(questions, "new-build"),
    }
    blind = build_blind_rows(questions, arms, noise_arm="new-build", rng=random.Random(5))
    assert blind.rows
    for row in blind.rows:
        payload = f"{row.gid}{row.question}{row.source}{row.ground}{row.answer}"
        assert "tq000" not in payload
        for arm in arms:
            assert arm not in payload
        assert row.gid in blind.assignments


def test_count_questions_are_not_blind_judged():
    questions = [_question("ct000", QuestionKind.COUNT)]
    blind = build_blind_rows(
        questions,
        {"A": _answers(questions, "A"), "B": _answers(questions, "B")},
        noise_arm="B",
        rng=random.Random(5),
    )
    assert blind.rows == []
    assert blind.assignments == {}


def test_failed_and_missing_answers_prefail_instead_of_judging():
    questions = [_question("tq000"), _question("tq001")]
    answers_a = {
        "tq000": _answer("tq000", "A", answer="", error="boom"),
    }
    blind = build_blind_rows(
        questions,
        {"A": answers_a, "B": _answers(questions, "B")},
        noise_arm="B",
        rng=random.Random(5),
    )
    assert len(blind.prefailed) == 2
    assert len(blind.rows) == 4
    for gid in blind.prefailed:
        assert blind.assignments[gid].arm == "A"


def test_build_blind_rows_is_deterministic_for_a_seed():
    questions = [_question("tq000"), _question("tq001")]
    arms = {"A": _answers(questions, "A"), "B": _answers(questions, "B")}
    first = build_blind_rows(questions, arms, noise_arm="B", rng=random.Random(9))
    second = build_blind_rows(questions, arms, noise_arm="B", rng=random.Random(9))
    assert [row.gid for row in first.rows] == [row.gid for row in second.rows]
    assert first.assignments == second.assignments


def test_unblind_regroups_by_arm_replicate_and_qid():
    questions = [_question("tq000")]
    blind = build_blind_rows(
        questions,
        {"A": _answers(questions, "A"), "B": _answers(questions, "B")},
        noise_arm="B",
        rng=random.Random(5),
    )
    grades = {gid: {"faithfulness": 2, "relevance": 1, "citation": 0} for gid in blind.assignments}
    regrouped = unblind(blind.assignments, grades)
    assert regrouped["A"][0]["tq000"]["faithfulness"] == 2
    assert regrouped["B"][0]["tq000"]["relevance"] == 1
    assert regrouped["B"][1]["tq000"]["citation"] == 0


def test_unblind_skips_unreturned_grades():
    questions = [_question("tq000")]
    blind = build_blind_rows(
        questions,
        {"A": _answers(questions, "A"), "B": _answers(questions, "B")},
        noise_arm="B",
        rng=random.Random(5),
    )
    regrouped = unblind(blind.assignments, {})
    assert regrouped["A"][0] == {}
