"""Blind ids identify content, not draw position.

The judge checkpoint is keyed on the gid and skips any gid already graded. If
the gid is a positional draw from a seeded RNG, then re-running with a changed
question set, a changed answer, or the two arms passed in the other order keeps
the k-th gid the same while the k-th row is now a different answer, and the
stale grade is silently attributed to it.
"""

import random

from evals.retrieval.answers import AnswerRow
from evals.retrieval.blinding import build_blind_rows
from evals.retrieval.questions import Question, QuestionKind


def _q(qid):
    return Question(
        qid=qid,
        kind=QuestionKind.TOPICAL,
        question=f"question {qid}?",
        source="a.txt",
        ground_passage="ground",
    )


def _a(qid, arm, text="an answer"):
    return AnswerRow(
        qid=qid,
        arm=arm,
        answer=text,
        sources=["a.txt"],
        cited_sources=["a.txt"],
        seconds=0.1,
        error=None,
    )


def _gid_of(blind, qid, arm, replicate=0):
    return next(
        gid
        for gid, a in blind.assignments.items()
        if a.qid == qid and a.arm == arm and a.replicate == replicate
    )


def _build(questions, answers, seed=7):
    return build_blind_rows(questions, answers, "B", random.Random(seed))


def test_same_content_yields_the_same_gid():
    # A genuine resume must reuse the checkpointed grade.
    qs = [_q("tq0"), _q("tq1")]
    ans = {"A": {q.qid: _a(q.qid, "A") for q in qs}, "B": {q.qid: _a(q.qid, "B") for q in qs}}
    first, second = _build(qs, ans), _build(qs, ans)
    assert _gid_of(first, "tq0", "A") == _gid_of(second, "tq0", "A")


def test_a_changed_answer_yields_a_different_gid():
    # The answer was re-run and changed, so its old grade must not be reused.
    qs = [_q("tq0")]
    before = {"A": {"tq0": _a("tq0", "A", "first answer")}, "B": {"tq0": _a("tq0", "B")}}
    after = {"A": {"tq0": _a("tq0", "A", "a different answer")}, "B": {"tq0": _a("tq0", "B")}}
    assert _gid_of(_build(qs, before), "tq0", "A") != _gid_of(_build(qs, after), "tq0", "A")


def test_dropping_a_question_does_not_shift_other_rows_gids():
    # Positional gids would slide every later row onto a previous row's grade.
    qs = [_q("tq0"), _q("tq1"), _q("tq2")]
    ans = {"A": {q.qid: _a(q.qid, "A") for q in qs}, "B": {q.qid: _a(q.qid, "B") for q in qs}}
    full = _build(qs, ans)
    fewer_qs = [_q("tq0"), _q("tq2")]
    fewer = _build(
        fewer_qs, {arm: {q.qid: rows[q.qid] for q in fewer_qs} for arm, rows in ans.items()}
    )
    assert _gid_of(full, "tq2", "A") == _gid_of(fewer, "tq2", "A")


def test_swapping_the_arm_order_does_not_reassign_gids():
    # judge and score can be invoked with the files in either order.
    qs = [_q("tq0")]
    a_row, b_row = _a("tq0", "A", "answer A"), _a("tq0", "B", "answer B")
    forward = _build(qs, {"A": {"tq0": a_row}, "B": {"tq0": b_row}})
    reversed_order = build_blind_rows(
        qs, {"B": {"tq0": b_row}, "A": {"tq0": a_row}}, "B", random.Random(7)
    )
    assert _gid_of(forward, "tq0", "A") == _gid_of(reversed_order, "tq0", "A")


def test_the_two_noise_replicates_get_distinct_gids():
    qs = [_q("tq0")]
    ans = {"A": {"tq0": _a("tq0", "A")}, "B": {"tq0": _a("tq0", "B")}}
    blind = _build(qs, ans)
    assert _gid_of(blind, "tq0", "B", 0) != _gid_of(blind, "tq0", "B", 1)


def test_gid_does_not_leak_the_arm_or_qid():
    # Realistic multi-character arm names: a one-letter arm would match any hex
    # digest by chance and prove nothing.
    qs = [_q("tq0")]
    ans = {
        "lilbee-parity": {"tq0": _a("tq0", "lilbee-parity")},
        "ragflow-default": {"tq0": _a("tq0", "ragflow-default")},
    }
    blind = build_blind_rows(qs, ans, "ragflow-default", random.Random(7))
    for gid, assignment in blind.assignments.items():
        digest = gid.removeprefix("g")
        assert assignment.qid not in gid
        assert assignment.arm not in gid
        # Opaque fixed-width hex, so nothing about the row is readable from it.
        assert len(digest) == 16
        assert all(c in "0123456789abcdef" for c in digest)
