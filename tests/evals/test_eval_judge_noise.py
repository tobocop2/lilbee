"""The judge's noise floor: what the second pass measures, and who grades.

The floor is the report's only significance threshold, so three things have to
hold: the second pass must actually be able to disagree with the first, rows
nobody graded must not count as agreement, and the judge must not be the model
that wrote the answers.
"""

import random

import pytest
from evals.retrieval.answers import AnswerRow
from evals.retrieval.blinding import (
    BlindAssignment,
    BlindRow,
    build_blind_rows,
    unblind,
)
from evals.retrieval.judging import JUDGE_PROMPTS, judge_prompt_for
from evals.retrieval.llm import JUDGE_BASE_URL_ENV, JUDGE_MODEL_ENV, judge_backend
from evals.retrieval.questions import Question, QuestionKind
from evals.retrieval.scoring import MissingNoiseReplicateError, build_results


def _row(variant):
    return BlindRow(
        gid="g1", question="Q?", source="a.txt", ground="ground", answer="ans", variant=variant
    )


def test_the_two_replicates_are_graded_under_different_prompts():
    # Identical prompts to a greedy decoder return identical grades, so the
    # measured "noise" would be zero by construction.
    assert judge_prompt_for(_row(0)) != judge_prompt_for(_row(1))


def test_both_prompt_variants_carry_the_same_content_and_rubric():
    first, second = judge_prompt_for(_row(0)), judge_prompt_for(_row(1))
    for fragment in ("Q?", "a.txt", "ground", "ans"):
        assert fragment in first
        assert fragment in second
    # Same scale and same dimensions requested, only the arrangement differs.
    for dimension in ("faithfulness", "relevance", "citation"):
        assert dimension in first
        assert dimension in second


def test_variant_selection_wraps_rather_than_indexing_out_of_range():
    assert judge_prompt_for(_row(len(JUDGE_PROMPTS))) == judge_prompt_for(_row(0))


def _answer(qid, arm, text="an answer", error=None):
    return AnswerRow(
        qid=qid,
        arm=arm,
        answer=text,
        sources=["a.txt"],
        cited_sources=["a.txt"],
        seconds=0.1,
        error=error,
    )


def test_the_noise_arm_replicates_get_different_variants():
    questions = [
        Question(
            qid="tq000",
            kind=QuestionKind.TOPICAL,
            question="Where?",
            source="a.txt",
            ground_passage="ground",
        )
    ]
    answers = {
        "A": {"tq000": _answer("tq000", "A")},
        "B": {"tq000": _answer("tq000", "B")},
    }
    blind = build_blind_rows(questions, answers, "B", random.Random(1))
    noise_rows = [row for row in blind.rows if blind.assignments[row.gid].arm == "B"]
    assert sorted(row.variant for row in noise_rows) == [0, 1]


def test_ungraded_rows_do_not_count_as_judge_agreement():
    # A question the judge never graded twice must not register as a zero
    # disagreement; with only prefail-shaped rows there is nothing to measure.
    questions = [
        Question(
            qid="tq000",
            kind=QuestionKind.TOPICAL,
            question="Where?",
            source="a.txt",
            ground_passage="g",
        )
    ]
    answers = {"A": {"tq000": _answer("tq000", "A")}, "B": {"tq000": _answer("tq000", "B")}}
    # Noise arm B has replicate 0 graded but no replicate 1 at all.
    unblinded = {
        "A": {0: {"tq000": {"faithfulness": 2, "relevance": 2, "citation": 2}}},
        "B": {0: {"tq000": {"faithfulness": 1, "relevance": 1, "citation": 1}}},
    }
    with pytest.raises(MissingNoiseReplicateError):
        build_results(questions, answers, unblinded, noise_arm="B", judged=unblinded)


def test_summary_records_the_judge_and_the_pairs_behind_the_floor():
    questions = [
        Question(
            qid="tq000",
            kind=QuestionKind.TOPICAL,
            question="Where?",
            source="a.txt",
            ground_passage="g",
        )
    ]
    answers = {"A": {"tq000": _answer("tq000", "A")}, "B": {"tq000": _answer("tq000", "B")}}
    grades = {"faithfulness": 1, "relevance": 1, "citation": 1}
    unblinded = {"A": {0: {"tq000": grades}}, "B": {0: {"tq000": grades}, 1: {"tq000": grades}}}
    summary = build_results(
        questions, answers, unblinded, noise_arm="B", judged=unblinded, judge_model="some-judge"
    )[-1]
    assert summary["judge_model"] == "some-judge"
    assert summary["noise_arm"] == "B"
    assert summary["noise_pairs"] == 1
    assert summary["judge_graded"] == {"A": 1, "B": 1}


def test_judge_refuses_to_fall_back_to_the_system_under_test(monkeypatch):
    # No judge endpoint configured: the harness must refuse rather than quietly
    # grade with the same model that generated the answers.
    monkeypatch.delenv(JUDGE_BASE_URL_ENV, raising=False)
    monkeypatch.delenv(JUDGE_MODEL_ENV, raising=False)
    with pytest.raises(RuntimeError, match="separate from the system under test"):
        judge_backend()


def test_judge_requires_a_named_model_alongside_the_base_url(monkeypatch):
    # A base URL with no model sends model="" and leaves grades unattributable.
    monkeypatch.setenv(JUDGE_BASE_URL_ENV, "http://judge")
    monkeypatch.delenv(JUDGE_MODEL_ENV, raising=False)
    with pytest.raises(RuntimeError):
        judge_backend()


def test_judge_backend_records_its_identity(monkeypatch):
    monkeypatch.setenv(JUDGE_BASE_URL_ENV, "http://judge")
    monkeypatch.setenv(JUDGE_MODEL_ENV, "fable-5")
    backend = judge_backend()
    assert backend.model == "fable-5"
    assert backend.base_url == "http://judge"


def test_unblind_keeps_replicates_separate():
    assignments = {
        "g0": BlindAssignment(qid="q1", arm="B", replicate=0),
        "g1": BlindAssignment(qid="q1", arm="B", replicate=1),
    }
    grades = {"g0": {"faithfulness": 2}, "g1": {"faithfulness": 1}}
    regrouped = unblind(assignments, grades)
    assert regrouped["B"][0]["q1"]["faithfulness"] == 2
    assert regrouped["B"][1]["q1"]["faithfulness"] == 1


def test_prefailed_answers_are_not_counted_as_judge_graded():
    # An answer that failed outright never reaches the judge. It scores zero
    # against its arm, but reporting it as judge-graded overstates the judge's n
    # and makes the report's own caveat false.
    questions = [
        Question(
            qid=f"tq{i}",
            kind=QuestionKind.TOPICAL,
            question="Where?",
            source="a.txt",
            ground_passage="g",
        )
        for i in range(2)
    ]
    answers = {
        "A": {"tq0": _answer("tq0", "A"), "tq1": _answer("tq1", "A", error="boom")},
        "B": {"tq0": _answer("tq0", "B"), "tq1": _answer("tq1", "B")},
    }
    grade = {"faithfulness": 1, "relevance": 1, "citation": 1}
    zero = {"faithfulness": 0, "relevance": 0, "citation": 0}
    # Arm A's tq1 prefailed: mechanically zeroed in `unblinded`, absent from `judged`.
    unblinded = {
        "A": {0: {"tq0": grade, "tq1": zero}},
        "B": {0: {"tq0": grade, "tq1": grade}, 1: {"tq0": grade, "tq1": grade}},
    }
    judged = {
        "A": {0: {"tq0": grade}},
        "B": {0: {"tq0": grade, "tq1": grade}, 1: {"tq0": grade, "tq1": grade}},
    }
    summary = build_results(questions, answers, unblinded, noise_arm="B", judged=judged)[-1]
    assert summary["judge_graded"]["A"] == 1
    # The mean still covers both, because the failure counts against the arm.
    assert summary["scored"]["A"] == 2
    assert summary["judgeable"] == 2


def test_arm_means_are_computed_over_the_same_question_set():
    # The judge returned junk for arm A on tq1, so that row is in neither the
    # judged nor the prefailed map. Averaging each arm over "whatever survived"
    # puts the two means on different question sets and biases the comparison
    # toward the arm the judge dislikes. The means must be paired.
    questions = [
        Question(
            qid=f"tq{i}",
            kind=QuestionKind.TOPICAL,
            question="Where?",
            source="a.txt",
            ground_passage="g",
        )
        for i in range(2)
    ]
    answers = {
        "A": {"tq0": _answer("tq0", "A"), "tq1": _answer("tq1", "A")},
        "B": {"tq0": _answer("tq0", "B"), "tq1": _answer("tq1", "B")},
    }
    high = {"faithfulness": 2, "relevance": 2, "citation": 2}
    low = {"faithfulness": 0, "relevance": 0, "citation": 0}
    # A graded only on tq0 (high). B graded on tq0 (high) and tq1 (low).
    unblinded = {
        "A": {0: {"tq0": high}},
        "B": {0: {"tq0": high, "tq1": low}, 1: {"tq0": high, "tq1": low}},
    }
    summary = build_results(questions, answers, unblinded, noise_arm="B", judged=unblinded)[-1]
    # Both means must cover the same questions, so B is not punished for tq1
    # while A silently skips it.
    assert summary["paired_questions"] == 1
    assert summary["arms"]["A"]["means"]["faithfulness"] == 2.0
    assert summary["arms"]["B"]["means"]["faithfulness"] == 2.0
    # The shortfall is still visible rather than hidden.
    assert summary["judge_graded"] == {"A": 1, "B": 2}
