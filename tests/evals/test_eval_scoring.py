"""Scoring: noise floor math, per-dimension means, and exact-truth checks."""

import pytest
from evals.retrieval.answers import AnswerRow
from evals.retrieval.questions import CountOracle, Question, QuestionKind
from evals.retrieval.scoring import (
    MissingNoiseReplicateError,
    ResultRowType,
    build_results,
    count_question_pass,
    dimension_means,
    known_item_pass,
    noise_floor,
)


def _grades(f: int, r: int, c: int) -> dict[str, int]:
    return {"faithfulness": f, "relevance": r, "citation": c}


def test_noise_floor_is_mean_absolute_disagreement_per_dimension():
    rep0 = {"q1": _grades(2, 2, 2), "q2": _grades(1, 1, 1)}
    rep1 = {"q1": _grades(2, 1, 0), "q2": _grades(1, 1, 1)}
    assert noise_floor(rep0, rep1) == pytest.approx(0.5)


def test_noise_floor_ignores_questions_missing_from_either_replicate():
    rep0 = {"q1": _grades(2, 2, 2), "only0": _grades(0, 0, 0)}
    rep1 = {"q1": _grades(2, 2, 2), "only1": _grades(2, 2, 2)}
    assert noise_floor(rep0, rep1) == 0.0


def test_noise_floor_without_a_second_replicate_fails_loudly():
    # A silent 0.0 here would mark every delta as outside the noise, so a run
    # that measured no judge variance at all must say so rather than score it.
    with pytest.raises(MissingNoiseReplicateError):
        noise_floor({}, {})


def test_noise_floor_fails_when_the_replicates_share_no_question():
    with pytest.raises(MissingNoiseReplicateError):
        noise_floor({"q1": _grades(2, 2, 2)}, {"q2": _grades(0, 0, 0)})


def test_dimension_means():
    means = dimension_means({"q1": _grades(2, 2, 0), "q2": _grades(0, 1, 1)})
    assert means == {"faithfulness": 1.0, "relevance": 1.5, "citation": 0.5}
    assert dimension_means({}) == {"faithfulness": 0.0, "relevance": 0.0, "citation": 0.0}


def test_count_question_pass_requires_the_document_count():
    # The question asks how many documents mention the term, so that is the
    # number the check verifies. A wrong chunk count alongside a right document
    # count still answers the question that was asked.
    oracle = CountOracle(term="lantern", chunks=14, sources=3)
    assert count_question_pass(oracle, "It appears in 14 chunks across 3 documents.")
    assert count_question_pass(oracle, "It appears 15 times in 3 documents.")
    assert not count_question_pass(oracle, "It appears in 14 chunks.")


def test_known_item_pass_requires_the_expected_citation():
    row = AnswerRow(
        qid="ki000",
        arm="A",
        answer="It is a book.",
        sources=["a.txt", "b.txt"],
        cited_sources=["a.txt"],
        seconds=0.1,
        error=None,
    )
    assert known_item_pass("a.txt", row)
    assert not known_item_pass("b.txt", row)


def _pipeline_fixture():
    questions = [
        Question(
            qid="tq000",
            kind=QuestionKind.TOPICAL,
            question="Where?",
            source="a.txt",
            ground_passage="ground",
        ),
        Question(
            qid="ki000",
            kind=QuestionKind.KNOWN_ITEM,
            question="What is a.txt about?",
            source="a.txt",
            ground_passage="head",
        ),
        Question(
            qid="ct000",
            kind=QuestionKind.COUNT,
            question="How many documents mention 'lantern'?",
            oracle=CountOracle(term="lantern", chunks=2, sources=1),
        ),
    ]

    def _answer(qid, arm, answer, cited):
        return AnswerRow(
            qid=qid,
            arm=arm,
            answer=answer,
            sources=cited,
            cited_sources=cited,
            seconds=0.2,
            error=None,
        )

    answers = {
        "A": {
            "tq000": _answer("tq000", "A", "By the pier.", ["a.txt"]),
            "ki000": _answer("ki000", "A", "A harbor log.", ["a.txt"]),
            "ct000": _answer("ct000", "A", "2 chunks in 1 document.", []),
        },
        "B": {
            "tq000": _answer("tq000", "B", "Near the docks.", ["a.txt"]),
            "ki000": _answer("ki000", "B", "A harbor log.", ["b.txt"]),
            "ct000": _answer("ct000", "B", "About 7 mentions.", []),
        },
    }
    unblinded = {
        "A": {0: {"tq000": _grades(2, 2, 2), "ki000": _grades(2, 2, 2)}},
        "B": {
            0: {"tq000": _grades(1, 1, 1), "ki000": _grades(1, 2, 1)},
            1: {"tq000": _grades(1, 1, 0), "ki000": _grades(1, 2, 1)},
        },
    }
    return questions, answers, unblinded


def test_build_results_emits_question_rows_and_a_summary():
    questions, answers, unblinded = _pipeline_fixture()
    rows = build_results(questions, answers, unblinded, noise_arm="B", judged=unblinded)
    summary = rows[-1]
    assert summary["row_type"] == ResultRowType.SUMMARY
    assert summary["noise_floor"] == pytest.approx(1 / 6, abs=1e-3)
    arm_a = summary["arms"]["A"]
    assert arm_a["means"]["faithfulness"] == 2.0
    assert arm_a["count_pass"] == [1, 1]
    assert arm_a["known_item_pass"] == [1, 1]
    assert arm_a["errors"] == 0
    arm_b = summary["arms"]["B"]
    assert arm_b["count_pass"] == [0, 1]
    assert arm_b["known_item_pass"] == [0, 1]

    question_rows = [r for r in rows if r["row_type"] == ResultRowType.QUESTION]
    assert len(question_rows) == 6
    topical_a = next(r for r in question_rows if r["qid"] == "tq000" and r["arm"] == "A")
    assert topical_a["grades"] == _grades(2, 2, 2)
    count_b = next(r for r in question_rows if r["qid"] == "ct000" and r["arm"] == "B")
    assert count_b["exact_pass"] is False


def test_build_results_counts_hard_failures():
    questions, answers, unblinded = _pipeline_fixture()
    answers["B"]["tq000"] = AnswerRow(
        qid="tq000",
        arm="B",
        answer="",
        sources=[],
        cited_sources=[],
        seconds=0.0,
        error="ConnectError: refused",
    )
    del answers["B"]["ct000"]
    rows = build_results(questions, answers, unblinded, noise_arm="B", judged=unblinded)
    summary = rows[-1]
    assert summary["arms"]["B"]["errors"] == 2


def test_count_pass_accepts_an_answer_to_the_question_actually_asked():
    # The generated question is "How many documents mention X?", one number.
    # Requiring the chunk count too fails a perfectly correct answer, which made
    # the metric read near zero for both arms regardless of retrieval quality.
    oracle = CountOracle(term="lantern", chunks=14, sources=3)
    assert count_question_pass(oracle, "It appears in 3 documents.")


def test_count_pass_still_accepts_an_answer_that_volunteers_the_chunk_count():
    oracle = CountOracle(term="lantern", chunks=14, sources=3)
    assert count_question_pass(oracle, "It appears in 14 chunks across 3 documents.")


def test_count_pass_rejects_the_wrong_document_count():
    oracle = CountOracle(term="lantern", chunks=14, sources=3)
    assert not count_question_pass(oracle, "It appears in 5 documents.")
    assert not count_question_pass(oracle, "It appears in 14 chunks.")
