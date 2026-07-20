"""Question generation: topical authoring, known-item, and count oracles."""

import random

import pytest
from evals.retrieval import questions as questions_mod
from evals.retrieval.questions import (
    CountOracle,
    Question,
    QuestionKind,
    author_topical,
    build_questions,
    parse_question,
    sample_terms,
)
from evals.retrieval.store_scan import ChunkRow


def test_parse_question_takes_first_line_and_strips_quotes():
    assert parse_question('"Where did the fleet anchor?"\nextra') == "Where did the fleet anchor?"


@pytest.mark.parametrize("text", ["", "not a question", "Why?", "Too short?"])
def test_parse_question_rejects_non_questions(text):
    assert parse_question(text) is None


def test_sample_terms_picks_mid_frequency_terms():
    passages = [(f"s{i}.txt", "common words everywhere") for i in range(10)]
    passages += [("s0.txt", "unique zebra sighting"), ("s1.txt", "zebra again here")]
    terms = sample_terms(passages, 5, random.Random(1))
    assert "common" not in terms
    assert "zebra" in terms


def test_sample_terms_respects_count():
    passages = [(f"s{i}.txt", f"word{i}alpha word{i}beta shared") for i in range(20)]
    assert len(sample_terms(passages, 3, random.Random(1))) <= 3


def test_author_topical_builds_questions_with_ground_passages():
    picks = [("a.txt", "p" * 500), ("b.txt", "q" * 500)]
    calls: list[str] = []

    def chat(prompt: str) -> str:
        calls.append(prompt)
        return "What color was the harbor light?"

    authored = author_topical(picks, chat, retry_delay=0)
    assert len(authored) == 2
    assert authored[0].kind is QuestionKind.TOPICAL
    assert authored[0].source == "a.txt"
    assert authored[0].ground_passage == "p" * 500
    assert "p" * 100 in calls[0]


def test_author_topical_retries_then_skips_hard_failures():
    attempts: list[int] = []

    def flaky_chat(prompt: str) -> str:
        attempts.append(1)
        raise RuntimeError("model busy")

    authored = author_topical([("a.txt", "p" * 500)], flaky_chat, attempts=3, retry_delay=0)
    assert authored == []
    assert len(attempts) == 3


def test_author_topical_drops_malformed_questions():
    authored = author_topical([("a.txt", "p" * 500)], lambda _prompt: "no.", retry_delay=0)
    assert authored == []


def test_question_round_trips_through_dict():
    question = Question(
        qid="ct000",
        kind=QuestionKind.COUNT,
        question="How many documents mention 'lantern'?",
        oracle=CountOracle(term="lantern", chunks=4, sources=2),
    )
    assert Question.from_dict(question.to_dict()) == question


def test_build_questions_assembles_all_three_kinds(monkeypatch):
    rare_words = ["lantern", "beacon", "harbor", "vessel", "anchor", "compass"]
    chunk_rows = [ChunkRow(f"doc{i}.txt", f"{rare_words[i]} " + "x" * 450, 0) for i in range(6)]
    monkeypatch.setattr(questions_mod, "iter_chunks", lambda _dir: iter(chunk_rows))
    monkeypatch.setattr(
        questions_mod, "iter_source_names", lambda _dir: iter(f"doc{i}.txt" for i in range(6))
    )

    def chat(prompt: str) -> str:
        return "Which lantern hung over the pier that night?"

    built = build_questions(lancedb_dir=None, chat=chat, topical=5, known_item=2, count=1, seed=11)
    kinds = [q.kind for q in built]
    assert kinds.count(QuestionKind.TOPICAL) == 5
    assert kinds.count(QuestionKind.KNOWN_ITEM) == 2
    assert kinds.count(QuestionKind.COUNT) == 1
    count_question = next(q for q in built if q.kind is QuestionKind.COUNT)
    assert count_question.oracle is not None
    assert count_question.oracle.chunks >= 1
    known = next(q for q in built if q.kind is QuestionKind.KNOWN_ITEM)
    assert known.source in known.question
    assert known.ground_passage
    assert len({q.qid for q in built}) == len(built)


def test_build_questions_skips_the_count_scan_when_no_terms_qualify(monkeypatch):
    scans: list[int] = []

    def counted_iter_chunks(_dir):
        scans.append(1)
        return iter([ChunkRow("doc0.txt", "y" * 450, 0)])

    monkeypatch.setattr(questions_mod, "iter_chunks", counted_iter_chunks)
    monkeypatch.setattr(questions_mod, "iter_source_names", lambda _dir: iter(["doc0.txt"]))
    built = build_questions(
        lancedb_dir=None,
        chat=lambda _p: "Where did the launch dock?",
        topical=1,
        known_item=1,
        count=4,
        seed=1,
    )
    assert all(q.kind is not QuestionKind.COUNT for q in built)
    assert len(scans) == 1
