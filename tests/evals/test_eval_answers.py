"""Answer collection over HTTP: warm-up, retries, and checkpointed resume."""

import json

import httpx
import pytest
from evals.retrieval.answers import AnswerRow, answer_questions, questions_digest, wait_for_server
from evals.retrieval.checkpoint import load_items
from evals.retrieval.questions import Question, QuestionKind


def _question(qid: str = "tq000") -> Question:
    return Question(qid=qid, kind=QuestionKind.TOPICAL, question="Where?", source="a.txt")


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")


def test_wait_for_server_returns_once_healthy():
    states = iter([503, 503, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/health"
        return httpx.Response(next(states))

    wait_for_server("http://test", _client(handler), attempts=5, poll=0)


def test_wait_for_server_raises_after_budget():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(RuntimeError, match="never became healthy"):
        wait_for_server("http://test", _client(handler), attempts=2, poll=0)


def _ask_response(answer: str = "By the pier.") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "answer": answer,
            "sources": [{"source": "a.txt", "content_type": "text", "chunk": "c"}],
            "cited_sources": [{"source": "a.txt", "content_type": "text", "chunk": "c"}],
        },
    )


def test_answer_questions_records_answer_sources_and_citations(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/health":
            return httpx.Response(200)
        if request.url.path == "/api/memories":
            return httpx.Response(404)  # memory subsystem off, the default
        assert json.loads(request.content)["question"] == "Where?"
        return _ask_response()

    out = tmp_path / "answers.jsonl"
    rows = answer_questions(
        [_question()], "http://test", "armA", out, retry_delay=0, client=_client(handler)
    )
    assert len(rows) == 1
    row = AnswerRow.from_dict(load_items(out)[0])
    assert row.arm == "armA"
    assert row.answer == "By the pier."
    assert row.sources == ["a.txt"]
    assert row.cited_sources == ["a.txt"]
    assert row.error is None


def test_answer_questions_retries_then_records_the_failure(tmp_path):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/health":
            return httpx.Response(200)
        if request.url.path == "/api/memories":
            return httpx.Response(404)  # memory subsystem off, the default
        seen.append(request.url.path)
        return httpx.Response(500)

    out = tmp_path / "answers.jsonl"
    rows = answer_questions(
        [_question()],
        "http://test",
        "armA",
        out,
        attempts=3,
        retry_delay=0,
        client=_client(handler),
    )
    assert len(seen) == 3
    assert rows[0].error is not None
    assert rows[0].answer == ""


def test_answer_questions_resumes_past_checkpointed_rows(tmp_path):
    out = tmp_path / "answers.jsonl"
    done = AnswerRow(
        qid="tq000",
        arm="armA",
        answer="cached",
        sources=[],
        cited_sources=[],
        seconds=0.1,
        error=None,
    )
    questions = [_question("tq000"), _question("tq001")]
    provenance = {
        "_checkpoint": {
            "arm": "armA",
            "base_url": "http://test",
            "top_k": 0,
            "questions": questions_digest(questions),
        }
    }
    out.write_text(json.dumps(provenance) + "\n" + json.dumps(done.to_dict()) + "\n")
    asked: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/health":
            return httpx.Response(200)
        if request.url.path == "/api/memories":
            return httpx.Response(404)  # memory subsystem off, the default
        asked.append(json.loads(request.content)["question"])
        return _ask_response("fresh")

    rows = answer_questions(
        questions, "http://test", "armA", out, retry_delay=0, client=_client(handler)
    )
    assert [row.qid for row in rows] == ["tq001"]
    assert asked == ["Where?"]
    assert len(load_items(out)) == 2


def test_collection_refuses_a_server_with_memory_enabled(tmp_path):
    # With memory on, the ask handler extracts memories that later questions and
    # the other arm read back, so the arms stop answering under identical
    # conditions. The memory routes 404 only when the subsystem is off.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/health":
            return httpx.Response(200)
        if request.url.path == "/api/memories":
            return httpx.Response(200, json={"memories": []})  # enabled
        return _ask_response("never reached")

    with pytest.raises(RuntimeError, match="memory subsystem enabled"):
        answer_questions(
            [_question()],
            "http://test",
            "armA",
            tmp_path / "answers.jsonl",
            retry_delay=0,
            client=_client(handler),
        )
