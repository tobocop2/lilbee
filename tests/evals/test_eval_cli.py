"""End-to-end pipeline on synthetic fixtures via the CLI subcommands."""

import json

from evals.retrieval import cli
from evals.retrieval.answers import AnswerRow
from evals.retrieval.checkpoint import load_items, load_jsonl
from evals.retrieval.questions import CountOracle, Question, QuestionKind
from evals.retrieval.scoring import ResultRowType


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class _JudgeBackend:
    """Stand-in for the real judge endpoint: a chat fn plus its identity."""

    def __init__(self, chat, model="test-judge", base_url="http://judge"):
        self.chat = chat
        self.model = model
        self.base_url = base_url


def _fixture_files(tmp_path):
    questions = [
        Question(
            qid="tq000",
            kind=QuestionKind.TOPICAL,
            question="Where was the light?",
            source="a.txt",
            ground_passage="ground passage",
        ),
        Question(
            qid="ct000",
            kind=QuestionKind.COUNT,
            question="How many documents mention 'lantern'?",
            oracle=CountOracle(term="lantern", chunks=2, sources=1),
        ),
    ]
    questions_path = tmp_path / "questions.jsonl"
    _write_jsonl(questions_path, [q.to_dict() for q in questions])

    def _answer(qid, arm, answer):
        return AnswerRow(
            qid=qid,
            arm=arm,
            answer=answer,
            sources=["a.txt"],
            cited_sources=["a.txt"],
            seconds=0.2,
            error=None,
        )

    answers_a = tmp_path / "answers-a.jsonl"
    _write_jsonl(
        answers_a,
        [
            _answer("tq000", "old", "On the hill.").to_dict(),
            _answer("ct000", "old", "2 chunks in 1 document.").to_dict(),
        ],
    )
    answers_b = tmp_path / "answers-b.jsonl"
    _write_jsonl(
        answers_b,
        [
            _answer("tq000", "new", "On the headland.").to_dict(),
            _answer("ct000", "new", "2 chunks in 1 document.").to_dict(),
        ],
    )
    return questions_path, answers_a, answers_b


def test_judge_score_report_pipeline(tmp_path, monkeypatch):
    questions_path, answers_a, answers_b = _fixture_files(tmp_path)
    work_dir = tmp_path / "work"

    fixed_grade = '{"faithfulness": 2, "relevance": 2, "citation": 1}'
    monkeypatch.setattr(cli, "judge_backend", lambda: _JudgeBackend(lambda _prompt: fixed_grade))
    monkeypatch.setattr(cli, "warm_chat", lambda chat: None)

    exit_code = cli.main(
        [
            "judge",
            "--questions",
            str(questions_path),
            "--answers-a",
            str(answers_a),
            "--answers-b",
            str(answers_b),
            "--work-dir",
            str(work_dir),
            "--seed",
            "3",
        ]
    )
    assert exit_code == 0
    assert len(load_jsonl(work_dir / "grades.jsonl")) == 3
    gid_map = json.loads((work_dir / "gid_map.json").read_text())
    assert len(gid_map) == 3

    results_path = tmp_path / "results.jsonl"
    exit_code = cli.main(
        [
            "score",
            "--questions",
            str(questions_path),
            "--answers-a",
            str(answers_a),
            "--answers-b",
            str(answers_b),
            "--work-dir",
            str(work_dir),
            "--out",
            str(results_path),
        ]
    )
    assert exit_code == 0
    rows = load_jsonl(results_path)
    summary = rows[-1]
    assert summary["row_type"] == ResultRowType.SUMMARY
    assert set(summary["arms"]) == {"old", "new"}
    assert summary["noise_floor"] == 0.0
    assert summary["arms"]["new"]["count_pass"] == [1, 1]

    report_path = tmp_path / "report.md"
    exit_code = cli.main(["report", "--results", str(results_path), "--out", str(report_path)])
    assert exit_code == 0
    text = report_path.read_text()
    assert "old" in text
    assert "new" in text
    assert "noise floor" in text.lower()


def test_judge_rejects_duplicate_arm_labels(tmp_path, monkeypatch):
    questions_path, answers_a, _ = _fixture_files(tmp_path)
    monkeypatch.setattr(cli, "judge_backend", lambda: _JudgeBackend(lambda _prompt: "{}"))
    monkeypatch.setattr(cli, "warm_chat", lambda chat: None)
    exit_code = cli.main(
        [
            "judge",
            "--questions",
            str(questions_path),
            "--answers-a",
            str(answers_a),
            "--answers-b",
            str(answers_a),
            "--work-dir",
            str(tmp_path / "w"),
        ]
    )
    assert exit_code == 1


def test_answer_subcommand_writes_checkpointed_answers(tmp_path, monkeypatch):
    questions_path, _, _ = _fixture_files(tmp_path)
    out = tmp_path / "answers.jsonl"

    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/health":
            return httpx.Response(200)
        if request.url.path == "/api/memories":
            return httpx.Response(404)  # memory subsystem off, the default
        return httpx.Response(200, json={"answer": "ok", "sources": [], "cited_sources": []})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    monkeypatch.setattr(cli, "make_http_client", lambda: client)
    exit_code = cli.main(
        [
            "answer",
            "--questions",
            str(questions_path),
            "--base-url",
            "http://test",
            "--arm",
            "old",
            "--out",
            str(out),
        ]
    )
    assert exit_code == 0
    assert len(load_items(out)) == 2
