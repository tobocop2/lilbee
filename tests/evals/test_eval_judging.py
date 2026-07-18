"""Judge plumbing: grade parsing, clamping, and checkpointed judging."""

from evals.retrieval.blinding import BlindRow
from evals.retrieval.checkpoint import load_jsonl
from evals.retrieval.judging import judge_rows, parse_grade


def _row(gid: str) -> BlindRow:
    return BlindRow(gid=gid, question="Where?", source="a.txt", ground="g", answer="a")


def test_parse_grade_extracts_json_from_noise():
    text = 'Sure. {"faithfulness": 2, "relevance": 1, "citation": 0} done.'
    assert parse_grade(text) == {"faithfulness": 2, "relevance": 1, "citation": 0}


def test_parse_grade_clamps_out_of_range_scores():
    text = '{"faithfulness": 9, "relevance": -3, "citation": 1}'
    assert parse_grade(text) == {"faithfulness": 2, "relevance": 0, "citation": 1}


def test_parse_grade_rejects_missing_dimensions_and_garbage():
    assert parse_grade('{"faithfulness": 2}') is None
    assert parse_grade("no json here") is None
    assert parse_grade('{"faithfulness": "high", "relevance": 1, "citation": 1}') is None


def test_judge_rows_grades_and_checkpoints(tmp_path):
    out = tmp_path / "grades.jsonl"
    grades = judge_rows(
        [_row("g1"), _row("g2")],
        lambda _prompt: '{"faithfulness": 2, "relevance": 2, "citation": 1}',
        out,
        retry_delay=0,
    )
    assert set(grades) == {"g1", "g2"}
    assert len(load_jsonl(out)) == 2


def test_judge_rows_resumes_and_keeps_existing_grades(tmp_path):
    out = tmp_path / "grades.jsonl"
    judge_rows(
        [_row("g1")],
        lambda _prompt: '{"faithfulness": 1, "relevance": 1, "citation": 1}',
        out,
        retry_delay=0,
    )
    judged: list[str] = []

    def chat(prompt: str) -> str:
        judged.append(prompt)
        return '{"faithfulness": 2, "relevance": 2, "citation": 2}'

    grades = judge_rows([_row("g1"), _row("g2")], chat, out, retry_delay=0)
    assert len(judged) == 1
    assert grades["g1"] == {"faithfulness": 1, "relevance": 1, "citation": 1}
    assert grades["g2"] == {"faithfulness": 2, "relevance": 2, "citation": 2}


def test_judge_rows_retries_and_skips_unparseable_rows(tmp_path):
    calls: list[str] = []

    def bad_judge(prompt: str) -> str:
        calls.append(prompt)
        return "not json"

    grades = judge_rows([_row("g1")], bad_judge, tmp_path / "g.jsonl", attempts=2, retry_delay=0)
    assert grades == {}
    assert len(calls) == 2
