"""Checkpointed JSONL writing: interrupted runs resume without redoing work."""

import json

from evals.retrieval.checkpoint import JsonlCheckpoint, load_jsonl


def test_load_jsonl_missing_file_returns_empty(tmp_path):
    assert load_jsonl(tmp_path / "absent.jsonl") == []


def test_load_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"qid": "a"}\n\n{"qid": "b"}\n')
    assert load_jsonl(path) == [{"qid": "a"}, {"qid": "b"}]


def test_append_records_and_marks_done(tmp_path):
    path = tmp_path / "out.jsonl"
    checkpoint = JsonlCheckpoint(path, "qid")
    assert "q1" not in checkpoint
    checkpoint.append({"qid": "q1", "answer": "x"})
    assert "q1" in checkpoint
    assert load_jsonl(path) == [{"qid": "q1", "answer": "x"}]


def test_resume_skips_previously_written_keys(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_text(json.dumps({"qid": "q1"}) + "\n")
    checkpoint = JsonlCheckpoint(path, "qid")
    assert "q1" in checkpoint
    assert "q2" not in checkpoint
    checkpoint.append({"qid": "q2"})
    assert {row["qid"] for row in load_jsonl(path)} == {"q1", "q2"}


def test_done_returns_a_copy(tmp_path):
    checkpoint = JsonlCheckpoint(tmp_path / "out.jsonl", "qid")
    checkpoint.done.add("phantom")
    assert "phantom" not in checkpoint
