"""Judge plumbing: what the harness still owns now that ragas grades.

Grade parsing, range checking, and retry-on-unparseable used to live here and
are gone: ragas requests a typed ``RubricScoreOutput`` through instructor, which
validates the score and retries a response that does not conform. Testing our
own copy of that would have been testing code that no longer exists.

What remains ours, and is what these exercise: one grade per dimension per row,
checkpointing and resume, and a row whose grading raised staying unreturned
rather than being scored.
"""

import pytest
from evals.retrieval.blinding import BlindRow
from evals.retrieval.checkpoint import load_jsonl
from evals.retrieval.judging import DIMENSIONS, PRESENTATIONS, RUBRICS, build_graders, judge_rows

from tests.evals.stub_judge import install_stub_graders

pytest.importorskip("ragas")


@pytest.fixture
def graders(monkeypatch):
    """Install stub graders in place of ragas' and hand them back."""

    def install(score: int = 4, fail_times: int = 0):
        return install_stub_graders(monkeypatch, score=score, fail_times=fail_times)

    return install


def _row(gid: str, variant: int = 0) -> BlindRow:
    return BlindRow(
        gid=gid, question="Where?", source="a.txt", ground="g", answer="a", variant=variant
    )


def test_judge_rows_grades_every_dimension_and_checkpoints(tmp_path, graders):
    graders()
    out = tmp_path / "grades.jsonl"
    grades = judge_rows([_row("g1"), _row("g2")], None, out)
    assert set(grades) == {"g1", "g2"}
    assert set(grades["g1"]) == set(DIMENSIONS)
    assert len(load_jsonl(out)) == 2


def test_each_row_is_graded_under_its_own_presentation(tmp_path, graders):
    # The noise arm's two replicates must reach different graders, or the second
    # pass is the first pass again and the floor it measures is zero.
    built = graders()
    judge_rows([_row("g1", variant=0), _row("g2", variant=1)], None, tmp_path / "g.jsonl")
    assert len(built[("faithfulness", 0)].calls) == 1
    assert len(built[("faithfulness", 1)].calls) == 1


def test_judge_rows_resumes_and_keeps_existing_grades(tmp_path, graders):
    out = tmp_path / "grades.jsonl"
    graders(score=1)
    judge_rows([_row("g1")], None, out)
    second = graders(score=5)
    grades = judge_rows([_row("g1"), _row("g2")], None, out)
    # Only the new row was sent to the judge; g1's earlier grade is kept.
    assert len(second[("faithfulness", 0)].calls) == 1
    assert grades["g1"]["faithfulness"] == 1
    assert grades["g2"]["faithfulness"] == 5


def test_a_row_whose_grading_raised_is_left_unreturned(tmp_path, graders):
    # Not scored, not retried here: instructor already retried a response that
    # failed validation, and a transport error the client retried will not
    # answer on a third ask from this loop. An ungraded row is data the run does
    # not have, and scoring it would invent one.
    graders(fail_times=99)
    assert judge_rows([_row("g1")], None, tmp_path / "g.jsonl") == {}


def test_one_failed_row_does_not_stop_the_rest(tmp_path, graders):
    graders(fail_times=1)
    grades = judge_rows([_row("g1"), _row("g2")], None, tmp_path / "g.jsonl")
    assert set(grades) == {"g2"}


def test_build_graders_binds_one_ragas_metric_per_dimension_and_presentation():
    # Built against the real ragas API, so a signature or scale change upstream
    # fails here rather than on the pod after a run has started. The client is
    # never called: only construction is exercised.
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    llm = llm_factory(
        "judge-model", provider="openai", client=AsyncOpenAI(base_url="http://unused", api_key="x")
    )
    graders = build_graders(llm)
    assert set(graders) == {
        (dimension, variant) for dimension in DIMENSIONS for variant in range(PRESENTATIONS)
    }
    # Each metric carries its own rubric text in the prompt it will send.
    for (dimension, variant), grader in graders.items():
        instruction = grader.scoring_prompt.instruction
        for description in RUBRICS[dimension][variant].values():
            assert description in instruction
