"""A grader set standing in for ragas' rubric metrics.

ragas refuses any LLM that is not its own ``InstructorLLM``, so a test cannot
reach the judge by handing in a fake model. The seam is the built graders
instead, which is the right level anyway: prompt construction, structured
output, and retry belong to ragas now, and what the harness still owns is the
loop around them.

A plain module rather than a conftest fixture, so every test file that needs it
imports it the same way whatever the runner's conftest situation is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evals.retrieval import judging
from evals.retrieval.judging import DIMENSIONS, PRESENTATIONS


@dataclass
class StubResult:
    """What ragas' ``ascore`` returns: a metric result carrying a value."""

    value: float


class StubGrader:
    """One dimension's grader under one rubric presentation."""

    def __init__(self, score: int = 4, fail_times: int = 0) -> None:
        self.score = score
        self.fail_times = fail_times
        self.calls: list[dict[str, Any]] = []

    async def ascore(self, **kwargs: Any) -> StubResult:
        self.calls.append(kwargs)
        if len(self.calls) <= self.fail_times:
            raise RuntimeError("connection reset")
        return StubResult(value=float(self.score))


def stub_graders(score: int = 4, fail_times: int = 0) -> dict[tuple[str, int], StubGrader]:
    """A full grader set, shaped as ``build_graders`` returns it."""
    return {
        (dimension, variant): StubGrader(score=score, fail_times=fail_times)
        for dimension in DIMENSIONS
        for variant in range(PRESENTATIONS)
    }


def install_stub_graders(
    monkeypatch: Any, score: int = 4, fail_times: int = 0
) -> dict[tuple[str, int], StubGrader]:
    """Patch ragas' graders out of ``judge_rows`` and return the stubs."""
    built = stub_graders(score=score, fail_times=fail_times)
    monkeypatch.setattr(judging, "build_graders", lambda _llm: built)
    return built
