"""Blind rubric grading, delegated to ragas.

``ragas.metrics.collections.DomainSpecificRubrics`` owns the whole grading call:
it builds the prompt, requests a typed ``RubricScoreOutput`` through instructor,
and retries a response that does not validate. What used to live here -- a
prompt template, a last-brace-wins regex, a hand-written range check, and a
three-attempt retry loop -- was a reimplementation of that, and each piece had
its own way of being wrong.

What stays here is the part ragas does not ship: the three dimensions this study
grades on, the rubric text for each, and the two equivalent presentations of
each rubric that make the noise floor measurable.
"""

from __future__ import annotations

import asyncio
import sys
from enum import StrEnum
from pathlib import Path
from typing import Any

from evals.retrieval.blinding import BlindRow
from evals.retrieval.checkpoint import JsonlCheckpoint, load_jsonl
from evals.retrieval.llm import RAGAS_INSTALL_HINT

# ragas' rubric metrics score on a fixed 1-5 scale (``allowed_values``), so the
# rubrics below spell out five levels rather than the three this study used
# while it owned the prompt.
SCORE_MIN = 1
SCORE_MAX = 5


class Dimension(StrEnum):
    FAITHFULNESS = "faithfulness"
    RELEVANCE = "relevance"
    CITATION = "citation"


DIMENSIONS = tuple(str(dimension) for dimension in Dimension)

# Two presentations of each rubric, carrying identical criteria. They differ
# only in wording and in the order ``format_rubrics`` lays the levels out.
#
# This is what makes the second judging pass a measurement rather than a second
# copy of the first. Both chat backends decode greedily at temperature 0, so
# re-sending a byte-identical prompt is the same computation twice and its
# "disagreement" is exactly zero: it measures decoder determinism, not the
# judge. Grading the same content under an equivalent but differently-arranged
# rubric measures the judge's sensitivity to presentation, which is real judge
# instability and is still fully reproducible.
#
# Detail merely absent from the ground material is not a fault in any of these:
# the ground material is one passage, not the whole corpus, and the answer may
# draw on passages the judge was not shown.
_FAITHFULNESS_ASCENDING = {
    "score1_description": "The response contradicts the ground material or invents specifics.",
    "score2_description": "The response's main claim is unsupported by the ground material.",
    "score3_description": "The response's main claim is supported, but a detail is unsupported.",
    "score4_description": "The response is supported apart from a minor unverifiable detail.",
    "score5_description": "Every claim in the response is supported by the ground material.",
}
_FAITHFULNESS_DESCENDING = {
    "score5_description": "Nothing in the response goes beyond what the ground material states.",
    "score4_description": "The response is faithful except for one small unverifiable detail.",
    "score3_description": "The central claim holds up, though some detail is not borne out.",
    "score2_description": "The ground material does not bear out what the response mainly claims.",
    "score1_description": "The response invents specifics or says the opposite of the material.",
}

_RELEVANCE_ASCENDING = {
    "score1_description": "The response misses the question entirely.",
    "score2_description": "The response addresses the topic but not the question asked.",
    "score3_description": "The response answers part of the question.",
    "score4_description": "The response answers the question but leaves a gap.",
    "score5_description": "The response directly and completely answers the question.",
}
_RELEVANCE_DESCENDING = {
    "score5_description": "The question is answered directly, with nothing left outstanding.",
    "score4_description": "The question is answered, though something asked for is missing.",
    "score3_description": "Only part of what was asked is answered.",
    "score2_description": "The subject matter is right but the question is not engaged.",
    "score1_description": "The question is not answered at all.",
}

_CITATION_ASCENDING = {
    "score1_description": "The response cites a document that is not the correct one.",
    "score2_description": "The response cites several documents, the correct one among them.",
    "score3_description": "The response cites nothing.",
    "score4_description": "The response points at the correct document indirectly.",
    "score5_description": "The response names or cites the correct document.",
}
_CITATION_DESCENDING = {
    "score5_description": "The correct document is named or cited outright.",
    "score4_description": "The correct document is identifiable from what the response says.",
    "score3_description": "No document is cited either way.",
    "score2_description": "The correct document appears in a list of several cited.",
    "score1_description": "A wrong document is cited.",
}

# dimension -> presentation variant -> rubric. Indexed by BlindRow.variant.
RUBRICS: dict[str, tuple[dict[str, str], ...]] = {
    Dimension.FAITHFULNESS: (_FAITHFULNESS_ASCENDING, _FAITHFULNESS_DESCENDING),
    Dimension.RELEVANCE: (_RELEVANCE_ASCENDING, _RELEVANCE_DESCENDING),
    Dimension.CITATION: (_CITATION_ASCENDING, _CITATION_DESCENDING),
}

PRESENTATIONS = len(RUBRICS[Dimension.FAITHFULNESS])


def rubric_for(dimension: str, variant: int) -> dict[str, str]:
    """The rubric this dimension is graded under for a given presentation variant.

    Wraps rather than indexing out of range, so adding a third replicate cycles
    through the presentations instead of raising.
    """
    variants = RUBRICS[dimension]
    return variants[variant % len(variants)]


def build_graders(llm: Any) -> dict[tuple[str, int], Any]:
    """One ragas rubric metric per (dimension, presentation variant).

    Built once per run: each instance formats its rubric into the prompt at
    construction, so rebuilding per row would repeat that work for every grade.
    """
    try:
        from ragas.metrics.collections import DomainSpecificRubrics
    except ImportError as exc:
        raise RuntimeError(RAGAS_INSTALL_HINT) from exc
    graders = {
        (dimension, variant): DomainSpecificRubrics(
            llm=llm,
            rubrics=rubric,
            with_reference=True,
            name=f"{dimension}_v{variant}",
        )
        for dimension, variants in RUBRICS.items()
        for variant, rubric in enumerate(variants)
    }
    # ragas fixes the rubric scale itself. The rubrics above spell out exactly
    # those levels and the report prints that range beside every mean, so a ragas
    # release that moves it must fail here rather than silently grade on one
    # scale while the report claims another.
    expected = (float(SCORE_MIN), float(SCORE_MAX))
    mismatched = sorted(
        f"{name[0]}_v{name[1]}"
        for name, grader in graders.items()
        if tuple(float(bound) for bound in grader.allowed_values) != expected
    )
    if mismatched:
        raise RuntimeError(
            f"ragas grades on a different scale than this harness declares "
            f"({expected}): {', '.join(mismatched)}. Update the rubrics and "
            "SCORE_MIN/SCORE_MAX together, since the report labels its means "
            "with that range."
        )
    return graders


async def _grade_row(graders: dict[tuple[str, int], Any], row: BlindRow) -> dict[str, int]:
    """Every dimension's score for one blind row, under that row's variant."""
    variant = row.variant % PRESENTATIONS
    results = await asyncio.gather(
        *(
            graders[(dimension, variant)].ascore(
                user_input=row.question,
                response=row.answer,
                reference=row.ground,
                retrieved_contexts=[row.ground],
            )
            for dimension in DIMENSIONS
        )
    )
    return {
        dimension: int(result.value) for dimension, result in zip(DIMENSIONS, results, strict=True)
    }


def judge_rows(rows: list[BlindRow], llm: Any, out_path: Path) -> dict[str, dict[str, int]]:
    """Grade every row not already checkpointed; return all grades on disk."""
    checkpoint = JsonlCheckpoint(out_path, "gid")
    graders = build_graders(llm)

    async def run() -> None:
        for row in rows:
            if row.gid in checkpoint:
                continue
            try:
                grade = await _grade_row(graders, row)
            except Exception as exc:
                # The row is left unreturned rather than scored. Retrying here
                # would duplicate what instructor already does for a response
                # that fails validation, and a transport failure the OpenAI
                # client has already retried will not answer on a third ask.
                print(f"no grade for {row.gid}: {exc}", file=sys.stderr)
                continue
            checkpoint.append({"gid": row.gid, **grade})

    asyncio.run(run())
    return {
        record["gid"]: {dimension: record[dimension] for dimension in DIMENSIONS}
        for record in load_jsonl(out_path)
    }
