"""Blind judging: one answer at a time against its ground truth."""

from __future__ import annotations

import json
import re
import sys
import time
from enum import StrEnum
from pathlib import Path

from evals.retrieval.blinding import BlindRow
from evals.retrieval.checkpoint import JsonlCheckpoint, load_jsonl
from evals.retrieval.llm import ChatFn

SCORE_MIN = 0
SCORE_MAX = 2
JUDGE_ATTEMPTS = 3
JUDGE_RETRY_DELAY_SECONDS = 5.0

# The LAST brace group, not the first: a judge that emits a worked example or a
# reasoning artifact before its answer would otherwise have that parsed as the
# grade.
_JSON_RE = re.compile(r"\{[^{}]*\}")


class Dimension(StrEnum):
    FAITHFULNESS = "faithfulness"
    RELEVANCE = "relevance"
    CITATION = "citation"


DIMENSIONS = tuple(str(dimension) for dimension in Dimension)

_RUBRIC = (
    "faithfulness: 2 = nothing in the answer contradicts the ground material and "
    "its main claim is supported by it, 1 = a minor detail the ground material "
    "neither supports nor contradicts, 0 = contradicts the ground material or "
    "invents specifics. Detail that is merely absent from the ground material is "
    "not itself a fault: the ground material is one passage, not the whole "
    "corpus, and the answer may draw on other passages you were not shown. "
    "relevance: 2 = directly answers the question, 1 = partial, 0 = misses. "
    "citation: 2 = names or cites the correct document, 1 = cites nothing, "
    "0 = cites a wrong document. Return ONLY the JSON."
)

# Two presentations of one grading task. Both carry identical content and the
# identical rubric; they differ only in the order the material is laid out and
# the order the JSON fields are requested.
#
# This is what makes the second judging pass a measurement. Both chat backends
# decode greedily at temperature 0, so re-sending a byte-identical prompt is the
# same computation twice and its "disagreement" is exactly zero -- it measures
# decoder determinism, not the judge. Grading the same content under an
# equivalent but differently-arranged prompt measures the judge's sensitivity to
# presentation, which is real judge instability and is still fully reproducible.
JUDGE_PROMPTS = (
    "You are grading an answer produced by a document-search assistant.\n"
    "Question: {question}\n\n"
    "Ground-truth material the answer should be based on (from document "
    "{source}):\n{ground}\n\n"
    "Answer to grade:\n{answer}\n\n"
    "Score STRICTLY as JSON with integer fields 0-2 each: "
    '{{"faithfulness": _, "relevance": _, "citation": _}}. ' + _RUBRIC,
    "You are grading an answer produced by a document-search assistant.\n"
    "Answer to grade:\n{answer}\n\n"
    "Question it was asked: {question}\n\n"
    "Ground-truth material the answer should be based on (from document "
    "{source}):\n{ground}\n\n"
    "Score STRICTLY as JSON with integer fields 0-2 each: "
    '{{"citation": _, "relevance": _, "faithfulness": _}}. ' + _RUBRIC,
)


def judge_prompt_for(row: BlindRow) -> str:
    """The prompt presentation this row's variant calls for."""
    template = JUDGE_PROMPTS[row.variant % len(JUDGE_PROMPTS)]
    return template.format(
        question=row.question, source=row.source, ground=row.ground, answer=row.answer
    )


def parse_grade(text: str) -> dict[str, int] | None:
    """Scores clamped to the 0-2 scale, or None when the judge returned junk."""
    matches = _JSON_RE.findall(text)
    if not matches:
        return None
    match = matches[-1]
    try:
        scores = json.loads(match)
        graded = {dimension: scores[dimension] for dimension in DIMENSIONS}
    except (KeyError, TypeError, ValueError):
        return None
    # Reject rather than coerce. A 5 clamped to 2, or a 1.7 truncated to 1, is a
    # judge that did not follow the rubric being turned into a plausible score
    # that then flows into the published means. An unusable grade is data the
    # run does not have.
    if any(not isinstance(value, int) or isinstance(value, bool) for value in graded.values()):
        return None
    if any(value < SCORE_MIN or value > SCORE_MAX for value in graded.values()):
        return None
    return graded


def judge_rows(
    rows: list[BlindRow],
    chat: ChatFn,
    out_path: Path,
    *,
    attempts: int = JUDGE_ATTEMPTS,
    retry_delay: float = JUDGE_RETRY_DELAY_SECONDS,
) -> dict[str, dict[str, int]]:
    """Grade every row not already checkpointed; return all grades on disk."""
    checkpoint = JsonlCheckpoint(out_path, "gid")
    for row in rows:
        if row.gid in checkpoint:
            continue
        grade: dict[str, int] | None = None
        prompt = judge_prompt_for(row)
        # Only transport failures are retried, and always under this row's own
        # presentation. Both backends decode greedily, so an unparseable
        # response is deterministic and re-asking cannot change it, which is why
        # the old three identical attempts were three calls for one result.
        # Re-asking under the *other* presentation would be worse still: a
        # retried noise replicate would then be graded under exactly the
        # presentation its twin got, so that pair would contribute zero
        # disagreement and quietly deflate the floor it exists to measure.
        for attempt in range(attempts):
            try:
                grade = parse_grade(chat(prompt))
            except Exception as exc:
                print(f"judge call failed for {row.gid}: {exc}", file=sys.stderr)
                if attempt + 1 < attempts:
                    time.sleep(retry_delay)
                continue
            break
        if grade is None:
            print(f"no parseable grade for {row.gid}; leaving unreturned", file=sys.stderr)
            continue
        checkpoint.append({"gid": row.gid, **grade})
    return {
        record["gid"]: {dimension: record[dimension] for dimension in DIMENSIONS}
        for record in load_jsonl(out_path)
    }
