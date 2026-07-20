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

_JSON_RE = re.compile(r"\{[^{}]*\}")


class Dimension(StrEnum):
    FAITHFULNESS = "faithfulness"
    RELEVANCE = "relevance"
    CITATION = "citation"


DIMENSIONS = tuple(str(dimension) for dimension in Dimension)

_RUBRIC = (
    "faithfulness: 2 = every claim supported by the ground material, "
    "1 = minor unsupported detail, 0 = contradicts or invents. "
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
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        scores = json.loads(match.group(0))
        return {
            dimension: max(SCORE_MIN, min(SCORE_MAX, int(scores[dimension])))
            for dimension in DIMENSIONS
        }
    except (KeyError, TypeError, ValueError):
        return None


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
        prompt = judge_prompt_for(row)
        grade: dict[str, int] | None = None
        for attempt in range(attempts):
            try:
                grade = parse_grade(chat(prompt))
            except Exception as exc:  # a failed grade is skipped, not fatal
                print(f"judge call failed for {row.gid}: {exc}", file=sys.stderr)
                grade = None
            if grade is not None:
                break
            if attempt + 1 < attempts:
                time.sleep(retry_delay)
        if grade is None:
            print(f"no parseable grade for {row.gid}; leaving unreturned", file=sys.stderr)
            continue
        checkpoint.append({"gid": row.gid, **grade})
    return {
        record["gid"]: {dimension: record[dimension] for dimension in DIMENSIONS}
        for record in load_jsonl(out_path)
    }
