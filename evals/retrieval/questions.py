"""Corpus-agnostic question generation from an existing lilbee index.

Three kinds, each with ground truth attached at authoring time so judging
never needs the index again:

- topical: written by the chat model FROM a sampled stored passage, so the
  passage that must support the answer is known.
- known_item: asks what a sampled document is about; ground truth is the
  document's head chunks.
- count: asks how many chunks and documents mention a term; ground truth is
  an exact streaming scan of the store, no judge involved.
"""

from __future__ import annotations

import collections
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from evals.retrieval.llm import ChatFn
from evals.retrieval.store_scan import (
    count_term_hits,
    iter_chunks,
    iter_source_names,
    reservoir_sample,
    scan_passages_and_heads,
)

TOPICAL_QUESTIONS = 60
KNOWN_ITEM_QUESTIONS = 20
COUNT_QUESTIONS = 8
MIN_PASSAGE_CHARS = 400
PASSAGE_PROMPT_CHARS = 1800
MIN_QUESTION_CHARS = 16
TERM_DF_LOW = 0.05
TERM_DF_HIGH = 0.4
AUTHOR_ATTEMPTS = 3
AUTHOR_RETRY_DELAY_SECONDS = 5.0
DEFAULT_SEED = 20260714

_WORD_RE = re.compile(r"[a-z]{5,16}")

QUESTION_PROMPT = (
    "Below is a passage from a document. Write ONE specific question that this "
    "passage answers. The question must be answerable from the passage alone, "
    "must not mention 'the passage' or 'the text', and must not quote more "
    "than three consecutive words from it. Return ONLY the question.\n\n"
    "Passage:\n{passage}"
)


class QuestionKind(StrEnum):
    TOPICAL = "topical"
    KNOWN_ITEM = "known_item"
    COUNT = "count"


@dataclass
class CountOracle:
    """Exact scan result a count answer is checked against."""

    term: str
    chunks: int
    sources: int


@dataclass
class Question:
    """One eval question with its ground truth attached."""

    qid: str
    kind: QuestionKind
    question: str
    source: str = ""
    ground_passage: str = ""
    oracle: CountOracle | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Question:
        oracle = data.get("oracle")
        return cls(
            qid=data["qid"],
            kind=QuestionKind(data["kind"]),
            question=data["question"],
            source=data.get("source", ""),
            ground_passage=data.get("ground_passage", ""),
            oracle=CountOracle(**oracle) if oracle else None,
        )


def parse_question(text: str) -> str | None:
    """The first line as a question, or None when the model returned junk."""
    lines = text.strip().splitlines()
    if not lines:
        return None
    candidate = lines[0].strip().strip('"')
    if candidate.endswith("?") and len(candidate) >= MIN_QUESTION_CHARS:
        return candidate
    return None


def author_topical(
    passages: list[tuple[str, str]],
    chat: ChatFn,
    *,
    attempts: int = AUTHOR_ATTEMPTS,
    retry_delay: float = AUTHOR_RETRY_DELAY_SECONDS,
) -> list[Question]:
    """One question per sampled passage; a bad passage never kills the set."""
    questions: list[Question] = []
    for index, (source, passage) in enumerate(passages):
        prompt = QUESTION_PROMPT.format(passage=passage[:PASSAGE_PROMPT_CHARS])
        response = None
        for attempt in range(attempts):
            try:
                response = chat(prompt)
                break
            except Exception as exc:  # a failed question is skipped, not fatal
                print(
                    f"authoring attempt {attempt + 1} failed for {source}: {exc}", file=sys.stderr
                )
                time.sleep(retry_delay)
        if response is None:
            continue
        question = parse_question(response)
        if question is None:
            continue
        questions.append(
            Question(
                qid=f"tq{index:03d}",
                kind=QuestionKind.TOPICAL,
                question=question,
                source=source,
                ground_passage=passage,
            )
        )
    return questions


def sample_terms(passages: list[tuple[str, str]], count: int, rng: random.Random) -> list[str]:
    """Mid-document-frequency terms drawn from the sampled passages.

    Document frequency is estimated over the sample, never the full index;
    the exact oracle counts come from a full streaming scan afterwards.
    """
    per_source_terms: dict[str, set[str]] = collections.defaultdict(set)
    for source, passage in passages:
        per_source_terms[source].update(_WORD_RE.findall(passage.lower()))
    doc_freq: collections.Counter[str] = collections.Counter()
    for terms in per_source_terms.values():
        doc_freq.update(terms)
    total = len(per_source_terms)
    if not total:
        return []
    band = sorted(t for t, df in doc_freq.items() if TERM_DF_LOW <= df / total <= TERM_DF_HIGH)
    rng.shuffle(band)
    return band[:count]


def build_questions(
    lancedb_dir: Path,
    chat: ChatFn,
    *,
    topical: int = TOPICAL_QUESTIONS,
    known_item: int = KNOWN_ITEM_QUESTIONS,
    count: int = COUNT_QUESTIONS,
    seed: int = DEFAULT_SEED,
    min_passage_chars: int = MIN_PASSAGE_CHARS,
) -> list[Question]:
    """The full question battery: sampled sources, two streaming chunk scans."""
    rng = random.Random(seed)
    known_sources = sorted(reservoir_sample(iter_source_names(lancedb_dir), known_item, rng))
    scan = scan_passages_and_heads(
        iter_chunks(lancedb_dir),
        passage_count=topical,
        min_passage_chars=min_passage_chars,
        head_sources=set(known_sources),
        rng=rng,
    )
    questions = author_topical(scan.passages, chat)
    for index, name in enumerate(known_sources):
        questions.append(
            Question(
                qid=f"ki{index:03d}",
                kind=QuestionKind.KNOWN_ITEM,
                question=f"What is {name} about? Summarize it briefly.",
                source=name,
                ground_passage=scan.doc_heads.get(name, ""),
            )
        )
    terms = sample_terms(scan.passages, count, rng)
    hits = count_term_hits(iter_chunks(lancedb_dir), terms) if terms else {}
    for index, term in enumerate(terms):
        questions.append(
            Question(
                qid=f"ct{index:03d}",
                kind=QuestionKind.COUNT,
                question=f"How many documents mention {term}?",
                oracle=CountOracle(term=term, chunks=hits[term].chunks, sources=hits[term].sources),
            )
        )
    return questions
