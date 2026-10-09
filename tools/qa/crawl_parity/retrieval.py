"""Yardstick C: does lilbee's search return the page a visible sentence came from."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from tools.qa.crawl_parity.model import Difference, Kind, Layer, Mode, Page
from tools.qa.crawl_parity.sides import DRIVERS_DIR
from tools.qa.crawl_parity.thresholds import RetrievalLimits
from tools.qa.crawl_parity.tokens import tokenize
from tools.qa.crawl_parity.truth import Truth

INDEX_DRIVER = "lilbee_index.py"
MIN_WORDS = 3
MAX_WORDS = 40
NAME_CHARS = 16
RECALL_FEATURE = "recall-at-k"
RUN_TIMEOUT_SECONDS = 3600


@dataclass(frozen=True)
class RetrievalConfig:
    """The lilbee interpreter and embedding model both indexes are built with."""

    python: Path
    embedding_model: str


@dataclass(frozen=True)
class Question:
    """One visible sentence and the page it came from."""

    id: str
    text: str
    page: str


@dataclass(frozen=True)
class RetrievalResult:
    """Recall at k for one side's markdown, and how long lilbee took."""

    asked: int
    answered: tuple[str, ...]
    indexed_pages: int
    failed_pages: int
    index_seconds: float
    search_seconds: float

    @property
    def recall(self) -> float:
        """The share of questions whose page was among the results."""
        return len(self.answered) / self.asked if self.asked else 0.0


def document_name(page: str) -> str:
    """The file name a page is indexed under, the same on both sides."""
    return f"p-{hashlib.sha256(page.encode()).hexdigest()[:NAME_CHARS]}.md"


def _sample_rank(seed: int, page: str, sentence: str) -> str:
    """A sort key that orders sentences by a hash of the seed, so a sample repeats."""
    return hashlib.sha256(f"{seed}\n{page}\n{sentence}".encode()).hexdigest()


def questions(truths: dict[str, Truth], sample_size: int, seed: int) -> list[Question]:
    """A seeded sample of visible sentences that occur on exactly one page."""
    occurrences: Counter[str] = Counter()
    for truth in truths.values():
        occurrences.update(set(truth.sentences))
    candidates = [
        (path, sentence)
        for path, truth in sorted(truths.items())
        if truth.scorable()
        for sentence in truth.sentences
        if occurrences[sentence] == 1 and MIN_WORDS <= len(tokenize(sentence).words) <= MAX_WORDS
    ]
    chosen = sorted(candidates, key=lambda candidate: _sample_rank(seed, *candidate))
    chosen = chosen[:sample_size]
    return [Question(f"q{index}", text, page) for index, (page, text) in enumerate(chosen)]


def measure(
    pages: dict[str, Page], asked: list[Question], config: RetrievalConfig, top_k: int, work: Path
) -> RetrievalResult:
    """Index one side's saved pages with lilbee and ask every question."""
    work.parent.mkdir(parents=True, exist_ok=True)
    # An index is built in a directory no earlier run has used: lilbee keeps what it finds.
    work = Path(tempfile.mkdtemp(prefix=f"{work.name}.", dir=work.parent))
    documents = work / "data" / "documents"
    documents.mkdir(parents=True)
    for path, page in pages.items():
        if page.markdown:
            (documents / document_name(path)).write_text(page.markdown, encoding="utf-8")
    questions_file = work / "questions.jsonl"
    questions_file.write_text(
        "".join(json.dumps({"id": q.id, "text": q.text}) + "\n" for q in asked), encoding="utf-8"
    )
    out = work / "answers.json"
    command = [
        str(config.python),
        str(DRIVERS_DIR / INDEX_DRIVER),
        "--data",
        str(work / "data"),
        "--questions",
        str(questions_file),
        "--embedding-model",
        config.embedding_model,
        "--top-k",
        str(top_k),
        "--out",
        str(out),
    ]
    with (work / "index-stderr.txt").open("wb") as stderr:
        subprocess.run(
            command,
            stdout=stderr,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
            cwd=work,
            check=True,
            timeout=RUN_TIMEOUT_SECONDS,
        )
    record = json.loads(out.read_text(encoding="utf-8"))
    answered = tuple(
        q.id
        for q in asked
        if any(source.endswith(document_name(q.page)) for source in record["answers"][q.id])
    )
    return RetrievalResult(
        asked=len(asked),
        answered=answered,
        indexed_pages=record["added"],
        failed_pages=len(record["failed"]) + len(record["skipped"]),
        index_seconds=record["index_seconds"],
        search_seconds=record["search_seconds"],
    )


def compare(
    oracle: RetrievalResult,
    candidate: RetrievalResult,
    asked: list[Question],
    limits: RetrievalLimits,
    layer: Layer,
    mode: Mode,
) -> list[Difference]:
    """A difference when the candidate's recall is below the oracle's by more than the limit."""
    drop = oracle.recall - candidate.recall
    if drop <= limits.recall_drop_max:
        return []
    only_oracle = sorted(set(oracle.answered) - set(candidate.answered))
    by_id = {question.id: question for question in asked}
    detail = "; ".join(f"{by_id[i].page}: {by_id[i].text[:60]}" for i in only_oracle[:5])
    return [Difference(Kind.RECALL, mode, layer, RECALL_FEATURE, amount=drop, detail=detail)]
