"""TREC run files and the chunk-to-document collapse both arms share.

A run file is the standard six-column TREC format::

    query_id Q0 doc_id rank score run_tag

Retrieval returns chunks; scoring happens at the document level, so chunks are
collapsed to their parent document (best score wins) and re-ranked before the
run file is written. Everything here is pure so it can be fully unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TREC_UNUSED_COLUMN = "Q0"


@dataclass(frozen=True)
class ChunkHit:
    """One retrieved chunk, tagged with the document it belongs to."""

    query_id: str
    doc_id: str
    score: float


@dataclass(frozen=True)
class RunEntry:
    """One ranked document for one query, as written to a TREC run file."""

    query_id: str
    doc_id: str
    rank: int
    score: float
    run_tag: str

    def to_line(self) -> str:
        return (
            f"{self.query_id} {TREC_UNUSED_COLUMN} {self.doc_id} "
            f"{self.rank} {self.score:.6f} {self.run_tag}"
        )

    @classmethod
    def from_line(cls, line: str) -> RunEntry:
        query_id, _unused, doc_id, rank, score, run_tag = line.split()
        return cls(
            query_id=query_id,
            doc_id=doc_id,
            rank=int(rank),
            score=float(score),
            run_tag=run_tag,
        )


def collapse_hits(hits: list[ChunkHit], run_tag: str) -> list[RunEntry]:
    """Collapse chunk hits to documents (best score wins) and re-rank per query.

    Multiple chunks from one document keep only that document's best score.
    Within each query, documents are ranked by descending score; ties break on
    doc_id so the ordering is deterministic and reproducible.
    """
    best: dict[str, dict[str, float]] = {}
    for hit in hits:
        per_query = best.setdefault(hit.query_id, {})
        if hit.doc_id not in per_query or hit.score > per_query[hit.doc_id]:
            per_query[hit.doc_id] = hit.score
    entries: list[RunEntry] = []
    for query_id in sorted(best):
        ranked = sorted(best[query_id].items(), key=lambda item: (-item[1], item[0]))
        for rank, (doc_id, score) in enumerate(ranked, start=1):
            entries.append(RunEntry(query_id, doc_id, rank, score, run_tag))
    return entries


def write_run(path: Path, entries: list[RunEntry]) -> None:
    """Serialize run entries to a TREC run file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(entry.to_line() + "\n" for entry in entries))


def read_run(path: Path) -> list[RunEntry]:
    """Parse a TREC run file; blank lines are skipped."""
    return [RunEntry.from_line(line) for line in path.read_text().splitlines() if line.strip()]


def run_to_pytrec(entries: list[RunEntry]) -> dict[str, dict[str, float]]:
    """Shape run entries as pytrec_eval expects: query_id -> doc_id -> score."""
    run: dict[str, dict[str, float]] = {}
    for entry in entries:
        run.setdefault(entry.query_id, {})[entry.doc_id] = entry.score
    return run
