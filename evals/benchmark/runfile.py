"""TREC run files and the chunk-to-document collapse both arms share.

Reading is ``ir_measures.read_trec_run``: the format is standard and its parser
is the one the scorer already ships. What stays here is the part no library
owns, because it is about this system rather than about TREC: retrieval returns
chunks, scoring happens at the document level, so chunks are collapsed to their
parent document (best score wins) and re-ranked before the run file is written.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evals.deps import install_hint

TREC_UNUSED_COLUMN = "Q0"

RUN_INSTALL_HINT = install_hint("ir_measures", "to read run and qrels files")


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


def rank_documents(scored: dict[str, float]) -> list[tuple[str, float]]:
    """One query's documents ordered by descending score, ties broken on doc_id descending.

    That tie rule is trec_eval's, and it is chosen to match rather than to be
    merely deterministic: the scorer is handed a doc_id to score map and re-sorts
    it with its own rule. An ascending tie-break here would write a rank column
    stating one order while the scorer used the reverse, which matters wherever
    rank fusion puts equal scores near a metric's cut depth.

    Every place that assigns a rank goes through here. A second copy of this sort
    is a second chance to pick the other tie rule, and the two would disagree
    only on the ties, which is exactly where it is hardest to notice.
    """
    return sorted(scored.items(), key=lambda item: (item[1], item[0]), reverse=True)


def collapse_hits(
    hits: list[ChunkHit], run_tag: str, *, limit: int | None = None
) -> list[RunEntry]:
    """Collapse chunk hits to documents (best score wins) and re-rank per query.

    Multiple chunks from one document keep only that document's best score, and
    ``rank_documents`` orders what survives.

    ``limit`` caps each query at that many documents after ranking. A chunk-level
    arm over-fetches chunks to reach the target document depth and can overshoot
    on the final page, so capping here is what makes both arms' runs the same
    document depth rather than whatever their pagination happened to land on.
    """
    best: dict[str, dict[str, float]] = {}
    for hit in hits:
        per_query = best.setdefault(hit.query_id, {})
        if hit.doc_id not in per_query or hit.score > per_query[hit.doc_id]:
            per_query[hit.doc_id] = hit.score
    entries: list[RunEntry] = []
    for query_id in sorted(best):
        ranked = rank_documents(best[query_id])
        if limit is not None:
            ranked = ranked[:limit]
        for rank, (doc_id, score) in enumerate(ranked, start=1):
            entries.append(RunEntry(query_id, doc_id, rank, score, run_tag))
    return entries


def write_run(path: Path, entries: list[RunEntry]) -> None:
    """Serialize run entries to a TREC run file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(entry.to_line() + "\n" for entry in entries))


def read_run(path: Path) -> dict[str, dict[str, float]]:
    """Parse a TREC run file to the ``query_id -> doc_id -> score`` map scorers take."""
    try:
        import ir_measures
    except ImportError as exc:
        raise RuntimeError(RUN_INSTALL_HINT) from exc
    run: dict[str, dict[str, float]] = {}
    for scored in ir_measures.read_trec_run(str(path)):
        run.setdefault(scored.query_id, {})[scored.doc_id] = float(scored.score)
    return run


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Parse a TREC qrels file to the ``query_id -> doc_id -> grade`` map scorers take."""
    try:
        import ir_measures
    except ImportError as exc:
        raise RuntimeError(RUN_INSTALL_HINT) from exc
    qrels: dict[str, dict[str, int]] = {}
    for judged in ir_measures.read_trec_qrels(str(path)):
        if judged.relevance > 0:
            qrels.setdefault(judged.query_id, {})[judged.doc_id] = int(judged.relevance)
    return qrels
