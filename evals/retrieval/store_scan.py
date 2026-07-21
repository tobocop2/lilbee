"""Streaming access to a lilbee LanceDB index; nothing materializes the table."""

from __future__ import annotations

import random
import re
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import NamedTuple, TypeVar

SCAN_BATCH_ROWS = 8192
HEAD_CHUNK_COUNT = 3
PASSAGE_OVERSAMPLE = 4

T = TypeVar("T")


class ChunkRow(NamedTuple):
    source: str
    chunk: str
    chunk_index: int


class TermCounts(NamedTuple):
    chunks: int
    sources: int


class PassageScan(NamedTuple):
    passages: list[tuple[str, str]]
    doc_heads: dict[str, str]


def iter_chunks(lancedb_dir: Path) -> Iterator[ChunkRow]:
    """Stream (source, chunk, chunk_index) rows batch by batch."""
    import lancedb  # heavy: arrow + datafusion

    from lilbee.core.config.defaults import CHUNKS_TABLE

    table = lancedb.connect(str(lancedb_dir)).open_table(CHUNKS_TABLE)
    columns = ["source", "chunk", "chunk_index"]
    for batch in table.search().select(columns).to_batches(batch_size=SCAN_BATCH_ROWS):
        yield from (
            ChunkRow(source, chunk, index)
            for source, chunk, index in zip(
                batch.column("source").to_pylist(),
                batch.column("chunk").to_pylist(),
                batch.column("chunk_index").to_pylist(),
                strict=True,
            )
        )


def iter_source_names(lancedb_dir: Path) -> Iterator[str]:
    """Stream indexed source filenames batch by batch."""
    import lancedb  # heavy: arrow + datafusion

    from lilbee.core.config.defaults import SOURCES_TABLE

    table = lancedb.connect(str(lancedb_dir)).open_table(SOURCES_TABLE)
    for batch in table.search().select(["filename"]).to_batches(batch_size=SCAN_BATCH_ROWS):
        yield from batch.column("filename").to_pylist()


def reservoir_sample(items: Iterable[T], k: int, rng: random.Random) -> list[T]:
    """Uniform sample of up to k items in one streaming pass (Algorithm R)."""
    reservoir: list[T] = []
    for seen, item in enumerate(items):
        if seen < k:
            reservoir.append(item)
            continue
        slot = rng.randrange(seen + 1)
        if slot < k:
            reservoir[slot] = item
    return reservoir


def count_term_hits(rows: Iterable[ChunkRow], terms: Sequence[str]) -> dict[str, TermCounts]:
    """Exact chunk and distinct-source hit counts for every term in one pass."""
    # Word-boundary, not substring. "How many documents mention X?" is a
    # question about word-level mentions, but unanchored containment also counts
    # "reported", "reports" and "reporting" for the term "report", so a system
    # that answers the question correctly is marked wrong against ground truth
    # defined a different way.
    needles = {term: re.compile(rf"\b{re.escape(term.lower())}\b") for term in terms}
    chunk_hits = dict.fromkeys(terms, 0)
    source_hits: dict[str, set[str]] = {term: set() for term in terms}
    for row in rows:
        if not row.chunk:
            continue
        text = row.chunk.lower()
        for term, needle in needles.items():
            if needle.search(text):
                chunk_hits[term] += 1
                source_hits[term].add(row.source)
    return {term: TermCounts(chunk_hits[term], len(source_hits[term])) for term in terms}


def scan_passages_and_heads(
    rows: Iterable[ChunkRow],
    *,
    passage_count: int,
    min_passage_chars: int,
    head_sources: set[str],
    rng: random.Random,
) -> PassageScan:
    """One streaming pass: sample candidate passages and collect document heads.

    Passages are reservoir-sampled with oversampling, then deduplicated to at
    most one per source, so the pass never holds more than a few hundred rows.
    """
    reservoir: list[ChunkRow] = []
    candidate_cap = passage_count * PASSAGE_OVERSAMPLE
    heads: dict[str, dict[int, str]] = {source: {} for source in head_sources}
    seen = 0
    for row in rows:
        if row.source in heads and row.chunk_index < HEAD_CHUNK_COUNT:
            heads[row.source][row.chunk_index] = row.chunk
        if not row.chunk or len(row.chunk) < min_passage_chars:
            continue
        if seen < candidate_cap:
            reservoir.append(row)
        else:
            slot = rng.randrange(seen + 1)
            if slot < candidate_cap:
                reservoir[slot] = row
        seen += 1

    # Algorithm R leaves slots never hit by a replacement holding the first
    # candidate_cap rows in stream order, and only about a quarter of the
    # reservoir survives the per-source dedupe below. Consuming it in slot order
    # would therefore draw preferentially from whatever the table returned
    # first. The sample is uniform only once the order is discarded.
    rng.shuffle(reservoir)
    passages: list[tuple[str, str]] = []
    picked_sources: set[str] = set()
    for row in reservoir:
        if row.source in picked_sources:
            continue
        picked_sources.add(row.source)
        passages.append((row.source, row.chunk))
        if len(passages) == passage_count:
            break
    doc_heads = {
        source: "\n".join(chunk for _, chunk in sorted(indexed.items()))
        for source, indexed in heads.items()
        if indexed
    }
    return PassageScan(passages=passages, doc_heads=doc_heads)
