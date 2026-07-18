"""Streaming scan helpers: sampling and counting without materializing an index."""

import random

from evals.retrieval.store_scan import (
    ChunkRow,
    count_term_hits,
    reservoir_sample,
    scan_passages_and_heads,
)


def _rows(spec: dict[str, list[str]]) -> list[ChunkRow]:
    return [
        ChunkRow(source, chunk, index)
        for source, chunks in spec.items()
        for index, chunk in enumerate(chunks)
    ]


def test_reservoir_sample_returns_all_when_fewer_than_k():
    assert sorted(reservoir_sample(iter("abc"), 10, random.Random(1))) == ["a", "b", "c"]


def test_reservoir_sample_is_deterministic_for_a_seed():
    items = list(range(1000))
    first = reservoir_sample(iter(items), 5, random.Random(7))
    second = reservoir_sample(iter(items), 5, random.Random(7))
    assert first == second
    assert len(first) == 5
    assert all(item in items for item in first)


def test_reservoir_sample_draws_from_the_whole_stream():
    samples = reservoir_sample(iter(range(10_000)), 50, random.Random(3))
    assert any(item > 5000 for item in samples)


def test_count_term_hits_counts_chunks_and_distinct_sources():
    rows = _rows(
        {
            "a.txt": ["the whale swam", "deep water"],
            "b.txt": ["a whale breached", "another Whale sighting"],
            "c.txt": ["nothing relevant"],
        }
    )
    counts = count_term_hits(rows, ["whale", "water", "absent"])
    assert counts["whale"].chunks == 3
    assert counts["whale"].sources == 2
    assert counts["water"] == (1, 1)
    assert counts["absent"] == (0, 0)


def test_scan_collects_one_passage_per_source_and_respects_min_chars():
    long_a = "a" * 500
    long_b = "b" * 500
    rows = _rows({"a.txt": [long_a, long_a], "b.txt": ["short", long_b]})
    scan = scan_passages_and_heads(
        rows, passage_count=5, min_passage_chars=400, head_sources=set(), rng=random.Random(1)
    )
    assert sorted(source for source, _ in scan.passages) == ["a.txt", "b.txt"]
    assert all(len(passage) >= 400 for _, passage in scan.passages)


def test_scan_caps_passages_at_requested_count():
    rows = _rows({f"s{i}.txt": ["x" * 450] for i in range(30)})
    scan = scan_passages_and_heads(
        rows, passage_count=4, min_passage_chars=400, head_sources=set(), rng=random.Random(2)
    )
    assert len(scan.passages) == 4
    assert len({source for source, _ in scan.passages}) == 4


def test_scan_builds_doc_heads_in_chunk_order():
    rows = [
        ChunkRow("k.txt", "third", 2),
        ChunkRow("k.txt", "first", 0),
        ChunkRow("k.txt", "second", 1),
        ChunkRow("k.txt", "far past the head", 9),
        ChunkRow("other.txt", "ignored", 0),
    ]
    scan = scan_passages_and_heads(
        rows, passage_count=0, min_passage_chars=400, head_sources={"k.txt"}, rng=random.Random(1)
    )
    assert scan.doc_heads == {"k.txt": "first\nsecond\nthird"}
