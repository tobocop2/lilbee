from __future__ import annotations

import json

from lilbee.results import group, to_dicts


def _chunk(
    source: str = "doc.md",
    content_type: str = "text/markdown",
    chunk: str = "hello world",
    distance: float = 0.5,
    page_start: int = 0,
    page_end: int = 0,
    line_start: int = 0,
    line_end: int = 0,
    chunk_index: int = 0,
) -> dict[str, object]:
    return {
        "source": source,
        "content_type": content_type,
        "chunk": chunk,
        "_distance": distance,
        "page_start": page_start,
        "page_end": page_end,
        "line_start": line_start,
        "line_end": line_end,
        "chunk_index": chunk_index,
        "vector": [0.1, 0.2, 0.3],
    }


def test_empty_input() -> None:
    assert group([]) == []


def test_single_chunk() -> None:
    results = group([_chunk(distance=1.0, chunk="text", line_start=5, line_end=10)])
    assert len(results) == 1
    doc = results[0]
    assert doc.source == "doc.md"
    assert doc.content_type == "text/markdown"
    assert len(doc.excerpts) == 1
    ex = doc.excerpts[0]
    assert ex.content == "text"
    assert ex.relevance == 0.5
    assert ex.line_start == 5
    assert ex.line_end == 10


def test_multiple_chunks_same_source() -> None:
    chunks = [
        _chunk(source="a.md", distance=0.5, chunk="first"),
        _chunk(source="a.md", distance=0.2, chunk="second"),
    ]
    results = group(chunks)
    assert len(results) == 1
    assert len(results[0].excerpts) == 2


def test_multiple_sources_sorted_by_best_relevance() -> None:
    chunks = [
        _chunk(source="low.md", distance=1.0, chunk="low"),
        _chunk(source="high.md", distance=0.0, chunk="high"),
    ]
    results = group(chunks)
    assert len(results) == 2
    assert results[0].source == "high.md"
    assert results[1].source == "low.md"
    assert results[0].best_relevance > results[1].best_relevance


def test_distance_zero_gives_relevance_one() -> None:
    results = group([_chunk(distance=0.0)])
    assert results[0].excerpts[0].relevance == 1.0


def test_distance_one_gives_relevance_half() -> None:
    results = group([_chunk(distance=1.0)])
    assert results[0].excerpts[0].relevance == 0.5


def test_zero_page_line_values_become_none() -> None:
    results = group([_chunk(page_start=0, page_end=0, line_start=0, line_end=0)])
    ex = results[0].excerpts[0]
    assert ex.page_start is None
    assert ex.page_end is None
    assert ex.line_start is None
    assert ex.line_end is None


def test_nonzero_page_line_values_preserved() -> None:
    results = group([_chunk(page_start=1, page_end=3, line_start=10, line_end=20)])
    ex = results[0].excerpts[0]
    assert ex.page_start == 1
    assert ex.page_end == 3
    assert ex.line_start == 10
    assert ex.line_end == 20


def test_vector_field_stripped() -> None:
    results = group([_chunk()])
    doc_dict = to_dicts(results)[0]
    for excerpt in doc_dict["excerpts"]:
        assert "vector" not in excerpt


def test_to_dicts_json_serializable() -> None:
    results = group([_chunk(distance=0.3, page_start=1, line_start=5)])
    dicts = to_dicts(results)
    serialized = json.dumps(dicts)
    assert isinstance(serialized, str)
    parsed = json.loads(serialized)
    assert len(parsed) == 1
    assert parsed[0]["source"] == "doc.md"


def test_excerpts_sorted_by_relevance_descending() -> None:
    chunks = [
        _chunk(source="a.md", distance=1.0, chunk="worst"),
        _chunk(source="a.md", distance=0.0, chunk="best"),
        _chunk(source="a.md", distance=0.5, chunk="middle"),
    ]
    results = group(chunks)
    excerpts = results[0].excerpts
    assert excerpts[0].content == "best"
    assert excerpts[1].content == "middle"
    assert excerpts[2].content == "worst"


def test_best_relevance_matches_top_excerpt() -> None:
    chunks = [
        _chunk(source="a.md", distance=1.0, chunk="low"),
        _chunk(source="a.md", distance=0.2, chunk="high"),
    ]
    results = group(chunks)
    assert results[0].best_relevance == results[0].excerpts[0].relevance
