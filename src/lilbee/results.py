from __future__ import annotations

from pydantic import BaseModel

from lilbee.store import SearchChunk


class Excerpt(BaseModel):
    content: str
    page_start: int | None
    page_end: int | None
    line_start: int | None
    line_end: int | None
    relevance: float  # 0.0-1.0 (1 = best match)


class DocumentResult(BaseModel):
    source: str
    content_type: str
    excerpts: list[Excerpt]
    best_relevance: float


def _zero_to_none(val: int) -> int | None:
    return None if val == 0 else val


def _to_excerpt(chunk: SearchChunk) -> Excerpt:
    if "_relevance_score" in chunk:
        relevance = float(chunk["_relevance_score"])
    else:
        distance = float(chunk["_distance"])
        relevance = 1.0 / (1.0 + distance)
    return Excerpt(
        content=str(chunk["chunk"]),
        page_start=_zero_to_none(chunk["page_start"]),
        page_end=_zero_to_none(chunk["page_end"]),
        line_start=_zero_to_none(chunk["line_start"]),
        line_end=_zero_to_none(chunk["line_end"]),
        relevance=relevance,
    )


def group(chunks: list[SearchChunk]) -> list[DocumentResult]:
    """Group raw LanceDB chunks into document-centric results."""
    by_source: dict[str, list[SearchChunk]] = {}
    for chunk in chunks:
        source = str(chunk["source"])
        by_source.setdefault(source, []).append(chunk)

    results: list[DocumentResult] = []
    for source, source_chunks in by_source.items():
        excerpts = sorted(
            [_to_excerpt(c) for c in source_chunks],
            key=lambda e: e.relevance,
            reverse=True,
        )
        results.append(
            DocumentResult(
                source=source,
                content_type=str(source_chunks[0]["content_type"]),
                excerpts=excerpts,
                best_relevance=excerpts[0].relevance,
            )
        )

    results.sort(key=lambda r: r.best_relevance, reverse=True)
    return results


def to_dicts(results: list[DocumentResult]) -> list[dict[str, object]]:
    """Serialize DocumentResults to JSON-safe dicts."""
    return [r.model_dump() for r in results]
