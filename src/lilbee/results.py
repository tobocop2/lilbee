from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Excerpt:
    content: str
    page_start: int | None
    page_end: int | None
    line_start: int | None
    line_end: int | None
    relevance: float  # 0.0-1.0 (1 = best match)


@dataclass
class DocumentResult:
    source: str
    content_type: str
    excerpts: list[Excerpt]
    best_relevance: float


def _zero_to_none(val: int) -> int | None:
    return None if val == 0 else val


def _to_excerpt(chunk: dict[str, object]) -> Excerpt:
    distance = float(chunk["_distance"])  # type: ignore[arg-type]
    relevance = 1.0 / (1.0 + distance)
    return Excerpt(
        content=str(chunk["chunk"]),
        page_start=_zero_to_none(int(chunk["page_start"])),  # type: ignore[call-overload]
        page_end=_zero_to_none(int(chunk["page_end"])),  # type: ignore[call-overload]
        line_start=_zero_to_none(int(chunk["line_start"])),  # type: ignore[call-overload]
        line_end=_zero_to_none(int(chunk["line_end"])),  # type: ignore[call-overload]
        relevance=relevance,
    )


def group(chunks: list[dict[str, object]]) -> list[DocumentResult]:
    """Group raw LanceDB chunks into document-centric results."""
    by_source: dict[str, list[dict[str, object]]] = {}
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
    return [asdict(r) for r in results]
