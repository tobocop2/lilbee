from __future__ import annotations

from pydantic import BaseModel

from lilbee.data.store import SearchChunk


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
    # Vault-relative path for clients to deep-link into the native UI.
    # ``None`` when the server can't resolve the source under ``cfg.vault_base``.
    vault_path: str | None = None


def _zero_to_none(val: int) -> int | None:
    return None if val == 0 else val


def _to_excerpt(chunk: SearchChunk) -> Excerpt:
    # The canonical [0, 1] score is what every retrieval path stamps; the
    # distance fallback (which read keyword-only rows as a perfect 1.0)
    # covers only hand-built chunks that never went through retrieval.
    fallback = 1.0 / (1.0 + (chunk.distance or 0))
    relevance = chunk.score if chunk.score is not None else fallback
    return Excerpt(
        content=chunk.chunk,
        page_start=_zero_to_none(chunk.page_start),
        page_end=_zero_to_none(chunk.page_end),
        line_start=_zero_to_none(chunk.line_start),
        line_end=_zero_to_none(chunk.line_end),
        relevance=relevance,
    )


def _best_content_type(source_chunks: list[SearchChunk]) -> str:
    """Content type of the highest-scoring chunk for a source.

    score is optional on the model, so an unscored chunk sorts last rather
    than raising.
    """
    return max(source_chunks, key=lambda c: c.score if c.score is not None else -1.0).content_type


def group(chunks: list[SearchChunk]) -> list[DocumentResult]:
    """Group raw LanceDB chunks into document-centric results."""
    from lilbee.app.search import resolve_vault_path

    by_source: dict[str, list[SearchChunk]] = {}
    for chunk in chunks:
        source = chunk.source
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
                # From the best-scoring chunk, not whichever the store returned
                # first: excerpts are already sorted by relevance and a source
                # can carry chunks of more than one type.
                content_type=_best_content_type(source_chunks),
                excerpts=excerpts,
                best_relevance=excerpts[0].relevance,
                vault_path=resolve_vault_path(source),
            )
        )

    results.sort(key=lambda r: r.best_relevance, reverse=True)
    return results


def to_dicts(results: list[DocumentResult]) -> list[dict[str, object]]:
    """Serialize DocumentResults to JSON-safe dicts."""
    return [r.model_dump() for r in results]
