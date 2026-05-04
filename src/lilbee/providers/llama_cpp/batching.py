"""Single-text llama-cpp embed and rerank helpers used inside worker subprocesses."""

from __future__ import annotations

from typing import Any

from lilbee.providers.base import ProviderError

_RERANK_PAIR_SEPARATOR = "</s></s>"


def embed_one(llm: Any, text: str) -> list[float]:
    """Embed a single text. Caller must run with fd 2 already redirected.

    Either ``stderr_suppressed()`` must be held in-process or the caller
    must run inside a subprocess where ``_redirect_stdio()`` ran at start.
    Per-text wrapping is what made the TUI appear frozen during multi-page
    PDF ingest.
    """
    response = llm.create_embedding(input=[text])
    result: list[float] = response["data"][0]["embedding"]
    return result


def compute_rerank_scores(llm: Any, query: str, candidates: list[str]) -> list[float]:
    """Score *candidates* against *query* via llama.cpp reranker embeddings.

    ``pooling_type=LLAMA_POOLING_TYPE_RANK`` requires the pair pre-joined
    as ``query</s></s>candidate``; passing them as two inputs makes
    ``llama_decode`` fail with ``-1``.

    Caller must run with fd 2 already redirected (see :func:`embed_one`).
    """
    scores: list[float] = []
    for candidate in candidates:
        pair = f"{query}{_RERANK_PAIR_SEPARATOR}{candidate}"
        response = llm.create_embedding(input=pair)
        score = _extract_rerank_score(response)
        scores.append(score)
    return scores


def _extract_rerank_score(response: dict[str, Any]) -> float:
    """Extract a single relevance score from a pooling_type=RANK response."""
    data = response.get("data") or []
    if not data:
        raise ProviderError("Reranker returned no data", provider="llama-cpp")
    embedding = data[-1].get("embedding")
    if isinstance(embedding, list) and embedding and isinstance(embedding[0], (int, float)):
        return float(embedding[0])
    raise ProviderError(
        "Reranker returned unexpected score shape "
        f"(got {type(embedding).__name__}: {embedding!r}); "
        "llama-cpp-python may have changed its response format",
        provider="llama-cpp",
    )
