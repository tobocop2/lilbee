"""Batched llama-cpp embed and rerank helpers used inside worker subprocesses."""

from __future__ import annotations

from typing import Any

from lilbee.providers.base import ProviderError

_RERANK_PAIR_SEPARATOR = "</s></s>"

EMBED_N_SEQ_MAX = 64
"""Max parallel sequences per ``create_embedding`` call.

The embed/rerank Llama contexts are loaded with
``context_params.n_seq_max = EMBED_N_SEQ_MAX`` (see
``lilbee.providers.llama_cpp.provider._llama_n_seq_max``). llama-cpp-
python's inner ``Llama.embed`` flushes its accumulated batch when the
TOKEN budget is exceeded, but never on sequence-count, so a caller
that hands it more than ``n_seq_max`` short inputs in one call will
trip the C-level ``invalid seq_id[N][0] = K >= K`` assertion and get
``llama_decode returned -1`` (upstream issue #2051, PR #2058 still
open as of May 2026). Both ``embed_batch`` and ``compute_rerank_scores``
flush at this sequence count to avoid that.
"""


def embed_batch(llm: Any, texts: list[str]) -> list[list[float]]:
    """Embed *texts* in as few llama-cpp calls as the model's batch budget allows.

    One ``llm.create_embedding(input=sub_batch)`` per sub-batch instead of
    one per text. Sub-batches respect ``llm.n_batch`` (token budget,
    already clamped to ``min(n_ctx, n_batch)`` at model load) AND
    ``EMBED_N_SEQ_MAX`` (sequence-count cap; see the constant docstring
    for the upstream bug we work around). Vectors come back in input
    order. Caller must run inside a worker subprocess where
    ``redirect_stdio_to_devnull()`` ran at startup so fd 2 is already
    redirected.
    """
    if not texts:
        return []
    token_cap = max(1, int(llm.n_batch))
    vectors: list[list[float]] = []
    sub_batch: list[str] = []
    sub_tokens = 0
    for text in texts:
        token_count = max(1, len(llm.tokenize(text.encode("utf-8"))))
        if sub_batch and (
            sub_tokens + token_count > token_cap or len(sub_batch) >= EMBED_N_SEQ_MAX
        ):
            vectors.extend(_embed_one_call(llm, sub_batch))
            sub_batch = []
            sub_tokens = 0
        sub_batch.append(text)
        sub_tokens += token_count
    if sub_batch:
        vectors.extend(_embed_one_call(llm, sub_batch))
    return vectors


def compute_rerank_scores(llm: Any, query: str, candidates: list[str]) -> list[float]:
    """Score *candidates* against *query* via llama.cpp reranker embeddings.

    ``pooling_type=LLAMA_POOLING_TYPE_RANK`` requires the pair pre-joined
    as ``query</s></s>candidate``; passing them as two inputs makes
    ``llama_decode`` fail with ``-1``. Pairs are batched together so one
    rerank call decodes many candidates in a single ggml graph, capped
    by both the token and sequence budgets ``embed_batch`` uses.
    """
    if not candidates:
        return []
    pairs = [f"{query}{_RERANK_PAIR_SEPARATOR}{candidate}" for candidate in candidates]
    token_cap = max(1, int(llm.n_batch))
    scores: list[float] = []
    sub_batch: list[str] = []
    sub_tokens = 0
    for pair in pairs:
        token_count = max(1, len(llm.tokenize(pair.encode("utf-8"))))
        if sub_batch and (
            sub_tokens + token_count > token_cap or len(sub_batch) >= EMBED_N_SEQ_MAX
        ):
            scores.extend(_rerank_one_call(llm, sub_batch))
            sub_batch = []
            sub_tokens = 0
        sub_batch.append(pair)
        sub_tokens += token_count
    if sub_batch:
        scores.extend(_rerank_one_call(llm, sub_batch))
    return scores


def _embed_one_call(llm: Any, sub_batch: list[str]) -> list[list[float]]:
    response = llm.create_embedding(input=sub_batch)
    data = response.get("data") or []
    return [item["embedding"] for item in data]


def _rerank_one_call(llm: Any, sub_batch: list[str]) -> list[float]:
    response = llm.create_embedding(input=sub_batch)
    data = response.get("data") or []
    if len(data) != len(sub_batch):
        raise ProviderError(
            f"Reranker returned {len(data)} entries for {len(sub_batch)} pairs; "
            "llama-cpp-python may have changed its response format",
            provider="llama-cpp",
        )
    return [_extract_rerank_score(item) for item in data]


def _extract_rerank_score(item: dict[str, Any]) -> float:
    """Pull a single relevance score from one pooling_type=RANK response item."""
    embedding = item.get("embedding")
    if isinstance(embedding, list) and embedding and isinstance(embedding[0], (int, float)):
        return float(embedding[0])
    raise ProviderError(
        "Reranker returned unexpected score shape "
        f"(got {type(embedding).__name__}: {embedding!r}); "
        "llama-cpp-python may have changed its response format",
        provider="llama-cpp",
    )
