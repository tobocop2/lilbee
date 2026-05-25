"""Batched llama-cpp embed and rerank helpers used inside worker subprocesses."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from lilbee.providers.base import ProviderError

log = logging.getLogger(__name__)

_RERANK_PAIR_SEPARATOR = "</s></s>"

EMBED_N_SEQ_MAX = 64
"""Max parallel sequences per ``create_embedding`` call.

llama-cpp-python's inner ``Llama.embed`` flushes its batch on token
budget but not on sequence count, so caller-side batches above the
context's ``n_seq_max`` trip a C-level assertion. Workaround for
upstream issue #2051 / PR #2058 (still open as of May 2026).
"""


def _truncate_to_budget(llm: Any, text: str, token_cap: int) -> str:
    """Tokenize *text*, keep the first ``token_cap`` tokens, detokenize back.

    Token-aware truncation is needed because the chunker's 4-chars-per-token
    heuristic underestimates dense input (medical codes, JSON, source code).
    Bytes-level slicing would split mid-token and confuse the embedder.
    """
    tokens = llm.tokenize(text.encode("utf-8"))
    if len(tokens) <= token_cap:
        return text
    truncated: bytes = llm.detokenize(tokens[:token_cap])
    return truncated.decode("utf-8", errors="replace")


def _split_into_sub_batches(llm: Any, items: list[str]) -> Iterator[list[str]]:
    """Yield sub-batches respecting both the token budget and ``EMBED_N_SEQ_MAX``.

    A single item longer than ``llm.n_batch`` tokens is truncated to that
    budget with a warning. Without the truncation, llama-cpp's ``llama_decode``
    returns -1 and the whole call fails, which used to surface as
    "Embedding worker reported an error: RuntimeError: llama_decode
    returned -1" on every file in a token-dense corpus.
    """
    token_cap = max(1, int(llm.n_batch))
    sub_batch: list[str] = []
    sub_tokens = 0
    for raw_item in items:
        raw_tokens = max(1, len(llm.tokenize(raw_item.encode("utf-8"))))
        if raw_tokens > token_cap:
            log.warning(
                "Truncating oversize input: %d tokens > cap %d (chars/token heuristic too loose)",
                raw_tokens,
                token_cap,
            )
            item = _truncate_to_budget(llm, raw_item, token_cap)
            token_count = token_cap
        else:
            item = raw_item
            token_count = raw_tokens
        if sub_batch and (
            sub_tokens + token_count > token_cap or len(sub_batch) >= EMBED_N_SEQ_MAX
        ):
            yield sub_batch
            sub_batch = []
            sub_tokens = 0
        sub_batch.append(item)
        sub_tokens += token_count
    if sub_batch:
        yield sub_batch


def embed_batch(llm: Any, texts: list[str]) -> list[list[float]]:
    """Embed *texts* in as few llama-cpp calls as the model's batch budget allows.

    One ``llm.create_embedding(input=sub_batch)`` per sub-batch instead of
    one per text. Vectors come back in input order. Caller must run inside
    a worker subprocess where ``redirect_stdio_to_devnull()`` ran at
    startup so fd 2 is already redirected.
    """
    if not texts:
        return []
    vectors: list[list[float]] = []
    for sub_batch in _split_into_sub_batches(llm, texts):
        vectors.extend(_embed_one_call(llm, sub_batch))
    return vectors


def compute_rerank_scores(llm: Any, query: str, candidates: list[str]) -> list[float]:
    """Score *candidates* against *query* via llama.cpp reranker embeddings.

    ``pooling_type=LLAMA_POOLING_TYPE_RANK`` requires the pair pre-joined
    as ``query</s></s>candidate``; passing them as two inputs makes
    ``llama_decode`` fail with ``-1``. Pairs share ``embed_batch``'s
    sub-batching so one rerank call decodes many candidates per graph.
    """
    if not candidates:
        return []
    pairs = [f"{query}{_RERANK_PAIR_SEPARATOR}{candidate}" for candidate in candidates]
    scores: list[float] = []
    for sub_batch in _split_into_sub_batches(llm, pairs):
        scores.extend(_rerank_one_call(llm, sub_batch))
    return scores


def _embed_one_call(llm: Any, sub_batch: list[str]) -> list[list[float]]:
    response = llm.create_embedding(input=sub_batch)
    data = response.get("data") or []
    if len(data) != len(sub_batch):
        raise ProviderError(
            f"Embedder returned {len(data)} vectors for {len(sub_batch)} inputs; "
            "llama-cpp-python may have changed its response format",
            provider="llama-cpp",
        )
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
