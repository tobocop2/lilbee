"""Vector ranking primitives: cosine similarity and Maximal Marginal Relevance reranking."""

from __future__ import annotations

import numpy as np

from lilbee.core.config import cfg

from .types import SearchChunk


def cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    norm_a = float(np.linalg.norm(arr_a))
    norm_b = float(np.linalg.norm(arr_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(arr_a, arr_b) / (norm_a * norm_b))


def mmr_rerank(
    query_vector: list[float],
    results: list[SearchChunk],
    top_k: int,
    mmr_lambda: float | None = None,
) -> list[SearchChunk]:
    """Maximal Marginal Relevance: select diverse results.
    Algorithm: Carbonell & Goldstein 1998,
    "The Use of MMR, Diversity-Based Reranking for Reordering Documents
    and Producing Summaries."

    ``mmr_lambda`` controls the relevance/diversity tradeoff:
    0.0 = maximum diversity, 1.0 = pure relevance.
    Defaults to ``cfg.mmr_lambda`` (0.5).

    Complexity: O(top_k · N · D) time, O(N · D) space for N candidates
    of dimension D. Each outer iteration updates a running max-redundancy
    vector via one matmul rather than recomputing pairs pairwise.
    Candidate vectors run through numpy in ``float32``, which can pick a
    different candidate than the pure-Python ``float64`` loop on
    ties within ~1e-7; distinct in principle, unobservable in practice
    since sub-float32 differences are below retrieval signal.
    """
    if mmr_lambda is None:
        mmr_lambda = cfg.mmr_lambda
    if len(results) <= top_k:
        return results

    candidate_vecs = np.asarray([r.vector for r in results], dtype=np.float32)
    query = np.asarray(query_vector, dtype=np.float32)
    # L2-normalize once so cosine becomes a plain dot product.
    cand_norms = np.linalg.norm(candidate_vecs, axis=1, keepdims=True)
    cand_norms[cand_norms == 0] = 1.0
    cand_unit = candidate_vecs / cand_norms
    query_norm = float(np.linalg.norm(query)) or 1.0
    query_unit = query / query_norm

    relevance = cand_unit @ query_unit  # shape (N,)

    n = len(results)
    max_redundancy = np.zeros(n, dtype=np.float32)
    available = np.ones(n, dtype=bool)
    selected: list[SearchChunk] = []

    for picks in range(top_k):
        redundancy_term = max_redundancy if picks > 0 else np.zeros(n, dtype=np.float32)
        score = mmr_lambda * relevance - (1.0 - mmr_lambda) * redundancy_term
        # Mask already-picked candidates so argmax skips them.
        score = np.where(available, score, -np.inf)
        best = int(np.argmax(score))
        selected.append(results[best])
        available[best] = False
        # Update running max redundancy against the newly-selected vector.
        similarity = cand_unit @ cand_unit[best]
        max_redundancy = np.maximum(max_redundancy, similarity)

    return selected
