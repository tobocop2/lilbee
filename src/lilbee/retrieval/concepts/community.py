"""Concept community dataclass and PMI / Leiden helpers."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

_MIN_LEIDEN_WEIGHT = 0.01
# Fixed seed so Leiden returns the same communities for the same edge set;
# the algorithm is randomized and would otherwise drift between runs.
_LEIDEN_SEED = 42


@dataclass
class Community:
    """A cluster of related concepts from Leiden partitioning."""

    cluster_id: int
    size: int
    concepts: list[str]


def _compute_pmi(
    cooccurrences: Counter[tuple[str, str]],
    concept_counts: Counter[str],
    total_chunks: int,
) -> dict[tuple[str, str], float]:
    """Compute PPMI (Positive PMI) weights for concept co-occurrence pairs.
    PPMI = max(0, log2(P(a,b) / (P(a) * P(b)))).
    Based on Church & Hanks 1990, "Word Association Norms, Mutual Information,
    and Lexicography." Negative values are clamped to zero to discard
    anti-correlated pairs.
    """
    pmi: dict[tuple[str, str], float] = {}
    for (a, b), count in cooccurrences.items():
        p_a = concept_counts[a] / total_chunks
        p_b = concept_counts[b] / total_chunks
        if p_a == 0 or p_b == 0:
            continue
        p_ab = count / total_chunks
        pmi[(a, b)] = max(0.0, math.log2(p_ab / (p_a * p_b)))
    return pmi


def _leiden_partition(
    edge_rows: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    """Run Leiden clustering on edge rows. Returns (partition, degree_map).
    Uses graspologic-native's Rust implementation (Traag et al. 2019,
    "From Louvain to Leiden: guaranteeing well-connected communities").
    """
    from graspologic_native import leiden

    edges: list[tuple[str, str, float]] = [
        (row["source"], row["target"], max(_MIN_LEIDEN_WEIGHT, row["weight"])) for row in edge_rows
    ]
    _modularity, partition = leiden(edges=edges, seed=_LEIDEN_SEED)  # type: ignore[call-arg]

    degree_map: dict[str, int] = Counter()
    for row in edge_rows:
        degree_map[row["source"]] += 1
        degree_map[row["target"]] += 1
    return partition, dict(degree_map)
