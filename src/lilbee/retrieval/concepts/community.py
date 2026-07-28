"""Concept community dataclass and PMI / Leiden helpers."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, NamedTuple

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
    Based on Church & Hanks 1990, "Word Association Norms, Mutual Information,
    and Lexicography." Pairs with non-positive PMI (co-occurring at or below
    chance) are dropped entirely: a stored 0.0 would later be floored to a
    positive Leiden weight, turning anti-correlation into attraction.
    """
    pmi: dict[tuple[str, str], float] = {}
    for (a, b), count in cooccurrences.items():
        p_a = concept_counts[a] / total_chunks
        p_b = concept_counts[b] / total_chunks
        if p_a == 0 or p_b == 0:
            continue
        p_ab = count / total_chunks
        value = math.log2(p_ab / (p_a * p_b))
        if value > 0:
            pmi[(a, b)] = value
    return pmi


class _LeidenResult(NamedTuple):
    """Leiden output: node -> community id, and node -> incident-edge count.

    ``degrees`` is an unweighted count of edges incident to each node (edge
    weights are not summed), which is what label ranking consumes.
    """

    partition: dict[str, int]
    degrees: dict[str, int]


def _leiden_partition(
    edge_rows: list[dict[str, Any]],
) -> _LeidenResult:
    """Run Leiden clustering on edge rows.
    Uses graspologic-native's Rust implementation (Traag et al. 2019,
    "From Louvain to Leiden: guaranteeing well-connected communities").
    """
    from graspologic_native import leiden

    edges: list[tuple[str, str, float]] = [
        (row["source"], row["target"], max(_MIN_LEIDEN_WEIGHT, row["weight"])) for row in edge_rows
    ]
    _modularity, partition = leiden(edges=edges, seed=_LEIDEN_SEED)  # type: ignore[call-arg]

    degree_map: Counter[str] = Counter()
    for row in edge_rows:
        degree_map[row["source"]] += 1
        degree_map[row["target"]] += 1
    return _LeidenResult(partition, dict(degree_map))
