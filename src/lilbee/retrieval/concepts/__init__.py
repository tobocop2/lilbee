"""Concept graph for LazyGraphRAG-style index-time knowledge extraction.

Extracts noun-phrase concepts from chunks via spaCy, builds a PPMI-weighted
co-occurrence graph (Church & Hanks 1990), and clusters with Leiden
(Traag et al. 2019, graspologic-native). Used to boost search results by
concept overlap and expand queries via graph traversal.

Requires optional ``graph`` extra: ``pip install lilbee[graph]``.
When dependencies are missing, all public functions degrade gracefully.
"""

from __future__ import annotations

from lilbee.retrieval.concepts.clusterer import ConceptGraphClusterer
from lilbee.retrieval.concepts.community import (
    _MIN_LEIDEN_WEIGHT,
    Community,
    _compute_pmi,
    _leiden_partition,
)
from lilbee.retrieval.concepts.graph import ConceptGraph
from lilbee.retrieval.concepts.nlp import (
    _ensure_spacy_model,
    _filter_noun_chunks,
    concepts_available,
    load_spacy_pipeline,
)

__all__ = [
    "_MIN_LEIDEN_WEIGHT",
    "Community",
    "ConceptGraph",
    "ConceptGraphClusterer",
    "_compute_pmi",
    "_ensure_spacy_model",
    "_filter_noun_chunks",
    "_leiden_partition",
    "concepts_available",
    "load_spacy_pipeline",
]
