"""Per-family embedding instruction profiles for asymmetric query/document embedding."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingProfile:
    """Instruction prefixes an embedder applies to a query vs a document."""

    query_instruction: str = ""
    doc_prefix: str = ""


# Instruction-tuned embedders (Qwen3-Embedding, *-instruct: e5/gte/...) share the
# Instruct/Query format with no document prefix. Base e5 uses query:/passage:.
# Everything else (bge-m3, nomic, gte-large, ...) stays symmetric: correct-but-
# symmetric, never silently wrong. Order is specific-first.
_INSTRUCT = EmbeddingProfile(
    query_instruction=(
        "Instruct: Given a web search query, retrieve relevant passages that "
        "answer the query\nQuery: "
    ),
)
_E5 = EmbeddingProfile(query_instruction="query: ", doc_prefix="passage: ")
_FAMILY_PROFILES: tuple[tuple[str, EmbeddingProfile], ...] = (
    ("qwen3-embedding", _INSTRUCT),
    ("instruct", _INSTRUCT),  # any instruction-tuned embedder: Instruct/Query, no doc prefix
    ("multilingual-e5", _E5),
    ("e5-", _E5),
)


def resolve_embedding_profile(model_ref: str) -> EmbeddingProfile:
    """Profile for *model_ref*, or a symmetric (empty) profile when unrecognized."""
    ref = model_ref.lower()
    for needle, profile in _FAMILY_PROFILES:
        if needle in ref:
            return profile
    return EmbeddingProfile()
