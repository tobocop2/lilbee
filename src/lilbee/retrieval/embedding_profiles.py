"""Per-family embedding instruction profiles for asymmetric query/document embedding."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingProfile:
    """Instruction prefixes an embedder applies to a query vs a document."""

    query_instruction: str = ""
    doc_prefix: str = ""
    # META_SCHEMA_VERSION that first stamped this doc_prefix at ingest; an
    # older store warns to rebuild rather than silently mixing prefixed
    # queries with unprefixed documents.
    doc_prefix_since: int = 1


# Instruction-tuned embedders (Qwen3-Embedding, *-instruct: e5/gte/...) share the
# Instruct/Query format with no document prefix. Base e5 uses query:/passage:,
# nomic-embed requires search_query:/search_document: (trained with them), bge
# v1/v1.5 wants a query-side instruction only. Everything else (bge-m3,
# gte-large, ...) stays symmetric: correct-but-symmetric, never silently wrong.
# Order is specific-first.
_INSTRUCT = EmbeddingProfile(
    query_instruction=(
        "Instruct: Given a web search query, retrieve relevant passages that "
        "answer the query\nQuery: "
    ),
)
_E5 = EmbeddingProfile(query_instruction="query: ", doc_prefix="passage: ")
_NOMIC = EmbeddingProfile(
    query_instruction="search_query: ", doc_prefix="search_document: ", doc_prefix_since=2
)
_BGE_V1 = EmbeddingProfile(
    query_instruction="Represent this sentence for searching relevant passages: "
)
_SYMMETRIC = EmbeddingProfile()
_FAMILY_PROFILES: tuple[tuple[str, EmbeddingProfile], ...] = (
    ("qwen3-embedding", _INSTRUCT),
    # "instructor" contains "instruct" but is a different dialect: the Instructor
    # family (hkunlp/instructor-*, a t5encoder the engine can load) prefixes a
    # "Represent the ... for retrieval:" instruction on the document as well as
    # the query. Handing it the Instruct/Query query prefix with no doc prefix
    # would be the asymmetric-in-the-wrong-dialect case this module promises not
    # to produce, so it takes the symmetric fallback until the real prefixes are
    # wired.
    ("instructor", _SYMMETRIC),
    ("instruct", _INSTRUCT),  # any instruction-tuned embedder: Instruct/Query, no doc prefix
    ("multilingual-e5", _E5),
    ("e5-", _E5),
    ("nomic-embed", _NOMIC),
    ("bge-m3", _SYMMETRIC),  # genuinely symmetric; must not fall through to bge v1
    ("bge-", _BGE_V1),
)


def resolve_embedding_profile(model_ref: str) -> EmbeddingProfile:
    """Profile for *model_ref*, or a symmetric (empty) profile when unrecognized."""
    ref = model_ref.lower()
    for needle, profile in _FAMILY_PROFILES:
        if needle in ref:
            return profile
    return EmbeddingProfile()
