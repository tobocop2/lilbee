"""Tests for per-family embedding instruction profiles."""

from __future__ import annotations

import pytest

from lilbee.retrieval.embedding_profiles import EmbeddingProfile, resolve_embedding_profile


def test_qwen3_embedding_uses_instruct_query() -> None:
    p = resolve_embedding_profile("Qwen/Qwen3-Embedding-8B-GGUF/q.gguf")
    assert p.query_instruction.startswith("Instruct:")
    assert p.doc_prefix == ""


@pytest.mark.parametrize(
    "ref",
    [
        "intfloat/multilingual-e5-large-instruct-GGUF/m.gguf",
        "Alibaba-NLP/gte-Qwen2-7B-instruct-GGUF/g.gguf",
    ],
)
def test_instruct_embedders_use_instruct_query(ref) -> None:
    p = resolve_embedding_profile(ref)
    assert p.query_instruction.startswith("Instruct:")
    assert p.doc_prefix == ""


@pytest.mark.parametrize(
    "ref",
    ["hkunlp/instructor-large-GGUF/i.gguf", "hkunlp/instructor-xl-GGUF/i.gguf"],
)
def test_instructor_family_stays_symmetric(ref) -> None:
    """The Instructor family's name contains "instruct" but it wants a document
    instruction too, in a different dialect. Giving it the Instruct/Query query
    prefix and no document prefix is asymmetric prompting in the wrong format --
    the silently-wrong case the symmetric fallback exists to avoid."""
    assert resolve_embedding_profile(ref) == EmbeddingProfile()


@pytest.mark.parametrize(
    "ref",
    ["intfloat/e5-large-v2-GGUF/e.gguf", "intfloat/multilingual-e5-large-GGUF/m.gguf"],
)
def test_base_e5_uses_query_passage_prefixes(ref) -> None:
    p = resolve_embedding_profile(ref)
    assert p.query_instruction == "query: "
    assert p.doc_prefix == "passage: "


@pytest.mark.parametrize(
    "ref",
    [
        "gpustack/bge-m3-GGUF/b.gguf",
        "Alibaba-NLP/gte-large-GGUF/g.gguf",
        "some/unknown-embedder/x.gguf",
    ],
)
def test_unrecognized_models_stay_symmetric(ref) -> None:
    assert resolve_embedding_profile(ref) == EmbeddingProfile()


@pytest.mark.parametrize(
    "ref",
    ["nomic-ai/nomic-embed-text-v1.5-GGUF/n.gguf", "nomic-ai/nomic-embed-text-v2-moe-GGUF/n.gguf"],
)
def test_nomic_uses_its_required_search_prefixes(ref) -> None:
    p = resolve_embedding_profile(ref)
    assert p.query_instruction == "search_query: "
    assert p.doc_prefix == "search_document: "
    assert p.doc_prefix_since == 2  # pre-prefix stores warn to rebuild


def test_bge_v1_gets_the_query_instruction_only() -> None:
    p = resolve_embedding_profile("BAAI/bge-large-en-v1.5-GGUF/b.gguf")
    assert p.query_instruction.startswith("Represent this sentence")
    assert p.doc_prefix == ""


def test_bge_m3_is_not_caught_by_the_bge_v1_needle() -> None:
    assert resolve_embedding_profile("gpustack/bge-m3-GGUF/b.gguf") == EmbeddingProfile()
