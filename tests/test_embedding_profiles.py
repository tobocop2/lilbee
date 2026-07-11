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
        "nomic-ai/nomic-embed-text-v1.5-GGUF/n.gguf",
        "Alibaba-NLP/gte-large-GGUF/g.gguf",
        "some/unknown-embedder/x.gguf",
    ],
)
def test_unrecognized_models_stay_symmetric(ref) -> None:
    assert resolve_embedding_profile(ref) == EmbeddingProfile()
