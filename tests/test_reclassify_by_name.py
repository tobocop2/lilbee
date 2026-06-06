"""Tests for ``reclassify_by_name``: defends against mis-tagged manifests."""

from __future__ import annotations

from lilbee.catalog.types import ModelTask
from lilbee.modelhub.model_manager.discovery import reclassify_by_name


def test_passes_through_when_name_has_no_special_pattern() -> None:
    assert reclassify_by_name("Qwen/Qwen3-0.6B-GGUF", "chat") == "chat"


def test_overrides_to_rerank_when_name_contains_reranker() -> None:
    assert reclassify_by_name("bge-reranker-v2-gemma", "chat") == ModelTask.RERANK


def test_overrides_to_rerank_when_name_contains_rerank() -> None:
    assert reclassify_by_name("baai/rerank-base", "chat") == ModelTask.RERANK


def test_overrides_to_rerank_when_name_contains_cross_encoder() -> None:
    assert reclassify_by_name("ms-marco-cross-encoder", "chat") == ModelTask.RERANK


def test_overrides_to_vision_when_name_contains_llava() -> None:
    assert reclassify_by_name("llava-1.5-7b", "chat") == ModelTask.VISION


def test_overrides_to_vision_when_name_contains_moondream() -> None:
    assert reclassify_by_name("moondream2", "chat") == ModelTask.VISION


def test_passes_through_embedding() -> None:
    assert reclassify_by_name("nomic-embed-text-v1.5", "embedding") == "embedding"


def test_overrides_to_embedding_when_name_contains_embedding() -> None:
    # Qwen3-Embedding is a qwen3 decoder arch (classifies as chat) + a pooling head,
    # so the name is the only signal short of probing the GGUF pooling type (bb-m3b).
    ref = "Qwen/Qwen3-Embedding-8B-GGUF/Qwen3-Embedding-8B-Q8_0.gguf"
    assert reclassify_by_name(ref, "chat") == ModelTask.EMBEDDING


def test_reranker_wins_over_embedding_pattern() -> None:
    # bge-reranker matches both the bge- embedder pattern and the rerank pattern;
    # rerank is checked first so it stays a reranker.
    assert reclassify_by_name("gpustack/bge-reranker-v2-m3-GGUF", "chat") == ModelTask.RERANK
