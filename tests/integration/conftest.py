"""Integration test configuration — shared fixtures for real-backend tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from lilbee.config import cfg
from lilbee.platform import canonical_models_dir

FIXTURES_DIR = Path(__file__).parent / "fixtures"
DOCS_DIR = FIXTURES_DIR / "docs"
TEST_DOCS = {f.name: f.read_text() for f in sorted(DOCS_DIR.iterdir()) if f.is_file()}


@pytest.fixture(autouse=True)
def _preserve_models_dir():
    """Ensure models_dir stays at canonical location for integration tests."""
    cfg.models_dir = canonical_models_dir()
    yield


@pytest.fixture(scope="session")
def rag_pipeline(tmp_path_factory):
    """Set up a real RAG pipeline with downloaded models and test documents.
    Session-scoped: downloads models once, creates documents, runs sync,
    yields pipeline data, then restores config.
    """
    from lilbee.catalog import FEATURED_CHAT, FEATURED_EMBEDDING, download_model
    from lilbee.ingest import sync
    from lilbee.model_manager import reset_model_manager
    from lilbee.services import reset_services as reset_provider

    snapshot = cfg.model_copy()
    tmp = tmp_path_factory.mktemp("rag_integration")
    docs_dir = tmp / "documents"
    data_dir = tmp / "data"
    lancedb_dir = data_dir / "lancedb"

    docs_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    for name, content in TEST_DOCS.items():
        (docs_dir / name).write_text(content)

    cfg.llm_provider = "llama-cpp"
    cfg.models_dir = canonical_models_dir()
    cfg.documents_dir = docs_dir
    cfg.data_dir = data_dir
    cfg.data_root = tmp
    cfg.lancedb_dir = lancedb_dir
    cfg.query_expansion_count = 0
    cfg.concept_graph = False
    cfg.hyde = False
    cfg.max_tokens = 512  # keep inference fast on slow CI runners

    reset_provider()
    reset_model_manager()

    embed_entry = FEATURED_EMBEDDING[0]
    download_model(embed_entry)
    cfg.embedding_model = embed_entry.ref

    chat_entry = next(m for m in FEATURED_CHAT if m.name == "qwen3" and m.tag == "0.6b")
    download_model(chat_entry)
    cfg.chat_model = chat_entry.ref

    result = asyncio.run(sync(quiet=True))

    yield {
        "tmp": tmp,
        "docs_dir": docs_dir,
        "data_dir": data_dir,
        "lancedb_dir": lancedb_dir,
        "sync_result": result,
        "test_docs": TEST_DOCS,
    }

    reset_provider()
    reset_model_manager()
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture(scope="session")
def wiki_pipeline(tmp_path_factory):
    """Set up a real pipeline with wiki enabled.
    Session-scoped: downloads models once, creates documents + wiki dir,
    runs sync, yields pipeline data, then restores config.
    """
    from lilbee.catalog import FEATURED_CHAT, FEATURED_EMBEDDING, download_model
    from lilbee.ingest import sync
    from lilbee.model_manager import reset_model_manager
    from lilbee.services import reset_services as reset_provider

    snapshot = cfg.model_copy()
    tmp = tmp_path_factory.mktemp("wiki_integration")
    docs_dir = tmp / "documents"
    data_dir = tmp / "data"
    lancedb_dir = data_dir / "lancedb"

    docs_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    for name, content in TEST_DOCS.items():
        (docs_dir / name).write_text(content)

    cfg.llm_provider = "llama-cpp"
    cfg.models_dir = canonical_models_dir()
    cfg.documents_dir = docs_dir
    cfg.data_dir = data_dir
    cfg.data_root = tmp
    cfg.lancedb_dir = lancedb_dir
    cfg.query_expansion_count = 0
    cfg.concept_graph = False
    cfg.hyde = False
    cfg.max_tokens = 512
    cfg.wiki = True
    cfg.wiki_dir = "wiki"
    (tmp / "wiki").mkdir(parents=True, exist_ok=True)

    reset_provider()
    reset_model_manager()

    embed_entry = FEATURED_EMBEDDING[0]
    download_model(embed_entry)
    cfg.embedding_model = embed_entry.ref

    chat_entry = next(m for m in FEATURED_CHAT if m.name == "qwen3" and m.tag == "0.6b")
    download_model(chat_entry)
    cfg.chat_model = chat_entry.ref

    result = asyncio.run(sync(quiet=True))

    yield {
        "tmp": tmp,
        "docs_dir": docs_dir,
        "data_dir": data_dir,
        "lancedb_dir": lancedb_dir,
        "sync_result": result,
        "test_docs": TEST_DOCS,
    }

    reset_provider()
    reset_model_manager()
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))
