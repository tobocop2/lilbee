"""Integration test configuration — shared fixtures for real-backend tests."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from lilbee.config import cfg
from lilbee.platform import canonical_models_dir

# macOS CI runners use CPU-only inference (no Metal GPU passthrough).
# SmolLM2-135M is fast enough; Qwen3-0.6B is too slow.
_DEFAULT_CHAT_REPO = "Qwen/Qwen3-0.6B-GGUF"
_CI_CHAT_REPO = os.environ.get("LILBEE_TEST_CHAT_MODEL", _DEFAULT_CHAT_REPO)

# Assertions that depend on the LLM producing specific factual content are
# unreliable on 135M-param models, which collapse into repetition even with
# correct retrieval context. Used to skip those content-assertions on macOS
# CI while still exercising the full pipeline on Ubuntu + Windows.
_SMALL_CHAT_REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
skip_if_small_chat_model = pytest.mark.skipif(
    _CI_CHAT_REPO == _SMALL_CHAT_REPO,
    reason=f"{_SMALL_CHAT_REPO} too small for reliable factual RAG answers",
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
DOCS_DIR = FIXTURES_DIR / "docs"
TEST_DOCS = {f.name: f.read_text() for f in sorted(DOCS_DIR.iterdir()) if f.is_file()}


def _resolve_installed_ref(hf_repo: str) -> str:
    """Return the canonical ``hf_repo/filename`` ref for whichever quant of
    *hf_repo* is currently installed in the registry."""
    from lilbee.registry import ModelRegistry

    for manifest in ModelRegistry(cfg.models_dir).list_installed():
        if manifest.hf_repo == hf_repo:
            return manifest.ref
    raise RuntimeError(f"No installed manifest for {hf_repo}")


# Integration tests run real LLM inference; the global 60s unit-test cap is too
# aggressive. 180s is ~4x the slowest observed test on Metal. Any test exceeding
# it on CPU-only CI runners is a hang, not a slow pass. Tests can opt in to a
# different cap with @pytest.mark.timeout(X).
_INTEGRATION_TIMEOUT_SECONDS = 180


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.timeout(_INTEGRATION_TIMEOUT_SECONDS))


@pytest.fixture(autouse=True)
def _preserve_models_dir():
    """Ensure models_dir stays at canonical location for integration tests."""
    cfg.models_dir = canonical_models_dir()
    yield


@pytest.fixture(scope="session")
def _integration_loop():
    """Session-scoped event loop shared by pipeline fixtures."""
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(asyncio.sleep(0.1))
        loop.close()


@pytest.fixture
def run_async(_integration_loop):
    """Run a coroutine on the shared session loop.

    Tests must NOT use asyncio.run() directly. A fresh loop tears down
    call_soon_threadsafe plumbing that the llama-cpp provider's daemon-thread
    embed/rerank workers depend on, wedging subsequent awaits on CPU-only CI.
    """

    def _run(coro):
        return _integration_loop.run_until_complete(coro)

    return _run


@pytest.fixture(scope="session")
def rag_pipeline(tmp_path_factory, _integration_loop):
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
    cfg.wiki = False
    cfg.max_tokens = 512  # keep inference fast on slow CI runners

    reset_provider()
    reset_model_manager()

    embed_entry = FEATURED_EMBEDDING[0]
    download_model(embed_entry)
    cfg.embedding_model = _resolve_installed_ref(embed_entry.hf_repo)

    chat_entry = next(m for m in FEATURED_CHAT if m.hf_repo == _CI_CHAT_REPO)
    download_model(chat_entry)
    cfg.chat_model = _resolve_installed_ref(_CI_CHAT_REPO)

    result = _integration_loop.run_until_complete(sync(quiet=True))

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
def wiki_pipeline(tmp_path_factory, _integration_loop):
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
    cfg.embedding_model = _resolve_installed_ref(embed_entry.hf_repo)

    chat_entry = next(m for m in FEATURED_CHAT if m.hf_repo == _CI_CHAT_REPO)
    download_model(chat_entry)
    cfg.chat_model = _resolve_installed_ref(_CI_CHAT_REPO)

    result = _integration_loop.run_until_complete(sync(quiet=True))

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
