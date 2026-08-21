"""Integration test configuration: shared fixtures for real-backend tests."""

from __future__ import annotations

import asyncio
import gc
import os
from pathlib import Path

import pytest

from lilbee.catalog import CatalogModel
from lilbee.core.config import cfg
from lilbee.core.system import canonical_models_dir

_DEFAULT_CHAT_REPO = "Qwen/Qwen3-0.6B-GGUF"
_DEFAULT_CHAT_FILE = "Qwen3-0.6B-Q8_0.gguf"
_CI_CHAT_REPO = os.environ.get("LILBEE_TEST_CHAT_MODEL", _DEFAULT_CHAT_REPO)
# Exact filename, not a wildcard: a named file costs one header probe, while a
# pattern also lists the repo, and the extra call per entry rate-limits the
# matrix cells that run last. An overridden repo has no known filename, so it
# keeps the pattern.
_CI_CHAT_FILE = _DEFAULT_CHAT_FILE if _CI_CHAT_REPO == _DEFAULT_CHAT_REPO else "*Q8_0.gguf"
_EMBED_REPO = "nomic-ai/nomic-embed-text-v1.5-GGUF"
_EMBED_FILE = "nomic-embed-text-v1.5.Q4_K_M.gguf"


def _real_entry(hf_repo: str, gguf_filename: str, task: str, size_gb: float) -> CatalogModel:
    """A catalog entry for a repo that really exists on HuggingFace.

    Integration tests perform real downloads, so they need real repos and
    cannot use the picks: those are whatever is trending today, which is both
    unstable and potentially enormous. Pinned repo ids are the point here, the
    same way ``tools/qa`` pins them.
    """
    return CatalogModel(
        hf_repo=hf_repo,
        gguf_filename=gguf_filename,
        size_gb=size_gb,
        min_ram_gb=2.0,
        description="",
        featured=False,
        downloads=0,
        task=task,
    )


CHAT_ENTRY = _real_entry(_CI_CHAT_REPO, _CI_CHAT_FILE, "chat", 0.5)
EMBED_ENTRY = _real_entry(_EMBED_REPO, _EMBED_FILE, "embedding", 0.3)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
DOCS_DIR = FIXTURES_DIR / "docs"
TEST_DOCS = {f.name: f.read_text() for f in sorted(DOCS_DIR.iterdir()) if f.is_file()}


def _resolve_installed_ref(hf_repo: str) -> str:
    """Return the canonical ``hf_repo/filename`` ref for whichever quant of
    *hf_repo* is currently installed in the registry."""
    from lilbee.modelhub.registry import ModelRegistry

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
        # Only default items that carry no timeout of their own: add_marker
        # prepends, so adding unconditionally makes the default the closest
        # marker and silently overrides every per-test opt-in.
        if item.get_closest_marker("timeout") is None:
            item.add_marker(pytest.mark.timeout(_INTEGRATION_TIMEOUT_SECONDS))


@pytest.fixture(autouse=True)
def _sealed_engine_resolution():
    """Unseal engine resolution: integration tests run the engine CI puts on PATH."""
    yield


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
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(asyncio.sleep(0.1))
        # Windows + Python 3.11: subprocess/pipe transports left to the garbage
        # collector finalize after the loop closes, and their __del__ then blows
        # up during interpreter shutdown (pytest dies on stdout.flush() at
        # exit). Collect while the proactor is still alive so every transport
        # closes through the loop, then give the loop one last spin to run the
        # close callbacks. Harmless on other platforms/versions.
        gc.collect()
        loop.run_until_complete(asyncio.sleep(0))
        loop.close()


@pytest.fixture
def run_async(_integration_loop):
    """Run a coroutine on the shared session loop.

    Tests must NOT use asyncio.run() directly: the pipeline runs blocking
    provider/embedder calls via asyncio.to_thread on this shared loop, and a
    fresh per-test loop would strand that plumbing and wedge subsequent awaits.
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
    from lilbee.app.services import reset_services as reset_provider
    from lilbee.catalog import download_model
    from lilbee.data.ingest import sync

    snapshot = cfg.model_copy()
    tmp = tmp_path_factory.mktemp("rag_integration")
    docs_dir = tmp / "documents"
    data_dir = tmp / "data"
    lancedb_dir = data_dir / "lancedb"

    docs_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    for name, content in TEST_DOCS.items():
        (docs_dir / name).write_text(content)

    cfg.llm_provider = "auto"  # local GGUF refs route to the llama-server fleet
    # Lazy server spawn: get_services() must not block tests that don't infer
    # (e.g. the status screen) on the fleet warm-up; inference tests spawn on
    # first call.
    cfg.worker_pool_eager_start = False
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

    embed_entry = EMBED_ENTRY
    download_model(embed_entry)
    cfg.embedding_model = _resolve_installed_ref(embed_entry.hf_repo)

    chat_entry = CHAT_ENTRY
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
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture(scope="session")
def wiki_pipeline(tmp_path_factory, _integration_loop):
    """Set up a real pipeline with wiki enabled.
    Session-scoped: downloads models once, creates documents + wiki dir,
    runs sync, yields pipeline data, then restores config.
    """
    from lilbee.app.services import reset_services as reset_provider
    from lilbee.catalog import download_model
    from lilbee.data.ingest import sync

    snapshot = cfg.model_copy()
    tmp = tmp_path_factory.mktemp("wiki_integration")
    docs_dir = tmp / "documents"
    data_dir = tmp / "data"
    lancedb_dir = data_dir / "lancedb"

    docs_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    for name, content in TEST_DOCS.items():
        (docs_dir / name).write_text(content)

    cfg.llm_provider = "auto"  # local GGUF refs route to the llama-server fleet
    # Lazy server spawn: get_services() must not block tests that don't infer
    # (e.g. the status screen) on the fleet warm-up; inference tests spawn on
    # first call.
    cfg.worker_pool_eager_start = False
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

    embed_entry = EMBED_ENTRY
    download_model(embed_entry)
    cfg.embedding_model = _resolve_installed_ref(embed_entry.hf_repo)

    chat_entry = CHAT_ENTRY
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
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))
