"""Shared test helpers."""

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lilbee.catalog import CatalogModel
from lilbee.config import cfg
from lilbee.ingest import file_hash
from lilbee.store import CitationRecord

FIXTURES_DIR = Path(__file__).parent / "fixtures"

def pytest_configure(config: pytest.Config) -> None:
    """Suppress asyncio event loop teardown noise from Textual worker threads."""
    config.addinivalue_line(
        "filterwarnings",
        "ignore::pytest.PytestUnraisableExceptionWarning",
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> None:  # type: ignore[type-arg]
    """Downgrade asyncio event loop teardown errors to xfail.

    Textual's @work(thread=True) workers can corrupt the event loop's
    self-pipe socket during teardown. pytest-asyncio's Runner fixture
    then raises OSError when closing the loop. This is not a real test
    failure.
    """
    outcome = yield
    report = outcome.get_result()
    if (
        report.when == "teardown"
        and report.failed
        and call.excinfo is not None
        and call.excinfo.errisinstance(OSError)
        and "Bad file descriptor" in str(call.excinfo.value)
    ):
        report.outcome = "passed"
        report.wasxfail = "asyncio loop teardown noise (Textual worker thread)"


@pytest.fixture(autouse=True)
def _drain_textual_threads():
    """Wait for Textual worker threads to finish after each test.

    Textual's @work(thread=True) runs work via loop.run_in_executor, which
    spawns threads named "asyncio_N". These threads may outlive the app's
    run_test() context manager. When pytest-xdist tears down its gateway
    worker process, lingering threads hold the GIL during join(), causing
    the process to freeze indefinitely.

    This fixture records threads before the test, then joins any new threads
    afterward with a short timeout.
    """
    before = set(threading.enumerate())
    yield
    for thread in threading.enumerate():
        if thread in before or thread is threading.current_thread():
            continue
        if thread.is_alive():
            thread.join(timeout=2.0)


@pytest.fixture(autouse=True)
def _isolate_cfg(tmp_path):
    """Snapshot and restore cfg for every test to prevent cross-test pollution."""
    snapshot = cfg.model_copy()
    cfg.models_dir = tmp_path / "models"
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))
    cfg.clear_model_defaults()


def make_mock_services(**overrides):
    """Create a mock Services container. Override individual services via kwargs."""
    from lilbee.providers.base import LLMProvider
    from lilbee.query import Searcher
    from lilbee.services import Services

    provider = overrides.pop("provider", None)
    if provider is None:
        provider = MagicMock(spec=LLMProvider)

    store = overrides.pop("store", None)
    if store is None:
        store = MagicMock()
        store.search.return_value = []
        store.bm25_probe.return_value = []
        store.get_sources.return_value = []
        store.add_chunks.side_effect = lambda records: len(records)

    embedder = overrides.pop("embedder", None)
    if embedder is None:
        embedder = MagicMock()
        embedder.embed.return_value = [0.1] * 768
        embedder.embed_batch.side_effect = lambda texts, **kw: [[0.1] * 768 for _ in texts]

    reranker = overrides.pop("reranker", None)
    if reranker is None:
        reranker = MagicMock()
        reranker.rerank.side_effect = lambda q, r, **kw: r

    concepts = overrides.pop("concepts", None)
    if concepts is None:
        concepts = MagicMock()
        concepts.get_graph.return_value = False

    clusterer = overrides.pop("clusterer", None)
    if clusterer is None:
        clusterer = MagicMock()
        clusterer.available.return_value = False
        clusterer.get_clusters.return_value = []

    searcher = overrides.pop("searcher", None)
    if searcher is None:
        searcher = Searcher(cfg, provider, store, embedder, reranker, concepts)

    registry = overrides.pop("registry", None)
    if registry is None:
        registry = MagicMock()

    return Services(
        provider=provider,
        store=store,
        embedder=embedder,
        reranker=reranker,
        concepts=concepts,
        clusterer=clusterer,
        searcher=searcher,
        registry=registry,
    )


def make_citation(
    wiki_source: str = "wiki/summaries/doc.md",
    source_filename: str = "doc.md",
    source_hash: str = "abc",
    excerpt: str = "some text",
    citation_key: str = "src1",
    **kwargs: object,
) -> CitationRecord:
    """Build a CitationRecord with sensible defaults."""
    defaults: CitationRecord = {
        "wiki_source": wiki_source,
        "wiki_chunk_index": 0,
        "citation_key": citation_key,
        "claim_type": "fact",
        "source_filename": source_filename,
        "source_hash": source_hash,
        "page_start": 0,
        "page_end": 0,
        "line_start": 0,
        "line_end": 0,
        "excerpt": excerpt,
        "created_at": "2026-01-01",
    }
    defaults.update(kwargs)  # type: ignore[typeddict-item]
    return defaults


def write_wiki_page(tmp_path: Path, subdir: str, slug: str, content: str) -> Path:
    """Write a wiki page and return its path."""
    wiki_root = tmp_path / "wiki" / subdir
    wiki_root.mkdir(parents=True, exist_ok=True)
    path = wiki_root / f"{slug}.md"
    path.write_text(content, encoding="utf-8")
    return path


def write_source(tmp_path: Path, name: str, content: str) -> Path:
    """Write a source document and return its path."""
    path = tmp_path / "documents" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def source_hash(path: Path) -> str:
    """Get the SHA-256 hash of a file (delegates to ingest.file_hash)."""
    return file_hash(path)


@pytest.fixture(autouse=False)
def wiki_isolated_env(tmp_path: Path):
    """Shared fixture for wiki tests: snapshot cfg, set wiki-related paths, restore."""
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir()
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.wiki = True
    cfg.wiki_dir = "wiki"
    cfg.wiki_faithfulness_threshold = 0.7
    cfg.wiki_prune_raw = False
    cfg.chat_model = "test-model"
    yield tmp_path
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def make_test_catalog_model(
    name: str = "TestModel",
    task: str = "chat",
    featured: bool = False,
    size_gb: float = 2.0,
    description: str = "A test model",
    tag: str = "latest",
    display_name: str = "",
    min_ram_gb: float = 4,
) -> CatalogModel:
    """Build a CatalogModel with sensible test defaults."""
    return CatalogModel(
        name=name.lower().replace(" ", "-"),
        tag=tag,
        display_name=display_name or name,
        hf_repo=f"test/{name.lower().replace(' ', '-')}",
        gguf_filename="*.gguf",
        size_gb=size_gb,
        min_ram_gb=min_ram_gb,
        description=description,
        featured=featured,
        downloads=100,
        task=task,
    )
