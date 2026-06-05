"""Shared test helpers."""

import os
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Opt tests out of the per-role catalog-task validator at import time so
# the many ``cfg.chat_model = "test-model"``-style fixtures don't trip
# over the production guard. The bypass is a two-signal gate: the env
# var alone is NOT enough. The validator also checks
# ``sys.modules["pytest"]`` to confirm the process is actually a test
# run -- so setting ``LILBEE_SKIP_MODEL_TASK_VALIDATION`` in a shell
# profile or Dockerfile cannot silently disable role validation in
# production. Because pytest has already imported itself by the time
# conftest runs, the sentinel half of the gate is satisfied here.
os.environ.setdefault("LILBEE_SKIP_MODEL_TASK_VALIDATION", "1")

from lilbee.catalog import CatalogModel
from lilbee.catalog.refs import format_native_gguf_ref
from lilbee.core.config import cfg
from lilbee.data.ingest import file_hash
from lilbee.data.store import CitationRecord
from lilbee.modelhub.registry import ModelManifest, ModelRegistry

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _patch_executor_daemon_threads() -> None:
    """Make ThreadPoolExecutor workers and LanceDB event-loop threads daemon on 3.11.

    Any asyncio loop we start (CLI asyncio.run, uvicorn, the TUI background
    loop) spins up a ThreadPoolExecutor. On 3.11 those executor threads are
    non-daemon and block xdist worker process exit. On 3.12+ interpreter
    shutdown handles this correctly.

    LanceDB spawns a non-daemon ``LanceDBBackgroundEventLoop`` tokio thread
    with no close() API. On ubuntu 3.11 these accumulate across tests in
    test_store.py and the process wedges shortly after. On 3.12+ interpreter
    shutdown handles it. Daemonify both at Thread.__init__ so start() runs
    them as daemons.
    """
    if sys.version_info >= (3, 12):
        return
    import concurrent.futures.thread as _tmod

    _real_init = threading.Thread.__init__

    def _init_with_daemon(self: threading.Thread, *args: object, **kwargs: object) -> None:
        _real_init(self, *args, **kwargs)  # type: ignore[misc]
        if getattr(self, "_target", None) is _tmod._worker or "LanceDB" in self.name:
            self.daemon = True

    threading.Thread.__init__ = _init_with_daemon  # type: ignore[assignment]


# Apply at import time so xdist workers get the patch immediately.
_patch_executor_daemon_threads()


# Silence stray lancedb thread shutdown errors globally so they can't wedge
# the test runner via threading.excepthook propagation on ubuntu 3.11.
if sys.version_info < (3, 12):
    from lilbee.data.store import install_lancedb_thread_error_suppressor

    install_lancedb_thread_error_suppressor()


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
def _suppress_model_scan(request, monkeypatch):
    """Prevent ModelBar._scan_models from doing real work in tests.

    ModelBar.on_mount calls _scan_models which is @work(thread=True).
    The real function does registry scans, HTTP calls, and litellm imports.
    Mocking it to return empty results makes the worker thread complete
    instantly, avoiding both thread accumulation and per-test join overhead.

    Tests that need real classification use @pytest.mark.real_model_classify.
    """
    if "real_model_classify" not in {m.name for m in request.node.iter_markers()}:
        monkeypatch.setattr(
            "lilbee.cli.tui.widgets.model_bar._classify_installed_models",
            lambda: ([], []),
        )


@pytest.fixture(autouse=True)
def _assume_litellm_available(request, monkeypatch):
    """Assume the SDK extra is installed for unit tests.

    Production code now gates remote-model discovery on
    ``litellm_available()``. Dev environments without ``lilbee[litellm]``
    would otherwise short-circuit every remote-discovery test. Tests
    that cover the missing-extra path skip this fixture with
    ``@pytest.mark.real_litellm_probe``.
    """
    if "real_litellm_probe" in {m.name for m in request.node.iter_markers()}:
        return
    monkeypatch.setattr("lilbee.providers.litellm_sdk.litellm_available", lambda: True)


@pytest.fixture(autouse=True)
def _reset_services_after_test():
    """Drop any Services container ``set_services()`` left around.

    Tests that inject a mock via ``set_services(make_mock_services(...))``
    would otherwise leak into the next test's ``get_services()`` call,
    producing confusing cross-test failures. Shutting the provider down
    drops any running fleet; mocks pass through the same path harmlessly.
    """
    yield
    from contextlib import suppress

    from lilbee.app.services import peek_services, set_services

    svc = peek_services()
    if svc is not None:
        with suppress(Exception):
            svc.provider.shutdown()
    set_services(None)


@pytest.fixture(autouse=True)
def _join_fleet_background_threads():
    """Join lingering fleet warm-up / reload daemon threads after each test.

    ``warm_up_pool`` / ``reload_role`` dispatch fire-and-forget daemon threads; on a
    host without the engine binary they fail fast and log a warning. Joining them
    here keeps that warning inside the test that started the thread instead of
    leaking it into a later test's log capture.
    """
    yield
    import threading

    for thread in threading.enumerate():
        if thread.name.startswith("fleet-") and thread.is_alive():
            thread.join(timeout=5.0)


@pytest.fixture(autouse=True)
def _ignore_user_global_config(monkeypatch):
    """Skip the platform-default config.toml for unit tests.

    A developer's persisted ``~/Library/Application Support/lilbee/config.toml``
    can hold values from a previous schema. ``Config()`` would crash at
    construction. Setting this env var tells ``settings_customise_sources``
    not to add the toml source: env + defaults only.
    """
    monkeypatch.setenv("LILBEE_SKIP_TOML_CONFIG", "1")


@pytest.fixture(autouse=True)
def _drain_textual_threads():
    """Safety net: join non-daemon threads that outlive the test.

    Daemon threads (executor workers, litestar QueueListeners) are safe to
    ignore since they won't block process exit. Only non-daemon threads need
    explicit joining to prevent xdist hangs.
    """
    before = set(threading.enumerate())
    yield
    for thread in threading.enumerate():
        if thread in before or thread is threading.current_thread():
            continue
        if thread.is_alive() and not thread.daemon:
            thread.join(timeout=2.0)


@pytest.fixture(autouse=True)
def _isolate_provider_env_keys():
    """Snapshot and restore per-provider API-key env vars for every test."""
    from lilbee.providers.sdk_backend import PROVIDER_KEYS

    env_vars = [env for _prov, _cfg, env, _label in PROVIDER_KEYS]
    snapshot = {var: os.environ.get(var) for var in env_vars}
    yield
    for var, value in snapshot.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value


@pytest.fixture(autouse=True)
def _isolate_cfg(tmp_path, request):
    """Snapshot and restore cfg for every test to prevent cross-test pollution.

    ``documents_dir`` is isolated to a scratch path so unit tests that drive
    ``/add``-style flows can't see files left behind in the dev
    ``.lilbee/documents/`` by an earlier run; without this, overwrite
    prompts fire on phantom "existing" files. ``data_root`` is also
    redirected so tests that persist settings via
    ``apply_settings_update`` never touch the developer's real
    ``config.toml``.

    Integration tests are opted out of the ``documents_dir`` override:
    their ``wiki_pipeline`` / ``rag_pipeline`` session fixtures set
    ``documents_dir`` to a real seeded directory, and a per-function
    override would break the contract every integration test assumes.
    """
    snapshot = cfg.model_copy()
    cfg.models_dir = tmp_path / "models"
    cfg.data_root = tmp_path / "data_root"
    if "integration" not in request.node.nodeid.split("/"):
        cfg.documents_dir = tmp_path / "documents"
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))
    cfg.clear_model_defaults()


def _default_provider_mock():
    """Return a ``LLMProvider``-spec'd MagicMock whose ``chat`` yields a ``ChatResult``.

    Without this default, ``MagicMock(spec=LLMProvider).chat(...)`` returns
    another MagicMock and the searcher's ``result.text`` access then breaks
    every downstream string consumer (regex, ``strip()``, etc.). Tests that
    care about the chat output override the return value explicitly.
    """
    from lilbee.providers.base import ChatResult, FinishReason, LLMProvider

    provider = MagicMock(spec=LLMProvider)
    provider.chat.return_value = ChatResult(text="", tool_calls=(), finish_reason=FinishReason.STOP)
    provider.supports_tools.return_value = False
    # role_ready feeds HealthResponse.chat_ready (a bool); default to warm so the
    # mock validates. Cold-start tests override this explicitly.
    provider.role_ready.return_value = True
    # Chat admission + context advertising read these; default to single-flight
    # with an unknown window so the gate and the /v1/models shape stay valid.
    provider.max_concurrent_chats.return_value = 1
    provider.served_chat_ctx.return_value = None
    return provider


def _default_store_mock():
    store = MagicMock()
    store.search.return_value = []
    store.bm25_probe.return_value = []
    store.get_sources.return_value = []
    store.count_sources.return_value = 0
    store.add_chunks.side_effect = len
    return store


def _default_embedder_mock():
    embedder = MagicMock()
    embedder.embed.return_value = [0.1] * 768
    embedder.embed_batch.side_effect = lambda texts, **kw: [[0.1] * 768 for _ in texts]
    # Production reads embedder.truncated_total to compute the per-sync delta; the
    # mock never truncates, so it must report a real 0 rather than a MagicMock.
    embedder.truncated_total = 0
    return embedder


def _default_reranker_mock():
    reranker = MagicMock()
    reranker.rerank.side_effect = lambda q, r, **kw: r
    return reranker


def _default_concepts_mock():
    concepts = MagicMock()
    concepts.get_graph.return_value = False
    return concepts


def _default_clusterer_mock():
    clusterer = MagicMock()
    clusterer.available.return_value = False
    clusterer.get_clusters.return_value = []
    return clusterer


def make_mock_services(**overrides):
    """Create a mock Services container. Override individual services via kwargs.

    The provider defaults to a ``MagicMock`` speccing :class:`LLMProvider`, so
    inference lifecycle calls (cancel / reload_role / add_spawn_listener) and the
    chat/embed/rerank methods are all stubbed. Override ``provider`` with a real
    :class:`FleetProvider` for tests that need genuine fleet behaviour.
    """
    from lilbee.app.services import CrawlerSyncState, Services
    from lilbee.catalog.hf_client import HfClient
    from lilbee.retrieval.query import Searcher
    from lilbee.runtime.ingest_lock import IngestLockRegistry

    provider = overrides.pop("provider", None) or _default_provider_mock()
    store = overrides.pop("store", None) or _default_store_mock()
    embedder = overrides.pop("embedder", None) or _default_embedder_mock()
    reranker = overrides.pop("reranker", None) or _default_reranker_mock()
    concepts = overrides.pop("concepts", None) or _default_concepts_mock()
    clusterer = overrides.pop("clusterer", None) or _default_clusterer_mock()
    searcher = overrides.pop("searcher", None) or Searcher(
        cfg, provider, store, embedder, reranker, concepts
    )
    registry = overrides.pop("registry", None) or MagicMock()
    hf_client = overrides.pop("hf_client", None) or HfClient()
    ingest_lock_registry = overrides.pop("ingest_lock_registry", None) or IngestLockRegistry()
    model_manager = overrides.pop("model_manager", None) or MagicMock()
    crawler_semaphore = overrides.pop("crawler_semaphore", None)
    crawler_sync_state = overrides.pop("crawler_sync_state", None) or CrawlerSyncState()
    known_models = overrides.pop("known_models", None) or _default_known_models_mock()

    return Services(
        provider=provider,
        store=store,
        embedder=embedder,
        reranker=reranker,
        concepts=concepts,
        clusterer=clusterer,
        searcher=searcher,
        registry=registry,
        hf_client=hf_client,
        ingest_lock_registry=ingest_lock_registry,
        model_manager=model_manager,
        crawler_semaphore=crawler_semaphore,
        crawler_sync_state=crawler_sync_state,
        known_models=known_models,
    )


def _default_known_models_mock():
    """KnownModelCache double whose ``refs`` / ``resolve`` return empty by default.

    Tests that need the cache to recognize specific refs override ``refs``
    and ``resolve`` on the returned mock.
    """
    cache = MagicMock()
    cache.refs.return_value = set()
    cache.resolve.return_value = None
    return cache


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
    path = tmp_path / "wiki" / subdir / f"{slug}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
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


@pytest.fixture
def wiki_enabled():
    """Toggle ``cfg.wiki`` on for the duration of a test, then restore."""
    previous = cfg.wiki
    cfg.wiki = True
    yield
    cfg.wiki = previous


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
    cfg.wiki_embedding_faithfulness_threshold = 0.5
    cfg.wiki_prune_raw = False
    cfg.chat_model = TEST_LOCAL_REF
    yield tmp_path
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


# Reusable canonical-shape refs for tests. Anything that needs a concrete
# native ref string can pull these so the shape stays in one place.
TEST_LOCAL_REPO = "test/Test-Model-GGUF"
TEST_LOCAL_FILE = "test-Q4_K_M.gguf"
TEST_LOCAL_REF = f"{TEST_LOCAL_REPO}/{TEST_LOCAL_FILE}"
TEST_EMBED_REPO = "test/Test-Embed-GGUF"
TEST_EMBED_FILE = "test-embed-Q4_K_M.gguf"
TEST_EMBED_REF = f"{TEST_EMBED_REPO}/{TEST_EMBED_FILE}"


def make_test_catalog_model(
    name: str = "TestModel",
    task: str = "chat",
    featured: bool = False,
    size_gb: float = 2.0,
    description: str = "A test model",
    min_ram_gb: float = 4,
    hf_repo: str | None = None,
    gguf_filename: str = "*.gguf",
) -> CatalogModel:
    """Build a CatalogModel with sensible test defaults."""
    return CatalogModel(
        hf_repo=hf_repo or f"test/{name.replace(' ', '-')}",
        gguf_filename=gguf_filename,
        size_gb=size_gb,
        min_ram_gb=min_ram_gb,
        description=description,
        featured=featured,
        downloads=100,
        task=task,
    )


def install_fake_model(hf_repo: str, gguf_filename: str, task: str) -> str:
    """Install a tiny fake GGUF under ``cfg.models_dir`` and return its canonical ref."""
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    source = cfg.models_dir / f"_seed-{gguf_filename}"
    source.write_bytes(b"GGUF\x00")
    registry = ModelRegistry(cfg.models_dir)
    registry.install(
        hf_repo,
        gguf_filename,
        source,
        ModelManifest(
            hf_repo=hf_repo,
            gguf_filename=gguf_filename,
            size_bytes=source.stat().st_size,
            task=task,
            downloaded_at="2026-05-15T00:00:00+00:00",
            blob="",
        ),
    )
    return format_native_gguf_ref(hf_repo, gguf_filename)
