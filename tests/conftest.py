"""Shared test helpers."""

import os
import shutil
import sys
import threading
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
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

# Build the cfg singleton hermetically: the first ``import lilbee.core.config``
# below constructs ``cfg`` from env + defaults, so skip the developer's real
# platform config.toml here (before that import) rather than only in an autouse
# fixture that runs after cfg is already loaded. Without this, values like
# ``vision_model`` / ``memory_enabled`` leak from the dev's machine into tests
# that assert defaults. Matches CI, which has no config.toml. (bb-e7d)
os.environ.setdefault("LILBEE_SKIP_TOML_CONFIG", "1")

from lilbee.catalog import CatalogModel
from lilbee.catalog.refs import format_native_gguf_ref
from lilbee.catalog.types import ModelCompat, ModelTask
from lilbee.core.config import cfg
from lilbee.data.extract import xberg as _xberg_extract
from lilbee.data.ingest import file_hash
from lilbee.data.store import CitationRecord
from lilbee.modelhub.registry import ModelManifest, ModelRegistry
from lilbee.providers.fleet.binary import EngineTool

# Pristine extraction entry points, captured before any test can patch them.
_PRISTINE_EXTRACT_DOCUMENT = _xberg_extract.extract_document
_PRISTINE_AEXTRACT_DOCUMENT = _xberg_extract.aextract_document
# Stack-dump watchdog for wedged tests (opt-in via LILBEE_TEST_HANG_DUMP_S).
pytest_plugins = ["tests._hang_watchdog"]

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _patch_executor_daemon_threads() -> None:
    """Make ThreadPoolExecutor workers and LanceDB event-loop threads daemon on 3.11.

    Any asyncio loop we start (CLI asyncio.run, uvicorn, the TUI background
    loop) spins up a ThreadPoolExecutor. On 3.11 those executor threads are
    non-daemon and block xdist worker process exit. On 3.12+ interpreter
    shutdown handles this correctly.

    LanceDB's ``LanceDBBackgroundEventLoop`` thread used to need daemonizing
    here too, but the pinned lancedb already creates it ``daemon=True``, so only
    the executor workers remain.
    """
    if sys.version_info >= (3, 12):
        return
    import concurrent.futures.thread as _tmod

    _real_init = threading.Thread.__init__

    def _init_with_daemon(self: threading.Thread, *args: object, **kwargs: object) -> None:
        _real_init(self, *args, **kwargs)  # type: ignore[misc]
        if getattr(self, "_target", None) is _tmod._worker:
            self.daemon = True

    threading.Thread.__init__ = _init_with_daemon  # type: ignore[assignment]


# Apply at import time so xdist workers get the patch immediately.
_patch_executor_daemon_threads()


def _use_selector_loop_on_windows() -> None:
    """Run the Windows test process on the selector loop, not the proactor one.

    The proactor loop's transport teardown leaks resources across the many
    Textual ``run_test`` app cycles this suite drives; they pile up on the
    xdist worker until it wedges holding the GIL, and the job hangs to
    timeout-minutes (Windows py3.12/3.13; 3.11 is worse and runs serially).
    The selector loop that Linux and macOS already use does not accumulate.

    Safe here because the proactor loop's one hard advantage, asyncio
    subprocess support, is never exercised by the tests: the sole caller
    (``crawler.bootstrap``) is monkeypatched to a fake in every test that
    reaches it. Production Windows keeps the proactor loop for real crawler
    subprocesses; this touches the test process only. Set at import, before
    any loop is created, so every xdist worker inherits it.
    """
    if sys.platform != "win32":
        return
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


_use_selector_loop_on_windows()


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
    """Downgrade asyncio event loop self-pipe corruption to xfail.

    Textual's @work(thread=True) workers can corrupt the event loop's
    self-pipe socket: a worker leaked by an earlier test writes to its
    closed loop's fd after the OS has reused the number for the current
    test's selector. pytest-asyncio's Runner then raises OSError from the
    loop's own selector, at close (teardown) or mid-poll (call), depending
    on when the stray write lands. Neither is a real test failure. The
    call-phase downgrade only applies when the error comes out of the
    selector itself, so a genuine EBADF raised by product code still fails.
    """
    outcome = yield
    report = outcome.get_result()
    if (
        report.when in ("call", "teardown")
        and report.failed
        and call.excinfo is not None
        and call.excinfo.errisinstance(OSError)
        and "Bad file descriptor" in str(call.excinfo.value)
        and (report.when == "teardown" or _raised_in_loop_selector(call.excinfo))
    ):
        report.outcome = "passed"
        report.wasxfail = "asyncio loop self-pipe noise (Textual worker thread)"


def _raised_in_loop_selector(excinfo: pytest.ExceptionInfo) -> bool:  # type: ignore[type-arg]
    """Whether the OSError came out of the event loop's selector poll."""
    tail = excinfo.traceback[-1]
    return str(tail.path).endswith("selectors.py")


@pytest.fixture(autouse=True)
def _suppress_model_scan(request, monkeypatch):
    """Prevent ModelBar._scan_models from doing real work in tests.

    ModelBar.on_mount spawns a @work(thread=True) scan that does registry
    scans, HTTP calls, and litellm imports. Mocking the classify function it
    imports to return empty results makes the worker complete instantly,
    avoiding thread accumulation and per-test join overhead.

    Tests that need real classification use @pytest.mark.real_model_classify.
    """
    if "real_model_classify" not in {m.name for m in request.node.iter_markers()}:
        monkeypatch.setattr(
            "lilbee.cli.tui.widgets.model_bar.classify_installed_models_full",
            lambda: {},
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
def _sealed_engine_resolution(request, monkeypatch):
    """Seal engine-binary resolution so every machine behaves like CI.

    resolve_engine_tool falls back to PATH when the bundled wheel's bin/ is
    empty (always, outside a release build), so a developer with llama-server
    installed resolves a real binary where CI raises ProviderError. Blocking
    the three engine names in shutil.which and the LILBEE_LLAMA_SERVER_PATH
    override forces a test that needs a binary to plant its own (a tmp-file
    ``cfg.llama_server_path``, a fake ``lilbee_engine``, or a
    ``shutil.which`` patch) or use ``@pytest.mark.real_engine_resolution``.
    """
    if "real_engine_resolution" in {m.name for m in request.node.iter_markers()}:
        return
    monkeypatch.setattr(cfg, "llama_server_path", "")
    sealed = {tool.value for tool in EngineTool}
    real_which = shutil.which

    def _engineless_which(cmd, *args, **kwargs):
        return None if cmd in sealed else real_which(cmd, *args, **kwargs)

    monkeypatch.setattr(shutil, "which", _engineless_which)


@pytest.fixture(autouse=True)
def _reset_services_after_test():
    """Drop any Services container ``set_services()`` left around.

    Tests that inject a mock via ``set_services(make_mock_services(...))``
    would otherwise leak into the next test's ``get_services()`` call,
    producing confusing cross-test failures. Shutting the provider down
    drops any running fleet; mocks pass through the same path harmlessly.

    ``lilbee.mcp_server`` is imported eagerly in the setup phase (before the
    test body runs) so that the MCP library's ``FallbackProcess`` class body
    -- which contains a ``subprocess.Popen[bytes]`` type annotation evaluated
    at class-definition time -- is parsed while the real ``subprocess.Popen``
    is still in place.  Tests that monkeypatch ``subprocess.Popen`` to a
    lambda would otherwise cause a ``TypeError`` when teardown first imports
    ``mcp_server`` and triggers that class body.
    """
    from lilbee.mcp_server import set_http_mounted

    yield
    from contextlib import suppress

    from lilbee.app.services import peek_services, set_services

    svc = peek_services()
    if svc is not None:
        with suppress(Exception):
            svc.provider.shutdown()
    set_services(None)
    # build_mcp_mount() flips this process-global True; clear it so a mount test
    # never leaves init/reset gated for an unrelated later test.
    set_http_mounted(False)


@pytest.fixture(autouse=True)
def _join_fleet_background_threads():
    """Retire the fleet's background threads after each test.

    Two shapes need retiring. ``warm_up_pool`` / ``reload_role`` dispatch
    fire-and-forget daemon threads that end on their own once joined; on a host
    without the engine binary they fail fast, and joining keeps that warning
    inside the test that started them. The child-guard spawner is different: its
    ``fleet-spawner`` worker is a process-lifetime executor thread that a join
    never ends (it parks waiting for the next spawn), so any test that ran a real
    probe or launch must close it, or the leak guard flags it against whatever
    test happens to run next.
    """
    yield
    import threading

    from lilbee.providers.fleet import child_guard

    child_guard._spawner.close()
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


@pytest.fixture(scope="session")
def _playwright_browsers_root(tmp_path_factory):
    """One throwaway browser cache for the whole session.

    Session-scoped on purpose: a per-test directory would mean one mkdir per
    test, and a temp root holding thousands of sibling directories is slow
    enough to push the timing-sensitive TUI tests into their timeout.
    """
    return tmp_path_factory.mktemp("ms-playwright")


@pytest.fixture(autouse=True)
def _isolate_playwright_browsers_path(_playwright_browsers_root, monkeypatch):
    """Keep the browser cache out of the developer's real Playwright directory.

    Tests that fake ``chromium_installed() -> False`` drive the install path,
    which creates and locks the browsers directory. Without this they would
    reach ``~/Library/Caches/ms-playwright`` on the machine running them.
    ``_browsers_cache_path`` honors this env var ahead of the platform default.
    """
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(_playwright_browsers_root))


@pytest.fixture
def overlay_reads_config_toml(monkeypatch):
    """Opt a test back into the config.toml overlay path.

    The suite runs with ``LILBEE_SKIP_TOML_CONFIG=1`` for hermeticity, and
    ``overlay_persisted_settings`` honors that flag. Tests that specifically
    exercise the overlay-applies behavior (writing a config.toml to a controlled
    root and asserting it lands on cfg) must clear the flag so overlay runs.
    """
    monkeypatch.delenv("LILBEE_SKIP_TOML_CONFIG", raising=False)


@pytest.fixture(autouse=True)
def _reset_xberg_extract_globals():
    """Start every test with the real extraction functions.

    ``extract_document`` (sync) resolves ``aextract_document`` via module-global
    lookup, so a ``mock.patch`` of ``lilbee.data.extract.xberg.aextract_document``
    intercepts both the async and the sync path -- and the sync path is what
    ``chunk_text`` uses. The ingest/handler suites patch that global heavily; under
    the parallel run a patch occasionally stays active into an unrelated test, and
    ``chunk_text`` then silently returns mock content ('Some extracted text.'),
    a nondeterministic cross-test failure. Resetting the globals before each test
    makes that leak impossible regardless of how a patch escaped. (bb-ql1)
    """
    _xberg_extract.extract_document = _PRISTINE_EXTRACT_DOCUMENT
    _xberg_extract.aextract_document = _PRISTINE_AEXTRACT_DOCUMENT


@pytest.fixture(autouse=True)
def _no_leaked_task_workers():
    """Fail the test that leaves a live task-bar worker thread behind.

    A leaked ``task-*`` worker (an unmocked model download, most expensively)
    outlives its test by minutes and starves whichever TUI test shares the
    xdist worker next, so the suite fails on a random victim instead of the
    owner. A short grace join absorbs workers that are finishing a mocked
    no-op target.
    """
    before = {t.name for t in threading.enumerate() if t.name.startswith("task-")}
    yield
    leaked: list[str] = []
    for thread in threading.enumerate():
        if not thread.name.startswith("task-") or thread.name in before:
            continue
        thread.join(timeout=2.0)
        if thread.is_alive():
            leaked.append(thread.name)
    if leaked:
        pytest.fail(
            f"Test leaked live task-bar worker thread(s) {leaked}: the task "
            "target (often a model download) must be mocked or joined before "
            "the test returns, or it will starve later TUI tests on this worker."
        )


@pytest.fixture(scope="session", autouse=True)
def _shutdown_ingest_pool():
    """Join the shared ingest pool before coverage stops, so the total is stable.

    ``offload._ingest_executor`` is a cached pool of daemon ``lilbee-ingest``
    workers. Left running, they are still tracing when ``coverage.stop()`` combines
    thread data at process exit, and a worker mid-line during that combine
    intermittently loses its lines -- the flake that reads the 100% gate as 99% on
    an unlucky run. A session finalizer runs before coverage's atexit, so joining
    the pool here removes those threads from the combine and makes coverage
    deterministic. No-op when ingest never ran.
    """
    yield
    from lilbee.data import offload

    if offload._ingest_executor.cache_info().currsize:
        offload._ingest_executor().shutdown(wait=True)
        offload._ingest_executor.cache_clear()


@pytest.fixture(autouse=True)
def _drain_textual_threads():
    """Join non-daemon threads that outlive the test, warning on any that survive.

    Daemon threads (executor workers, litestar QueueListeners) are safe to
    ignore since they won't block process exit. Only non-daemon threads need
    explicit joining to prevent xdist hangs.

    A non-daemon thread still alive after the join is the precondition for the
    Windows xdist wedge: it survives loop teardown and keeps posting to the
    closing loop's self-pipe. Warn (naming the test and the threads) rather than
    fail: several suites -- ingest workers, litellm's executor -- legitimately
    leave such threads, so a hard failure would just be noise. The warning makes
    the leakers greppable in CI so a real wedge can be traced to its owner.
    """
    before = set(threading.enumerate())
    yield
    leaked: list[str] = []
    for thread in threading.enumerate():
        if thread in before or thread is threading.current_thread():
            continue
        if thread.is_alive() and not thread.daemon:
            thread.join(timeout=2.0)
            if thread.is_alive():
                leaked.append(thread.name)
    if leaked:
        # warnings.warn (not print/stderr, which pytest's capture swallows on a
        # passing test) so pytest aggregates it into the end-of-run warnings
        # summary and the leaking thread names stay greppable in CI.
        warnings.warn(
            f"_drain_textual_threads: non-daemon thread(s) still alive after a 2s join: {leaked}",
            stacklevel=2,
        )


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
    # data_dir defaults to the real platform dir; isolate it like its siblings so
    # tests that persist under it (e.g. chat session auto-save) stay hermetic.
    cfg.data_dir = tmp_path / "data"
    # Clear any provider API keys the developer has in their real config.toml
    # so tests run hermetically, as CI does (no keys). Otherwise a configured
    # key makes a cloud model "available" and leaks into model discovery and
    # the chat-model availability fallback, breaking tests that assume a clean
    # environment. Tests that exercise key-dependent paths set the key themselves.
    from lilbee.providers.sdk_backend import PROVIDER_API_KEY_FIELD

    for field in PROVIDER_API_KEY_FIELD.values():
        setattr(cfg, field, "")
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
    # warm_progress feeds the /api/warm/stream handler (WarmProgress | None);
    # default to None (idle). Warm-stream tests override this.
    provider.warm_progress.return_value = None
    return provider


def _default_store_mock():
    store = MagicMock()
    store.has_chunks.return_value = True
    # No entity schema induced unless a test persists one.
    store.entity_schema_state.return_value = None
    store.search.return_value = []
    store.bm25_probe.return_value = []
    store.get_sources.return_value = []
    store.count_sources.return_value = 0
    store.get_page_texts.return_value = []
    store.add_chunks.side_effect = len
    return store


def _default_embedder_mock():
    embedder = MagicMock()
    embedder.embed.return_value = np.full(768, 0.1, dtype=np.float32)
    embedder.embed_query.return_value = np.full(768, 0.1, dtype=np.float32)
    embedder.embed_batch.side_effect = lambda texts, **kw: [
        np.full(768, 0.1, dtype=np.float32) for _ in texts
    ]
    embedder.embed_query_batch.side_effect = lambda texts, **kw: [
        np.full(768, 0.1, dtype=np.float32) for _ in texts
    ]
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


def _pick(
    hf_repo: str,
    task: ModelTask,
    params: int,
    size_gb: float,
    min_ram_gb: float,
    gguf_filename: str = "*Q4_K_M.gguf",
) -> CatalogModel:
    """Build one entry of :data:`SAMPLE_PICKS`."""
    return CatalogModel(
        hf_repo=hf_repo,
        gguf_filename=gguf_filename,
        size_gb=size_gb,
        min_ram_gb=min_ram_gb,
        description=f"Sample {task} pick",
        featured=True,
        downloads=1000,
        task=ModelTask(task),
        compat=ModelCompat.SUPPORTED,
        params=params,
    )


def make_test_catalog_model(
    name: str = "TestModel",
    task: str = "chat",
    featured: bool = False,
    size_gb: float = 2.0,
    description: str = "A test model",
    min_ram_gb: float = 4,
    hf_repo: str | None = None,
    gguf_filename: str = "*.gguf",
    compat: ModelCompat = ModelCompat.SUPPORTED,
) -> CatalogModel:
    """Build a CatalogModel with sensible test defaults.

    ``compat`` defaults to SUPPORTED because most callers want a model the
    engine can actually run; the setup wizard refuses to recommend anything
    else.
    """
    return CatalogModel(
        hf_repo=hf_repo or f"test/{name.replace(' ', '-')}",
        gguf_filename=gguf_filename,
        size_gb=size_gb,
        min_ram_gb=min_ram_gb,
        description=description,
        featured=featured,
        downloads=100,
        task=task,
        compat=compat,
    )


# A deterministic stand-in for the live HuggingFace picks. Unit tests must not
# depend on what is trending, and must not reach the network at all, so the
# autouse fixture below seeds these instead. Chat entries cover all four
# parameter tiers so tier-sensitive assertions have something to bite on.
SAMPLE_PICKS: tuple[CatalogModel, ...] = (
    _pick("tiny/Tiny-1B-GGUF", "chat", 1_000_000_000, 0.6, 2.0),
    _pick("small/Small-3B-GGUF", "chat", 3_000_000_000, 1.8, 2.7),
    _pick("mid/Mid-8B-GGUF", "chat", 8_000_000_000, 4.6, 6.9),
    _pick("mid/Mid-13B-GGUF", "chat", 13_000_000_000, 7.4, 11.1),
    _pick("large/Large-27B-GGUF", "chat", 27_000_000_000, 15.4, 23.1),
    _pick("large/Large-34B-GGUF", "chat", 34_000_000_000, 19.4, 29.1),
    _pick("huge/Huge-70B-GGUF", "chat", 70_000_000_000, 40.0, 60.0),
    _pick("huge/Huge-120B-GGUF", "chat", 120_000_000_000, 68.6, 102.9),
    _pick(
        "embed/Test-Embedding-GGUF",
        "embedding",
        300_000_000,
        0.2,
        2.0,
        gguf_filename="test-embedding-Q4_K_M.gguf",
    ),
    _pick("embed/Other-Embedding-GGUF", "embedding", 500_000_000, 0.3, 2.0),
    _pick("embed/Third-Embedding-GGUF", "embedding", 700_000_000, 0.4, 2.0),
    _pick("vision/Test-VL-GGUF", "vision", 4_000_000_000, 2.3, 3.5),
    _pick("rerank/bge-reranker-test-GGUF", "rerank", 600_000_000, 0.4, 2.0),
)


PICKS_CHAT: tuple[CatalogModel, ...] = tuple(m for m in SAMPLE_PICKS if m.task == "chat")
PICKS_EMBEDDING: tuple[CatalogModel, ...] = tuple(m for m in SAMPLE_PICKS if m.task == "embedding")
PICKS_VISION: tuple[CatalogModel, ...] = tuple(m for m in SAMPLE_PICKS if m.task == "vision")
PICKS_RERANK: tuple[CatalogModel, ...] = tuple(m for m in SAMPLE_PICKS if m.task == "rerank")


@pytest.fixture(autouse=True)
def _seed_model_picks(request):
    """Seed deterministic picks so no unit test fetches HuggingFace trending.

    Opt out with ``@pytest.mark.live_picks`` when a test drives the real
    resolution path itself.
    """
    from lilbee.catalog import picks as picks_mod

    if request.node.get_closest_marker("live_picks"):
        yield
        return
    picks_mod.seed_picks(SAMPLE_PICKS)
    yield
    picks_mod.reset_picks()


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


async def one_plan_batch(files):
    """Present a known file list to ``ingest_stream`` as a single-shard stream."""
    yield list(files)


def make_pdf(*, pages: int = 1, title: str | None = None, author: str | None = None) -> bytes:
    """A born-digital PDF with a real text layer, for extraction tests.

    reportlab always writes a ``/Title``, defaulting to "untitled", so a PDF built
    without an explicit ``title`` still carries one rather than reporting none.
    """
    import io

    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    if title is not None:
        c.setTitle(title)
    if author is not None:
        c.setAuthor(author)
    for i in range(pages):
        c.drawString(72, 720, f"Page {i + 1} with a perfectly clean native text layer.")
        c.showPage()
    c.save()
    return buf.getvalue()


_STUB_HF_SEARCH_HIT = CatalogModel(
    hf_repo="Qwen/Qwen3-0.6B-GGUF",
    gguf_filename="Qwen3-0.6B-Q4_K_M.gguf",
    size_gb=0.4,
    min_ram_gb=2.0,
    description="Stub HuggingFace search result.",
    featured=False,
    downloads=1000,
    task="chat",
    params=600_000_000,
)


@pytest.fixture(autouse=True)
def _hermetic_hf_client(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the unit suite off the HuggingFace API.

    Search results were coming from the live API, so a rate-limited runner
    turned "search finds something" into "search finds nothing" and the matrix
    cells that ran last failed. Returns an empty page as the catalog tests
    already stub by hand, and one deterministic hit for a Qwen search, which is
    the only query any test asserts must match.
    """
    if request.node.get_closest_marker("real_hf_client"):
        return  # exercises fetch_models itself, and stubs the transport already

    from lilbee.catalog.hf_client import HfClient
    from lilbee.catalog.models import HfPage

    def _fetch(_self: object, **kwargs: object) -> HfPage:
        search = str(kwargs.get("search") or "").lower()
        hit = "qwen" in search
        return HfPage(models=[_STUB_HF_SEARCH_HIT] if hit else [], has_more=False)

    # On the class, not on get_services().hf_client: this runs for every test in
    # the suite, and building the services container each time is not free.
    monkeypatch.setattr(HfClient, "fetch_models", _fetch)
