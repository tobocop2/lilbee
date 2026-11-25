"""Shared test helpers."""

import importlib.util
import io
import os
import shutil
import signal
import sys
import tempfile
import threading
import warnings
from collections.abc import Callable, Generator, Iterator
from contextlib import AbstractContextManager, contextmanager
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest
from filelock import AsyncReadWriteLock, FileLock, ReadWriteLock
from filelock import Timeout as FileLockTimeout
from PIL import Image, ImageDraw

import tests._suite_data_root  # noqa: F401  # must run before lilbee and huggingface_hub load
from tests import _stated_sizes as stated_sizes_module

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
from lilbee.cli.app import clear_overrides
from lilbee.core import profile_files
from lilbee.core.config import cfg
from lilbee.core.profile_files import profile_folders
from lilbee.core.system import canonical_models_dir, default_data_dir
from lilbee.data.extract import xberg as _xberg_extract
from lilbee.data.ingest import file_hash
from lilbee.data.store import CitationRecord, SourceSummary
from lilbee.modelhub.registry import ModelManifest, ModelRegistry
from lilbee.providers.availability import provider_sdk_available as _real_provider_sdk_available
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet.binary import EngineTool, resolve_engine_tool
from lilbee.providers.fleet.engine_install import engine_install_hint
from lilbee.providers.fleet.first_use import installed_tool
from lilbee.runtime.lock import _no_sync_running

# The literal name, not read from the code under test: a wrong name must fail the tests.
SDK_DIST = "liter-llm"

# Pristine extraction entry points, captured before any test can patch them.
_PRISTINE_EXTRACT_DOCUMENT = _xberg_extract.extract_document
_PRISTINE_AEXTRACT_DOCUMENT = _xberg_extract.aextract_document
# Stack-dump watchdog for wedged tests (opt-in via LILBEE_TEST_HANG_DUMP_S).
pytest_plugins = ["tests._hang_watchdog", "pytester"]

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# The developer's platform data root, read before any test redirects it.
REAL_GLOBAL_ROOT = default_data_dir()
# The developer's models folder; the engine pack installs beside it.
REAL_MODELS_DIR = canonical_models_dir()
# The package's own profiles folder; the suite reads a session copy of it.
REAL_PACKAGE_PROFILES_DIR = profile_files.PACKAGE_PROFILES_DIR
# The variables that root derives from, with their real values.
HOME_VARIABLES = ("HOME", "USERPROFILE", "XDG_DATA_HOME", "LOCALAPPDATA")
REAL_HOME_ENVIRONMENT = {name: os.environ.get(name) for name in HOME_VARIABLES}
_SESSION_SCRATCH = pytest.StashKey[tempfile.TemporaryDirectory[str]]()
_PACKAGE_PROFILES_STATE = pytest.StashKey[dict[str, bytes]]()


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

    The selector loop cannot start asyncio subprocesses, so the browser-mode
    crawl tests, which launch Chromium, run their crawl loops on the proactor
    loop through ``windows_proactor_loop`` in ``tests/_windows_loop.py``.
    Production Windows keeps the proactor loop; this touches the test process
    only. Set at import, before any loop is created, so every xdist worker
    inherits it.
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


@pytest.fixture
def stated_sizes(monkeypatch) -> Generator[None]:
    """Switch on ``tests._stated_sizes.state_size`` for one test."""
    stated_sizes_module.install(monkeypatch)
    yield
    stated_sizes_module.uninstall()


@pytest.fixture(autouse=True)
def _serve_never_ends_the_test_process(monkeypatch):
    """Keep ``serve`` from reaching the real process exit while a library open runs.

    A CliRunner serve test that meets a running open would end its xdist worker.
    A test that checks the exit patches ``os._exit`` and restores the real function.
    """
    from lilbee.cli.commands import servers

    monkeypatch.setattr(servers, "exit_if_opens_remain", lambda status: None)


@pytest.fixture(autouse=True)
def _no_real_browser(monkeypatch):
    """Stop a test from opening a page on the machine that runs the suite."""
    monkeypatch.setattr("webbrowser.open", MagicMock())


@pytest.fixture(autouse=True)
def _no_held_install_warnings() -> Generator[None]:
    """Start and end each test with no install warning held, so one test's warning never leaks."""
    from lilbee.providers.fleet import first_use

    first_use.take_install_warnings()
    yield
    first_use.take_install_warnings()


@pytest.fixture(autouse=True)
def _fresh_nvidia_probes() -> Generator[None]:
    """Start and end each test with no cached nvidia-smi answer, so one test's fake never leaks."""
    from lilbee.providers.fleet.gpu_backends import nvidia

    probes = (nvidia._cache, nvidia._cards_probe, nvidia._compute_probe, nvidia._cuda_probe)
    for probe in probes:
        probe.reset()
    yield
    for probe in probes:
        probe.reset()


@pytest.fixture(autouse=True)
def _sealed_host_cards(request, monkeypatch, tmp_path_factory) -> Generator[None]:
    """Run each test on a host with no card: no nvidia-smi, no NVLink topology, no PCI device.

    Every read of the host's cards answers "none", so no count depends on the machine that
    runs the test. A test that needs cards patches the same reader itself, which wins over
    this seal. ``real_nvidia_smi`` opts out for a test that drives ``run_tool`` for real, and
    ``real_windows_registry`` for a test that injects its own registry module.
    """
    from lilbee.providers.fleet import devices, gpu_hardware
    from lilbee.providers.fleet.gpu_backends import nvidia

    if not request.node.get_closest_marker("real_nvidia_smi"):
        monkeypatch.setattr(nvidia, "run_tool", lambda tool, args, timeout: ("", None))

    run_bounded = devices.run_bounded

    def _without_topology(argv, **kwargs):
        if list(argv[:3]) == ["nvidia-smi", "topo", "-m"]:
            return "", None
        return run_bounded(argv, **kwargs)

    monkeypatch.setattr(devices, "run_bounded", _without_topology)
    monkeypatch.setattr(
        gpu_hardware, "_SYSFS_PCI_DEVICE_DIR", tmp_path_factory.getbasetemp() / "no-pci-bus"
    )
    if not request.node.get_closest_marker("real_windows_registry"):
        monkeypatch.setattr(gpu_hardware, "_windows_adapters", lambda: iter(()))
    yield


@pytest.fixture(autouse=True)
def _download_without_a_child_process(request, monkeypatch):
    """Run a download's transfer in this process so a test's patches reach it.

    Every download now runs in a spawned child, and that child re-imports the
    catalog modules without the test's monkeypatches, so a test that stubs the
    transfer would drive a real one. Tests marked ``download_child`` drive the
    relay itself and keep the real child.
    """
    if request.node.get_closest_marker("download_child"):
        return
    from lilbee.catalog import download_process
    from lilbee.runtime.cancellation import TaskCancelledError

    def _in_process(entry, models_dir, token, engine, *, on_progress, on_retry=None, cancel):
        from lilbee.catalog.download import fetch_model_files

        if cancel.is_set():
            raise TaskCancelledError
        return fetch_model_files(
            entry, models_dir, token, on_progress=on_progress, engine=engine, on_retry=on_retry
        )

    monkeypatch.setattr(download_process, "download_in_subprocess", _in_process)


@pytest.fixture(autouse=True)
def _restore_interactive_intent():
    """Snapshot the interactive flag, the upgrade reporter and the config refresh per test.

    mark_interactive_session flips process state; a leak arms the TUI's
    straggler exit inside a test worker, which then exits the worker. The CLI
    callback installs its stderr reporter, which must not reach a later test.
    """
    from lilbee.app import services as services_mod

    before = services_mod._state.interactive
    reporter = services_mod._state.on_upgrade
    config_refresh = services_mod._state.config_refresh
    watchers = list(services_mod._state.pool_watchers)
    yield
    services_mod._state.interactive = before
    services_mod._state.on_upgrade = reporter
    services_mod._state.config_refresh = config_refresh
    services_mod._state.pool_watchers[:] = watchers


@pytest.fixture(autouse=True)
def _restore_server_stop_flag():
    """Snapshot the flag that says the server is stopping around every test.

    A serve teardown sets it and only the next serve clears it, so a leak makes
    a later test's cancelled chat stream end with the stop error.
    """
    from lilbee.app import services as services_mod

    before = services_mod._server_exit.stopping
    yield
    services_mod._server_exit.stopping = before


@pytest.fixture(autouse=True)
def _forget_engine_stop(_restore_server_stop_flag):
    """Clear every engine stop request after each test.

    A stop runs once per process and refuses new containers, so a leak breaks
    every later test that builds the services. A stop still running would set
    its done event during the next test, so it fails this test instead.

    The stop thread sets the server-stopping flag before it sets its done
    event. Asking for the flag fixture makes this teardown run first, so the
    flag is restored after the wait and not before a late stop sets it.

    The fixture holds the process's own lifecycle from before the test. Fixture
    teardown order depends on which autouse fixture asks for ``monkeypatch``
    first, and a test that swaps in a lifecycle of its own must not become the
    one this fixture waits on.
    """
    from lilbee.app import services as services_mod

    lifecycle = services_mod._lifecycle
    yield
    if lifecycle.stop_requested:
        assert lifecycle.wait_stopped(5.0), "the test left an engine stop running"
    lifecycle.reset()


@pytest.fixture(autouse=True)
def _suppress_model_scan(request, monkeypatch):
    """Prevent ModelBar._scan_models from doing real work in tests.

    ModelBar.on_mount spawns a @work(thread=True) scan that does registry
    scans, HTTP calls, and SDK imports. Mocking the classify function it
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
def _no_engine_model_lists(request, monkeypatch):
    """Keep a catalog browse from reading an engine's published model list over the network.

    Tests of that read use ``@pytest.mark.real_engine_model_lists`` and patch ``httpx.get``.
    """
    if "real_engine_model_lists" in {m.name for m in request.node.iter_markers()}:
        return
    from lilbee.catalog.hf_client import HfClient

    monkeypatch.setattr(HfClient, "fetch_engine_models", lambda self, engine: [])


@pytest.fixture(autouse=True)
def _no_router_download(monkeypatch, tmp_path):
    """Keep an engine install from downloading the router; a test of that step patches it back."""
    monkeypatch.setattr(
        "lilbee.app.engine_service.install_router", lambda **_signals: tmp_path / "llama-swap"
    )


@pytest.fixture(autouse=True)
def _assume_provider_sdk_available(request, monkeypatch):
    """Assume the SDK extra is installed for unit tests.

    Production code gates remote-model discovery on ``provider_sdk_available()``.
    Dev environments without ``lilbee[providers]`` would otherwise
    short-circuit every remote-discovery test. Tests that cover the
    missing-extra path skip this fixture with
    ``@pytest.mark.real_provider_sdk_probe``.
    """
    if "real_provider_sdk_probe" in {m.name for m in request.node.iter_markers()}:
        return
    monkeypatch.setattr("lilbee.providers.availability.provider_sdk_available", lambda: True)


def _only_providers_installed(dist: str) -> str:
    if dist == SDK_DIST:
        return "2.5.0"
    raise PackageNotFoundError(dist)


@pytest.fixture(autouse=True)
def _seal_provider_sdk_distribution(monkeypatch):
    """Report the SDK installed, instead of the host's packages."""
    monkeypatch.setattr("lilbee.providers.availability._dist_version", _only_providers_installed)
    _real_provider_sdk_available.cache_clear()
    yield
    _real_provider_sdk_available.cache_clear()


@pytest.fixture(autouse=True)
def _record_toasts(monkeypatch):
    """Record every toast Textual shows; tests read it with ``tests._toasts.posted_toasts``."""
    from textual.notifications import Notifications

    from tests._toasts import recording_add, reset_posted_toasts

    reset_posted_toasts()
    monkeypatch.setattr(Notifications, "add", recording_add(Notifications.add))


@pytest.fixture(autouse=True)
def _fresh_session_host(request, monkeypatch):
    """Give each test a fixed Linux x86-64 host with no GPU; ``real_host_read`` reads the live one.

    The live read starts a child process for the CPU probe, so a read per test is slow. The cache
    is cleared around each test, so an engine install that refreshes the host reads it afresh.
    The remembered device probe is dropped on entry only: a test may still hold a fake cache in
    its place when teardown runs.
    """
    from lilbee.catalog import runnable
    from lilbee.providers.fleet import engine_manifest, planning
    from lilbee.providers.fleet.host_caps import collect

    planning._read_device_cache.clear()
    session_host = runnable._read_host
    session_host.cache_clear()
    if "real_host_read" not in {m.name for m in request.node.iter_markers()}:
        fixed = collect(system="Linux", machine="x86_64", cpu_info={"flags": ["avx2"]}, nvidia=None)
        monkeypatch.setattr(runnable, "detect", lambda: fixed)
        # The OS floor reads the OS version live; a fixed host meets every floor on any runner.
        monkeypatch.setattr(engine_manifest, "host_os_version", lambda _os: "99.0")
    yield
    session_host.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_ingest_pool_size():
    """Size the ingest pool afresh per test; a cached count would fix it for later tests."""
    from lilbee.data.offload import reset_max_workers

    reset_max_workers()
    yield
    reset_max_workers()


@pytest.fixture(autouse=True)
def _fresh_block_bar_answer():
    """Probe the terminal afresh per test; a cached block bar answer would leak a fake terminal."""
    from lilbee.cli.tui.color_compat import draws_block_bars

    draws_block_bars.cache_clear()
    yield
    draws_block_bars.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_engine_help():
    """Read each engine binary's --help afresh per test; a cached text would leak across tests."""
    from lilbee.providers.fleet.adapters import _warn_once
    from lilbee.providers.fleet.binary import _help_timeouts, _read_help, _router_release_of

    _read_help.cache_clear()
    _router_release_of.cache_clear()
    _help_timeouts.clear()
    _warn_once.cache_clear()
    yield
    _read_help.cache_clear()
    _router_release_of.cache_clear()
    _help_timeouts.clear()
    _warn_once.cache_clear()


@pytest.fixture(scope="session")
def _engine_free_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """The ``PATH`` of this session with every engine tool hidden and the other tools kept.

    A folder that holds an engine tool is swapped for a scratch folder of links to
    the other files in it, so a child finds no engine on ``PATH`` and still finds
    ``git`` or ``pdfseparate`` from the same folder.
    """
    sealed = {tool.value for tool in EngineTool}
    scratch = tmp_path_factory.mktemp("engine-free-path")
    folders = []
    for index, folder in enumerate(os.environ.get("PATH", "").split(os.pathsep)):
        if not any(shutil.which(name, path=folder) for name in sealed):
            folders.append(folder)
            continue
        shadow = scratch / str(index)
        shadow.mkdir()
        for entry in Path(folder).resolve().iterdir():
            if entry.stem not in sealed:
                _link_or_copy(entry, shadow / entry.name)
        folders.append(str(shadow))
    return os.pathsep.join(folders)


def _link_or_copy(entry: Path, link: Path) -> None:
    """Link *link* to *entry*; copy a file when the host grants no symlink right (Windows)."""
    try:
        os.symlink(entry, link, target_is_directory=entry.is_dir())
    except OSError:
        if entry.is_file():
            shutil.copy2(entry, link)


@pytest.fixture(scope="session")
def _engine_free_pythonpath(tmp_path_factory: pytest.TempPathFactory) -> str | None:
    """A folder with the engine wheel package minus its ``bin/``, or ``None`` with no wheel.

    A child that finds this folder first on ``PYTHONPATH`` imports a wheel that holds no
    engine, as the wheel is outside a release build.
    """
    spec = importlib.util.find_spec("lilbee_engine_llama_cpp")
    if spec is None or spec.origin is None:
        return None
    shadow = tmp_path_factory.mktemp("engine-free-pythonpath")
    shutil.copytree(
        Path(spec.origin).parent,
        shadow / spec.name,
        ignore=shutil.ignore_patterns("bin", "__pycache__"),
    )
    return str(shadow)


@pytest.fixture(autouse=True)
def _sealed_engine_resolution(request, monkeypatch, _engine_free_path, _engine_free_pythonpath):
    """Seal engine-binary resolution so every machine behaves like CI.

    resolve_engine_tool falls back to PATH when the bundled wheel's bin/ is
    empty (always, outside a release build), so a developer with llama-server
    installed resolves a real binary where CI raises ProviderError. Resolution
    reads the LILBEE_LLAMA_SERVER_PATH override, the bundled wheel and the engine
    names on PATH. All three are sealed in the environment, which a child process
    a test spawns inherits (the wheel through a copy without bin/ that comes first
    on PYTHONPATH), and in shutil.which for this process. A test that needs a binary
    plants its own (a tmp-file ``cfg.llama_server_path``, a fake
    ``lilbee_engine_llama_cpp``, or a ``shutil.which`` patch) or uses
    ``@pytest.mark.real_engine_resolution``.
    """
    if "real_engine_resolution" in {m.name for m in request.node.iter_markers()}:
        return
    monkeypatch.setattr(cfg, "llama_server_path", "")
    monkeypatch.delenv("LILBEE_LLAMA_SERVER_PATH", raising=False)
    monkeypatch.setenv("PATH", _engine_free_path)
    if _engine_free_pythonpath is not None:
        inherited = os.environ.get("PYTHONPATH")
        monkeypatch.setenv(
            "PYTHONPATH", os.pathsep.join(filter(None, [_engine_free_pythonpath, inherited]))
        )
    sealed = {tool.value for tool in EngineTool}
    real_which = shutil.which

    def _engineless_which(cmd, *args, **kwargs):
        return None if cmd in sealed else real_which(cmd, *args, **kwargs)

    monkeypatch.setattr(shutil, "which", _engineless_which)


@pytest.fixture
def real_gguf_parser(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the test at the developer's own engine store and return its gguf-parser.

    The product lookup runs first, with no install: the bundled wheel, then PATH.
    If it raises, the fixture reads the installed engine pack itself, in the real
    engine store. Nothing is installed: a host with no gguf-parser skips, unless
    LILBEE_REQUIRE_GGUF_PARSER is set, which turns the skip into a failure.
    """
    monkeypatch.setattr(cfg, "models_dir", REAL_MODELS_DIR)
    try:
        return resolve_engine_tool(EngineTool.GGUF_PARSER, install=False)
    except ProviderError:
        packed = installed_tool(EngineTool.GGUF_PARSER.value)
    if packed is None:
        message = (
            "gguf-parser is not on PATH and the engine pack is not installed under "
            f"{REAL_MODELS_DIR.parent}. {engine_install_hint()}"
        )
        if os.environ.get("LILBEE_REQUIRE_GGUF_PARSER"):
            pytest.fail(message, pytrace=False)
        pytest.skip(message)
    return packed


@pytest.fixture(autouse=True)
def _sealed_gguf_header_probe(monkeypatch):
    """Seal the GGUF header probe so no unit test reaches HuggingFace.

    Resolving a pull reads each candidate's header over the network. An empty
    header is the documented offline verdict: the file stays eligible and
    selection falls back to the quant ranking. Tests that exercise the gate
    patch ``compat.probe_header`` or ``download.file_header`` themselves, which
    wins over this default; ``header_probe``'s own tests stub ``httpx`` below it.
    """
    from lilbee.catalog import compat
    from lilbee.catalog.header_probe import GgufHeader

    monkeypatch.setattr(compat, "probe_header", lambda _url: GgufHeader())


@pytest.fixture(autouse=True)
def _sealed_provider_key_check(monkeypatch):
    """Answer every hosted-provider key check with 200 so no unit test reaches a provider.

    Tests of the check itself patch ``key_check._http_get`` again, which wins.
    """
    import httpx

    from lilbee.providers import key_check

    def _accepted(url: str, *, headers: dict[str, str]) -> httpx.Response:
        return httpx.Response(200, request=httpx.Request("GET", url))

    monkeypatch.setattr(key_check, "_http_get", _accepted)
    key_check._checked_key_status.cache_clear()
    yield
    key_check._checked_key_status.cache_clear()


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
def _no_unmarked_fleet_warm_up(request, monkeypatch):
    """Fail the test that runs a real fleet warm-up thread without declaring it.

    ``get_services()`` builds a real provider and, with ``worker_pool_eager_start``
    on, starts ``fleet-warm-up`` as a side effect, so a test that only reads the
    store ends up planning and launching an engine on a second thread. That
    thread reads ``os.environ`` and loads the Vulkan loader while the test body
    and its teardown change the environment and the engine lock. On a CI host
    with a GPU stack it killed the worker with a segmentation fault. A test that
    means to exercise the warm-up carries ``@pytest.mark.fleet_warm_up``.
    """
    from lilbee.providers.fleet.provider import FleetProvider

    started: list[str] = []
    real = FleetProvider._warm_up_blocking

    def recording(self):
        name = threading.current_thread().name
        if name == "fleet-warm-up":
            started.append(name)
        return real(self)

    monkeypatch.setattr(FleetProvider, "_warm_up_blocking", recording)
    yield
    if started and "fleet_warm_up" not in {m.name for m in request.node.iter_markers()}:
        pytest.fail(
            "Test ran a real fleet warm-up thread without @pytest.mark.fleet_warm_up: "
            "it plans and launches an engine beside the test body. Keep "
            "cfg.worker_pool_eager_start off or mock the provider's warm_up_pool."
        )


@pytest.fixture(autouse=True)
def _join_fleet_background_threads():
    """Retire the fleet's background threads after each test.

    Two shapes need retiring. ``warm_up_pool`` / ``reload_role`` dispatch
    fire-and-forget daemon threads that end on their own once joined; on a host
    without the engine binary they fail fast, and joining keeps that warning
    inside the test that started them. The child-guard spawner is different: its
    ``fleet-spawner`` thread is a process-lifetime daemon that a join
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
def _drop_the_fleet_plan_snapshot():
    """Clear the process-wide plan snapshot, so no later test's reload probes real devices."""
    yield
    from lilbee.providers.fleet import planning

    planning.clear_plan_probe()


@pytest.fixture(autouse=True)
def _ignore_user_global_config(monkeypatch, tmp_path, request):
    """Point every config.toml read at the test's own data root, never the developer's.

    ``LILBEE_DATA`` matches the ``cfg.data_root`` that ``_isolate_cfg`` sets, so a
    fresh ``Config()``, the CLI overlay and the settings resolver all read the same
    scratch directory. The platform default root (``--global``) is a second scratch
    directory, reached by redirecting the environment it derives from.
    Integration tests keep the real one, which holds their models. The
    skip flag set at import is cleared: a settings write must read back the
    config.toml it just wrote.
    """
    monkeypatch.delenv("LILBEE_SKIP_TOML_CONFIG", raising=False)
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path / "data_root"))
    if "integration" not in request.node.nodeid.split("/"):
        redirect_global_root(monkeypatch, tmp_path / "home")


@pytest.fixture(autouse=True)
def _package_profiles_untouched(request: pytest.FixtureRequest) -> Iterator[None]:
    """Fail the test that changes a file in the package's own profiles folder."""
    yield
    after = folder_state(REAL_PACKAGE_PROFILES_DIR)
    if after != request.config.stash[_PACKAGE_PROFILES_STATE]:
        request.config.stash[_PACKAGE_PROFILES_STATE] = after
        pytest.fail(
            f"The test changed {REAL_PACKAGE_PROFILES_DIR}; "
            "tests read and write the session copy at profile_files.PACKAGE_PROFILES_DIR"
        )


def redirect_global_root(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    """Point every variable the platform data root derives from under *home*."""
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / ".local" / "share"))
    monkeypatch.setenv("LOCALAPPDATA", str(home / "AppData" / "Local"))


def restore_real_home(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give back the real value of every variable the platform data root derives from."""
    for name, value in REAL_HOME_ENVIRONMENT.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


def folder_state(root: Path) -> dict[str, bytes]:
    """Every file under *root*, by its relative path, with its bytes."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def profile_folders_under_real_root(data_root: Path) -> list[Path]:
    """The profile folders for *data_root* that resolve under the real global root."""
    folders = [path for _, path in profile_folders(data_root)]
    return [path for path in folders if path.is_relative_to(REAL_GLOBAL_ROOT)]


@pytest.hookimpl(wrapper=True)
def pytest_sessionstart(session: pytest.Session) -> Iterator[None]:
    """Redirect the global root and the package profiles folder before collection.

    Stops when a profile folder escapes the redirected global root.

    Runs after xdist starts its workers, so each worker reads the real root at import.
    """
    yield
    scratch = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    session.config.stash[_SESSION_SCRATCH] = scratch
    redirect_global_root(pytest.MonkeyPatch(), Path(scratch.name) / "home")
    package_copy = Path(scratch.name) / "package-profiles"
    shutil.copytree(REAL_PACKAGE_PROFILES_DIR, package_copy)
    pytest.MonkeyPatch().setattr(profile_files, "PACKAGE_PROFILES_DIR", package_copy)
    session.config.stash[_PACKAGE_PROFILES_STATE] = folder_state(REAL_PACKAGE_PROFILES_DIR)
    leaks = profile_folders_under_real_root(Path(scratch.name) / "data_root")
    if leaks:
        raise pytest.UsageError(f"Profile folders resolve under the real global root: {leaks}")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Iterator[None]:
    """Run an integration test, and the fixtures it sets up at any scope, on the real root."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        if "integration" in item.nodeid.split("/"):
            restore_real_home(monkeypatch)
        yield


@pytest.fixture(autouse=True)
def _forget_shown_load_warnings():
    """Start each test with no load warning marked as already logged by this process."""
    from lilbee.core.config.load_warnings import forget_shown

    forget_shown()


@pytest.fixture(scope="session")
def _playwright_browsers_root(tmp_path_factory):
    """One throwaway browser cache for the whole session.

    Session-scoped on purpose: a per-test directory would mean one mkdir per
    test, and a temp root holding thousands of sibling directories is slow
    enough to push the timing-sensitive TUI tests into their timeout.
    """
    return tmp_path_factory.mktemp("ms-playwright")


@pytest.fixture(autouse=True)
def _clear_gpu_memory_cache():
    """Clear the GPU memory probe cache so each test starts uncached.

    The cache is module-global and keyed on gpu_memory_fraction; without this,
    a test that mocks the probe can hit a stale entry from a previous test and
    never exercise the mock.
    """
    from lilbee.runtime import hardware

    hardware._available_memory_cache.clear()
    yield
    hardware._available_memory_cache.clear()


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
def unreadable_folder(tmp_path: Path) -> Iterator[Path]:
    """A folder the current user cannot open; skips where the mode bits do not apply."""
    folder = tmp_path / "locked"
    folder.mkdir()
    (folder / "note.txt").write_text("hidden", encoding="utf-8")
    folder.chmod(0)
    try:
        if os.access(folder, os.R_OK):
            pytest.skip("this user can open a mode 0 folder")
        yield folder
    finally:
        folder.chmod(0o700)


@pytest.fixture
def permissive_umask():
    """Run the body under umask 0 so anything not created owner-only lands world-readable."""
    previous = os.umask(0)
    yield
    os.umask(previous)


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


@pytest.fixture(scope="session")
def tessdata_ready(tmp_path_factory) -> Iterator[None]:
    """Fetch Tesseract's language data once per run, and point every OCR call at it.

    xberg downloads a missing language file on first use through one fixed temp
    name and no lock, so OCR calls that run at once on a cold cache can each fail
    the download, and a page comes back without its OCR text. Each test has its
    own HOME, so xberg's default cache is cold in every test. The run's folder
    sits beside the workers' temp folders, is filled under a lock every worker
    shares, and is named by TESSDATA_PREFIX, which xberg searches before its cache.
    """
    from xberg import ExtractionConfig, OcrConfig

    run_root = tmp_path_factory.getbasetemp().parent
    cache = run_root / "xberg-cache"
    tessdata = cache / "tessdata"
    image = Image.new("RGB", (400, 100), "white")
    ImageDraw.Draw(image).text((10, 40), "tessdata ready", fill="black")
    png = io.BytesIO()
    image.save(png, format="PNG")
    config = ExtractionConfig(ocr=OcrConfig(backend="tesseract", language=["eng"]), force_ocr=True)
    with FileLock(run_root / "tessdata.lock"), pytest.MonkeyPatch.context() as env:
        # xberg writes a download under XBERG_CACHE_DIR/tessdata.
        env.setenv("XBERG_CACHE_DIR", str(cache))
        _PRISTINE_EXTRACT_DOCUMENT(png.getvalue(), "image/png", filename="ready.png", config=config)
    with pytest.MonkeyPatch.context() as env:
        env.setenv("TESSDATA_PREFIX", str(tessdata))
        yield


CODE_PARSER_LANGUAGES = ("python", "javascript", "typescript", "go", "rust", "java", "c", "cpp")


@pytest.fixture(scope="session")
def code_parsers_ready(tmp_path_factory) -> Iterator[None]:
    """Download the tree-sitter parsers once per run, and point every parse at them.

    The parser pack caches its downloads under the home folder. Each test has its
    own HOME, so that cache is cold in every test, and each test that parses code
    downloads its parser from the network inside the test timeout. On a gate run
    a stalled download timed out three sync tests, and on a loaded host it took
    the worker down. The run's folder sits beside the workers' temp folders, is
    filled under a lock every worker shares, and is the cache the pack reads.
    """
    from importlib.metadata import version

    from tree_sitter_language_pack import PackConfig, ProcessConfig, configure, process

    run_root = tmp_path_factory.getbasetemp().parent
    # The pack version is in the name: a run without xdist keeps this folder across runs.
    cache = run_root / f"parser-cache-{version('tree-sitter-language-pack')}"
    configure(PackConfig(cache_dir=str(cache)))
    with FileLock(run_root / "parser-cache.lock"):
        for language in CODE_PARSER_LANGUAGES:
            # The pack fetches a parser on its first parse, not in has_language or init.
            try:
                process("x\n", ProcessConfig(language))  # type: ignore[arg-type]
            except Exception as error:
                pytest.fail(f"The {language} parser is not available in {cache}: {error}")
    yield


def shutdown_ingest_pool() -> None:
    """Shut down and clear the shared ingest pool, if one is cached."""
    from lilbee.data import offload

    if offload._ingest_executor.cache_info().currsize:
        offload._ingest_executor().shutdown(wait=True)
        offload._ingest_executor.cache_clear()


@pytest.fixture(autouse=True)
def _shutdown_ingest_pool():
    """Shut down and clear the shared ingest pool after every test, so none carries over."""
    yield
    shutdown_ingest_pool()


@pytest.fixture(autouse=True)
def _drain_textual_threads():
    """Join non-daemon threads that outlive the test, warning on any that survive.

    Daemon threads (executor workers) are safe to
    ignore since they won't block process exit. Only non-daemon threads need
    explicit joining to prevent xdist hangs.

    A non-daemon thread still alive after the join is the precondition for the
    Windows xdist wedge: it survives loop teardown and keeps posting to the
    closing loop's self-pipe. Warn (naming the test and the threads) rather than
    fail: several suites -- ingest workers, the background loop -- legitimately
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
    clear_overrides()
    # get_services() would otherwise start a real fleet warm-up thread in any test
    # that only reads the store; tests of the warm-up turn it on themselves.
    cfg.worker_pool_eager_start = False
    yield
    clear_overrides()
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture
def monkeypatch(_isolate_cfg: None) -> Iterator[pytest.MonkeyPatch]:
    """pytest's ``monkeypatch``, undone before ``_isolate_cfg`` restores cfg."""
    with pytest.MonkeyPatch.context() as patcher:
        yield patcher


def _default_provider_mock():
    """Return a ``LLMProvider``-spec'd MagicMock whose ``chat`` yields a ``ChatResult``.

    Without this default, ``MagicMock(spec=LLMProvider).chat(...)`` returns
    another MagicMock and the searcher's ``result.text`` access then breaks
    every downstream string consumer (regex, ``strip()``, etc.). Tests that
    care about the chat output override the return value explicitly.
    """
    from lilbee.providers.base import ChatResult, FinishReason, LLMProvider, byte_token_estimate

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
    provider.served_chat_slots.return_value = None
    provider.chat_prefill_progress.return_value = None
    # The chunker bounds its budget to the embed cap; None keeps the configured budget.
    provider.embed_token_cap.return_value = None
    # Four characters to a token, as the chunker assumed before it counted tokens; the
    # one-character registration probe xberg sends counts one.
    provider.count_tokens.side_effect = lambda text: max(1, len(text) // 4)
    # Retrieval budgets with the chat tokenizer when it answers; None keeps the character rule.
    provider.count_chat_text_tokens.return_value = None
    # The count route's estimate reads an int from here; the default counts bytes.
    provider.estimate_chat_tokens.side_effect = lambda text, *, model=None: byte_token_estimate(
        text
    )
    provider.health_warnings.return_value = []
    # warm_progress feeds the /api/warm/stream handler (WarmProgress | None);
    # default to None (idle). Warm-stream tests override this.
    provider.warm_progress.return_value = None
    return provider


def summary_of(heads: list[dict]) -> SourceSummary:
    """A source summary over *heads*, as the store reports it."""
    return SourceSummary(len(heads), sum(h["chunk_count"] for h in heads), {}, heads)


def _default_store_mock():
    store = MagicMock()
    store.has_chunks.return_value = True
    # No entity schema induced unless a test persists one.
    store.entity_schema_state.return_value = None
    store.search.return_value = []
    store.bm25_probe.return_value = []
    store.get_sources.return_value = []
    store.web_titles.return_value = {}
    store.source_summary.return_value = SourceSummary(0, 0, {}, [])
    store.count_sources.return_value = 0
    store.get_page_texts.return_value = []
    store.add_chunks.side_effect = len
    # No index identity persisted unless a test writes one.
    store.get_meta.return_value = None
    store.count_chunks.return_value = 0
    store.index_mismatch.return_value = None
    store.embedding_mismatch.return_value = None
    store.canary_mismatch.return_value = None
    store.upgrade_progress.return_value = None
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


@contextmanager
def reset_held_on_a_thread(
    data_root: Path, ends_after_s: float = 30.0
) -> Generator[threading.Thread, None, None]:
    """Hold a reset on *data_root* on a thread until the block ends, *ends_after_s* at most."""
    held = threading.Event()
    release = threading.Event()

    def _reset() -> None:
        with _no_sync_running(data_root):
            held.set()
            release.wait(ends_after_s)

    resetter = threading.Thread(target=_reset)
    resetter.start()
    try:
        assert held.wait(5)
        yield resetter
    finally:
        release.set()
        resetter.join(timeout=5)


@contextmanager
def ctrl_c_reaches_python() -> Generator[None, None, None]:
    """Make ``_thread.interrupt_main`` raise ``KeyboardInterrupt``, as a terminal's Ctrl+C does.

    ``interrupt_main`` runs the handler of SIGINT and does nothing when the process
    started with SIGINT ignored, as a background job of a non-interactive shell does.
    """
    assert threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


ReadTake = Callable[
    [ReadWriteLock | AsyncReadWriteLock, float | None], AbstractContextManager[None]
]


@contextmanager
def read_takes_hooked(around: ReadTake) -> Generator[None, None, None]:
    """Run each take of the read side of a lock inside ``around(lock, timeout)``.

    A sync takes the read side through the async lock class, which does not call the
    thread class's ``acquire_read``, so both classes are hooked. On the async class
    *around* runs on the event loop, so it must not block.
    """
    real_take = ReadWriteLock.acquire_read
    real_async_take = AsyncReadWriteLock.acquire_read

    def _take(
        lock: ReadWriteLock, timeout: float | None = None, *, blocking: bool | None = None
    ) -> object:
        with around(lock, timeout):
            return real_take(lock, timeout, blocking=blocking)

    async def _async_take(
        lock: AsyncReadWriteLock, timeout: float | None = None, *, blocking: bool | None = None
    ) -> object:
        with around(lock, timeout):
            return await real_async_take(lock, timeout, blocking=blocking)

    with (
        mock.patch.object(ReadWriteLock, "acquire_read", _take),
        mock.patch.object(AsyncReadWriteLock, "acquire_read", _async_take),
    ):
        yield


@contextmanager
def cancel_set_when_the_sync_lock_is_granted(
    cancel: threading.Event,
) -> Generator[list[str], None, None]:
    """Set *cancel* as each take of the read side of a sync lock is granted; yield the grants."""
    grants: list[str] = []

    @contextmanager
    def _granted(lock: ReadWriteLock | AsyncReadWriteLock, _timeout: float | None):
        yield
        if Path(lock.lock_file).name == "sync.lock":
            grants.append("granted")
            cancel.set()

    with read_takes_hooked(_granted):
        yield grants


@contextmanager
def sync_lock_not_a_lock_file(data_root: Path) -> Generator[None, None, None]:
    """Put bytes that are not a lock file where *data_root* keeps its sync lock, for the block."""
    lock_file = data_root / "sync.lock"
    data_root.mkdir(parents=True, exist_ok=True)
    lock_file.write_bytes(b"this is not a database\n" * 400)
    try:
        yield
    finally:
        lock_file.unlink()


@contextmanager
def filesystem_without_locking() -> Generator[None, None, None]:
    """Report every lock probe busy for the block, as a filesystem that cannot lock files does."""
    real = ReadWriteLock.acquire_write

    def _probe_busy(lock: ReadWriteLock, *args: object, **kwargs: object) -> object:
        if Path(lock.lock_file).name.startswith("sync.probe."):
            raise FileLockTimeout(lock.lock_file)
        return real(lock, *args, **kwargs)

    with mock.patch.object(ReadWriteLock, "acquire_write", _probe_busy):
        yield


@pytest.fixture
def counting_provider():
    """Services whose provider counts tokens, for tests that chunk text through the real chunker."""
    import lilbee.app.services as svc_mod

    svc_mod.set_services(make_mock_services())
    yield
    svc_mod.set_services(None)


def make_mock_services(**overrides):
    """Create a mock Services container. Override individual services via kwargs.

    The provider defaults to a ``MagicMock`` speccing :class:`LLMProvider`, so
    inference lifecycle calls (cancel / reload_role / add_spawn_listener) and the
    chat/embed/rerank methods are all stubbed. Override ``provider`` with a real
    :class:`FleetProvider` for tests that need genuine fleet behaviour.
    """
    from lilbee.app.engine_choice import EngineBook
    from lilbee.app.services import CrawlerSyncState, Services
    from lilbee.catalog.hf_client import HfClient
    from lilbee.retrieval.query import Searcher
    from lilbee.runtime.ingest_lock import IngestLockRegistry

    engines = overrides.pop("engines", None) or EngineBook(cfg)
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
    hf_client = overrides.pop("hf_client", None) or HfClient(cfg)
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
        engines=engines,
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


def child_pythonpath(*folders: str | Path) -> str:
    """PYTHONPATH for a child process: the sealed value, then *folders*.

    The sealing fixture puts a wheel without an engine first on PYTHONPATH. A child
    that gets its own PYTHONPATH from a test must start from it, or it imports the
    host's engine wheel.
    """
    return os.pathsep.join(filter(None, [os.environ.get("PYTHONPATH"), *map(str, folders)]))


def clean_env(tmp_path: Path | None = None) -> dict[str, str]:
    """Return os.environ with all LILBEE_* vars removed.

    If tmp_path is given, sets LILBEE_DATA to it so no existing config.toml
    is accidentally picked up. Sets ``LILBEE_SKIP_MODEL_TASK_VALIDATION=1``
    so tests using placeholder model names don't trip the per-role
    catalog-task validator; pop it explicitly to exercise that validator.

    The ambient environment is kept because the platform data directory
    resolves through it, and dropping it leaves Windows with no home directory.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("LILBEE_")}
    env["LILBEE_SKIP_MODEL_TASK_VALIDATION"] = "1"
    if tmp_path is not None:
        env["LILBEE_DATA"] = str(tmp_path)
    return env


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
def wiki_isolated_env(tmp_path: Path, counting_provider):
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


def stub_hf_listing(
    monkeypatch: pytest.MonkeyPatch, listings: dict[str, list[CatalogModel]]
) -> list[dict[str, object]]:
    """Serve *listings* (pipeline tag to rows, most downloaded first) in cursor pages.

    Honors ``limit`` and ``cursor`` like the HuggingFace listing API and returns
    the list of recorded fetch arguments.
    """
    from lilbee.app.services import get_services
    from lilbee.catalog.models import HfPage

    calls: list[dict[str, object]] = []

    def _fetch(**kwargs: object) -> HfPage:
        calls.append(kwargs)
        rows = listings.get(str(kwargs["pipeline_tag"]), [])
        start = int(str(kwargs.get("cursor") or 0))
        stop = start + int(str(kwargs["limit"]))
        return HfPage(models=rows[start:stop], next_cursor=str(stop) if stop < len(rows) else None)

    monkeypatch.setattr(get_services().hf_client, "fetch_models", _fetch)
    return calls


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
        return HfPage(models=[_STUB_HF_SEARCH_HIT] if hit else [])

    # On the class, not on get_services().hf_client: this runs for every test in
    # the suite, and building the services container each time is not free.
    monkeypatch.setattr(HfClient, "fetch_models", _fetch)


@pytest.fixture
def index_holds_chunks() -> Iterator[None]:
    """An index with chunks: the rebuild check finds one."""
    with (
        mock.patch("lilbee.app.settings.index_has_chunks", return_value=True),
        mock.patch("lilbee.app.profiles.index_has_chunks", return_value=True),
    ):
        yield


@pytest.fixture
def index_is_empty() -> Iterator[None]:
    """An index with no chunks: the rebuild check finds none."""
    with (
        mock.patch("lilbee.app.settings.index_has_chunks", return_value=False),
        mock.patch("lilbee.app.profiles.index_has_chunks", return_value=False),
    ):
        yield


BROKEN_CRAWLBERG_REASON = "libcrawlberg.so: cannot open shared object file"


class _BrokenCrawlbergFinder:
    """A meta-path finder whose ``crawlberg`` is found but raises ``OSError`` when it loads."""

    def find_spec(self, name: str, path: object = None, target: object = None) -> object:
        if name != "crawlberg":
            return None
        import importlib.abc
        import importlib.machinery

        class _Loader(importlib.abc.Loader):
            def exec_module(self, module: object) -> None:
                raise OSError(BROKEN_CRAWLBERG_REASON)

        return importlib.machinery.ModuleSpec("crawlberg", _Loader())


@pytest.fixture
def broken_crawlberg(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """The crawler extra is installed, but crawlberg raises ``OSError`` on import.

    Yields the error text. The availability probe starts empty before and after the test.
    """
    from lilbee.crawler import availability

    for name in [n for n in sys.modules if n == "crawlberg" or n.startswith("crawlberg.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [_BrokenCrawlbergFinder(), *sys.meta_path])
    availability.crawler_unavailable_reason.cache_clear()
    yield BROKEN_CRAWLBERG_REASON
    availability.crawler_unavailable_reason.cache_clear()
