"""QA matrix pytest configuration.

Fixtures here are the contract between scenarios and the runner. Lifecycles
are documented inline; load-bearing for cross-worker isolation under xdist.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import httpx
import pytest
from drivers.mcp import MCPStdioClient
from drivers.tui import TuiSession
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

# Pull by HuggingFace repo ID rather than friendly alias. Friendly aliases
# (qwen3:0.6b, nomic-embed-text:v1.5) are only registered in lilbee builds
# that include FEATURED_ALL. Repo IDs go straight to the catalog and work
# across every published version.
#
# Defaults are the smallest models that produce assertable output on a
# free GHA runner: Qwen3 0.6B for chat, nomic-embed-text-v1.5 for
# embedding, bge-reranker-v2-m3 Q8_0 (~0.4 GB) for the reranker lane.
_DEFAULT_CHAT_MODEL = "Qwen/Qwen3-0.6B-GGUF"
_DEFAULT_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5-GGUF"
_DEFAULT_RERANKER_MODEL = "gpustack/bge-reranker-v2-m3-GGUF"
# Public so xfail decorators that read the lane name at module-import time
# (before any fixture runs) can compare against the same constant the
# `lane` fixture uses. See `current_lane_name()` for the helper that
# returns the resolved LaneName for use in those decorators.
LANE_ENV_VAR = "LILBEE_QA_LANE"

_BIN_ENV_VAR = "LILBEE_QA_BIN"
_MODELS_DIR_ENV_VAR = "LILBEE_QA_MODELS_DIR"
_SERVER_PORT_BASE = 5000
_SERVER_BOOT_TIMEOUT = 60.0
_SERVER_HEALTH_POLL = 0.25
_SERVER_TEARDOWN_GRACE = 5.0
_MCP_STARTUP_TIMEOUT = 60.0
MODEL_PULL_TIMEOUT = 240.0


def worker_port_offset() -> int:
    """Translate PYTEST_XDIST_WORKER (gw0/gw1/.../master) into a port offset."""
    raw = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
    if raw == "master" or not raw.startswith("gw"):
        return 0
    return int(raw.removeprefix("gw"))


def lilbee_env(
    data_dir: Path,
    *,
    models_dir: Path | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a deterministic environment for spawning lilbee under QA.

    ``models_dir`` points the runtime at the shared QA model cache; pass
    it for any test that pulls or uses a model so the cache survives
    across tests. ``extra`` overrides individual keys, applied last.
    """
    env = os.environ.copy()
    env["LILBEE_DATA"] = str(data_dir)
    env["LILBEE_NO_SPLASH"] = "1"
    env["LILBEE_LOG_LEVEL"] = "WARNING"
    if models_dir is not None:
        env["LILBEE_MODELS_DIR"] = str(models_dir)
    if extra:
        env.update(extra)
    return env


# Public timeouts shared across e2e tests so the writer files don't each
# redefine the same constant with drifting values.
SYNC_TIMEOUT = 240.0
ASK_TIMEOUT = 320.0
SEARCH_TIMEOUT = 90.0
STATUS_TIMEOUT = 60.0
TUI_BOOT_TIMEOUT = 60.0
TUI_SCREEN_TIMEOUT = 15.0
TUI_RESPONSE_TIMEOUT = 360.0
SERVER_BOOT_TIMEOUT_WITH_MODELS = 180.0
HTTP_FAST_TIMEOUT = 15.0
HTTP_SLOW_TIMEOUT = 30.0
CLI_FAST_TIMEOUT = 60.0
# Token / extras probes can run cold-start on Windows binary; bump
# accordingly so a slow but-not-hung process doesn't trip pytest-timeout.
TOKEN_FETCH_TIMEOUT = 90.0
EXTRAS_PROBE_TIMEOUT = 120.0
# `lilbee --json model list` walks the on-disk registry; allow headroom
# for a cold-start binary on Windows so a slow enumeration doesn't trip.
MODEL_LIST_TIMEOUT = 180.0


class LaneName(StrEnum):
    """The artifact under test for this run.

    ``L1_SOURCE`` is the local-dev default (whatever ``lilbee`` is on PATH).
    ``L1_PYPI`` is the CI lane that installs from PyPI / a sibling-run wheel
    artifact. ``L2_BINARY`` is the CI lane that runs the released onefile
    binary downloaded from a GH release / sibling-run binary artifact.
    """

    L1_SOURCE = "l1-source"
    L1_PYPI = "l1-pypi"
    L2_BINARY = "l2-binary"


class ModelTask(StrEnum):
    """Task kinds reported by ``lilbee --json model list`` for each row.

    Mirrors ``src/lilbee/catalog/types.ModelTask``. Keep the variant set
    aligned with that source-of-truth so a row whose ``task`` field is
    ``"vision"`` doesn't fall outside any harness enum.
    """

    CHAT = "chat"
    EMBEDDING = "embedding"
    RERANK = "rerank"
    VISION = "vision"


def current_lane_name() -> LaneName | None:
    """Resolve the active lane from ``LILBEE_QA_LANE`` for use at module
    import time (before fixtures run), e.g. inside ``@pytest.mark.xfail``
    decorators. Returns ``None`` if the env var is unset or holds a value
    that isn't a known lane (the ``lane`` fixture fails fast on that case;
    here we just decline to xfail rather than raise during collection).
    """
    raw = os.environ.get(LANE_ENV_VAR)
    if raw is None:
        return None
    try:
        return LaneName(raw)
    except ValueError:
        return None


@dataclass(frozen=True)
class Lane:
    """The artifact under test for this run."""

    name: LaneName
    lilbee_bin: str

    @property
    def is_binary(self) -> bool:
        return self.name is LaneName.L2_BINARY


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Assign xdist groups: writers serialize, others group by file.

    Enforces the invariant that writer-marked tests live in dedicated files. A
    file mixing writer and non-writer tests would fork `lilbee serve` twice for
    one file (writer group + file group), defeating the file-scoped fixture.
    """
    files_with_writers = {item.path for item in items if "writer" in item.keywords}
    files_with_non_writers = {item.path for item in items if "writer" not in item.keywords}
    mixed = files_with_writers & files_with_non_writers
    if mixed:
        names = sorted(p.name for p in mixed)
        raise pytest.UsageError(
            f"writer-marked tests must live in dedicated files; found mixed: {names}"
        )

    for item in items:
        if "writer" in item.keywords:
            item.add_marker(pytest.mark.xdist_group("writers"))
        else:
            item.add_marker(pytest.mark.xdist_group(item.path.name))


@pytest.fixture(scope="session")
def qa_chat_model() -> str:
    return _DEFAULT_CHAT_MODEL


@pytest.fixture(scope="session")
def qa_embedding_model() -> str:
    return _DEFAULT_EMBEDDING_MODEL


@pytest.fixture(scope="session")
def qa_reranker_model() -> str:
    """HF repo for the reranker model.

    Not pre-pulled by ``models_pulled``: the reranker is heavier than the
    chat / embed models and only the rerank-lane tests need it, so the
    pull is per-test. Tests that put the reranker into
    ``lilbee_env_with_models`` must pull it explicitly first.
    """
    return _DEFAULT_RERANKER_MODEL


@pytest.fixture(scope="session")
def qa_models_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Shared, cacheable models directory for e2e tests.

    Resolution order:
      1. LILBEE_QA_MODELS_DIR (CI sets this to a path covered by actions/cache)
      2. ~/.lilbee-qa-models (local dev default; persists across runs)
      3. tmp_path_factory base (last-resort ephemeral; defeats caching)
    """
    explicit = os.environ.get(_MODELS_DIR_ENV_VAR)
    if explicit:
        path = Path(explicit)
    else:
        home = Path(os.path.expanduser("~"))
        path = (
            home / ".lilbee-qa-models"
            if home.exists()
            else tmp_path_factory.getbasetemp() / "models"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _pull_model(lilbee_bin: str, ref: str, env: dict[str, str]) -> None:
    """Run `lilbee model pull <ref>` and raise with a full diagnostic if it fails.

    Captures BOTH stdout and stderr because the rich progress bar paints to
    stderr first and the real error message ends up on stdout. Wrapped in
    tenacity so a transient HF Hub 503 doesn't fail the cell on one hiccup.
    """

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=5, min=5, max=60),
        reraise=True,
    )
    def _attempt() -> None:
        result = subprocess.run(
            [lilbee_bin, "model", "pull", ref],
            env=env,
            capture_output=True,
            text=True,
            timeout=MODEL_PULL_TIMEOUT,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"lilbee model pull {ref} failed: rc={result.returncode}\n"
                f"--- stdout tail ---\n{result.stdout[-1500:]}\n"
                f"--- stderr tail ---\n{result.stderr[-1500:]}"
            )

    _attempt()


def resolve_registered_name(
    lilbee_bin: str, env: dict[str, str], task: ModelTask, repo_substring: str
) -> str:
    """Return the registry name (incl. `.gguf` filename) for a pulled model.

    `lilbee model pull <hf_repo>` registers the model under a key like
    `<owner>/<repo>/<filename>.gguf`. Role assignment needs that full key,
    not the bare repo ID. Walks `lilbee --json model list` and returns the
    first entry whose `task` matches and whose `name` contains
    `repo_substring`.
    """
    result = subprocess.run(
        [lilbee_bin, "--json", "model", "list"],
        env=env,
        capture_output=True,
        text=True,
        timeout=MODEL_LIST_TIMEOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"model list failed: rc={result.returncode}\n{result.stderr}")
    payload = json.loads(result.stdout)
    models = payload.get("models", [])
    matches = [
        m["name"]
        for m in models
        if m.get("task") == task.value and repo_substring in m.get("name", "")
    ]
    if not matches:
        raise RuntimeError(
            f"no {task.value} model registered matching {repo_substring!r}; got {models!r}"
        )
    return matches[0]


@pytest.fixture(scope="session")
def models_pulled(
    lane: Lane,
    qa_models_dir: Path,
    qa_chat_model: str,
    qa_embedding_model: str,
) -> dict[str, str]:
    """Pull chat + embedding models once per session and return their
    registered names (full `<owner>/<repo>/<filename>.gguf` keys).

    Hard-fails on pull failure. If `lilbee model pull` doesn't work for the
    artifact under test, that IS the regression this matrix is designed to
    catch. Masquerading it as a skip lets a fundamental install bug ride
    green CI.
    """
    env = lilbee_env(qa_models_dir / "data", models_dir=qa_models_dir)
    for ref in (qa_chat_model, qa_embedding_model):
        try:
            _pull_model(lane.lilbee_bin, ref, env)
        except (RetryError, RuntimeError) as exc:
            pytest.fail(f"could not pull {ref} after retries: {exc}")
    chat_name = resolve_registered_name(lane.lilbee_bin, env, ModelTask.CHAT, qa_chat_model)
    embed_name = resolve_registered_name(
        lane.lilbee_bin, env, ModelTask.EMBEDDING, qa_embedding_model
    )
    return {"chat": chat_name, "embedding": embed_name}


@pytest.fixture
def lilbee_env_with_models(
    lilbee_data: Path,
    qa_models_dir: Path,
    models_pulled: dict[str, str],
) -> dict[str, str]:
    """Env pointing lilbee at the QA models cache and the resolved role models.

    Uses the registered names from `models_pulled` (full `<owner>/<repo>/<filename>.gguf`
    keys) so role assignment resolves regardless of build-specific friendly aliases.
    """
    return lilbee_env(
        lilbee_data,
        models_dir=qa_models_dir,
        extra={
            "LILBEE_CHAT_MODEL": models_pulled["chat"],
            "LILBEE_EMBEDDING_MODEL": models_pulled["embedding"],
            "LILBEE_QUERY_EXPANSION_COUNT": "0",  # avoid loading chat model on search
        },
    )


@pytest.fixture(scope="session")
def lane() -> Lane:
    raw_name = os.environ.get(LANE_ENV_VAR, LaneName.L1_SOURCE.value)
    try:
        name = LaneName(raw_name)
    except ValueError:
        pytest.fail(
            f"{LANE_ENV_VAR}={raw_name!r} is not a known lane; "
            f"valid values: {[m.value for m in LaneName]}"
        )
    explicit = os.environ.get(_BIN_ENV_VAR)
    if explicit:
        bin_path = explicit
    else:
        discovered = shutil.which("lilbee")
        if not discovered:
            pytest.skip(f"lilbee binary not found; set {_BIN_ENV_VAR} or install lilbee")
        bin_path = discovered
    return Lane(name=name, lilbee_bin=bin_path)


@pytest.fixture
def lilbee_data(tmp_path: Path) -> Path:
    """Per-test data directory; isolates LanceDB across xdist workers."""
    data = tmp_path / "lilbee-data"
    data.mkdir(exist_ok=True)
    return data


_FIXTURE_NOTES_DIR = Path(__file__).parent / "fixtures" / "notes"


def extract_search_results(payload: Any) -> list[dict[str, Any]]:
    """Coalesce ``/api/search`` response shapes across releases.

    Older builds wrapped chunks in ``{"results": [...]}`` or
    ``{"chunks": [...]}``. Current builds return a bare list. Return a
    list either way; callers iterate over chunk dicts.
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list):
            return results
        chunks = payload.get("chunks")
        if isinstance(chunks, list):
            return chunks
    return []


def skip_if_search_unauthenticated(response: httpx.Response) -> None:
    """``/api/search`` returns 401 in builds that enforce auth on the
    public reads; the CLI lane covers the same flow without auth, so
    the HTTP test skips rather than fails."""
    if response.status_code == httpx.codes.UNAUTHORIZED:
        pytest.skip("HTTP /api/search returned 401: auth is enforced in this build")


def seed_fixture_corpus(lilbee_data: Path) -> Path:
    """Copy the shared fixture corpus (coffee + EV notes) into a test's
    documents directory and return the directory path."""
    documents = lilbee_data / "documents"
    documents.mkdir(parents=True, exist_ok=True)
    for path in _FIXTURE_NOTES_DIR.glob("*.md"):
        shutil.copy(path, documents / path.name)
    return documents


# Env-var names for the ollama-up workflow step / harness. Keep here so
# a rename only needs to touch this file plus the workflow.
OLLAMA_HOST_ENV_VAR = "LILBEE_QA_OLLAMA_HOST"
OLLAMA_PORT_ENV_VAR = "LILBEE_QA_OLLAMA_PORT"
OLLAMA_MODEL_ENV_VAR = "LILBEE_QA_OLLAMA_MODEL"


def run_lilbee_with_env(
    lane: Lane,
    args: list[str],
    *,
    env: dict[str, str],
    timeout: float = 60.0,
) -> subprocess.CompletedProcess[str]:
    """Run a lilbee CLI command with a fully-built env, capture stdout/stderr.

    Use this when the caller already has a model-aware env (e.g. from
    ``lilbee_env_with_models``). For the simpler "build env on the fly"
    case use ``run_lilbee`` instead.
    """
    return subprocess.run(
        [lane.lilbee_bin, *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def run_lilbee(
    lane: Lane,
    args: list[str],
    *,
    data_dir: Path,
    timeout: float = 60.0,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a lilbee CLI command and capture stdout/stderr."""
    return run_lilbee_with_env(
        lane,
        args,
        env=lilbee_env(data_dir, extra=extra_env),
        timeout=timeout,
    )


@pytest.fixture
def tui(lane: Lane, lilbee_data: Path) -> Iterator[TuiSession]:
    """Spawn `lilbee` as a TUI in a PTY; tear down on exit."""
    session = TuiSession([lane.lilbee_bin], env=lilbee_env(lilbee_data))
    try:
        yield session
    finally:
        session.close()


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def allocate_server_port() -> int:
    """Pick a free port deterministically per xdist worker, falling back
    to an OS-allocated ephemeral port. Public so test files that spawn
    their own model-aware ``lilbee serve`` use the same port-pick logic
    as the ``server_url`` fixture and avoid xdist port collisions.
    """
    candidate = _SERVER_PORT_BASE + worker_port_offset()
    if _port_is_free(candidate):
        return candidate
    # Fallback if the deterministic slot collides (e.g. with another runner job).
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_for_server(url: str, timeout: float) -> None:
    """Block until ``url`` returns 200, swallowing the four ``httpx``
    transport errors that fire during cold-start on slow runners. Raises
    ``TimeoutError`` with the last transport error attached when the
    deadline elapses.
    """
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url, timeout=2.0)
        except (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
        ) as exc:
            last_err = exc
        else:
            if response.status_code == httpx.codes.OK:
                return
            last_err = httpx.HTTPStatusError(
                f"unexpected status {response.status_code}",
                request=response.request,
                response=response,
            )
        time.sleep(_SERVER_HEALTH_POLL)
    raise TimeoutError(
        f"lilbee serve at {url} not ready within {timeout:.0f}s; last error: {last_err}"
    )


@contextlib.contextmanager
def serve_lilbee_with(
    lane: Lane,
    env: dict[str, str],
    *,
    boot_timeout: float = _SERVER_BOOT_TIMEOUT,
) -> Iterator[str]:
    """Spawn ``lilbee serve`` with the given env, wait for /api/health,
    yield the base URL, and tear down on exit. Use this when the caller
    needs a model-aware env (e.g. ``lilbee_env_with_models``); the
    ``server_url`` fixture covers the empty-store case.
    """
    port = allocate_server_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [lane.lilbee_bin, "serve", "--host", "127.0.0.1", "--port", str(port)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_for_server(f"{base_url}/api/health", timeout=boot_timeout)
        yield base_url
    finally:
        with contextlib.suppress(Exception):
            proc.terminate()
            proc.wait(timeout=_SERVER_TEARDOWN_GRACE)
        with contextlib.suppress(Exception):
            proc.kill()


@pytest.fixture
def server_url(lane: Lane, lilbee_data: Path) -> Iterator[str]:
    """Spawn `lilbee serve` on a per-worker port; yield base URL.

    Function-scoped so each test gets a clean data dir + cold-start server.
    Cheap enough at the smoke/walk tier that file-scoped reuse isn't worth
    the cross-test state coupling.
    """
    with serve_lilbee_with(lane, lilbee_env(lilbee_data)) as base_url:
        yield base_url


@pytest.fixture
def mcp_client(lane: Lane, lilbee_data: Path) -> Iterator[MCPStdioClient]:
    """Spawn `lilbee mcp` and yield a JSON-RPC client over its stdio."""
    client = MCPStdioClient(
        [lane.lilbee_bin, "mcp"],
        env=lilbee_env(lilbee_data),
        startup_timeout=_MCP_STARTUP_TIMEOUT,
    )
    try:
        yield client
    finally:
        client.close()
