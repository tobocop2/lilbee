"""QA matrix pytest configuration.

Fixtures here are the contract between scenarios and the runner. Lifecycles
are documented inline; load-bearing for cross-worker isolation under xdist.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from drivers.mcp import MCPStdioClient
from drivers.tui import TuiSession, lilbee_env, worker_port_offset
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

# Pull by HuggingFace repo ID rather than friendly alias. Friendly aliases
# (smollm2:135m, nomic-embed-text:v1.5) are only registered in lilbee builds
# that include FEATURED_ALL — older releases (e.g. b455) reject them. Repo
# IDs go straight to the catalog and work across every published version.
_DEFAULT_CHAT_MODEL = "bartowski/SmolLM2-135M-Instruct-GGUF"
_DEFAULT_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5-GGUF"
_LANE_ENV_VAR = "LILBEE_QA_LANE"
_BIN_ENV_VAR = "LILBEE_QA_BIN"
_CHAT_MODEL_ENV_VAR = "LILBEE_QA_CHAT_MODEL"
_EMBEDDING_MODEL_ENV_VAR = "LILBEE_QA_EMBEDDING_MODEL"
_MODELS_DIR_ENV_VAR = "LILBEE_QA_MODELS_DIR"
_SERVER_PORT_BASE = 5000
_SERVER_BOOT_TIMEOUT = 60.0
_SERVER_HEALTH_POLL = 0.25
_SERVER_TEARDOWN_GRACE = 5.0
_MCP_STARTUP_TIMEOUT = 60.0
_MODEL_PULL_TIMEOUT = 240.0


@dataclass(frozen=True)
class Lane:
    """The artifact under test for this run."""

    name: str
    lilbee_bin: str

    @property
    def is_binary(self) -> bool:
        return self.name == "l2-binary"


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
    return os.environ.get(_CHAT_MODEL_ENV_VAR, _DEFAULT_CHAT_MODEL)


@pytest.fixture(scope="session")
def qa_embedding_model() -> str:
    return os.environ.get(_EMBEDDING_MODEL_ENV_VAR, _DEFAULT_EMBEDDING_MODEL)


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
            timeout=_MODEL_PULL_TIMEOUT,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"lilbee model pull {ref} failed: rc={result.returncode}\n"
                f"--- stdout tail ---\n{result.stdout[-1500:]}\n"
                f"--- stderr tail ---\n{result.stderr[-1500:]}"
            )

    _attempt()


def _resolve_registered_name(
    lilbee_bin: str, env: dict[str, str], task: str, repo_substring: str
) -> str:
    """Return the registry name (incl. `.gguf` filename) for a pulled model.

    `lilbee model pull <hf_repo>` registers the model under a key like
    `<owner>/<repo>/<filename>.gguf`. The chat / embedding role assignment
    needs that full key, not the bare repo ID. This walks `model list`
    output to find the entry matching `repo_substring` for the right task.
    """
    result = subprocess.run(
        [lilbee_bin, "--json", "model", "list"],
        env=env,
        capture_output=True,
        text=True,
        # PyInstaller binary cold-start on Windows can run 30+ seconds while it
        # unpacks itself into a temp dir on every invocation (tracked as
        # bb-rjez). 30s was racy; 180s leaves headroom for the slowest cell
        # without masking a real hang.
        timeout=180,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"model list failed: rc={result.returncode}\n{result.stderr}")
    payload = json.loads(result.stdout)
    models = payload.get("models", [])
    matches = [
        m["name"] for m in models if m.get("task") == task and repo_substring in m.get("name", "")
    ]
    if not matches:
        raise RuntimeError(
            f"no {task} model registered matching {repo_substring!r}; got {models!r}"
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
    env = os.environ.copy()
    env["LILBEE_DATA"] = str(qa_models_dir / "data")
    env["LILBEE_MODELS_DIR"] = str(qa_models_dir)
    env["LILBEE_NO_SPLASH"] = "1"
    env["LILBEE_LOG_LEVEL"] = "WARNING"
    for ref in (qa_chat_model, qa_embedding_model):
        try:
            _pull_model(lane.lilbee_bin, ref, env)
        except (RetryError, RuntimeError) as exc:
            pytest.fail(f"could not pull {ref} after retries: {exc}")
    chat_name = _resolve_registered_name(lane.lilbee_bin, env, "chat", qa_chat_model)
    embed_name = _resolve_registered_name(lane.lilbee_bin, env, "embedding", qa_embedding_model)
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
        extra={
            "LILBEE_MODELS_DIR": str(qa_models_dir),
            "LILBEE_CHAT_MODEL": models_pulled["chat"],
            "LILBEE_EMBEDDING_MODEL": models_pulled["embedding"],
            "LILBEE_QUERY_EXPANSION_COUNT": "0",  # avoid loading chat model on search
        },
    )


@pytest.fixture(scope="session")
def lane() -> Lane:
    name = os.environ.get(_LANE_ENV_VAR, "l1-source")
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
    data.mkdir()
    return data


def run_lilbee(
    lane: Lane,
    args: list[str],
    *,
    data_dir: Path,
    timeout: float = 60.0,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a lilbee CLI command and capture stdout/stderr.

    Spawns lilbee in its own process group / job object so a timeout can
    kill the whole tree, not just the parent. lilbee can fork worker
    processes (huggingface_hub progress, llama-cpp model loaders, etc.);
    if those orphan after a timeout they pile up over a long pytest run
    on a single VM and starve the runner of memory/PIDs (the symptom on
    GHA's native ubuntu/macOS runners is the runner heartbeat dropping
    after ~46 minutes of test execution).
    """
    popen_kwargs: dict[str, object] = {}
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(
        [lane.lilbee_bin, *args],
        env=lilbee_env(data_dir, extra=extra_env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **popen_kwargs,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(
            args=proc.args, returncode=proc.returncode, stdout=stdout, stderr=stderr
        )
    except subprocess.TimeoutExpired:
        if sys.platform == "win32":
            proc.kill()
        else:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        stdout, stderr = proc.communicate()
        raise subprocess.TimeoutExpired(
            cmd=proc.args, timeout=timeout, output=stdout, stderr=stderr
        ) from None


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


def _allocate_server_port() -> int:
    """Pick a free port deterministically per xdist worker, falling back to ephemeral."""
    candidate = _SERVER_PORT_BASE + worker_port_offset()
    if _port_is_free(candidate):
        return candidate
    # Fallback if the deterministic slot collides (e.g. with another runner job).
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_server(url: str, timeout: float) -> None:
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


@pytest.fixture
def server_url(lane: Lane, lilbee_data: Path) -> Iterator[str]:
    """Spawn `lilbee serve` on a per-worker port; yield base URL.

    Function-scoped so each test gets a clean data dir + cold-start server.
    Cheap enough at the smoke/walk tier that file-scoped reuse isn't worth
    the cross-test state coupling.
    """
    port = _allocate_server_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [lane.lilbee_bin, "serve", "--host", "127.0.0.1", "--port", str(port)],
        env=lilbee_env(lilbee_data),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_server(f"{base_url}/api/health", timeout=_SERVER_BOOT_TIMEOUT)
        yield base_url
    finally:
        with contextlib.suppress(Exception):
            proc.terminate()
            proc.wait(timeout=_SERVER_TEARDOWN_GRACE)
        with contextlib.suppress(Exception):
            proc.kill()


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
