"""QA matrix pytest configuration.

Fixtures here are the contract between scenarios and the runner. Lifecycles
are documented inline; load-bearing for cross-worker isolation under xdist.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import socket
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from drivers.mcp import MCPStdioClient
from drivers.tui import TuiSession, lilbee_env, worker_port_offset

_DEFAULT_CHAT_MODEL = "smollm2:135m"
_LANE_ENV_VAR = "LILBEE_QA_LANE"
_BIN_ENV_VAR = "LILBEE_QA_BIN"
_CHAT_MODEL_ENV_VAR = "LILBEE_QA_CHAT_MODEL"
_SERVER_PORT_BASE = 5000
_SERVER_BOOT_TIMEOUT = 60.0
_SERVER_HEALTH_POLL = 0.25
_SERVER_TEARDOWN_GRACE = 5.0
_MCP_STARTUP_TIMEOUT = 60.0


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
    """Run a lilbee CLI command and capture stdout/stderr."""
    return subprocess.run(
        [lane.lilbee_bin, *args],
        env=lilbee_env(data_dir, extra=extra_env),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
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
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
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
