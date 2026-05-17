"""Fixtures for the opencode QA matrix and protocol smoke suites.

Two surfaces share this conftest:

* The opencode matrix needs a real ``opencode`` binary and a long-lived
  ``lilbee serve`` subprocess so opencode can hit ``/v1`` over the wire.
  Those fixtures are session-scoped and skip cleanly when
  ``LILBEE_QA_OPENCODE`` is unset or ``opencode`` is not on ``PATH``.
* The protocol smoke suite drives a Litestar ``AsyncTestClient`` instead
  of spawning a subprocess: it asserts wire shapes only, so the
  in-process ASGI transport is sufficient and CI-cheap.
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
from pathlib import Path

import httpx
import pytest

QA_OPENCODE_ENV = "LILBEE_QA_OPENCODE"
TOOL_CAPABLE_MODEL = "bartowski/Qwen3-0.6B-GGUF::Q4_K_M"

_HEALTH_POLL_SECONDS = 0.25
_HEALTH_TIMEOUT_SECONDS = 60.0
_TEARDOWN_GRACE_SECONDS = 5.0


def _opencode_skip_reason() -> str | None:
    """Return the reason the opencode matrix should skip, or None to run."""
    if not os.environ.get(QA_OPENCODE_ENV):
        return f"set {QA_OPENCODE_ENV}=1 to run the opencode matrix"
    if shutil.which("opencode") is None:
        return "opencode binary not on PATH; install via `npm i -g opencode-ai`"
    return None


def _free_port() -> int:
    """Allocate an ephemeral port and release it for the subprocess to bind."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_health(url: str, timeout: float) -> None:
    """Block until *url* returns 200 or *timeout* elapses."""
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
            last_err = RuntimeError(f"unexpected status {response.status_code}")
        time.sleep(_HEALTH_POLL_SECONDS)
    raise TimeoutError(
        f"lilbee serve at {url} not ready within {timeout:.0f}s; last error: {last_err}"
    )


@pytest.fixture(scope="session")
def opencode_binary() -> str:
    """Resolve the opencode binary path, skipping the suite if unavailable."""
    reason = _opencode_skip_reason()
    if reason is not None:
        pytest.skip(reason)
    path = shutil.which("opencode")
    assert path is not None  # narrowed by the skip above
    return path


@pytest.fixture(scope="session")
def qa_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped data dir for the QA matrix's lilbee serve."""
    return tmp_path_factory.mktemp("lilbee-qa-data")


@pytest.fixture(scope="session")
def lilbee_serve(opencode_binary: str, qa_data_dir: Path) -> Iterator[str]:
    """Spawn ``lilbee serve`` for the QA matrix and yield its base URL.

    Depends on ``opencode_binary`` so it only boots when the matrix runs.
    Uses an ephemeral port to avoid colliding with the developer's own
    server. The server generates its own token and persists it under
    ``$LILBEE_DATA/data/server.json``; ``lilbee agent-config opencode``
    reads back that same path.
    """
    port = _free_port()
    env = os.environ.copy()
    env["LILBEE_DATA"] = str(qa_data_dir)
    env["LILBEE_NO_SPLASH"] = "1"
    env["LILBEE_LOG_LEVEL"] = "WARNING"
    proc = subprocess.Popen(
        ["lilbee", "serve", "--host", "127.0.0.1", "--port", str(port)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        try:
            _wait_for_health(f"{base_url}/api/health", timeout=_HEALTH_TIMEOUT_SECONDS)
        except TimeoutError as exc:
            proc.terminate()
            stdout, stderr = proc.communicate(timeout=_TEARDOWN_GRACE_SECONDS)
            raise TimeoutError(
                f"{exc}\n--- lilbee serve stdout tail ---\n{stdout[-1500:]}\n"
                f"--- lilbee serve stderr tail ---\n{stderr[-1500:]}"
            ) from exc
        yield base_url
    finally:
        with contextlib.suppress(Exception):
            proc.terminate()
            proc.wait(timeout=_TEARDOWN_GRACE_SECONDS)
        with contextlib.suppress(Exception):
            proc.kill()


@pytest.fixture
def opencode_config(
    tmp_path: Path,
    lilbee_serve: str,
    qa_data_dir: Path,
) -> Path:
    """Render ``lilbee agent-config opencode`` against the running server.

    The CLI reads ``server.json`` and ``server.port`` from
    ``LILBEE_DATA``; we point it at the same data dir the spawned server
    wrote those files into.
    """
    out_path = tmp_path / "opencode.json"
    env = os.environ.copy()
    env["LILBEE_DATA"] = str(qa_data_dir)
    result = subprocess.run(
        ["lilbee", "agent-config", "opencode"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"lilbee agent-config opencode failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}\n"
            f"base_url: {lilbee_serve}"
        )
    out_path.write_text(result.stdout)
    return out_path


@pytest.fixture
def auth_headers(qa_data_dir: Path, lilbee_serve: str) -> dict[str, str]:
    """Bearer headers built from the token the spawned server wrote out."""
    session_path = qa_data_dir / "data" / "server.json"
    token = json.loads(session_path.read_text())["token"]
    return {"Authorization": f"Bearer {token}", "x-api-key": token}
