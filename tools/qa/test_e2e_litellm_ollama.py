"""T5 e2e Ollama backend. Exercise lilbee with LILBEE_LLM_PROVIDER=remote
pointed at a local ollama daemon.

Catches the failure mode reported against the release executable: segfault
when picking an Ollama-backed chat model (bb-m234). On POSIX, a SIGSEGV in
the spawned lilbee process surfaces as a negative subprocess returncode;
asserting `returncode >= 0` distinguishes a clean failure from a crash.

Skips cleanly when no ollama daemon is reachable on the default port. The
matrix runs cells without ollama too, and the absence of the daemon is a
test-environment fact, not a regression.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
from pathlib import Path

import httpx
import pytest

from conftest import (
    OLLAMA_HOST_ENV_VAR,
    OLLAMA_MODEL_ENV_VAR,
    OLLAMA_PORT_ENV_VAR,
    STATUS_TIMEOUT,
    Lane,
    LaneName,
    current_lane_name,
    lilbee_env,
    run_lilbee_with_env,
)

_OLLAMA_DEFAULT_HOST = "127.0.0.1"
_OLLAMA_DEFAULT_PORT = 11434
_OLLAMA_HEALTHCHECK_TIMEOUT = 2.0
_CHAT_TIMEOUT = 360.0


def _ollama_base_url() -> str:
    host = os.environ.get(OLLAMA_HOST_ENV_VAR, _OLLAMA_DEFAULT_HOST)
    port = int(os.environ.get(OLLAMA_PORT_ENV_VAR, str(_OLLAMA_DEFAULT_PORT)))
    return f"http://{host}:{port}"


def _ollama_reachable() -> bool:
    """Probe the same host/port the URL builder will use, not the defaults."""
    host = os.environ.get(OLLAMA_HOST_ENV_VAR, _OLLAMA_DEFAULT_HOST)
    port = int(os.environ.get(OLLAMA_PORT_ENV_VAR, str(_OLLAMA_DEFAULT_PORT)))
    try:
        with socket.create_connection((host, port), timeout=_OLLAMA_HEALTHCHECK_TIMEOUT):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def ollama_url() -> str:
    """Resolve the ollama daemon URL or skip the suite."""
    if not _ollama_reachable():
        pytest.skip(
            "ollama daemon not reachable on 127.0.0.1:11434; "
            "set LILBEE_QA_OLLAMA_HOST/PORT or start `ollama serve`"
        )
    return _ollama_base_url()


@pytest.fixture(scope="session")
def ollama_chat_model(ollama_url: str) -> str:
    """Resolve which ollama-pulled chat model to drive against.

    Reads LILBEE_QA_OLLAMA_MODEL if set; otherwise picks the first model
    the daemon reports via /api/tags. Skips if no models are installed.
    """
    explicit = os.environ.get(OLLAMA_MODEL_ENV_VAR)
    if explicit:
        return explicit
    response = httpx.get(f"{ollama_url}/api/tags", timeout=10.0)
    response.raise_for_status()
    payload = response.json()
    models = payload.get("models", [])
    if not models:
        pytest.skip(
            f"no models installed in ollama; pull one (e.g. `ollama pull qwen3:0.6b`) "
            f"or set {OLLAMA_MODEL_ENV_VAR}"
        )
    return models[0]["name"]


@pytest.fixture
def lilbee_env_with_ollama(
    lilbee_data: Path, ollama_url: str, ollama_chat_model: str
) -> dict[str, str]:
    """Env that points lilbee at ollama as the remote LLM provider.

    lilbee's config validator requires remote-provider models to carry an
    `ollama/` (or `openai/`, `anthropic/`, `gemini/`) prefix; bare
    `qwen3:0.6b` is rejected as ambiguous. Prefix the ref before passing
    it to LILBEE_CHAT_MODEL.
    """
    chat_ref = ollama_chat_model
    if "/" not in chat_ref:
        chat_ref = f"ollama/{chat_ref}"
    return lilbee_env(
        lilbee_data,
        extra={
            "LILBEE_LLM_PROVIDER": "remote",
            "LILBEE_REMOTE_BASE_URL": ollama_url,
            "LILBEE_CHAT_MODEL": chat_ref,
        },
    )


def _has_litellm_extras(lane: Lane, lilbee_data: Path) -> bool:
    """Probe whether the artifact has the [litellm] extra wired at runtime.

    Runs ``lilbee --json self-check-extras`` against an isolated env
    (mirrors the per-test isolation invariant the rest of the harness
    follows) and reads the per-extra boolean directly from the JSON
    payload. Can't gate on exit-code alone because the command exits 1
    if ANY extra is missing, so a healthy litellm with a missing crawler
    would falsely mark this probe False.
    """
    result = run_lilbee_with_env(
        lane, ["--json", "self-check-extras"], env=lilbee_env(lilbee_data), timeout=60
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False
    return payload.get("litellm") is True


# Every ollama-backed test pays the same two gates: xfail on the bundled
# binary (bb-m234 startup segfault on ollama/* chat models) and skip when
# the artifact doesn't ship the [litellm] extra. Wrap once so the
# decorators don't drift between tests.
_xfail_on_binary_segfault = pytest.mark.xfail(
    current_lane_name() is LaneName.L2_BINARY,
    reason="bb-m234: binary segfaults at startup with an ollama/* LILBEE_CHAT_MODEL",
    strict=False,
)


def _skip_without_litellm(lane: Lane, lilbee_data: Path) -> None:
    if not _has_litellm_extras(lane, lilbee_data):
        pytest.skip("litellm not importable; ollama provider path not available")


def _assert_no_segfault(result: subprocess.CompletedProcess[str], context: str) -> None:
    """Negative returncode on POSIX means killed by signal; SIGSEGV is -11."""
    assert result.returncode >= 0, (
        f"{context} crashed with signal {-result.returncode}; "
        f"stdout tail: {result.stdout[-500:]}\nstderr tail: {result.stderr[-500:]}"
    )


@pytest.mark.wiki
@pytest.mark.writer
@pytest.mark.timeout(120)
@_xfail_on_binary_segfault
def test_status_with_ollama_backend_returns_chat_model(
    lane: Lane, lilbee_data: Path, lilbee_env_with_ollama: dict[str, str]
) -> None:
    """`lilbee --json status` configured for ollama exits 0 and includes the
    chat_model in the rendered config payload.

    bb-m234 reproduces a SIGSEGV in the release executable when the
    chat_model resolves to an ollama ref. Status doesn't run inference;
    it touches the registry / provider resolution path. Asserting both
    "no signal kill" and "config payload renders chat_model" so the test
    catches a silent regression where the path runs but drops the value.
    """
    _skip_without_litellm(lane, lilbee_data)

    result = run_lilbee_with_env(
        lane, ["--json", "status"], env=lilbee_env_with_ollama, timeout=STATUS_TIMEOUT
    )
    _assert_no_segfault(result, "lilbee --json status with ollama backend")
    assert result.returncode == 0, f"status failed: stderr={result.stderr}"
    payload = json.loads(result.stdout)
    config = payload.get("config", {})
    assert "chat_model" in config


@pytest.mark.wiki
@pytest.mark.writer
@pytest.mark.timeout(120)
@_xfail_on_binary_segfault
def test_model_list_includes_ollama_model(
    lane: Lane,
    lilbee_data: Path,
    lilbee_env_with_ollama: dict[str, str],
    ollama_chat_model: str,
) -> None:
    """`lilbee --json model list` surfaces ollama-installed models with source=remote."""
    _skip_without_litellm(lane, lilbee_data)

    result = run_lilbee_with_env(
        lane,
        ["--json", "model", "list"],
        env=lilbee_env_with_ollama,
        timeout=STATUS_TIMEOUT,
    )
    _assert_no_segfault(result, "lilbee --json model list with ollama backend")
    assert result.returncode == 0, f"model list failed: stderr={result.stderr}"
    payload = json.loads(result.stdout)
    models = payload.get("models", [])
    remote_names = {m["name"] for m in models if m.get("source") == "remote"}
    # The model may show up as 'qwen3:0.6b' or 'ollama/qwen3:0.6b' depending
    # on how lilbee normalises remote refs in its registry view.
    bare = ollama_chat_model.removeprefix("ollama/")
    assert bare in remote_names or any(bare in name for name in remote_names), (
        f"expected {bare} in remote models; got {remote_names}"
    )


@pytest.mark.wiki
@pytest.mark.writer
@pytest.mark.timeout(420)
@_xfail_on_binary_segfault
def test_ask_via_ollama_backend_completes(
    lane: Lane, lilbee_data: Path, lilbee_env_with_ollama: dict[str, str]
) -> None:
    """`lilbee --json ask` round-trip via ollama; no segfault, answer non-empty.

    Without a corpus this is a generic chat (no retrieval); we only assert
    that the provider stack rendered tokens and exited cleanly.
    """
    _skip_without_litellm(lane, lilbee_data)

    if not shutil.which(lane.lilbee_bin):
        pytest.skip(f"lilbee binary not found at {lane.lilbee_bin}")

    result = run_lilbee_with_env(
        lane,
        ["--json", "ask", "Reply with the single word 'ready'."],
        env=lilbee_env_with_ollama,
        timeout=_CHAT_TIMEOUT,
    )
    _assert_no_segfault(result, "lilbee --json ask via ollama")
    if result.returncode != 0:
        # Surfaces a non-segfault crash (e.g. unexpected exit code from a
        # deeper provider error) with full diagnostics.
        pytest.fail(
            f"ask failed: rc={result.returncode}\n"
            f"stdout: {result.stdout[-1500:]}\nstderr: {result.stderr[-1500:]}"
        )
    payload = json.loads(result.stdout)
    answer = payload.get("answer", "")
    assert isinstance(answer, str) and answer.strip(), payload
