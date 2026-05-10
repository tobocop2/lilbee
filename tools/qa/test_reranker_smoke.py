"""T2 reranker effect smoke.

Pulls a small cross-encoder reranker (gpustack/bge-reranker-v2-m3-GGUF, ~400MB
Q8_0), assigns it as the active reranker via env, syncs the standard fixture
corpus, and runs `/api/search` plus `lilbee --json search` to verify the
reranker pipeline doesn't crash search and produces results.

We don't assert ordering changes vs. no-reranker. The 2-document corpus has
too few candidates to give a deterministic flip; that test would be flaky
without a curated corpus and curated query. The contract this test gates is
"set a reranker, search still works end to end". Future work in bb-p6sy can
add an ordering-change test once the corpus is expanded.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from conftest import Lane

_FIXTURES = Path(__file__).parent / "fixtures" / "notes"
_PULL_TIMEOUT = 240.0
_SYNC_TIMEOUT = 240.0
_SEARCH_TIMEOUT = 120.0
_SERVER_BOOT_TIMEOUT = 60.0


def _seed_corpus(lilbee_data: Path) -> None:
    documents = lilbee_data / "documents"
    documents.mkdir(parents=True, exist_ok=True)
    for path in _FIXTURES.glob("*.md"):
        shutil.copy(path, documents / path.name)


def _resolve_registered_reranker(lane: Lane, env: dict[str, str], hf_repo: str) -> str:
    """Return the full registered key for a pulled reranker model."""
    result = subprocess.run(
        [lane.lilbee_bin, "--json", "model", "list"],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    models = json.loads(result.stdout).get("models", [])
    matches = [
        m["name"] for m in models if m.get("task") == "rerank" and hf_repo in m.get("name", "")
    ]
    assert matches, f"no rerank model registered matching {hf_repo!r}; got {models!r}"
    return matches[0]


@pytest.mark.http
@pytest.mark.writer
@pytest.mark.timeout(540)
def test_search_with_reranker_returns_results(
    lane: Lane,
    lilbee_data: Path,
    qa_models_dir: Path,
    qa_reranker_model: str,
    lilbee_env_with_models: dict[str, str],
    models_pulled: dict[str, str],
) -> None:
    """End-to-end: pull reranker, configure it active, sync corpus, search.

    Asserts the reranker pipeline integrates cleanly (no crash) and search
    returns at least one chunk. The chat / embed models come from the
    session-scoped models_pulled fixture; this test just adds a reranker
    pull on top.
    """
    pull_env = os.environ.copy()
    pull_env["LILBEE_DATA"] = str(qa_models_dir / "data")
    pull_env["LILBEE_MODELS_DIR"] = str(qa_models_dir)
    pull_env["LILBEE_NO_SPLASH"] = "1"
    pull_env["LILBEE_LOG_LEVEL"] = "WARNING"

    pull = subprocess.run(
        [lane.lilbee_bin, "model", "pull", qa_reranker_model],
        env=pull_env,
        capture_output=True,
        text=True,
        timeout=_PULL_TIMEOUT,
        check=False,
    )
    if pull.returncode != 0:
        # Network flakiness: HF Hub 503 etc. The matrix isn't here to test
        # HF availability; surface as skip with a tail of the failure.
        pytest.skip(
            f"reranker pull from HF failed (likely transient network): "
            f"{pull.stderr[-300:] or pull.stdout[-300:]}"
        )

    reranker_name = _resolve_registered_reranker(lane, pull_env, qa_reranker_model)

    env_with_reranker = dict(lilbee_env_with_models)
    env_with_reranker["LILBEE_RERANKER_MODEL"] = reranker_name

    _seed_corpus(lilbee_data)
    sync = subprocess.run(
        [lane.lilbee_bin, "sync"],
        env=env_with_reranker,
        capture_output=True,
        text=True,
        timeout=_SYNC_TIMEOUT,
        check=False,
    )
    assert sync.returncode == 0, sync.stderr

    search = subprocess.run(
        [lane.lilbee_bin, "--json", "search", "lithium battery", "--top-k", "3"],
        env=env_with_reranker,
        capture_output=True,
        text=True,
        timeout=_SEARCH_TIMEOUT,
        check=False,
    )
    assert search.returncode == 0, search.stderr
    payload = json.loads(search.stdout)
    results = payload.get("results", payload.get("chunks", []))
    assert isinstance(results, list), payload
    assert results, f"search returned zero results with reranker active: {payload}"


@pytest.mark.http
@pytest.mark.writer
@pytest.mark.timeout(540)
def test_http_search_with_reranker_set(
    lane: Lane,
    lilbee_data: Path,
    qa_models_dir: Path,
    qa_reranker_model: str,
    lilbee_env_with_models: dict[str, str],
    models_pulled: dict[str, str],
) -> None:
    """HTTP /api/search route works with reranker assigned via PUT.

    Boots a server, PUTs the reranker model, runs a search, asserts
    structurally-valid results. Doesn't assert exact ordering. Tests the
    PUT -> GET wiring in the running server, not just the env-var path.
    """
    pull_env = os.environ.copy()
    pull_env["LILBEE_DATA"] = str(qa_models_dir / "data")
    pull_env["LILBEE_MODELS_DIR"] = str(qa_models_dir)
    pull_env["LILBEE_NO_SPLASH"] = "1"
    pull_env["LILBEE_LOG_LEVEL"] = "WARNING"

    pull = subprocess.run(
        [lane.lilbee_bin, "model", "pull", qa_reranker_model],
        env=pull_env,
        capture_output=True,
        text=True,
        timeout=_PULL_TIMEOUT,
        check=False,
    )
    if pull.returncode != 0:
        pytest.skip(f"reranker pull from HF failed: {pull.stderr[-300:] or pull.stdout[-300:]}")
    reranker_name = _resolve_registered_reranker(lane, pull_env, qa_reranker_model)

    _seed_corpus(lilbee_data)
    pre_sync = subprocess.run(
        [lane.lilbee_bin, "sync"],
        env=lilbee_env_with_models,
        capture_output=True,
        text=True,
        timeout=_SYNC_TIMEOUT,
        check=False,
    )
    assert pre_sync.returncode == 0, pre_sync.stderr

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    proc = subprocess.Popen(
        [lane.lilbee_bin, "serve", "--host", "127.0.0.1", "--port", str(port)],
        env=lilbee_env_with_models,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + _SERVER_BOOT_TIMEOUT
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{base_url}/api/health", timeout=2.0).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.3)
        else:
            pytest.fail("server never came up")

        put = httpx.put(
            f"{base_url}/api/models/reranker",
            json={"model": reranker_name},
            timeout=30.0,
        )
        assert put.status_code in (httpx.codes.OK, httpx.codes.ACCEPTED), put.text

        response = httpx.get(
            f"{base_url}/api/search",
            params={"q": "lithium battery", "top_k": 3},
            timeout=60.0,
        )
        if response.status_code == httpx.codes.UNAUTHORIZED:
            pytest.skip("HTTP search requires auth in this build; CLI lane covers it")
        assert response.status_code == httpx.codes.OK, response.text
        payload = response.json()
        results = (
            payload
            if isinstance(payload, list)
            else payload.get("results", payload.get("chunks", []))
        )
        assert isinstance(results, list), payload
        assert results, f"HTTP search with reranker returned no rows: {payload}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
