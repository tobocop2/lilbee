"""T2 reranker pipeline smoke.

Pulls a small cross-encoder reranker (gpustack/bge-reranker-v2-m3-GGUF, ~0.4 GB
Q8_0), assigns it as the active reranker via env or via PUT, runs search,
asserts the request completes cleanly with non-empty results.

What these tests do NOT prove: that the reranker actually loaded into memory,
that it ran on the candidates, or that it changed result ordering. The 2-doc
fixture corpus is too small for a deterministic ordering flip, and lilbee may
silently fall back to embedding ranking on a cross-encoder load failure. The
contract gated here is "search with a reranker configured returns the same
shape of results it does without one". An ordering-effect test belongs in a
follow-up with a curated multi-doc corpus.

Reranker pull failures are hard-failed (not skipped) to match the conftest
``models_pulled`` policy: a broken pull is the regression the matrix exists
to catch, not a reason to ride green.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from conftest import (
    HTTP_SLOW_TIMEOUT,
    MODEL_PULL_TIMEOUT,
    STATUS_TIMEOUT,
    SYNC_TIMEOUT,
    Lane,
    ModelTask,
    extract_search_results,
    lilbee_env,
    resolve_registered_name,
    run_lilbee_with_env,
    seed_fixture_corpus,
    serve_lilbee_with,
    skip_if_search_unauthenticated,
)

# Reranker search is slower than the bare semantic search; bump above the
# shared SEARCH_TIMEOUT to absorb the cross-encoder load + per-candidate scoring.
_RERANK_SEARCH_TIMEOUT = 120.0


@pytest.mark.http
@pytest.mark.writer
@pytest.mark.timeout(540)
def test_cli_search_with_reranker_set_returns_results(
    lane: Lane,
    lilbee_data: Path,
    qa_models_dir: Path,
    qa_reranker_model: str,
    lilbee_env_with_models: dict[str, str],
) -> None:
    """Pull the reranker, point lilbee at it via env, sync, run a CLI search.

    The chat / embed models come from the session-scoped ``models_pulled``
    fixture; this test pulls the reranker on top. Asserts the search request
    completes with rc=0 and returns a non-empty results list. Does not
    assert that the reranker actually loaded or that it changed ordering;
    see the file docstring for the scope rationale.
    """
    pull_env = lilbee_env(qa_models_dir / "data", models_dir=qa_models_dir)

    pull = run_lilbee_with_env(
        lane, ["model", "pull", qa_reranker_model], env=pull_env, timeout=MODEL_PULL_TIMEOUT
    )
    assert pull.returncode == 0, (
        f"reranker pull from HF failed; treating as a hard failure to match "
        f"the conftest models_pulled policy. stdout tail:\n"
        f"{pull.stdout[-500:]}\nstderr tail:\n{pull.stderr[-500:]}"
    )

    reranker_name = resolve_registered_name(
        lane.lilbee_bin, pull_env, ModelTask.RERANK, qa_reranker_model
    )

    env_with_reranker = dict(lilbee_env_with_models)
    env_with_reranker["LILBEE_RERANKER_MODEL"] = reranker_name

    seed_fixture_corpus(lilbee_data)
    sync = run_lilbee_with_env(lane, ["sync"], env=env_with_reranker, timeout=SYNC_TIMEOUT)
    assert sync.returncode == 0, sync.stderr

    search = run_lilbee_with_env(
        lane,
        ["--json", "search", "lithium battery", "--top-k", "3"],
        env=env_with_reranker,
        timeout=_RERANK_SEARCH_TIMEOUT,
    )
    assert search.returncode == 0, search.stderr
    payload = json.loads(search.stdout)
    results = extract_search_results(payload)
    assert results, f"search returned zero results with reranker active: {payload}"


@pytest.mark.http
@pytest.mark.writer
@pytest.mark.timeout(540)
def test_http_search_with_reranker_set_returns_results(
    lane: Lane,
    lilbee_data: Path,
    qa_models_dir: Path,
    qa_reranker_model: str,
    lilbee_env_with_models: dict[str, str],
) -> None:
    """`PUT /api/models/reranker` followed by `GET /api/search` completes
    with non-empty results. Tests the PUT -> GET wiring in the running
    server (not just the env-var path).
    """
    pull_env = lilbee_env(qa_models_dir / "data", models_dir=qa_models_dir)

    pull = run_lilbee_with_env(
        lane, ["model", "pull", qa_reranker_model], env=pull_env, timeout=MODEL_PULL_TIMEOUT
    )
    assert pull.returncode == 0, (
        f"reranker pull from HF failed (hard fail per conftest policy):\n"
        f"stdout tail: {pull.stdout[-500:]}\nstderr tail: {pull.stderr[-500:]}"
    )
    reranker_name = resolve_registered_name(
        lane.lilbee_bin, pull_env, ModelTask.RERANK, qa_reranker_model
    )

    seed_fixture_corpus(lilbee_data)
    pre_sync = run_lilbee_with_env(lane, ["sync"], env=lilbee_env_with_models, timeout=SYNC_TIMEOUT)
    assert pre_sync.returncode == 0, pre_sync.stderr

    with serve_lilbee_with(lane, lilbee_env_with_models) as base_url:
        put = httpx.put(
            f"{base_url}/api/models/reranker",
            json={"model": reranker_name},
            timeout=HTTP_SLOW_TIMEOUT,
        )
        assert put.status_code in (httpx.codes.OK, httpx.codes.ACCEPTED), put.text

        response = httpx.get(
            f"{base_url}/api/search",
            params={"q": "lithium battery", "top_k": 3},
            timeout=STATUS_TIMEOUT,
        )
        skip_if_search_unauthenticated(response)
        assert response.status_code == httpx.codes.OK, response.text
        results = extract_search_results(response.json())
        assert results, f"HTTP search with reranker returned no rows: {response.json()}"
