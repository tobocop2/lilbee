"""T5 e2e chat. A real question through CLI ask + HTTP /api/ask + MCP search.

Cite-the-source assertion: ask a question only the corpus can answer, verify
the source filename appears in the response. Doesn't assert on token content
(non-deterministic across model versions); asserts retrieval routed correctly.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from drivers.mcp import MCPStdioClient

from conftest import (
    ASK_TIMEOUT,
    SEARCH_TIMEOUT,
    SYNC_TIMEOUT,
    Lane,
    extract_search_results,
    run_lilbee_with_env,
    seed_fixture_corpus,
    serve_lilbee_with,
    skip_if_search_unauthenticated,
)


@pytest.mark.wiki
@pytest.mark.writer
@pytest.mark.timeout(360)
def test_cli_ask_cites_correct_source(
    lane: Lane,
    lilbee_data: Path,
    lilbee_env_with_models: dict[str, str],
) -> None:
    """`lilbee --json ask "battery question"` returns a result citing ev-notes."""
    seed_fixture_corpus(lilbee_data)
    sync = run_lilbee_with_env(lane, ["sync"], env=lilbee_env_with_models, timeout=SYNC_TIMEOUT)
    assert sync.returncode == 0, sync.stderr

    ask = run_lilbee_with_env(
        lane,
        ["--json", "ask", "Answer in one short sentence: which document covers EV batteries?"],
        env=lilbee_env_with_models,
        timeout=ASK_TIMEOUT,
    )
    assert ask.returncode == 0, ask.stderr
    payload = json.loads(ask.stdout)
    answer = payload.get("answer", "")
    sources = payload.get("sources", [])
    assert isinstance(answer, str) and len(answer) > 0, payload
    cited = [s.get("source", s.get("source_path", s.get("filename", ""))) for s in sources]
    assert any("ev-notes" in c for c in cited), {"sources": sources, "answer_len": len(answer)}


@pytest.mark.wiki
@pytest.mark.writer
@pytest.mark.timeout(360)
def test_http_search_returns_battery_source(
    lane: Lane,
    lilbee_data: Path,
    lilbee_env_with_models: dict[str, str],
) -> None:
    """POST /api/search routes the query through embedding and returns ev-notes."""
    seed_fixture_corpus(lilbee_data)
    sync = run_lilbee_with_env(lane, ["sync"], env=lilbee_env_with_models, timeout=SYNC_TIMEOUT)
    assert sync.returncode == 0, sync.stderr

    # Spawn server with the model-aware env (the bare `server_url` fixture
    # uses an empty data dir and would not find the just-synced corpus).
    with serve_lilbee_with(lane, lilbee_env_with_models) as base_url:
        # GET /api/search?q=...&top_k=N. POST returns 405; query param key is `q`.
        response = httpx.get(
            f"{base_url}/api/search",
            params={"q": "lithium-ion battery technology", "top_k": 3},
            timeout=SEARCH_TIMEOUT,
        )
        assert response.status_code in (
            httpx.codes.OK,
            httpx.codes.UNAUTHORIZED,
        ), response.text
        skip_if_search_unauthenticated(response)
        payload = response.json()
        results = extract_search_results(payload)
        sources = [r.get("source", r.get("source_path", "")) for r in results]
        assert any("ev-notes" in s for s in sources), payload


# Cold-start budget for `lilbee mcp` in a frozen binary: spawn + import +
# get_services() pre-warm before stdio attach, then the first `search` call
# pays the embedding-model load (the pre-warm constructs the embedder but
# doesn't load weights until first embed). On a cold Windows binary that
# sequence runs several minutes; the timeouts below absorb it.
_MCP_BINARY_STARTUP_TIMEOUT = 300.0
_MCP_BINARY_CALL_TIMEOUT = 300.0


@pytest.mark.wiki
@pytest.mark.writer
@pytest.mark.timeout(900)
def test_mcp_search_routes_battery_to_ev_notes(
    lane: Lane,
    lilbee_data: Path,
    lilbee_env_with_models: dict[str, str],
) -> None:
    """MCP `search` tool routes the battery query to ev-notes."""
    seed_fixture_corpus(lilbee_data)
    sync = run_lilbee_with_env(lane, ["sync"], env=lilbee_env_with_models, timeout=SYNC_TIMEOUT)
    assert sync.returncode == 0, sync.stderr

    client = MCPStdioClient(
        [lane.lilbee_bin, "mcp"],
        env=lilbee_env_with_models,
        startup_timeout=_MCP_BINARY_STARTUP_TIMEOUT,
    )
    try:
        result = client.call_tool(
            "search",
            {"query": "lithium-ion battery technology", "top_k": 3},
            timeout=_MCP_BINARY_CALL_TIMEOUT,
        )
        assert isinstance(result, dict), result
        # MCP returns content as text blocks; extract sources from the JSON-like text
        text = json.dumps(result)
        assert "ev-notes" in text, result
    finally:
        client.close()
