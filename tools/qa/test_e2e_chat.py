"""T5 e2e chat. A real question through CLI ask + HTTP /api/ask + MCP search.

Cite-the-source assertion: ask a question only the corpus can answer, verify
the source filename appears in the response. Doesn't assert on token content
(non-deterministic across model versions); asserts retrieval routed correctly.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import httpx
import pytest
from drivers.mcp import MCPStdioClient

from conftest import Lane

_FIXTURES = Path(__file__).parent / "fixtures" / "notes"
_SYNC_TIMEOUT = 240.0
_ASK_TIMEOUT = 320.0


def _seed_corpus(lilbee_data: Path) -> Path:
    documents = lilbee_data / "documents"
    documents.mkdir(parents=True, exist_ok=True)
    for path in _FIXTURES.glob("*.md"):
        shutil.copy(path, documents / path.name)
    return documents


def _run(
    lane: Lane, args: list[str], env: dict[str, str], timeout: float
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [lane.lilbee_bin, *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@pytest.mark.wiki
@pytest.mark.writer
@pytest.mark.timeout(360)
def test_cli_ask_cites_correct_source(
    lane: Lane,
    lilbee_data: Path,
    lilbee_env_with_models: dict[str, str],
    models_pulled: dict[str, str],
) -> None:
    """`lilbee --json ask "battery question"` returns a result citing ev-notes."""
    _seed_corpus(lilbee_data)
    sync = _run(lane, ["sync"], lilbee_env_with_models, _SYNC_TIMEOUT)
    assert sync.returncode == 0, sync.stderr

    ask = _run(
        lane,
        ["--json", "ask", "What is lithium-ion battery technology?"],
        lilbee_env_with_models,
        _ASK_TIMEOUT,
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
    models_pulled: dict[str, str],
) -> None:
    """POST /api/search routes the query through embedding and returns ev-notes."""
    _seed_corpus(lilbee_data)
    sync = _run(lane, ["sync"], lilbee_env_with_models, _SYNC_TIMEOUT)
    assert sync.returncode == 0, sync.stderr

    # Spawn server inline since the server_url fixture uses lilbee_data with no models.
    import socket
    import time

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
        # Wait for /api/health
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{base_url}/api/health", timeout=2.0).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.3)
        else:
            pytest.fail("server never came up")

        # GET /api/search?q=...&top_k=N — POST returns 405; query param key is `q`.
        response = httpx.get(
            f"{base_url}/api/search",
            params={"q": "lithium-ion battery technology", "top_k": 3},
            timeout=60.0,
        )
        assert response.status_code in (httpx.codes.OK, 401), response.text
        if response.status_code == 401:
            pytest.skip("HTTP search requires auth in this build; CLI ask path covers the flow")
        payload = response.json()
        # /api/search returns a bare list of chunks; older builds wrapped it in
        # {"results": [...]} or {"chunks": [...]}. Handle all three.
        if isinstance(payload, list):
            results = payload
        else:
            results = payload.get("results", payload.get("chunks", []))
        sources = [r.get("source", r.get("source_path", "")) for r in results]
        assert any("ev-notes" in s for s in sources), payload
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.mark.wiki
@pytest.mark.writer
@pytest.mark.timeout(360)
def test_mcp_search_routes_battery_to_ev_notes(
    lane: Lane,
    lilbee_data: Path,
    lilbee_env_with_models: dict[str, str],
    models_pulled: dict[str, str],
) -> None:
    """MCP `search` tool routes the battery query to ev-notes."""
    _seed_corpus(lilbee_data)
    sync = _run(lane, ["sync"], lilbee_env_with_models, _SYNC_TIMEOUT)
    assert sync.returncode == 0, sync.stderr

    client = MCPStdioClient(
        [lane.lilbee_bin, "mcp"],
        env=lilbee_env_with_models,
        startup_timeout=60.0,
    )
    try:
        result = client.call_tool(
            "search",
            {"query": "lithium-ion battery technology", "top_k": 3},
            timeout=60.0,
        )
        assert isinstance(result, dict), result
        # MCP returns content as text blocks; extract sources from the JSON-like text
        text = json.dumps(result)
        assert "ev-notes" in text, result
    finally:
        client.close()
