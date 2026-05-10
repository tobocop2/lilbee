"""T2 HTTP status. Mirrors the CLI status payload via the HTTP surface."""

from __future__ import annotations

import httpx
import pytest

from conftest import HTTP_FAST_TIMEOUT


@pytest.mark.http
def test_status_returns_200(server_url: str) -> None:
    response = httpx.get(f"{server_url}/api/status", timeout=HTTP_FAST_TIMEOUT)
    assert response.status_code == httpx.codes.OK


@pytest.mark.http
def test_status_payload_has_command_and_config(server_url: str) -> None:
    response = httpx.get(f"{server_url}/api/status", timeout=HTTP_FAST_TIMEOUT)
    payload = response.json()
    assert payload.get("command") == "status"
    assert isinstance(payload.get("config"), dict)


@pytest.mark.http
def test_status_payload_zero_chunks_initially(server_url: str) -> None:
    response = httpx.get(f"{server_url}/api/status", timeout=HTTP_FAST_TIMEOUT)
    payload = response.json()
    assert payload["total_chunks"] == 0
    assert payload["sources"] == []


@pytest.mark.http
def test_status_config_lists_model_roles(server_url: str) -> None:
    response = httpx.get(f"{server_url}/api/status", timeout=HTTP_FAST_TIMEOUT)
    config = response.json()["config"]
    for role in ("chat_model", "embedding_model", "vision_model", "reranker_model"):
        assert role in config, f"missing role in status config: {role}"
