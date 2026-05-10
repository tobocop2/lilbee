"""T2 HTTP config. GET round-trips the in-memory config; key shape stable."""

from __future__ import annotations

import httpx
import pytest

from conftest import HTTP_FAST_TIMEOUT

_REQUIRED_CONFIG_KEYS = (
    "documents_dir",
    "chat_model",
    "embedding_model",
    "chunk_size",
    "chunk_overlap",
    "top_k",
    "max_distance",
)


@pytest.mark.http
def test_config_returns_200(server_url: str) -> None:
    response = httpx.get(f"{server_url}/api/config", timeout=HTTP_FAST_TIMEOUT)
    assert response.status_code == httpx.codes.OK


@pytest.mark.http
def test_config_payload_shape(server_url: str) -> None:
    response = httpx.get(f"{server_url}/api/config", timeout=HTTP_FAST_TIMEOUT)
    payload = response.json()
    assert isinstance(payload, dict)
    for key in _REQUIRED_CONFIG_KEYS:
        assert key in payload, f"config missing required key: {key}"


@pytest.mark.http
def test_config_numeric_fields_have_numeric_types(server_url: str) -> None:
    response = httpx.get(f"{server_url}/api/config", timeout=HTTP_FAST_TIMEOUT)
    payload = response.json()
    assert isinstance(payload["chunk_size"], int)
    assert isinstance(payload["chunk_overlap"], int)
    assert isinstance(payload["top_k"], int)
    assert isinstance(payload["max_distance"], int | float)
