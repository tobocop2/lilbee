"""T2 HTTP models. List-style endpoints + the role assignment 422 negative path."""

from __future__ import annotations

import httpx
import pytest


@pytest.mark.http
def test_models_installed_returns_200(server_url: str) -> None:
    response = httpx.get(f"{server_url}/api/models/installed", timeout=30.0)
    assert response.status_code == httpx.codes.OK
    payload = response.json()
    # Response shape varies (sometimes a list, sometimes {"models": [...]}).
    assert isinstance(payload, dict | list), payload


@pytest.mark.http
def test_models_catalog_returns_200(server_url: str) -> None:
    """Featured catalog should always be available; doesn't depend on installed models."""
    response = httpx.get(f"{server_url}/api/models/catalog", timeout=30.0)
    assert response.status_code == httpx.codes.OK
    payload = response.json()
    assert isinstance(payload, dict | list)


@pytest.mark.http
def test_unknown_role_assignment_rejected(server_url: str) -> None:
    """PUT /api/models/<role> with an unknown role does not 5xx; the surface
    rejects the request via auth / validation / method-not-allowed / not-found.
    Any 4xx is acceptable; the contract is "doesn't crash the server"."""
    response = httpx.put(
        f"{server_url}/api/models/not-a-real-role",
        json={"model": "anything"},
        timeout=15.0,
    )
    assert httpx.codes.BAD_REQUEST <= response.status_code < httpx.codes.INTERNAL_SERVER_ERROR
