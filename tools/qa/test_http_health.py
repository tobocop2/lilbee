"""T2 HTTP health and version. Cheapest endpoints, no model required."""

from __future__ import annotations

import httpx
import pytest


@pytest.mark.http
def test_health_returns_ok(server_url: str) -> None:
    response = httpx.get(f"{server_url}/api/health", timeout=10.0)
    assert response.status_code == httpx.codes.OK
    payload = response.json()
    assert payload.get("status") == "ok"


@pytest.mark.http
def test_health_reports_version(server_url: str) -> None:
    response = httpx.get(f"{server_url}/api/health", timeout=10.0)
    payload = response.json()
    version = payload.get("version")
    assert isinstance(version, str)
    assert version, f"empty version string: {payload}"


@pytest.mark.http
def test_unknown_route_returns_404(server_url: str) -> None:
    response = httpx.get(f"{server_url}/api/this-route-does-not-exist", timeout=10.0)
    assert response.status_code == httpx.codes.NOT_FOUND
