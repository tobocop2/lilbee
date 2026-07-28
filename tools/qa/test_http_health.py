"""T2 HTTP health and version. Cheapest endpoints, no model required.

Health is token-gated like every other route, so the reads below carry the
session token and the last test pins what a caller without one gets.
"""

from __future__ import annotations

import httpx
import pytest

from conftest import HTTP_FAST_TIMEOUT


@pytest.mark.http
def test_health_returns_ok(server_url: str, server_headers: dict[str, str]) -> None:
    response = httpx.get(
        f"{server_url}/api/health", timeout=HTTP_FAST_TIMEOUT, headers=server_headers
    )
    assert response.status_code == httpx.codes.OK
    payload = response.json()
    assert payload.get("status") == "ok"


@pytest.mark.http
def test_health_reports_version(server_url: str, server_headers: dict[str, str]) -> None:
    response = httpx.get(
        f"{server_url}/api/health", timeout=HTTP_FAST_TIMEOUT, headers=server_headers
    )
    payload = response.json()
    version = payload.get("version")
    assert isinstance(version, str)
    assert version, f"empty version string: {payload}"


@pytest.mark.http
def test_health_without_a_token_returns_401(server_url: str) -> None:
    """Even health needs the token: nothing on the surface answers anonymously."""
    response = httpx.get(f"{server_url}/api/health", timeout=HTTP_FAST_TIMEOUT)
    assert response.status_code == httpx.codes.UNAUTHORIZED


@pytest.mark.http
def test_unknown_route_returns_404(server_url: str, server_headers: dict[str, str]) -> None:
    response = httpx.get(
        f"{server_url}/api/this-route-does-not-exist",
        timeout=HTTP_FAST_TIMEOUT,
        headers=server_headers,
    )
    assert response.status_code == httpx.codes.NOT_FOUND
