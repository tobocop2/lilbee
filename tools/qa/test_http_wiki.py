"""T2 HTTP wiki. Empty-store reads against whatever wiki surface the binary exposes.

Wiki routes have evolved across releases (paths, slug parsing, status route);
these tests assert the contract that the surface either responds with a
2xx/4xx the client can handle, never 5xx, and that empty-store reads return
empty arrays where the route exists.
"""

from __future__ import annotations

import httpx
import pytest

from conftest import HTTP_FAST_TIMEOUT


def _is_4xx_or_2xx(status: int) -> bool:
    return httpx.codes.OK <= status < httpx.codes.INTERNAL_SERVER_ERROR


@pytest.mark.http
def test_wiki_list_returns_empty_or_404(server_url: str) -> None:
    """`GET /api/wiki` either lists pages (200 + array) or 404s if the
    binary doesn't expose this route. Either is fine; 5xx is not."""
    response = httpx.get(f"{server_url}/api/wiki", timeout=HTTP_FAST_TIMEOUT)
    assert _is_4xx_or_2xx(response.status_code), response.text
    if response.status_code == httpx.codes.OK:
        payload = response.json()
        if isinstance(payload, list):
            assert payload == []


@pytest.mark.http
def test_wiki_drafts_returns_empty_or_404(server_url: str) -> None:
    response = httpx.get(f"{server_url}/api/wiki/drafts", timeout=HTTP_FAST_TIMEOUT)
    assert _is_4xx_or_2xx(response.status_code), response.text


@pytest.mark.http
def test_unknown_wiki_slug_returns_404(server_url: str) -> None:
    response = httpx.get(
        f"{server_url}/api/wiki/this-page-does-not-exist", timeout=HTTP_FAST_TIMEOUT
    )
    assert response.status_code == httpx.codes.NOT_FOUND
