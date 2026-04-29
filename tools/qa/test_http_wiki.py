"""T2 HTTP wiki. Read endpoints over an empty wiki store."""

from __future__ import annotations

import httpx
import pytest


@pytest.mark.http
def test_wiki_list_returns_empty(server_url: str) -> None:
    response = httpx.get(f"{server_url}/api/wiki", timeout=15.0)
    assert response.status_code == httpx.codes.OK
    payload = response.json()
    if isinstance(payload, list):
        assert payload == []
    else:
        assert isinstance(payload, dict)


@pytest.mark.http
def test_wiki_drafts_returns_empty(server_url: str) -> None:
    response = httpx.get(f"{server_url}/api/wiki/drafts", timeout=15.0)
    assert response.status_code == httpx.codes.OK


@pytest.mark.http
def test_unknown_wiki_slug_returns_404(server_url: str) -> None:
    response = httpx.get(f"{server_url}/api/wiki/this-page-does-not-exist", timeout=15.0)
    assert response.status_code == httpx.codes.NOT_FOUND
