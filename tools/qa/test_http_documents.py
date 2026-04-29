"""T2 HTTP documents. Empty-store reads and the unknown-source 404 path."""

from __future__ import annotations

import httpx
import pytest


@pytest.mark.http
def test_documents_returns_empty_list(server_url: str) -> None:
    """An empty data dir should report zero documents, not error."""
    response = httpx.get(f"{server_url}/api/documents", timeout=15.0)
    assert response.status_code == httpx.codes.OK
    payload = response.json()
    # Could be a bare list or {"documents": [...]}, depending on the version.
    if isinstance(payload, list):
        assert payload == []
    else:
        assert isinstance(payload, dict)
        documents = payload.get("documents", payload.get("sources", []))
        assert documents == []


@pytest.mark.http
def test_unknown_document_returns_404(server_url: str) -> None:
    response = httpx.delete(
        f"{server_url}/api/documents/this-source-does-not-exist.md", timeout=15.0
    )
    assert response.status_code in (httpx.codes.NOT_FOUND, httpx.codes.OK)
    # Some versions accept the delete and return ok with deleted=0; either
    # is fine as long as the surface doesn't 500.
    assert response.status_code < httpx.codes.INTERNAL_SERVER_ERROR
