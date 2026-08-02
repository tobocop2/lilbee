"""T2 HTTP wiki. Empty-store reads against the wiki surface.

The wiki is opt-in and every wiki route is token-gated, so the listing
contract (200 + empty array on a fresh store) is only observable on a
server spawned with ``LILBEE_WIKI=1`` and read with the session token.
The wiki-enabled tests below cover that; the sweep over a default server
covers the rest, where the only guarantee is a status the client can act
on (401 without a token, 404 with the wiki off), never a 5xx.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import httpx
import pytest

from conftest import (
    HTTP_FAST_TIMEOUT,
    Lane,
    auth_headers,
    lilbee_env,
    serve_lilbee_with,
)

_WIKI_READ_PATHS = ("/api/wiki", "/api/wiki/drafts", "/api/wiki/this-page-does-not-exist")


class WikiServer(NamedTuple):
    """A running wiki-enabled server plus the headers its routes require."""

    base_url: str
    headers: dict[str, str]


@pytest.fixture
def wiki_server(lane: Lane, lilbee_data: Path) -> Iterator[WikiServer]:
    """Spawn `lilbee serve` with the wiki enabled and an empty store."""
    env = lilbee_env(lilbee_data, extra={"LILBEE_WIKI": "1"})
    with serve_lilbee_with(lane, env) as base_url:
        yield WikiServer(base_url=base_url, headers=auth_headers(env))


@pytest.mark.http
@pytest.mark.parametrize("path", ["/api/wiki", "/api/wiki/drafts"])
def test_wiki_reads_on_empty_store_return_empty_lists(wiki_server: WikiServer, path: str) -> None:
    """An enabled wiki with no generated pages lists nothing, and says so with a 200."""
    response = httpx.get(
        f"{wiki_server.base_url}{path}",
        timeout=HTTP_FAST_TIMEOUT,
        headers=wiki_server.headers,
    )
    assert response.status_code == httpx.codes.OK, response.text
    assert response.json() == []


@pytest.mark.http
def test_unknown_wiki_slug_returns_404(wiki_server: WikiServer) -> None:
    response = httpx.get(
        f"{wiki_server.base_url}/api/wiki/this-page-does-not-exist",
        timeout=HTTP_FAST_TIMEOUT,
        headers=wiki_server.headers,
    )
    assert response.status_code == httpx.codes.NOT_FOUND


@pytest.mark.http
@pytest.mark.parametrize("path", _WIKI_READ_PATHS)
def test_wiki_reads_without_a_token_never_return_5xx(server_url: str, path: str) -> None:
    """A default server refuses (401) or reports the wiki off (404). Either is a
    status the client can act on; a fault in the wiki layer is not."""
    response = httpx.get(f"{server_url}{path}", timeout=HTTP_FAST_TIMEOUT)
    assert response.status_code < httpx.codes.INTERNAL_SERVER_ERROR, response.text
