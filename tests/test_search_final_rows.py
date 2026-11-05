"""Every search surface returns the same rows, cut by top_k and the search settings.

Each test runs the real ``Searcher.search`` behind the real CLI command, HTTP route
and MCP tool. Only the store and the embedder are stubs.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator

import pytest
from litestar.testing import TestClient
from typer.testing import CliRunner

from lilbee.app.services import set_services
from lilbee.cli import app
from lilbee.core.config import cfg
from lilbee.data.store import SearchChunk
from lilbee.mcp_server import search as mcp_search
from lilbee.server.app import create_app
from lilbee.server.auth import session_manager
from tests.conftest import make_mock_services

Surface = Callable[[str, int | None], list[tuple[str, int]]]

runner = CliRunner()


def _row(source: str, index: int, score: float) -> SearchChunk:
    return SearchChunk(
        source=source,
        content_type="text",
        page_start=0,
        page_end=0,
        line_start=index + 1,
        line_end=index + 1,
        chunk=f"{source} passage {index}",
        chunk_index=index,
        distance=0.1,
        score=score,
        vector=[0.1],
    )


# Twenty rows in two files, best first: the shape a merged search pool takes.
POOL = [_row(f"doc{i % 2}.md", i, 1.0 - i * 0.04) for i in range(20)]
BY_LINE = {(r.source, r.line_start): r.chunk_index for r in POOL}


def _cli(query: str, top_k: int | None) -> list[tuple[str, int]]:
    args = ["--json", "search", query, *(["-k", str(top_k)] if top_k else [])]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    rows = json.loads(result.stdout)["results"]
    return [(r["source"], r["chunk_index"]) for r in rows]


def _http(query: str, top_k: int | None) -> list[tuple[str, int]]:
    params = {"q": query, **({"top_k": str(top_k)} if top_k else {})}
    response = TestClient(create_app()).get("/api/search", params=params)
    assert response.status_code == 200, response.text
    return [
        (document["source"], BY_LINE[(document["source"], excerpt["line_start"])])
        for document in response.json()
        for excerpt in document["excerpts"]
    ]


def _mcp(query: str, top_k: int | None) -> list[tuple[str, int]]:
    rows = mcp_search(query, top_k=top_k)
    assert isinstance(rows, list)
    return [(r["source"], r["chunk_index"]) for r in rows]


SURFACES = {"cli": _cli, "http": _http, "mcp": _mcp}


Settings = Callable[..., None]


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Set values in the live config and in the environment the CLI start reads again."""

    def put(**values: object) -> None:
        for key, value in values.items():
            monkeypatch.setattr(cfg, key, value)
            monkeypatch.setenv(f"LILBEE_{key.upper()}", str(value))

    return put


@pytest.fixture
def pool(settings: Settings) -> Iterator[None]:
    services = make_mock_services()
    services.store.search.return_value = list(POOL)
    services.store.bm25_probe.return_value = []
    services.store.get_sources.return_value = []
    services.embedder.embed_query.return_value = [0.1]
    set_services(services)
    session_manager.disable()
    settings(
        query_expansion_count=0,
        concept_graph=False,
        reranker_model="",
        diversity_max_per_source=100,
        min_relevance_score=0.0,
    )
    yield
    session_manager.cleanup()
    set_services(None)


@pytest.fixture(params=sorted(SURFACES))
def surface(request: pytest.FixtureRequest, pool: None) -> Surface:
    return SURFACES[request.param]


class TestEverySurface:
    def test_top_k_is_a_ceiling(self, surface: Surface) -> None:
        assert len(surface("valve maintenance notes", 5)) == 5

    def test_omitted_top_k_takes_the_configured_value(
        self, surface: Surface, settings: Settings
    ) -> None:
        settings(top_k=7)
        assert len(surface("valve maintenance notes", None)) == 7

    def test_relevance_floor_drops_rows(self, surface: Surface, settings: Settings) -> None:
        settings(min_relevance_score=0.9)
        assert sorted(i for _, i in surface("valve maintenance notes", 12)) == [0, 1, 2]

    def test_diversity_cap_limits_each_source(self, surface: Surface, settings: Settings) -> None:
        settings(diversity_max_per_source=1)
        assert sorted(surface("valve maintenance notes", 12)) == [("doc0.md", 0), ("doc1.md", 1)]

    def test_mode_prefix_takes_the_same_limits(self, surface: Surface, settings: Settings) -> None:
        settings(diversity_max_per_source=1)
        assert sorted(surface("vec: valve maintenance", 12)) == [("doc0.md", 0), ("doc1.md", 1)]


def test_surfaces_agree_on_the_rows(pool: None) -> None:
    answers = {name: sorted(call("valve maintenance notes", 6)) for name, call in SURFACES.items()}
    assert answers["cli"] == answers["http"] == answers["mcp"]
