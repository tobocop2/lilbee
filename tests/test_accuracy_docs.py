"""E2E accuracy tests for non-PDF document formats in the RAG pipeline.

Verifies that Markdown, plain text, CSV, and HTML documents are correctly
ingested, chunked, and retrievable via the search and ask APIs.

Requires Ollama running with mistral and nomic-embed-text models pulled.
"""

import asyncio
from dataclasses import dataclass

import pytest

from tests.conftest import batch_search, copy_fixtures_to, patched_lilbee_dirs, requires_models

_DOC_TEST_CASES = [
    ("Szechuan peppercorn tofu soy sauce", "recipes.md", ["soy sauce"], "text"),
    ("Proxima Centauri orbital period", "astronomy.txt", ["11.2"], "text"),
    ("part number ZX-7842 warehouse", "inventory.csv", ["ZX-7842"], "data"),
    ("Kraken espresso brew temperature", "manual.html", ["93"], "text"),
]

_ASK_QUERY = "Szechuan peppercorn tofu"


@dataclass
class DocTestData:
    """Pre-computed results from the ingested docs fixture."""

    tmp: object
    search_results: dict[str, list[dict]]
    ask_result: str


# ---------------------------------------------------------------------------
# Fixture — ingest once, batch-search once, ask once
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def doc_data(tmp_path_factory):
    """Ingest doc fixtures and pre-compute all Ollama-dependent results."""
    tmp = tmp_path_factory.mktemp("lilbee_docs_test")
    db_dir = tmp / "lancedb"
    docs_dir = tmp / "documents" / "doc-formats"
    docs_dir.mkdir(parents=True)

    with patched_lilbee_dirs(db_dir, docs_dir.parent):
        copy_fixtures_to("docs", docs_dir)

        from lilbee.ingest import sync

        asyncio.run(sync())

        queries = [q for q, _, _, _ in _DOC_TEST_CASES]
        search_results = batch_search(queries)

        from lilbee.query import ask

        ask_result = ask(_ASK_QUERY)

        yield DocTestData(tmp=tmp, search_results=search_results, ask_result=ask_result)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@requires_models
class TestDocumentAccuracy:
    @pytest.mark.parametrize(
        "query,expected_source,expected_terms,expected_type",
        _DOC_TEST_CASES,
    )
    def test_search_finds_correct_document(
        self, doc_data, query, expected_source, expected_terms, expected_type
    ):
        results = doc_data.search_results[query]
        assert len(results) > 0, f"No results returned for query: {query}"

        matching = [
            r
            for r in results
            if r.source.endswith(expected_source)
            and any(term in r.chunk for term in expected_terms)
        ]
        assert matching, (
            f"No result from '{expected_source}' containing {expected_terms} "
            f"for query '{query}'. Got sources: "
            f"{[r.source for r in results]}"
        )

    @pytest.mark.parametrize(
        "query,expected_source,expected_terms,expected_type",
        _DOC_TEST_CASES,
    )
    def test_content_type_metadata(
        self, doc_data, query, expected_source, expected_terms, expected_type
    ):
        results = doc_data.search_results[query]
        matching = [r for r in results if r.source.endswith(expected_source)]
        assert matching, f"No result from '{expected_source}' for query '{query}'"
        assert matching[0].content_type == expected_type, (
            f"Expected content_type '{expected_type}' for '{expected_source}', "
            f"got '{matching[0].content_type}'"
        )

    def test_document_source_attribution(self, doc_data):
        answer = doc_data.ask_result
        assert "Sources:" in answer, f"No 'Sources:' section in answer:\n{answer}"
        assert "recipes.md" in answer, f"'recipes.md' not cited in answer:\n{answer}"
