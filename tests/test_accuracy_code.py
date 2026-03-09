"""E2E accuracy tests for code files in the RAG pipeline.

Ingests 5 code files (Python, Go, Rust, Java, TypeScript) with known
function signatures and verifies that search retrieves the correct chunks,
metadata is valid, and source attribution works.

Requires Ollama running with mistral and nomic-embed-text models pulled.
"""

import asyncio
from dataclasses import dataclass

import pytest

from tests.conftest import batch_search, copy_fixtures_to, patched_lilbee_dirs, requires_models

CODE_FILES = ["calculator.py", "stack.go", "matrix.rs", "LinkedList.java", "validator.ts"]

_CODE_TEST_CASES = [
    ("fibonacci sequence calculation", "calculator.py", ["fibonacci"]),
    ("temperature conversion celsius fahrenheit", "calculator.py", ["celsius_to_fahrenheit"]),
    ("push item onto stack", "stack.go", ["Push"]),
    ("matrix transpose operation", "matrix.rs", ["transpose"]),
    ("linked list insert at head", "LinkedList.java", ["insertAtHead"]),
    ("validate email address format", "validator.ts", ["validateEmail"]),
    ("sanitize html input string", "validator.ts", ["sanitizeHtml"]),
]

_ASK_QUERY = "fibonacci sequence calculation"


@dataclass
class CodeTestData:
    """Pre-computed results from the ingested code fixture."""

    tmp: object
    search_results: dict[str, list[dict]]
    ask_result: str


# ---------------------------------------------------------------------------
# Fixture — ingest once, batch-search once, ask once
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def code_data(tmp_path_factory):
    """Ingest code fixtures and pre-compute all Ollama-dependent results."""
    tmp = tmp_path_factory.mktemp("lilbee_code_test")
    db_dir = tmp / "lancedb"
    docs_dir = tmp / "documents" / "code-samples"
    docs_dir.mkdir(parents=True)

    with patched_lilbee_dirs(db_dir, docs_dir.parent):
        copy_fixtures_to("code", docs_dir)

        from lilbee.ingest import sync

        asyncio.run(sync())

        queries = [q for q, _, _ in _CODE_TEST_CASES]
        search_results = batch_search(queries)

        from lilbee.query import ask

        ask_result = ask(_ASK_QUERY)

        yield CodeTestData(tmp=tmp, search_results=search_results, ask_result=ask_result)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@requires_models
class TestCodeAccuracy:
    @pytest.mark.parametrize("query,expected_source,expected_terms", _CODE_TEST_CASES)
    def test_search_finds_correct_code_chunk(
        self, code_data, query, expected_source, expected_terms
    ):
        results = code_data.search_results[query]
        matching = [
            r
            for r in results
            if r["source"].endswith(expected_source)
            and any(term in r["chunk"] for term in expected_terms)
        ]
        assert matching, (
            f"No chunk from '{expected_source}' containing {expected_terms} "
            f"for query '{query}'. Got sources: "
            f"{[r['source'] for r in results]}"
        )

    def test_code_chunks_have_valid_line_metadata(self, code_data):
        from lilbee.store import get_chunks_by_source

        for filename in CODE_FILES:
            source = f"code-samples/{filename}"
            chunks = get_chunks_by_source(source)
            assert chunks, f"No chunks found for {source}"
            for chunk in chunks:
                assert chunk["line_start"] > 0, (
                    f"{source}: line_start must be > 0, got {chunk['line_start']}"
                )
                assert chunk["line_end"] >= chunk["line_start"], (
                    f"{source}: line_end ({chunk['line_end']}) < line_start ({chunk['line_start']})"
                )
                assert chunk["content_type"] == "code", (
                    f"{source}: expected content_type 'code', got '{chunk['content_type']}'"
                )

    def test_code_source_attribution(self, code_data):
        answer = code_data.ask_result
        assert "Sources:" in answer, f"No 'Sources:' section in answer:\n{answer}"
        assert "calculator.py" in answer, f"'calculator.py' not in answer:\n{answer}"
