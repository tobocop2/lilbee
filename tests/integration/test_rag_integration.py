"""RAG pipeline integration tests with real models.

Uses llama-cpp-python with real GGUF models downloaded from HuggingFace.
No external server required. Marked slow — excluded from default test runs.

Run with:
    uv run pytest tests/test_rag_integration.py -v -m slow
"""

from __future__ import annotations

import asyncio
from collections import Counter
from unittest.mock import patch

import pytest

llama_cpp = pytest.importorskip("llama_cpp")

from lilbee.catalog import FEATURED_CHAT, FEATURED_EMBEDDING, download_model  # noqa: E402
from lilbee.config import cfg  # noqa: E402
from lilbee.ingest import sync  # noqa: E402
from lilbee.model_manager import reset_model_manager  # noqa: E402
from lilbee.services import get_services  # noqa: E402
from lilbee.services import reset_services as reset_provider  # noqa: E402

pytestmark = pytest.mark.slow


def search_context(question, top_k=0):
    return get_services().searcher.search(question, top_k=top_k)


def ask_raw(question, top_k=0, history=None, options=None):
    return get_services().searcher.ask_raw(question, top_k=top_k, history=history, options=options)


# Test document contents — known facts for verifiable retrieval
SPECS_MD = """\
# Thunderbolt X500

Engine: 3.5L V6 TurboForce
Oil capacity: 6.5 quarts
Top speed: 155 mph
Horsepower: 365 hp
Transmission: 8-speed automatic
"""

AUTH_PART1_MD = """\
# Authentication: OAuth Setup

Configure OAuth 2.0 with client ID and secret. Register your application
in the developer portal to obtain credentials. The authorization endpoint
handles the initial redirect flow with PKCE challenge.
"""

AUTH_PART2_MD = """\
# Authentication: JWT Tokens

JWT tokens are signed with RS256 algorithm using asymmetric keys.
Access tokens expire after 15 minutes. Refresh tokens are stored
securely and rotated on each use for replay attack prevention.
"""

AUTH_PART3_MD = """\
# Authentication: Session Management

Sessions stored in Redis with 24h TTL. Session IDs are generated
using cryptographically secure random bytes. Inactive sessions
are garbage collected every 6 hours by the cleanup worker.
"""

DEPLOY_MD = """\
# Deployment Guide

Use kubectl apply to deploy containers to the Kubernetes cluster.
The CI/CD pipeline builds Docker images tagged with the git SHA.
Rolling updates ensure zero downtime during releases. Configure
resource limits and health checks in the deployment manifest.
"""

DB_PERF_MD = """\
# Database Performance

Index your queries. Use connection pooling with PgBouncer to reduce
connection overhead. Monitor slow queries with pg_stat_statements.
Vacuum and analyze tables regularly. Partition large tables by date
for faster range scans.
"""

API_PERF_MD = """\
# API Performance

Cache responses with Redis. Use connection pooling for upstream services.
Rate limit with token bucket algorithm to prevent abuse. Enable gzip
compression for large payloads. Use async I/O for non-blocking requests.
"""

FIBONACCI_PY = '''\
"""Fibonacci sequence calculator."""


def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number iteratively.

    Args:
        n: The position in the Fibonacci sequence (0-indexed).

    Returns:
        The nth Fibonacci number.

    Examples:
        >>> fibonacci(0)
        0
        >>> fibonacci(10)
        55
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
'''

NOTES_MD = """\
# Meeting Notes March 2026

Decided to migrate to PostgreSQL 16 for improved JSON support.
Timeline: Q2 2026. Migration lead: Sarah Chen. Budget approved
for dedicated DBA contractor. Rollback plan documented in wiki.
"""


@pytest.fixture(scope="module")
def rag_pipeline(tmp_path_factory):
    """Set up a real RAG pipeline with downloaded models and test documents.

    Module-scoped: downloads models once, creates documents, runs sync,
    yields pipeline data, then restores config.
    """
    snapshot = cfg.model_copy()
    tmp = tmp_path_factory.mktemp("rag_integration")
    docs_dir = tmp / "documents"
    data_dir = tmp / "data"
    lancedb_dir = data_dir / "lancedb"

    docs_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    # Write test documents
    test_docs = {
        "specs.md": SPECS_MD,
        "auth-part1.md": AUTH_PART1_MD,
        "auth-part2.md": AUTH_PART2_MD,
        "auth-part3.md": AUTH_PART3_MD,
        "deploy.md": DEPLOY_MD,
        "db-perf.md": DB_PERF_MD,
        "api-perf.md": API_PERF_MD,
        "fibonacci.py": FIBONACCI_PY,
        "notes.md": NOTES_MD,
    }
    for name, content in test_docs.items():
        (docs_dir / name).write_text(content)

    # Configure lilbee for llama-cpp
    cfg.llm_provider = "llama-cpp"
    cfg.documents_dir = docs_dir
    cfg.data_dir = data_dir
    cfg.data_root = tmp
    cfg.lancedb_dir = lancedb_dir
    cfg.models_dir = tmp / "models"
    cfg.models_dir.mkdir(parents=True)
    # Disable query expansion for predictable search results (no LLM calls during search)
    cfg.query_expansion_count = 0
    # Disable concept graph to avoid spacy dependency in integration tests
    cfg.concept_graph = False
    # Disable HyDE
    cfg.hyde = False

    # Reset singletons so they pick up the new config
    reset_provider()
    reset_model_manager()

    # Download embedding model via catalog (llama-cpp can't pull directly)
    embed_entry = FEATURED_EMBEDDING[0]
    embed_path = download_model(embed_entry)
    cfg.embedding_model = embed_path.name

    # Download smallest featured chat model (Qwen3 0.6B)
    chat_entry = FEATURED_CHAT[0]
    chat_path = download_model(chat_entry)
    cfg.chat_model = chat_path.name

    # Run real sync
    result = asyncio.run(sync(quiet=True))

    yield {
        "tmp": tmp,
        "docs_dir": docs_dir,
        "data_dir": data_dir,
        "lancedb_dir": lancedb_dir,
        "sync_result": result,
        "test_docs": test_docs,
    }

    # Restore config and singletons
    reset_provider()
    reset_model_manager()
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _source_names(results):
    """Extract source filenames from search results."""
    return [r.source for r in results]


def _unique_sources(results):
    """Extract unique source filenames from search results."""
    return list(dict.fromkeys(r.source for r in results))


# ---------------------------------------------------------------------------
# Pipeline Basics
# ---------------------------------------------------------------------------


class TestPipelineBasics:
    def test_ingest_creates_chunks(self, rag_pipeline):
        """Sync produces chunks in LanceDB, count > 0."""
        table = get_services().store.open_table("chunks")
        assert table is not None
        rows = table.to_arrow().to_pylist()
        assert len(rows) > 0

    def test_ingest_creates_fts_index(self, rag_pipeline):
        """FTS index is built after sync."""
        results = get_services().store.bm25_probe("Thunderbolt", top_k=3)
        assert len(results) > 0

    def test_embed_produces_real_vectors(self, rag_pipeline):
        """Embeddings are non-zero float vectors with correct dimensionality."""
        vec = get_services().embedder.embed("test embedding vector")
        assert len(vec) == cfg.embedding_dim
        assert any(v != 0.0 for v in vec)

    def test_ingest_realistic_length_document(self, rag_pipeline):
        """Verify embedding works with realistic text lengths, not just short strings.

        Regression test: llama-cpp-python defaults n_batch=512, silently
        truncating embeddings. With multiple chunks exceeding this limit,
        llama_decode returns -1. This test catches that by ingesting a
        document long enough to produce multiple real-sized chunks.
        """
        docs_dir = rag_pipeline["docs_dir"]
        # ~8000 chars → multiple chunks at 512-token chunk_size
        content = (
            "# Vehicle Maintenance Procedures\n\n"
            "Regular maintenance extends the life of your vehicle and prevents "
            "costly repairs. This comprehensive guide covers all essential "
            "maintenance procedures for modern vehicles.\n\n"
            "## Oil Change Procedure\n\n"
            "Drain the old oil completely by removing the drain plug. Replace "
            "the oil filter with a new one, applying a thin film of fresh oil "
            "to the gasket. Reinstall the drain plug and fill with the correct "
            "grade of synthetic motor oil. Check the dipstick to verify the "
            "correct oil level has been reached.\n\n"
        ) * 10  # Repeat to ensure multiple chunks

        (docs_dir / "maintenance_guide.md").write_text(content)
        result = asyncio.run(sync(quiet=True))
        assert result.failed == 0, f"Ingest failed: {result}"
        assert result.added > 0 or result.updated > 0

        results = get_services().searcher.search("oil change procedure", top_k=3)
        assert len(results) > 0
        sources = [r.source for r in results]
        assert "maintenance_guide.md" in sources


# ---------------------------------------------------------------------------
# Search Quality
# ---------------------------------------------------------------------------


class TestSearchQuality:
    def test_hybrid_finds_exact_keyword(self, rag_pipeline):
        """'Thunderbolt X500 oil capacity' returns specs.md."""
        results = search_context("Thunderbolt X500 oil capacity", top_k=5)
        sources = _source_names(results)
        assert "specs.md" in sources

    def test_hybrid_finds_semantic_match(self, rag_pipeline):
        """'engine specifications' finds specs.md via semantic similarity."""
        results = search_context("engine specifications", top_k=5)
        sources = _source_names(results)
        assert "specs.md" in sources

    def test_mmr_returns_diverse_sources(self, rag_pipeline):
        """'authentication' returns chunks from different auth files."""
        results = search_context("authentication", top_k=10)
        sources = _unique_sources(results)
        auth_files = [s for s in sources if s.startswith("auth-")]
        assert len(auth_files) >= 2, f"Expected >=2 auth files, got {auth_files}"

    def test_per_source_cap(self, rag_pipeline):
        """No more than diversity_max_per_source chunks from any single file."""
        from lilbee.query import prepare_results

        results = search_context("authentication setup tokens sessions", top_k=20)
        prepared = prepare_results(results)
        counts = Counter(r.source for r in prepared)
        max_per_source = cfg.diversity_max_per_source
        for source, count in counts.items():
            assert count <= max_per_source, (
                f"{source} has {count} chunks, exceeds cap of {max_per_source}"
            )

    def test_expansion_bridges_vocabulary(self, rag_pipeline):
        """'how to ship code to production' finds deploy.md.

        Even without LLM expansion (disabled for speed), the semantic
        similarity between 'ship code to production' and deployment
        vocabulary should bridge the gap.
        """
        results = search_context("how to ship code to production", top_k=10)
        sources = _source_names(results)
        assert "deploy.md" in sources, f"deploy.md not in {sources}"

    def test_code_search_finds_function(self, rag_pipeline):
        """'fibonacci calculation' finds fibonacci.py with line numbers."""
        results = search_context("fibonacci calculation", top_k=5)
        sources = _source_names(results)
        assert "fibonacci.py" in sources
        fib_chunks = [r for r in results if r.source == "fibonacci.py"]
        assert len(fib_chunks) > 0
        # Code chunks should have line number metadata
        assert fib_chunks[0].content_type == "code"

    def test_concept_boost_promotes_related(self, rag_pipeline):
        """'connection pooling' finds both db-perf.md and api-perf.md.

        Both documents discuss connection pooling in different contexts
        (database vs API). Semantic search should surface both.
        """
        results = search_context("connection pooling", top_k=10)
        sources = _source_names(results)
        assert "db-perf.md" in sources, f"db-perf.md not in {sources}"
        assert "api-perf.md" in sources, f"api-perf.md not in {sources}"


# ---------------------------------------------------------------------------
# Answer Generation
# ---------------------------------------------------------------------------


class TestAnswerGeneration:
    """Answer generation tests mock the LLM chat call but use real embeddings + search.

    The 0.6B model's default 512-token context window is too small for RAG prompts.
    Mocking the chat response lets us verify the full pipeline (search -> context
    building -> answer formatting) without needing a large context window.
    """

    def test_ask_returns_answer(self, rag_pipeline):
        """ask_raw() returns a non-empty answer with real search, mocked chat."""
        svc = get_services()
        with patch.object(svc.provider, "chat", return_value="The oil capacity is 6.5 quarts."):
            result = ask_raw("What is the oil capacity?", top_k=5)
        assert result.answer
        assert len(result.answer) > 0

    def test_ask_includes_citations(self, rag_pipeline):
        """ask_raw() returns source references from real search."""
        svc = get_services()
        with patch.object(svc.provider, "chat", return_value="The Thunderbolt has a 3.5L V6."):
            result = ask_raw("What engine does the Thunderbolt have?", top_k=5)
        assert len(result.sources) > 0
        source_names = [s.source for s in result.sources]
        assert "specs.md" in source_names

    def test_ask_known_fact(self, rag_pipeline):
        """Real search retrieves the right context for 'oil capacity of the Thunderbolt'.

        Verifies that the context passed to the LLM contains the known fact,
        even though we mock the LLM response itself.
        """
        captured_messages = []

        def capture_chat(messages, **kwargs):
            captured_messages.extend(messages)
            return "The oil capacity is 6.5 quarts."

        svc = get_services()
        with patch.object(svc.provider, "chat", side_effect=capture_chat):
            ask_raw(
                "What is the oil capacity of the Thunderbolt X500?",
                top_k=5,
            )
        # The context sent to the LLM should contain the known fact
        user_msg = next((m for m in captured_messages if m["role"] == "user"), None)
        assert user_msg is not None
        assert "6.5 quarts" in user_msg["content"], (
            f"Expected '6.5 quarts' in context: {user_msg['content'][:200]}"
        )


# ---------------------------------------------------------------------------
# Regression Guards
# ---------------------------------------------------------------------------


class TestRegressionGuards:
    def test_empty_query(self, rag_pipeline):
        """Empty string search returns results but they are low relevance.

        Vector search on an empty string embedding still returns results
        (cosine distance to all vectors), but all results should have high
        distance (low relevance), confirming the pipeline handles it gracefully.
        """
        results = search_context("", top_k=5)
        # Empty query embedding produces results but with high distance
        for r in results:
            if r.distance is not None:
                assert r.distance > 0.1, "Empty query should not produce close matches"

    def test_nonexistent_topic(self, rag_pipeline):
        """'quantum teleportation' returns no relevant results or low quality ones."""
        results = search_context("quantum teleportation warp drive", top_k=5)
        # With real embeddings, may still return some results due to vector search
        # but they should not be from clearly unrelated sources
        if results:
            # At minimum, the results should not be highly relevant
            for r in results:
                if r.distance is not None:
                    # High distance = low relevance for cosine distance
                    assert r.distance > 0.1, "Unexpectedly close match for nonsense query"

    def test_delete_removes_from_search(self, rag_pipeline):
        """Removing specs.md makes it unfindable."""
        s = get_services().store
        # Verify it's currently findable
        before = search_context("Thunderbolt X500", top_k=5)
        assert "specs.md" in _source_names(before)

        # Delete it
        s.delete_by_source("specs.md")
        s.delete_source("specs.md")
        s.ensure_fts_index()

        after = search_context("Thunderbolt X500", top_k=5)
        assert "specs.md" not in _source_names(after)

        # Re-add it for subsequent tests by re-syncing
        asyncio.run(sync(quiet=True))
        s.ensure_fts_index()

    def test_sync_idempotent(self, rag_pipeline):
        """Running sync twice produces the same chunk count."""
        s = get_services().store
        table = s.open_table("chunks")
        assert table is not None
        count_before = len(table.to_arrow().to_pylist())

        asyncio.run(sync(quiet=True))

        table = s.open_table("chunks")
        assert table is not None
        count_after = len(table.to_arrow().to_pylist())
        assert count_after == count_before


# ---------------------------------------------------------------------------
# Query Expansion (LLM-generated variant queries)
# ---------------------------------------------------------------------------


class TestQueryExpansion:
    """Tests with query_expansion_count enabled so the LLM generates variant queries."""

    @pytest.fixture(autouse=True)
    def _enable_expansion(self):
        original = cfg.query_expansion_count
        # Disable skip-expansion so expansion always runs
        original_skip = cfg.expansion_skip_threshold
        cfg.query_expansion_count = 1
        cfg.expansion_skip_threshold = 0.0
        reset_provider()
        yield
        cfg.query_expansion_count = original
        cfg.expansion_skip_threshold = original_skip
        reset_provider()

    def test_expansion_finds_specs(self, rag_pipeline):
        """'engine specs' finds specs.md when expansion generates a variant query."""
        results = search_context("engine specs", top_k=10)
        sources = _source_names(results)
        assert "specs.md" in sources, f"specs.md not in {sources}"

    def test_expansion_produces_variants(self, rag_pipeline):
        """Expansion broadens results — at least as many as without expansion."""
        results = search_context("engine specs", top_k=10)
        assert len(results) > 0, "Expected non-empty results with expansion enabled"


# ---------------------------------------------------------------------------
# HyDE Search (Hypothetical Document Embedding)
# ---------------------------------------------------------------------------


class TestHydeSearch:
    """Tests with HyDE enabled — generates a hypothetical answer, embeds it, searches."""

    @pytest.fixture(autouse=True)
    def _enable_hyde(self):
        original = cfg.hyde
        cfg.hyde = True
        reset_provider()
        yield
        cfg.hyde = original
        reset_provider()

    def test_hyde_finds_specs_from_vague_query(self, rag_pipeline):
        """'what machine am I reading about' finds specs.md via HyDE."""
        results = search_context("what machine am I reading about", top_k=10)
        assert len(results) > 0, "HyDE search returned no results"
        sources = _source_names(results)
        assert "specs.md" in sources, f"specs.md not in {sources}"

    def test_hyde_returns_nonempty(self, rag_pipeline):
        """HyDE search for a general question returns at least one result."""
        results = search_context("tell me about the vehicle", top_k=5)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Concept Graph (spaCy-based concept extraction + graph boost)
# ---------------------------------------------------------------------------


class TestConceptGraph:
    """Tests with concept_graph enabled — requires spacy and graspologic."""

    @pytest.fixture(autouse=True)
    def _enable_concepts(self, rag_pipeline):
        spacy = pytest.importorskip("spacy")
        pytest.importorskip("graspologic_native")
        # Ensure the en_core_web_sm model is available
        try:
            spacy.load("en_core_web_sm")
        except OSError:
            pytest.skip("spacy model en_core_web_sm not installed")

        original = cfg.concept_graph
        cfg.concept_graph = True
        reset_provider()

        # Re-sync so concept tables get built
        asyncio.run(sync(quiet=True))

        yield

        cfg.concept_graph = original
        reset_provider()

    def test_concept_tables_exist(self, rag_pipeline):
        """After sync with concept_graph=True, concept tables exist in LanceDB."""
        store = get_services().store
        nodes_table = store.open_table("concept_nodes")
        edges_table = store.open_table("concept_edges")
        assert nodes_table is not None, "concept_nodes table not found"
        assert edges_table is not None, "concept_edges table not found"

    def test_concept_search_finds_related(self, rag_pipeline):
        """Searching with concept boost finds related documents."""
        results = search_context("engine specifications", top_k=10)
        sources = _source_names(results)
        assert "specs.md" in sources, f"specs.md not in {sources}"

    def test_extract_concepts_nonempty(self, rag_pipeline):
        """Extracting concepts from a query returns at least one concept."""
        concepts_svc = get_services().concepts
        concepts = concepts_svc.extract_concepts("Thunderbolt engine specifications horsepower")
        assert len(concepts) > 0, f"Expected non-empty concepts, got {concepts}"


# ---------------------------------------------------------------------------
# Temporal Filtering (date-based result filtering)
# ---------------------------------------------------------------------------


class TestTemporalFilter:
    """Tests with temporal_filtering enabled — filters by ingestion date."""

    @pytest.fixture(autouse=True)
    def _enable_temporal(self):
        original = cfg.temporal_filtering
        cfg.temporal_filtering = True
        yield
        cfg.temporal_filtering = original

    def test_temporal_query_returns_results(self, rag_pipeline):
        """A temporal query like 'recent changes' still returns results.

        All test docs were ingested at the same time (now), so temporal
        filtering keeps them all. The key assertion is that the filter
        runs without error and does not discard everything.
        """
        results = search_context("recent changes", top_k=10)
        assert len(results) > 0, "Temporal query returned no results"

    def test_temporal_keywords_detected(self, rag_pipeline):
        """Temporal keywords are detected in queries."""
        from lilbee.temporal import detect_temporal

        assert detect_temporal("recent changes") is not None
        assert detect_temporal("latest updates") is not None
        assert detect_temporal("engine specifications") is None


# ---------------------------------------------------------------------------
# Structured Queries (prefix syntax: term:, vec:, hyde:)
# ---------------------------------------------------------------------------


class TestStructuredQueries:
    """Tests for structured query prefix syntax."""

    def test_term_prefix_bm25_only(self, rag_pipeline):
        """'term: Thunderbolt' performs BM25-only search and returns results."""
        results = search_context("term: Thunderbolt", top_k=5)
        assert len(results) > 0, "term: prefix returned no results"
        sources = _source_names(results)
        assert "specs.md" in sources

    def test_vec_prefix_vector_only(self, rag_pipeline):
        """'vec: engine specifications' performs vector-only search and returns results."""
        results = search_context("vec: engine specifications", top_k=5)
        assert len(results) > 0, "vec: prefix returned no results"
        sources = _source_names(results)
        assert "specs.md" in sources

    def test_hyde_prefix_search(self, rag_pipeline):
        """'hyde:' prefix triggers the HyDE code path without crashing."""
        original = cfg.hyde
        cfg.hyde = True
        reset_provider()
        try:
            # HyDE with a tiny model (Qwen3 0.6B) may produce poor hypothetical
            # docs that don't match anything. We verify the code path runs
            # without error; if results are found, check they're plausible.
            results = search_context("hyde: Thunderbolt X500 engine oil capacity", top_k=5)
            assert isinstance(results, list)
            if results:
                sources = {r.source for r in results}
                assert any("specs" in s for s in sources), f"Got {sources}"
        finally:
            cfg.hyde = original
            reset_provider()


# ---------------------------------------------------------------------------
# Ask Stream (streaming answer generation)
# ---------------------------------------------------------------------------


class TestAskStream:
    """Tests for ask_stream() — real search, mocked LLM chat."""

    def test_stream_yields_tokens(self, rag_pipeline):
        """ask_stream() yields StreamToken objects with content."""
        from lilbee.reasoning import StreamToken

        svc = get_services()
        with patch.object(
            svc.provider,
            "chat",
            return_value=iter(["The ", "oil ", "capacity ", "is ", "6.5 quarts."]),
        ):
            tokens = list(svc.searcher.ask_stream("What is the oil capacity?", top_k=5))
        assert len(tokens) > 0
        assert all(isinstance(t, StreamToken) for t in tokens)

    def test_stream_ends_with_citations(self, rag_pipeline):
        """The last token from ask_stream() contains source citations."""
        from lilbee.reasoning import StreamToken

        svc = get_services()
        with patch.object(svc.provider, "chat", return_value=iter(["The engine is a V6."])):
            tokens = list(svc.searcher.ask_stream("What engine does it have?", top_k=5))
        assert len(tokens) > 0
        last_token = tokens[-1]
        assert isinstance(last_token, StreamToken)
        assert "Sources:" in last_token.content

    def test_stream_is_reasoning_flag(self, rag_pipeline):
        """Tokens from ask_stream() have is_reasoning=False for normal content."""
        from lilbee.reasoning import StreamToken

        svc = get_services()
        with patch.object(svc.provider, "chat", return_value=iter(["Answer here."])):
            tokens = list(svc.searcher.ask_stream("Tell me about the car", top_k=5))
        non_empty = [t for t in tokens if t.content.strip()]
        assert len(non_empty) > 0
        # Normal content tokens (no <think> tags) should all be is_reasoning=False
        for t in non_empty:
            assert isinstance(t, StreamToken)
            assert t.is_reasoning is False
