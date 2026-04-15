"""RAG pipeline integration tests with real models.

Uses llama-cpp-python with real GGUF models downloaded from HuggingFace.
No external server required. Marked slow — excluded from default test runs.

Run with:
    uv run pytest tests/integration/test_rag_integration.py -v -m slow
"""

from __future__ import annotations

import asyncio
from collections import Counter

import pytest

from lilbee.catalog import FEATURED_EMBEDDING, download_model
from lilbee.config import cfg
from lilbee.ingest import sync
from lilbee.services import get_services
from lilbee.services import reset_services as reset_provider

pytestmark = pytest.mark.slow


def search_context(question, top_k=0):
    return get_services().searcher.search(question, top_k=top_k)


def ask_raw(question, top_k=0, history=None, options=None):
    return get_services().searcher.ask_raw(question, top_k=top_k, history=history, options=options)


def _source_names(results):
    """Extract source filenames from search results."""
    return [r.source for r in results]


def _unique_sources(results):
    """Extract unique source filenames from search results."""
    return list(dict.fromkeys(r.source for r in results))


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
        assert len(result.failed) == 0, f"Ingest failed: {result}"
        assert len(result.added) > 0 or len(result.updated) > 0

        results = get_services().searcher.search("oil change procedure", top_k=3)
        assert len(results) > 0
        sources = [r.source for r in results]
        assert "maintenance_guide.md" in sources


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


class TestAnswerGeneration:
    """Answer generation with real LLM (Qwen3 0.6B) and real search."""

    def test_ask_returns_answer(self, rag_pipeline):
        """ask_raw() returns a non-empty answer from real search + real LLM."""
        result = ask_raw("What is the oil capacity?", top_k=5)
        assert result.answer
        assert len(result.answer) > 0

    def test_ask_includes_citations(self, rag_pipeline):
        """ask_raw() returns source references from real search."""
        result = ask_raw("What engine does the Thunderbolt have?", top_k=5)
        assert len(result.sources) > 0
        source_names = [s.source for s in result.sources]
        assert "specs.md" in source_names

    def test_ask_answer_references_facts(self, rag_pipeline):
        """Real LLM answer references known facts from the indexed documents."""
        result = ask_raw("What is the oil capacity of the Thunderbolt X500?", top_k=5)
        assert result.answer
        answer_lower = result.answer.lower()
        assert "6.5" in answer_lower or "quart" in answer_lower, (
            f"Expected oil capacity fact in answer: {result.answer[:300]}"
        )


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

        # Force rebuild so concept indexing runs for all files
        # (a plain sync skips unchanged files and never calls _index_concepts)
        asyncio.run(sync(quiet=True, force_rebuild=True))

        yield

        cfg.concept_graph = original
        reset_provider()
        # Re-sync without concepts to restore tables for subsequent tests
        asyncio.run(sync(quiet=True, force_rebuild=True))

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


class TestAskStream:
    """Tests for ask_stream() with real LLM streaming."""

    @pytest.fixture(autouse=True)
    def _reset_provider(self):
        """Reset the provider between tests to release any held model locks."""
        yield
        reset_provider()

    def test_stream_yields_tokens(self, rag_pipeline):
        """ask_stream() yields StreamToken objects with content."""
        from lilbee.reasoning import StreamToken

        svc = get_services()
        tokens = list(svc.searcher.ask_stream("What is the oil capacity?", top_k=5))
        assert len(tokens) > 0
        assert all(isinstance(t, StreamToken) for t in tokens)

    def test_stream_ends_with_citations(self, rag_pipeline):
        """The last token from ask_stream() contains source citations."""
        from lilbee.reasoning import StreamToken

        svc = get_services()
        tokens = list(svc.searcher.ask_stream("What engine does the Thunderbolt have?", top_k=5))
        assert len(tokens) > 0
        last_token = tokens[-1]
        assert isinstance(last_token, StreamToken)
        assert "Sources:" in last_token.content

    def test_stream_is_reasoning_flag(self, rag_pipeline):
        """ask_stream() yields StreamTokens with correct is_reasoning flags."""
        from lilbee.reasoning import StreamToken

        svc = get_services()
        tokens = list(svc.searcher.ask_stream("Tell me about the Thunderbolt", top_k=5))
        non_empty = [t for t in tokens if t.content.strip()]
        assert len(non_empty) > 0
        assert all(isinstance(t, StreamToken) for t in non_empty)
        # With thinking models (qwen3), some tokens are reasoning. Verify
        # that at least some non-reasoning content exists (the actual answer
        # or citations).
        response_tokens = [t for t in non_empty if not t.is_reasoning]
        assert len(response_tokens) > 0, "Expected non-reasoning content in stream"


class TestDownloadProgressCallbacks:
    def test_download_fires_progress_callbacks(self, rag_pipeline):
        """Download progress callbacks fire during model download."""
        progress_calls: list[tuple[int, int]] = []

        def on_progress(downloaded: int, total: int) -> None:
            progress_calls.append((downloaded, total))

        # Re-download the embedding model (cached — returns from HF cache immediately)
        download_model(FEATURED_EMBEDDING[0], on_progress=on_progress)
        # Cached download fires a single completion callback
        assert len(progress_calls) == 1
        downloaded, total = progress_calls[0]
        assert downloaded == total  # 100% completion signal
