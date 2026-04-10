"""Wiki clusterer integration tests with real embeddings and LanceDB.

Exercises the full chunk-level mutual kNN + Label Propagation + TF-IDF
pipeline against the real ``wiki_pipeline`` fixture corpus (3 auth docs
+ 2 perf docs + 4 singletons). The LLM is mocked so the tests are
fast, but everything between the store and the cluster output is real.

Run with:
    uv run pytest tests/integration/test_clusterer_integration.py -v -m slow
"""

from __future__ import annotations

import itertools
from unittest.mock import patch

import pytest

llama_cpp = pytest.importorskip("llama_cpp")

from lilbee.clustering import Clusterer  # noqa: E402
from lilbee.clustering_embedding import EmbeddingClusterer  # noqa: E402
from lilbee.config import ClustererBackend, cfg  # noqa: E402
from lilbee.services import get_services  # noqa: E402

pytestmark = pytest.mark.slow


_AUTH_SOURCES = frozenset({"auth-part1.md", "auth-part2.md", "auth-part3.md"})
_PERF_SOURCES = frozenset({"api-perf.md", "db-perf.md"})


class TestEmbeddingClustererRealBackend:
    """End-to-end clustering over a real LanceDB with real embeddings."""

    def test_available_on_populated_store(self, wiki_pipeline):
        clusterer = EmbeddingClusterer(cfg, get_services().store)
        assert clusterer.available() is True

    def test_auth_docs_cluster_together(self, wiki_pipeline):
        """The three auth-partN.md docs must land in one shared cluster.

        Each auth doc talks about OAuth / JWT / sessions so their
        embedding means sit close together in the model's semantic
        space. Mutual-kNN + label propagation should surface that.
        """
        clusterer = EmbeddingClusterer(cfg, get_services().store)
        clusters = clusterer.get_clusters(min_sources=3)

        # Find the cluster that contains all three auth docs.
        auth_cluster = next(
            (c for c in clusters if _AUTH_SOURCES.issubset(c.sources)),
            None,
        )
        assert auth_cluster is not None, (
            f"Expected a cluster containing all auth docs. Got: "
            f"{[sorted(c.sources) for c in clusters]}"
        )
        assert auth_cluster.label, "Cluster label should be non-empty"
        assert auth_cluster.cluster_id.startswith("embedding-")

    def test_perf_docs_cluster_together(self, wiki_pipeline):
        """api-perf.md + db-perf.md share performance vocabulary."""
        clusterer = EmbeddingClusterer(cfg, get_services().store)
        clusters = clusterer.get_clusters(min_sources=2)

        perf_cluster = next(
            (c for c in clusters if _PERF_SOURCES.issubset(c.sources)),
            None,
        )
        assert perf_cluster is not None, (
            f"Expected a cluster containing both perf docs. Got: "
            f"{[sorted(c.sources) for c in clusters]}"
        )

    def test_deterministic_across_runs(self, wiki_pipeline):
        """Running get_clusters twice on the same store returns identical output."""
        clusterer = EmbeddingClusterer(cfg, get_services().store)
        run_a = clusterer.get_clusters(min_sources=2)
        run_b = clusterer.get_clusters(min_sources=2)
        assert [(c.sources, c.label) for c in run_a] == [(c.sources, c.label) for c in run_b]


class TestClustererFacadeRealBackend:
    def test_default_facade_selects_embedding_backend(
        self, wiki_pipeline, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(cfg, "wiki_clusterer", ClustererBackend.EMBEDDING)
        clusterer = Clusterer(cfg, get_services().store)
        assert isinstance(clusterer.backend, EmbeddingClusterer)
        assert clusterer.available()

    def test_facade_rebuilds_backend_when_config_changes(
        self, wiki_pipeline, monkeypatch: pytest.MonkeyPatch
    ):
        """Flipping cfg.wiki_clusterer at runtime must take effect without
        reconstructing the Services container. Uses monkeypatch.setattr on
        cfg directly so the change is reverted after the test and does not
        leak into other session-scoped integration tests.
        """
        from lilbee.concepts import ConceptGraphClusterer

        monkeypatch.setattr(cfg, "wiki_clusterer", ClustererBackend.EMBEDDING)
        clusterer = Clusterer(cfg, get_services().store)
        first = clusterer.backend
        assert isinstance(first, EmbeddingClusterer)

        # Force the adapter to report available so the facade picks it,
        # then flip the config knob. Both changes auto-revert at
        # function teardown.
        monkeypatch.setattr(ConceptGraphClusterer, "available", lambda self: True)
        monkeypatch.setattr(cfg, "wiki_clusterer", ClustererBackend.CONCEPTS)

        second = clusterer.backend
        assert isinstance(second, ConceptGraphClusterer)
        assert second is not first


class TestSynthesisPagesEndToEnd:
    def test_generate_synthesis_pages_with_real_clusterer(self, wiki_pipeline):
        """Real clusterer + mocked LLM → synthesis pages written to disk."""
        from lilbee.wiki.gen import generate_synthesis_pages

        svc = get_services()
        wiki_text = (
            "# Authentication Architecture\n\n"
            "OAuth, JWT, and session management are covered across the auth docs.[^src1]\n\n"
            "---\n<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            '[^src1]: auth-part1.md, excerpt: "Configure OAuth 2.0 with client ID and secret"\n'
        )
        faithfulness = "0.85"

        # Only the LLM is mocked — embedder, store, and clusterer all run
        # against the real pipeline. ``itertools.cycle`` means the test
        # keeps working if the fixture corpus ever produces more than one
        # qualifying cluster: each cluster consumes a (wiki_text,
        # faithfulness) pair, and the cycle refills.
        mock_responses = itertools.cycle([wiki_text, faithfulness])
        with patch.object(svc.provider, "chat", side_effect=lambda *a, **kw: next(mock_responses)):
            pages = generate_synthesis_pages(svc.provider, svc.store, svc.clusterer)

        assert pages, "Expected at least one synthesis page to be generated"
        for path in pages:
            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert "sources:" in content  # frontmatter present
            assert "[^src" in content  # at least one citation anchor
