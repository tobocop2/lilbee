"""Tests for the concept graph module.

All heavy deps (spacy, graspologic-native) are mocked at the boundary so these
tests run without the ``graph`` extra installed.
"""

from dataclasses import fields
from unittest.mock import MagicMock, patch

import pytest

import lilbee.services as svc_mod
from lilbee.concepts import ConceptGraph
from lilbee.config import cfg
from lilbee.store import SearchChunk


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    """Redirect config paths to temp dir for every test."""
    snapshot = {name: getattr(cfg, name) for name in type(cfg).model_fields}
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir()
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.concept_graph = True
    cfg.concept_max_per_chunk = 10
    cfg.concept_boost_weight = 0.3
    yield
    for name, val in snapshot.items():
        setattr(cfg, name, val)


@pytest.fixture(autouse=True)
def mock_svc():
    """Provide a mock Services container for all concept tests."""
    from tests.conftest import make_mock_services

    mock_store = MagicMock()
    mock_store.search.return_value = []
    mock_store.bm25_probe.return_value = []
    mock_store.get_sources.return_value = []
    mock_store.open_table.return_value = None
    concepts = ConceptGraph(cfg, mock_store)
    services = make_mock_services(store=mock_store, concepts=concepts)
    svc_mod.set_services(services)
    yield services
    svc_mod.set_services(None)


@pytest.fixture(autouse=True)
def reset_singletons(mock_svc):
    """Reset ConceptGraph nlp cache between tests."""
    mock_svc.concepts.reset_nlp_cache()
    yield
    mock_svc.concepts.reset_nlp_cache()


@pytest.fixture()
def cg(mock_svc):
    """Return the real ConceptGraph from the mock services."""
    return mock_svc.concepts


def _make_mock_doc(noun_chunks):
    """Create a mock spaCy Doc with the given noun chunks."""
    doc = MagicMock()
    chunks = []
    for text in noun_chunks:
        chunk = MagicMock()
        chunk.text = text
        chunks.append(chunk)
    doc.noun_chunks = chunks
    return doc


def _make_mock_nlp(noun_chunks_per_doc):
    """Create a mock spaCy nlp that returns docs with specified noun chunks."""
    nlp = MagicMock()

    def call_fn(text):
        return _make_mock_doc(noun_chunks_per_doc.get(text, []))

    nlp.side_effect = call_fn

    def pipe_fn(texts):
        return [_make_mock_doc(noun_chunks_per_doc.get(t, [])) for t in texts]

    nlp.pipe = pipe_fn
    return nlp


def _make_result(
    source="test.pdf",
    chunk_index=0,
    chunk="some text",
    distance=0.5,
    relevance_score=None,
) -> SearchChunk:
    return SearchChunk(
        source=source,
        content_type="pdf",
        page_start=1,
        page_end=1,
        line_start=0,
        line_end=0,
        chunk=chunk,
        chunk_index=chunk_index,
        distance=distance,
        relevance_score=relevance_score,
        vector=[0.1],
    )


class TestConceptsAvailable:
    def test_returns_true_when_installed(self):
        mock_spacy = MagicMock()
        mock_graspologic = MagicMock()
        with patch.dict(
            "sys.modules", {"spacy": mock_spacy, "graspologic_native": mock_graspologic}
        ):
            from lilbee.concepts import concepts_available

            assert concepts_available() is True

    def test_returns_false_when_not_installed(self):
        with patch.dict("sys.modules", {"spacy": None}):
            from lilbee.concepts import concepts_available

            assert concepts_available() is False


class TestExtractConcepts:
    @patch("lilbee.concepts._ensure_spacy_model")
    def test_basic_extraction(self, mock_spacy, cg):
        mock_spacy.return_value = _make_mock_nlp(
            {"hello world": ["machine learning", "neural networks"]}
        )
        result = cg.extract_concepts("hello world")
        assert result == ["machine learning", "neural networks"]

    @patch("lilbee.concepts._ensure_spacy_model")
    def test_deduplication(self, mock_spacy, cg):
        mock_spacy.return_value = _make_mock_nlp({"text": ["Concept", "concept", "Other"]})
        result = cg.extract_concepts("text")
        assert result == ["concept", "other"]

    @patch("lilbee.concepts._ensure_spacy_model")
    def test_max_cap(self, mock_spacy, cg):
        mock_spacy.return_value = _make_mock_nlp({"text": ["alpha", "beta", "gamma", "delta"]})
        result = cg.extract_concepts("text", max_concepts=2)
        assert len(result) == 2

    @patch("lilbee.concepts._ensure_spacy_model")
    def test_empty_input(self, mock_spacy, cg):
        result = cg.extract_concepts("")
        assert result == []
        mock_spacy.assert_not_called()

    @patch("lilbee.concepts._ensure_spacy_model")
    def test_filters_short_concepts(self, mock_spacy, cg):
        mock_spacy.return_value = _make_mock_nlp({"text": ["a", "ok", "good concept"]})
        result = cg.extract_concepts("text")
        assert "a" not in result
        assert "ok" in result
        assert "good concept" in result


class TestExtractConceptsBatch:
    @patch("lilbee.concepts._ensure_spacy_model")
    def test_batch_extraction(self, mock_spacy, cg):
        mock_spacy.return_value = _make_mock_nlp(
            {"doc1": ["concept a"], "doc2": ["concept b", "concept c"]}
        )
        result = cg.extract_concepts_batch(["doc1", "doc2"])
        assert len(result) == 2
        assert result[0] == ["concept a"]
        assert result[1] == ["concept b", "concept c"]

    @patch("lilbee.concepts._ensure_spacy_model")
    def test_empty_input(self, mock_spacy, cg):
        result = cg.extract_concepts_batch([])
        assert result == []
        mock_spacy.assert_not_called()

    @patch("lilbee.concepts._ensure_spacy_model")
    def test_batch_filters_short_concepts(self, mock_spacy, cg):
        mock_spacy.return_value = _make_mock_nlp({"text": ["a", "ok", "good"]})
        result = cg.extract_concepts_batch(["text"])
        assert result == [["ok", "good"]]

    @patch("lilbee.concepts._ensure_spacy_model")
    def test_batch_deduplicates(self, mock_spacy, cg):
        mock_spacy.return_value = _make_mock_nlp({"text": ["Alpha", "alpha", "Beta"]})
        result = cg.extract_concepts_batch(["text"])
        assert result == [["alpha", "beta"]]

    @patch("lilbee.concepts._ensure_spacy_model")
    def test_batch_caps_at_max(self, mock_spacy, cg):
        cfg.concept_max_per_chunk = 2
        mock_spacy.return_value = _make_mock_nlp({"text": ["aa", "bb", "cc", "dd"]})
        result = cg.extract_concepts_batch(["text"])
        assert len(result[0]) == 2


class TestGetNlp:
    @patch("lilbee.concepts._ensure_spacy_model")
    def test_caches_nlp_model(self, mock_ensure, cg):
        """ConceptGraph._ensure_nlp caches the spaCy model after first call."""
        mock_ensure.return_value = MagicMock()
        cg.reset_nlp_cache()
        nlp1 = cg._ensure_nlp()
        nlp2 = cg._ensure_nlp()
        mock_ensure.assert_called_once()
        assert nlp1 is nlp2


class TestEnsureSpacyModel:
    def test_loads_existing(self):
        mock_spacy = MagicMock()
        mock_spacy.load.return_value = MagicMock()
        with patch.dict("sys.modules", {"spacy": mock_spacy, "spacy.cli": MagicMock()}):
            from lilbee.concepts import _ensure_spacy_model

            result = _ensure_spacy_model()
            mock_spacy.load.assert_called_once_with("en_core_web_sm")
            assert result is not None

    def test_downloads_on_oserror(self):
        mock_spacy = MagicMock()
        mock_spacy.load.side_effect = [OSError("not found"), MagicMock()]
        mock_cli = MagicMock()
        mock_spacy.cli = mock_cli
        with patch.dict("sys.modules", {"spacy": mock_spacy, "spacy.cli": mock_cli}):
            from lilbee.concepts import _ensure_spacy_model

            result = _ensure_spacy_model()
            mock_cli.download.assert_called_once_with("en_core_web_sm")
            assert result is not None

    def test_raises_import_error_when_download_fails(self):
        mock_spacy = MagicMock()
        mock_spacy.load.side_effect = OSError("not found")
        mock_cli = MagicMock()
        mock_cli.download.side_effect = SystemExit(1)
        mock_spacy.cli = mock_cli
        with patch.dict("sys.modules", {"spacy": mock_spacy, "spacy.cli": mock_cli}):
            from lilbee.concepts import _ensure_spacy_model

            with pytest.raises(ImportError, match="auto-download failed"):
                _ensure_spacy_model()


class TestGracefulDegradation:
    @patch("lilbee.concepts._ensure_spacy_model", side_effect=ImportError("no model"))
    def test_ensure_nlp_returns_none_on_failure(self, mock_spacy, cg):
        assert cg._ensure_nlp() is None

    @patch("lilbee.concepts._ensure_spacy_model", side_effect=ImportError("no model"))
    def test_caches_failure_state(self, mock_spacy, cg):
        cg._ensure_nlp()
        cg._ensure_nlp()
        mock_spacy.assert_called_once()

    @patch("lilbee.concepts._ensure_spacy_model")
    def test_caches_successful_load(self, mock_spacy, cg):
        mock_nlp = MagicMock()
        mock_spacy.return_value = mock_nlp
        assert cg._ensure_nlp() is mock_nlp
        assert cg._ensure_nlp() is mock_nlp
        mock_spacy.assert_called_once()

    @patch("lilbee.concepts._ensure_spacy_model", side_effect=ImportError("no model"))
    def test_extract_concepts_returns_empty(self, mock_spacy, cg):
        assert cg.extract_concepts("some text about python") == []

    @patch("lilbee.concepts._ensure_spacy_model", side_effect=ImportError("no model"))
    def test_extract_concepts_batch_returns_empty_lists(self, mock_spacy, cg):
        result = cg.extract_concepts_batch(["text one", "text two"])
        assert result == [[], []]

    @patch("lilbee.concepts._ensure_spacy_model", side_effect=ImportError("no model"))
    def test_expand_query_returns_empty(self, mock_spacy, cg):
        assert cg.expand_query("python frameworks") == []


class TestBuildFromChunks:
    @patch("lilbee.lock.write_lock")
    def test_build_from_chunks(self, mock_lock, cg):
        mock_lock.return_value.__enter__ = MagicMock()
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)

        chunk_ids = [("doc.md", 0), ("doc.md", 1)]
        concept_lists = [["python", "machine learning"], ["python", "deep learning"]]
        cg.build_from_chunks(chunk_ids, concept_lists)

    def test_build_empty_chunks(self, cg):
        cg.build_from_chunks([], [])


class TestBoostResults:
    def test_boost_results_with_overlap(self, cg, mock_svc):
        results = [_make_result(distance=0.5, chunk_index=0)]
        mock_table = MagicMock()
        mock_table.search.return_value.where.return_value.to_list.return_value = [
            {"concept": "python"},
            {"concept": "ml"},
        ]
        mock_svc.store.open_table.return_value = mock_table
        boosted = cg.boost_results(results, ["python", "java"])
        assert boosted[0].distance < 0.5

    def test_boost_results_relevance_score(self, cg, mock_svc):
        results = [_make_result(distance=None, relevance_score=0.8, chunk_index=0)]
        mock_table = MagicMock()
        mock_table.search.return_value.where.return_value.to_list.return_value = [
            {"concept": "python"},
        ]
        mock_svc.store.open_table.return_value = mock_table
        boosted = cg.boost_results(results, ["python"])
        assert boosted[0].relevance_score > 0.8

    def test_boost_results_no_overlap(self, cg, mock_svc):
        results = [_make_result(distance=0.5, chunk_index=0)]
        mock_table = MagicMock()
        mock_table.search.return_value.where.return_value.to_list.return_value = [
            {"concept": "java"},
        ]
        mock_svc.store.open_table.return_value = mock_table
        boosted = cg.boost_results(results, ["python"])
        assert boosted[0].distance == 0.5

    def test_boost_results_empty_query_concepts(self, cg):
        results = [_make_result()]
        boosted = cg.boost_results(results, [])
        assert boosted == results

    def test_boost_results_empty_results(self, cg):
        boosted = cg.boost_results([], ["python"])
        assert boosted == []


class TestExpandQuery:
    @patch("lilbee.concepts._ensure_spacy_model")
    def test_expand_query(self, mock_spacy, cg, mock_svc):
        mock_spacy.return_value = _make_mock_nlp({"python frameworks": ["python"]})
        mock_table = MagicMock()
        mock_table.search.return_value.where.return_value.to_list.return_value = [
            {"source": "python", "target": "django", "weight": 1.0},
            {"source": "python", "target": "flask", "weight": 0.8},
        ]
        mock_svc.store.open_table.return_value = mock_table
        related = cg.expand_query("python frameworks")
        assert "django" in related
        assert "flask" in related

    @patch("lilbee.concepts._ensure_spacy_model")
    def test_expand_query_no_concepts(self, mock_spacy, cg):
        mock_spacy.return_value = _make_mock_nlp({"???": []})
        assert cg.expand_query("???") == []


class TestGetRelatedConcepts:
    def test_get_related_concepts(self, cg, mock_svc):
        mock_table = MagicMock()
        mock_table.search.return_value.where.return_value.to_list.return_value = [
            {"source": "python", "target": "django", "weight": 1.0},
        ]
        mock_svc.store.open_table.return_value = mock_table

        related = cg.get_related_concepts("python")
        assert "django" in related

    def test_get_related_concepts_no_table(self, cg, mock_svc):
        mock_svc.store.open_table.return_value = None
        assert cg.get_related_concepts("python") == []

    def test_get_related_concepts_query_exception(self, cg, mock_svc):
        mock_table = MagicMock()
        mock_table.search.return_value.where.return_value.to_list.side_effect = RuntimeError(
            "query failed"
        )
        mock_svc.store.open_table.return_value = mock_table

        result = cg.get_related_concepts("python")
        assert result == []


class TestTopCommunities:
    def test_top_communities(self, cg, mock_svc):
        mock_table = MagicMock()
        mock_table.to_arrow.return_value.to_pylist.return_value = [
            {"concept": "python", "cluster_id": 0, "degree": 5},
            {"concept": "ml", "cluster_id": 0, "degree": 3},
            {"concept": "web", "cluster_id": 1, "degree": 2},
        ]
        mock_svc.store.open_table.return_value = mock_table

        communities = cg.top_communities(k=2)
        assert len(communities) == 2
        assert communities[0].size == 2
        assert communities[0].cluster_id == 0

    def test_top_communities_no_table(self, cg, mock_svc):
        mock_svc.store.open_table.return_value = None
        assert cg.top_communities() == []


class TestGetChunkConcepts:
    def test_get_chunk_concepts(self, cg, mock_svc):
        mock_table = MagicMock()
        mock_table.search.return_value.where.return_value.to_list.return_value = [
            {"concept": "python"},
            {"concept": "ml"},
        ]
        mock_svc.store.open_table.return_value = mock_table

        concepts = cg.get_chunk_concepts("doc.md", 0)
        assert concepts == ["python", "ml"]

    def test_get_chunk_concepts_no_table(self, cg, mock_svc):
        mock_svc.store.open_table.return_value = None
        assert cg.get_chunk_concepts("doc.md", 0) == []

    def test_get_chunk_concepts_exception(self, cg, mock_svc):
        mock_table = MagicMock()
        mock_table.search.side_effect = RuntimeError("query failed")
        mock_svc.store.open_table.return_value = mock_table

        assert cg.get_chunk_concepts("doc.md", 0) == []


class TestRebuildClusters:
    def test_rebuild_no_table(self, cg, mock_svc):
        mock_svc.store.open_table.return_value = None
        cg.rebuild_clusters()

    def test_rebuild_empty_edges(self, cg, mock_svc):
        mock_table = MagicMock()
        mock_table.to_arrow.return_value.to_pylist.return_value = []
        mock_svc.store.open_table.return_value = mock_table
        cg.rebuild_clusters()

    @patch("lilbee.lock.write_lock")
    @patch("lilbee.store.ensure_table")
    @patch("lilbee.concepts._leiden_partition")
    def test_rebuild_with_edges(self, mock_leiden, mock_ensure, mock_lock, cg, mock_svc):
        mock_lock.return_value.__enter__ = MagicMock()
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        mock_table = MagicMock()
        edge_rows = [
            {"source": "python", "target": "ml", "weight": 2.0},
            {"source": "ml", "target": "deep learning", "weight": 1.5},
        ]
        mock_table.to_arrow.return_value.to_pylist.return_value = edge_rows
        mock_svc.store.open_table.return_value = mock_table
        mock_svc.store.get_db.return_value = MagicMock()
        mock_leiden.return_value = (
            {"python": 0, "ml": 0, "deep learning": 1},
            {"python": 1, "ml": 2, "deep learning": 1},
        )
        mock_nodes_table = MagicMock()
        mock_ensure.return_value = mock_nodes_table

        cg.rebuild_clusters()
        mock_leiden.assert_called_once_with(edge_rows)
        mock_nodes_table.add.assert_called_once()


class TestGetGraph:
    def test_returns_true_when_enabled(self, cg, mock_svc):
        mock_svc.store.open_table.return_value = MagicMock()
        cfg.concept_graph = True
        assert cg.get_graph() is True

    def test_returns_false_when_disabled(self, cg):
        cfg.concept_graph = False
        assert cg.get_graph() is False

    def test_returns_false_when_no_tables(self, cg, mock_svc):
        mock_svc.store.open_table.return_value = None
        cfg.concept_graph = True
        assert cg.get_graph() is False


class TestResetGraph:
    def test_clears_nlp_cache(self, cg):
        """reset_nlp_cache clears the spaCy model cache."""
        cg._nlp = MagicMock()
        cg.reset_nlp_cache()
        assert cg._nlp is None


class TestComputePmi:
    def test_basic_ppmi(self):
        from collections import Counter

        from lilbee.concepts import _compute_pmi

        cooccurrences = Counter({("a", "b"): 5})
        concept_counts = Counter({"a": 8, "b": 6})
        pmi = _compute_pmi(cooccurrences, concept_counts, 10)
        assert ("a", "b") in pmi
        # PPMI: all values >= 0
        assert pmi[("a", "b")] >= 0.0

    def test_ppmi_clamps_negative(self):
        """Anti-correlated pairs should get PPMI = 0."""
        from collections import Counter

        from lilbee.concepts import _compute_pmi

        # a and b rarely co-occur but each appear often -> negative PMI -> clamped to 0
        cooccurrences = Counter({("a", "b"): 1})
        concept_counts = Counter({"a": 9, "b": 9})
        pmi = _compute_pmi(cooccurrences, concept_counts, 10)
        assert pmi[("a", "b")] == 0.0

    def test_ppmi_skips_zero_count_concepts(self):
        """Concepts with zero count are skipped (avoid division by zero)."""
        from collections import Counter

        from lilbee.concepts import _compute_pmi

        cooccurrences = Counter({("a", "b"): 1})
        concept_counts = Counter({"a": 0, "b": 5})
        pmi = _compute_pmi(cooccurrences, concept_counts, 10)
        assert ("a", "b") not in pmi


class TestLeidenPartition:
    def test_returns_partition_and_degrees(self):
        mock_graspologic = MagicMock()
        mock_graspologic.leiden.return_value = (0.5, {"a": 0, "b": 0, "c": 1})
        with patch.dict("sys.modules", {"graspologic_native": mock_graspologic}):
            from lilbee.concepts import _leiden_partition

            edge_rows = [
                {"source": "a", "target": "b", "weight": 2.0},
                {"source": "b", "target": "c", "weight": 1.5},
            ]
            partition, degrees = _leiden_partition(edge_rows)
            assert partition == {"a": 0, "b": 0, "c": 1}
            assert degrees["a"] == 1
            assert degrees["b"] == 2
            assert degrees["c"] == 1

    def test_clamps_low_weights(self):
        """Weights below _MIN_LEIDEN_WEIGHT are clamped up."""
        mock_graspologic = MagicMock()
        mock_graspologic.leiden.return_value = (0.5, {"a": 0, "b": 0})
        with patch.dict("sys.modules", {"graspologic_native": mock_graspologic}):
            from lilbee.concepts import _MIN_LEIDEN_WEIGHT, _leiden_partition

            edge_rows = [{"source": "a", "target": "b", "weight": 0.0}]
            _leiden_partition(edge_rows)
            call_args = mock_graspologic.leiden.call_args
            edges_passed = call_args[1]["edges"]
            assert edges_passed[0][2] == _MIN_LEIDEN_WEIGHT


class TestCommunityDataclass:
    def test_community_fields(self):
        from lilbee.concepts import Community

        c = Community(cluster_id=0, size=3, concepts=["a", "b", "c"])
        assert c.cluster_id == 0
        assert c.size == 3
        assert c.concepts == ["a", "b", "c"]

    def test_community_is_dataclass(self):
        from lilbee.concepts import Community

        assert len(fields(Community)) == 3


class TestGetClusterSources:
    def test_returns_clusters_spanning_min_sources(self, cg, mock_svc):
        nodes_table = MagicMock()
        nodes_table.to_arrow.return_value.to_pylist.return_value = [
            {"concept": "python", "cluster_id": 0, "degree": 3},
            {"concept": "ml", "cluster_id": 0, "degree": 2},
            {"concept": "web", "cluster_id": 1, "degree": 1},
        ]
        cc_table = MagicMock()
        cc_table.to_arrow.return_value.to_pylist.return_value = [
            {"chunk_source": "a.md", "chunk_index": 0, "concept": "python"},
            {"chunk_source": "b.md", "chunk_index": 0, "concept": "python"},
            {"chunk_source": "c.md", "chunk_index": 0, "concept": "ml"},
            {"chunk_source": "d.md", "chunk_index": 0, "concept": "web"},
        ]

        def open_table(name):
            from lilbee.config import CHUNK_CONCEPTS_TABLE, CONCEPT_NODES_TABLE

            if name == CONCEPT_NODES_TABLE:
                return nodes_table
            if name == CHUNK_CONCEPTS_TABLE:
                return cc_table
            return None

        mock_svc.store.open_table.side_effect = open_table
        result = cg.get_cluster_sources(min_sources=3)
        assert 0 in result
        assert result[0] == {"a.md", "b.md", "c.md"}
        assert 1 not in result

    def test_skips_orphan_concepts(self, cg, mock_svc):
        """Chunk-concepts referencing concepts not in any cluster are ignored."""
        nodes_table = MagicMock()
        nodes_table.to_arrow.return_value.to_pylist.return_value = [
            {"concept": "python", "cluster_id": 0, "degree": 3},
        ]
        cc_table = MagicMock()
        cc_table.to_arrow.return_value.to_pylist.return_value = [
            {"chunk_source": "a.md", "chunk_index": 0, "concept": "python"},
            {"chunk_source": "b.md", "chunk_index": 0, "concept": "orphan_concept"},
        ]

        def open_table(name):
            from lilbee.config import CHUNK_CONCEPTS_TABLE, CONCEPT_NODES_TABLE

            if name == CONCEPT_NODES_TABLE:
                return nodes_table
            if name == CHUNK_CONCEPTS_TABLE:
                return cc_table
            return None

        mock_svc.store.open_table.side_effect = open_table
        result = cg.get_cluster_sources(min_sources=1)
        assert 0 in result
        assert result[0] == {"a.md"}

    def test_returns_empty_when_no_tables(self, cg, mock_svc):
        mock_svc.store.open_table.return_value = None
        assert cg.get_cluster_sources() == {}

    def test_returns_empty_when_no_qualifying_clusters(self, cg, mock_svc):
        nodes_table = MagicMock()
        nodes_table.to_arrow.return_value.to_pylist.return_value = [
            {"concept": "python", "cluster_id": 0, "degree": 1},
        ]
        cc_table = MagicMock()
        cc_table.to_arrow.return_value.to_pylist.return_value = [
            {"chunk_source": "a.md", "chunk_index": 0, "concept": "python"},
        ]

        def open_table(name):
            from lilbee.config import CHUNK_CONCEPTS_TABLE, CONCEPT_NODES_TABLE

            if name == CONCEPT_NODES_TABLE:
                return nodes_table
            if name == CHUNK_CONCEPTS_TABLE:
                return cc_table
            return None

        mock_svc.store.open_table.side_effect = open_table
        assert cg.get_cluster_sources(min_sources=3) == {}


class TestGetClusterLabel:
    def test_returns_highest_degree_concept(self, cg, mock_svc):
        mock_table = MagicMock()
        mock_table.to_arrow.return_value.to_pylist.return_value = [
            {"concept": "python", "cluster_id": 0, "degree": 5},
            {"concept": "ml", "cluster_id": 0, "degree": 3},
            {"concept": "web", "cluster_id": 1, "degree": 2},
        ]
        mock_svc.store.open_table.return_value = mock_table
        assert cg.get_cluster_label(0) == "python"

    def test_returns_fallback_when_no_table(self, cg, mock_svc):
        mock_svc.store.open_table.return_value = None
        assert cg.get_cluster_label(42) == "cluster-42"

    def test_returns_fallback_for_unknown_cluster(self, cg, mock_svc):
        mock_table = MagicMock()
        mock_table.to_arrow.return_value.to_pylist.return_value = [
            {"concept": "python", "cluster_id": 0, "degree": 5},
        ]
        mock_svc.store.open_table.return_value = mock_table
        assert cg.get_cluster_label(99) == "cluster-99"


class TestFilterNounChunks:
    def test_filter_noun_chunks(self):
        from lilbee.concepts import _filter_noun_chunks

        doc = _make_mock_doc(["Hello World", "a", "Good Stuff", "Hello World"])
        result = _filter_noun_chunks(doc, max_concepts=10)
        assert result == ["hello world", "good stuff"]

    def test_filter_noun_chunks_max(self):
        from lilbee.concepts import _filter_noun_chunks

        doc = _make_mock_doc(["aa", "bb", "cc"])
        result = _filter_noun_chunks(doc, max_concepts=2)
        assert len(result) == 2
