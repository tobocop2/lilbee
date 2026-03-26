"""Tests for the concept graph module."""

from unittest import mock
from unittest.mock import MagicMock, patch

import pytest

from lilbee.config import cfg
from lilbee.store import SearchChunk


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    """Redirect config paths to temp dir for every test."""
    snapshot = cfg.model_copy()
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir()
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.concept_graph = True
    cfg.concept_max_per_chunk = 10
    cfg.concept_boost_weight = 0.3
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset module-level singletons between tests."""
    from lilbee.concepts import reset_graph

    reset_graph()
    yield
    reset_graph()


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


class TestExtractConcepts:
    @patch("lilbee.concepts._get_nlp")
    def test_basic_extraction(self, mock_get_nlp):
        mock_get_nlp.return_value = _make_mock_nlp(
            {"hello world": ["machine learning", "neural networks"]}
        )
        from lilbee.concepts import extract_concepts

        result = extract_concepts("hello world")
        assert result == ["machine learning", "neural networks"]

    @patch("lilbee.concepts._get_nlp")
    def test_deduplication(self, mock_get_nlp):
        mock_get_nlp.return_value = _make_mock_nlp(
            {"text": ["Concept", "concept", "Other"]}
        )
        from lilbee.concepts import extract_concepts

        result = extract_concepts("text")
        assert result == ["concept", "other"]

    @patch("lilbee.concepts._get_nlp")
    def test_max_cap(self, mock_get_nlp):
        mock_get_nlp.return_value = _make_mock_nlp(
            {"text": ["alpha", "beta", "gamma", "delta"]}
        )
        from lilbee.concepts import extract_concepts

        result = extract_concepts("text", max_concepts=2)
        assert len(result) == 2

    @patch("lilbee.concepts._get_nlp")
    def test_empty_input(self, mock_get_nlp):
        from lilbee.concepts import extract_concepts

        result = extract_concepts("")
        assert result == []
        mock_get_nlp.assert_not_called()

    @patch("lilbee.concepts._get_nlp")
    def test_filters_short_concepts(self, mock_get_nlp):
        mock_get_nlp.return_value = _make_mock_nlp(
            {"text": ["a", "ok", "good concept"]}
        )
        from lilbee.concepts import extract_concepts

        result = extract_concepts("text")
        assert "a" not in result
        assert "ok" in result
        assert "good concept" in result


class TestExtractConceptsBatch:
    @patch("lilbee.concepts._get_nlp")
    def test_batch_extraction(self, mock_get_nlp):
        mock_get_nlp.return_value = _make_mock_nlp(
            {"doc1": ["concept a"], "doc2": ["concept b", "concept c"]}
        )
        from lilbee.concepts import extract_concepts_batch

        result = extract_concepts_batch(["doc1", "doc2"])
        assert len(result) == 2
        assert result[0] == ["concept a"]
        assert result[1] == ["concept b", "concept c"]

    @patch("lilbee.concepts._get_nlp")
    def test_empty_input(self, mock_get_nlp):
        from lilbee.concepts import extract_concepts_batch

        result = extract_concepts_batch([])
        assert result == []
        mock_get_nlp.assert_not_called()


class TestEnsureSpacyModel:
    @patch("spacy.load")
    def test_loads_existing(self, mock_load):
        mock_load.return_value = MagicMock()
        from lilbee.concepts import _ensure_spacy_model

        result = _ensure_spacy_model()
        mock_load.assert_called_once_with("en_core_web_sm")
        assert result is not None

    @patch("spacy.load")
    @patch("spacy.cli.download")
    def test_downloads_on_oserror(self, mock_download, mock_load):
        mock_load.side_effect = [OSError("not found"), MagicMock()]
        from lilbee.concepts import _ensure_spacy_model

        result = _ensure_spacy_model()
        mock_download.assert_called_once_with("en_core_web_sm")
        assert result is not None


class TestConceptGraph:
    def test_build_from_chunks(self):
        from lilbee.concepts import ConceptGraph

        graph = ConceptGraph()
        chunk_ids = [("doc.md", 0), ("doc.md", 1)]
        concept_lists = [["python", "machine learning"], ["python", "deep learning"]]
        graph.build_from_chunks(chunk_ids, concept_lists)

    def test_build_empty_chunks(self):
        from lilbee.concepts import ConceptGraph

        graph = ConceptGraph()
        graph.build_from_chunks([], [])

    def test_boost_results_with_overlap(self):
        from lilbee.concepts import ConceptGraph

        graph = ConceptGraph()
        results = [_make_result(distance=0.5, chunk_index=0)]
        with patch.object(graph, "_get_chunk_concepts", return_value=["python", "ml"]):
            boosted = graph.boost_results(results, ["python", "java"])
        assert boosted[0].distance < 0.5

    def test_boost_results_relevance_score(self):
        from lilbee.concepts import ConceptGraph

        graph = ConceptGraph()
        results = [_make_result(distance=None, relevance_score=0.8, chunk_index=0)]
        with patch.object(graph, "_get_chunk_concepts", return_value=["python"]):
            boosted = graph.boost_results(results, ["python"])
        assert boosted[0].relevance_score > 0.8

    def test_boost_results_no_overlap(self):
        from lilbee.concepts import ConceptGraph

        graph = ConceptGraph()
        results = [_make_result(distance=0.5, chunk_index=0)]
        with patch.object(graph, "_get_chunk_concepts", return_value=["java"]):
            boosted = graph.boost_results(results, ["python"])
        assert boosted[0].distance == 0.5

    def test_boost_results_empty_query_concepts(self):
        from lilbee.concepts import ConceptGraph

        graph = ConceptGraph()
        results = [_make_result()]
        boosted = graph.boost_results(results, [])
        assert boosted == results

    def test_boost_results_empty_results(self):
        from lilbee.concepts import ConceptGraph

        graph = ConceptGraph()
        boosted = graph.boost_results([], ["python"])
        assert boosted == []

    @patch("lilbee.concepts.extract_concepts", return_value=["python"])
    def test_expand_query(self, mock_extract):
        from lilbee.concepts import ConceptGraph

        graph = ConceptGraph()
        with patch.object(graph, "get_related_concepts", return_value=["django", "flask"]):
            related = graph.expand_query("python frameworks")
        assert "django" in related
        assert "flask" in related

    @patch("lilbee.concepts.extract_concepts", return_value=[])
    def test_expand_query_no_concepts(self, mock_extract):
        from lilbee.concepts import ConceptGraph

        graph = ConceptGraph()
        assert graph.expand_query("???") == []

    @patch("lilbee.store._open_table")
    def test_get_related_concepts(self, mock_open):
        mock_table = MagicMock()
        mock_table.search.return_value.where.return_value.to_list.return_value = [
            {"source": "python", "target": "django", "weight": 1.0},
        ]
        mock_open.return_value = mock_table
        from lilbee.concepts import ConceptGraph

        graph = ConceptGraph()
        related = graph.get_related_concepts("python")
        assert "django" in related

    @patch("lilbee.store._open_table", return_value=None)
    def test_get_related_concepts_no_table(self, mock_open):
        from lilbee.concepts import ConceptGraph

        graph = ConceptGraph()
        assert graph.get_related_concepts("python") == []

    @patch("lilbee.store._open_table")
    def test_top_communities(self, mock_open):
        mock_table = MagicMock()
        mock_table.to_arrow.return_value.to_pylist.return_value = [
            {"concept": "python", "cluster_id": 0, "degree": 5},
            {"concept": "ml", "cluster_id": 0, "degree": 3},
            {"concept": "web", "cluster_id": 1, "degree": 2},
        ]
        mock_open.return_value = mock_table
        from lilbee.concepts import ConceptGraph

        graph = ConceptGraph()
        communities = graph.top_communities(k=2)
        assert len(communities) == 2
        assert communities[0]["size"] == 2
        assert communities[0]["cluster_id"] == 0

    @patch("lilbee.store._open_table", return_value=None)
    def test_top_communities_no_table(self, mock_open):
        from lilbee.concepts import ConceptGraph

        graph = ConceptGraph()
        assert graph.top_communities() == []

    @patch("lilbee.store._open_table")
    def test_get_chunk_concepts(self, mock_open):
        mock_table = MagicMock()
        mock_table.search.return_value.where.return_value.to_list.return_value = [
            {"concept": "python"},
            {"concept": "ml"},
        ]
        mock_open.return_value = mock_table
        from lilbee.concepts import ConceptGraph

        graph = ConceptGraph()
        concepts = graph._get_chunk_concepts("doc.md", 0)
        assert concepts == ["python", "ml"]

    @patch("lilbee.store._open_table", return_value=None)
    def test_get_chunk_concepts_no_table(self, mock_open):
        from lilbee.concepts import ConceptGraph

        graph = ConceptGraph()
        assert graph._get_chunk_concepts("doc.md", 0) == []

    @patch("lilbee.store._open_table")
    def test_get_chunk_concepts_exception(self, mock_open):
        mock_table = MagicMock()
        mock_table.search.side_effect = RuntimeError("query failed")
        mock_open.return_value = mock_table
        from lilbee.concepts import ConceptGraph

        graph = ConceptGraph()
        assert graph._get_chunk_concepts("doc.md", 0) == []


class TestRebuildClusters:
    @patch("lilbee.store._open_table")
    def test_rebuild_no_table(self, mock_open):
        mock_open.return_value = None
        from lilbee.concepts import ConceptGraph

        graph = ConceptGraph()
        graph.rebuild_clusters()

    @patch("lilbee.store._open_table")
    def test_rebuild_empty_edges(self, mock_open):
        mock_table = MagicMock()
        mock_table.to_arrow.return_value.to_pylist.return_value = []
        mock_open.return_value = mock_table
        from lilbee.concepts import ConceptGraph

        graph = ConceptGraph()
        graph.rebuild_clusters()

    @patch("lilbee.store.safe_delete")
    @patch("lilbee.store.ensure_table")
    @patch("lilbee.store.get_db")
    @patch("lilbee.concepts._leiden_partition")
    @patch("lilbee.store._open_table")
    def test_rebuild_with_edges(
        self, mock_open, mock_leiden, mock_get_db, mock_ensure, mock_safe_del
    ):
        mock_table = MagicMock()
        edge_rows = [
            {"source": "python", "target": "ml", "weight": 2.0},
            {"source": "ml", "target": "deep learning", "weight": 1.5},
        ]
        mock_table.to_arrow.return_value.to_pylist.return_value = edge_rows
        mock_open.return_value = mock_table
        mock_leiden.return_value = (
            {"python": 0, "ml": 0, "deep learning": 1},
            {"python": 1, "ml": 2, "deep learning": 1},
        )
        mock_nodes_table = MagicMock()
        mock_ensure.return_value = mock_nodes_table

        from lilbee.concepts import ConceptGraph

        graph = ConceptGraph()
        graph.rebuild_clusters()
        mock_leiden.assert_called_once_with(edge_rows)
        mock_nodes_table.add.assert_called_once()


class TestGetGraph:
    def test_returns_graph_when_enabled(self):
        from lilbee.concepts import get_graph

        cfg.concept_graph = True
        graph = get_graph()
        assert graph is not None

    def test_returns_none_when_disabled(self):
        from lilbee.concepts import get_graph

        cfg.concept_graph = False
        assert get_graph() is None

    def test_singleton(self):
        from lilbee.concepts import get_graph

        cfg.concept_graph = True
        g1 = get_graph()
        g2 = get_graph()
        assert g1 is g2


class TestResetGraph:
    def test_clears_singleton(self):
        from lilbee.concepts import get_graph, reset_graph

        cfg.concept_graph = True
        g1 = get_graph()
        reset_graph()
        g2 = get_graph()
        assert g1 is not g2


class TestComputePmi:
    def test_basic_pmi(self):
        from collections import Counter

        from lilbee.concepts import _compute_pmi

        cooccurrences = Counter({("a", "b"): 5})
        concept_counts = Counter({"a": 8, "b": 6})
        pmi = _compute_pmi(cooccurrences, concept_counts, 10)
        assert ("a", "b") in pmi
        assert isinstance(pmi[("a", "b")], float)
