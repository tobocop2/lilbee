"""Tests for the source clustering protocol and the embedding clusterer."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lilbee.clustering import SourceCluster, SourceClusterer
from lilbee.clustering_embedding import (
    EmbeddingClusterer,
    _aggregate_by_source,
    _connected_components,
    _heuristic_label,
    _mean_embeddings,
    _normalized_mean,
)
from lilbee.config import cfg


@pytest.fixture(autouse=True)
def isolated_cfg():
    snapshot = cfg.model_copy()
    cfg.wiki_clusterer_threshold = 0.6
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _row(source: str, vector: list[float], chunk: str = "text") -> dict[str, object]:
    return {"source": source, "vector": vector, "chunk": chunk}


class TestSourceCluster:
    def test_immutable_dataclass(self):
        cluster = SourceCluster(cluster_id="x", label="topic", sources=frozenset({"a"}))
        with pytest.raises(AttributeError):
            cluster.cluster_id = "y"  # type: ignore[misc]

    def test_protocol_runtime_check(self):
        class Dummy:
            def available(self) -> bool:
                return True

            def get_clusters(self, min_sources: int = 3) -> list[SourceCluster]:
                return []

        assert isinstance(Dummy(), SourceClusterer)


class TestNormalizedMean:
    def test_returns_unit_length_vector(self):
        result = _normalized_mean([[1.0, 0.0], [3.0, 0.0]])
        assert result is not None
        assert result == [1.0, 0.0]

    def test_zero_vector_returns_none(self):
        assert _normalized_mean([[0.0, 0.0]]) is None

    def test_empty_returns_none(self):
        assert _normalized_mean([]) is None


class TestAggregateBySource:
    def test_groups_vectors_and_texts(self):
        rows = [
            _row("a.md", [1.0, 0.0], chunk="alpha"),
            _row("a.md", [3.0, 0.0], chunk="beta"),
            _row("b.md", [0.0, 2.0], chunk="gamma"),
        ]
        vectors, texts = _aggregate_by_source(rows)
        assert sorted(vectors.keys()) == ["a.md", "b.md"]
        assert vectors["a.md"] == [[1.0, 0.0], [3.0, 0.0]]
        assert texts["a.md"] == ["alpha", "beta"]
        assert texts["b.md"] == ["gamma"]

    def test_skips_invalid_rows(self):
        rows: list[dict[str, object]] = [
            {"source": None, "vector": [1.0]},
            {"source": "a.md", "vector": None},
            _row("b.md", [1.0, 0.0]),
        ]
        vectors, _ = _aggregate_by_source(rows)
        assert list(vectors.keys()) == ["b.md"]


class TestMeanEmbeddings:
    def test_normalizes_each_source(self):
        sources, means = _mean_embeddings({"a.md": [[1.0, 0.0], [3.0, 0.0]], "b.md": [[0.0, 2.0]]})
        assert sorted(sources) == ["a.md", "b.md"]
        for vec in means:
            length = sum(v * v for v in vec) ** 0.5
            assert length == pytest.approx(1.0, abs=1e-6)

    def test_skips_zero_norm_sources(self):
        sources, means = _mean_embeddings({"a.md": [[0.0, 0.0]]})
        assert sources == []
        assert means == []


class TestConnectedComponents:
    def test_separates_orthogonal_vectors(self):
        embeddings = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        components = _connected_components(embeddings, threshold=0.5)
        # First and third have cosine -1.0, neither connects to second
        sizes = sorted(len(c) for c in components)
        assert sizes == [1, 1, 1]

    def test_merges_similar_vectors(self):
        embeddings = [[1.0, 0.0], [0.99, 0.14], [0.95, 0.31]]
        components = _connected_components(embeddings, threshold=0.9)
        assert len(components) == 1
        assert sorted(components[0]) == [0, 1, 2]


class TestHeuristicLabel:
    def test_returns_top_words(self):
        texts = {"a.md": ["python typing python typing example example example"]}
        label = _heuristic_label(["a.md"], texts, fallback="x")
        assert "python" in label or "typing" in label or "example" in label

    def test_filters_short_words(self):
        texts = {"a.md": ["the and from with python python python typing"]}
        label = _heuristic_label(["a.md"], texts, fallback="x")
        assert "python" in label
        assert "the" not in label

    def test_empty_returns_fallback(self):
        label = _heuristic_label(["a.md", "b.md"], {}, fallback="cluster-7")
        assert label == "cluster-7"


class TestEmbeddingClustererAvailable:
    def test_available_when_table_has_rows(self):
        store = MagicMock()
        table = MagicMock()
        table.count_rows.return_value = 5
        store.open_table.return_value = table
        clusterer = EmbeddingClusterer(cfg, store)
        assert clusterer.available() is True

    def test_unavailable_when_table_empty(self):
        store = MagicMock()
        table = MagicMock()
        table.count_rows.return_value = 0
        store.open_table.return_value = table
        clusterer = EmbeddingClusterer(cfg, store)
        assert clusterer.available() is False

    def test_unavailable_when_table_missing(self):
        store = MagicMock()
        store.open_table.return_value = None
        clusterer = EmbeddingClusterer(cfg, store)
        assert clusterer.available() is False

    def test_count_rows_failure_falls_back_to_true(self):
        store = MagicMock()
        table = MagicMock()
        table.count_rows.side_effect = RuntimeError("not supported")
        store.open_table.return_value = table
        clusterer = EmbeddingClusterer(cfg, store)
        assert clusterer.available() is True


class TestEmbeddingClustererGetClusters:
    def _store_with_rows(self, rows: list[dict[str, object]]):
        store = MagicMock()
        table = MagicMock()
        arrow = MagicMock()
        arrow.to_pylist.return_value = rows
        table.to_arrow.return_value = arrow
        store.open_table.return_value = table
        return store

    def test_returns_empty_when_no_table(self):
        store = MagicMock()
        store.open_table.return_value = None
        result = EmbeddingClusterer(cfg, store).get_clusters()
        assert result == []

    def test_returns_empty_below_min_sources(self):
        rows = [_row("a.md", [1.0, 0.0]), _row("b.md", [0.0, 1.0])]
        store = self._store_with_rows(rows)
        result = EmbeddingClusterer(cfg, store).get_clusters(min_sources=3)
        assert result == []

    def test_returns_cluster_for_similar_sources(self):
        cfg.wiki_clusterer_threshold = 0.5
        rows = [
            _row("a.md", [1.0, 0.0], chunk="python typing python typing python"),
            _row("b.md", [0.99, 0.14], chunk="python typing examples here"),
            _row("c.md", [0.95, 0.31], chunk="python typing more examples"),
            _row("d.md", [-1.0, 0.0], chunk="totally unrelated cooking recipes"),
        ]
        store = self._store_with_rows(rows)
        result = EmbeddingClusterer(cfg, store).get_clusters(min_sources=3)
        assert len(result) == 1
        assert result[0].sources == frozenset({"a.md", "b.md", "c.md"})
        assert result[0].cluster_id.startswith("embedding-")
        assert result[0].label  # non-empty label


class TestConceptGraphClusterer:
    """The concept graph adapter is exercised here to keep wiki test coverage."""

    def test_unavailable_without_graph(self, monkeypatch: pytest.MonkeyPatch):
        from lilbee.concepts import ConceptGraphClusterer

        monkeypatch.setattr("lilbee.concepts.concepts_available", lambda: False)
        store = MagicMock()
        store.open_table.return_value = None
        clusterer = ConceptGraphClusterer(cfg, store)
        assert clusterer.available() is False

    def test_get_clusters_maps_concept_graph_output(self, monkeypatch: pytest.MonkeyPatch):
        from lilbee.concepts import ConceptGraph, ConceptGraphClusterer

        monkeypatch.setattr(
            ConceptGraph, "get_cluster_sources", lambda self, **kw: {7: {"a.md", "b.md"}}
        )
        monkeypatch.setattr(ConceptGraph, "get_cluster_label", lambda self, cid: f"label-{cid}")
        store = MagicMock()
        clusterer = ConceptGraphClusterer(cfg, store)
        result = clusterer.get_clusters(min_sources=2)
        assert len(result) == 1
        assert result[0].cluster_id == "concept-7"
        assert result[0].label == "label-7"
        assert result[0].sources == frozenset({"a.md", "b.md"})
