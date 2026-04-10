"""Tests for the source clustering protocol and the embedding clusterer."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from lilbee.clustering import SourceCluster, SourceClusterer
from lilbee.clustering_embedding import (
    EmbeddingClusterer,
    _auto_k,
    _build_clusters,
    _ChunkRecord,
    _communities_by_label,
    _corpus_document_frequency,
    _filter_sources,
    _label_community,
    _label_propagation,
    _load_chunk_records,
    _mutual_knn,
    _normalize_rows,
    _source_totals,
    _warn_if_undersegmented,
)
from lilbee.config import cfg


@pytest.fixture(autouse=True)
def isolated_cfg():
    snapshot = cfg.model_copy()
    cfg.wiki_clusterer_k = 0
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _record(source: str, chunk_index: int = 0, text: str = "") -> _ChunkRecord:
    from lilbee.text import tokenize

    return _ChunkRecord(
        source=source,
        chunk_index=chunk_index,
        text=text,
        tokens=tokenize(text),
    )


def _row(
    source: str,
    vector: list[float],
    chunk: str = "text",
    chunk_index: int = 0,
) -> dict[str, object]:
    return {
        "source": source,
        "vector": vector,
        "chunk": chunk,
        "chunk_index": chunk_index,
    }


def _mock_table(rows: list[dict[str, object]]) -> MagicMock:
    table = MagicMock()
    arrow = MagicMock()
    arrow.to_pylist.return_value = rows
    table.to_arrow.return_value = arrow
    table.count_rows.return_value = len(rows)
    return table


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


class TestAutoK:
    @pytest.mark.parametrize(
        "n,expected",
        [
            (1, 5),
            (50, 8),
            (500, 11),
            (5000, 14),
            (50000, 18),
            (10_000_000, 20),  # clamped upper bound
        ],
    )
    def test_scales_with_corpus_size(self, n: int, expected: int) -> None:
        assert _auto_k(n) == expected


class TestNormalizeRows:
    def test_drops_zero_vectors_and_normalizes(self):
        matrix = np.array([[3.0, 0.0], [0.0, 0.0], [0.0, 4.0]], dtype=np.float32)
        normalized, keep = _normalize_rows(matrix)
        assert keep.tolist() == [True, False, True]
        assert normalized.shape == (2, 2)
        np.testing.assert_allclose(
            np.linalg.norm(normalized, axis=1),
            np.array([1.0, 1.0], dtype=np.float32),
            atol=1e-6,
        )

    def test_empty_matrix_is_noop(self):
        matrix = np.zeros((0, 0), dtype=np.float32)
        normalized, keep = _normalize_rows(matrix)
        assert normalized.shape == (0, 0)
        assert keep.shape == (0,)


class TestMutualKnn:
    def test_two_triangles_are_mutual(self):
        # Triangle A: 3 vectors near [1,0]. Triangle B: 3 near [0,1].
        matrix = np.array(
            [
                [1.0, 0.0],
                [0.98, 0.2],
                [0.95, 0.31],
                [0.0, 1.0],
                [0.2, 0.98],
                [0.31, 0.95],
            ],
            dtype=np.float32,
        )
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        adjacency = _mutual_knn(matrix, k=2)
        # Nodes 0,1,2 should be mutually connected; 3,4,5 likewise.
        for i in (0, 1, 2):
            assert any(j in adjacency[i] for j in (0, 1, 2) if j != i)
            assert all(j not in adjacency[i] for j in (3, 4, 5))

    def test_mutual_kNN_rejects_hub(self):
        # Node 0 is the hub: a ones-vector that sits at a modest cosine
        # similarity to every basis vector. Each basis vector's single
        # closest neighbor is some *other* basis vector in a small
        # cluster we plant (A), so the hub never earns a mutual edge
        # even though it would dominate a plain top-k list.
        cluster = np.array(
            [
                [1.0, 0.99, 0.0, 0.0, 0.0],
                [0.99, 1.0, 0.0, 0.0, 0.0],
                [0.98, 0.98, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        hub = np.ones((1, 5), dtype=np.float32)
        other = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        matrix = np.vstack([hub, cluster, other])
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        adjacency = _mutual_knn(matrix, k=1)
        # Hub must have zero mutual edges: each cluster row mutually
        # points to its in-cluster neighbor, and the singleton row has
        # no reciprocated partner.
        assert adjacency[0] == set()

    def test_empty_matrix(self):
        assert _mutual_knn(np.zeros((0, 0), dtype=np.float32), k=3) == {}

    def test_k_larger_than_population(self):
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        adjacency = _mutual_knn(matrix, k=10)
        # Both nodes should still end up as mutual neighbors.
        assert 1 in adjacency[0]
        assert 0 in adjacency[1]

    def test_single_row_has_no_neighbors(self):
        matrix = np.array([[1.0, 0.0]], dtype=np.float32)
        adjacency = _mutual_knn(matrix, k=3)
        assert adjacency == {0: set()}

    def test_k_zero_returns_empty(self):
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        assert _mutual_knn(matrix, k=0) == {}


class TestLabelPropagation:
    def test_two_cliques_converge(self):
        adjacency = {
            0: {1, 2},
            1: {0, 2},
            2: {0, 1},
            3: {4, 5},
            4: {3, 5},
            5: {3, 4},
        }
        labels = _label_propagation(adjacency, order=list(range(6)))
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] == labels[4] == labels[5]
        assert labels[0] != labels[3]

    def test_deterministic_with_fixed_order(self):
        adjacency = {
            0: {1},
            1: {0, 2},
            2: {1, 3},
            3: {2},
        }
        run_a = _label_propagation(adjacency, order=[0, 1, 2, 3])
        run_b = _label_propagation(adjacency, order=[0, 1, 2, 3])
        assert run_a == run_b

    def test_isolated_nodes_keep_initial_labels(self):
        adjacency = {0: set(), 1: set()}
        labels = _label_propagation(adjacency, order=[0, 1])
        assert labels == [0, 1]


class TestCommunitiesByLabel:
    def test_groups_by_label(self):
        labels = [0, 0, 1, 1, 1, 2]
        communities = _communities_by_label(labels)
        assert communities[0] == [0, 1]
        assert communities[1] == [2, 3, 4]
        assert communities[2] == [5]


class TestSourceTotals:
    def test_counts_per_source(self):
        records = [_record("a.md"), _record("a.md", 1), _record("b.md")]
        assert _source_totals(records) == {"a.md": 2, "b.md": 1}


class TestFilterSources:
    def test_rejects_stray_chunk(self):
        # Source "long.md" has 40 chunks corpus-wide; only chunk 0 is in
        # this community. 40 * 0.2 = 8 threshold, so 1 chunk is rejected.
        records = [_record("long.md"), _record("other.md")]
        totals = {"long.md": 40, "other.md": 5}
        kept = _filter_sources([0], records, totals)
        assert kept == frozenset()

    def test_accepts_when_min_sources_chunks_met(self):
        records = [
            _record("long.md", 0),
            _record("long.md", 1),
            _record("long.md", 2),
        ]
        totals = {"long.md": 40}
        kept = _filter_sources([0, 1, 2], records, totals)
        assert kept == frozenset({"long.md"})

    def test_accepts_when_fractional_cutoff_lower(self):
        # Short source: 4 chunks total, need min(3, ceil(4*0.2)=1) = 1
        records = [_record("short.md")]
        totals = {"short.md": 4}
        kept = _filter_sources([0], records, totals)
        assert kept == frozenset({"short.md"})


class TestCorpusDocumentFrequency:
    def test_counts_chunks_not_occurrences(self):
        records = [
            _record("a.md", text="python python python"),
            _record("b.md", text="kafka streams"),
            _record("c.md", text="python kafka"),
        ]
        df = _corpus_document_frequency(records)
        # "python" appears in 2 chunks, "kafka" in 2, "streams" in 1
        assert df["python"] == 2
        assert df["kafka"] == 2
        assert df["streams"] == 1


class TestLabelCommunity:
    def test_prefers_cluster_specific_terms(self):
        records = [
            _record("a.md", text="kubernetes operator reconciliation"),
            _record("b.md", text="kubernetes deployment autoscaling"),
            _record("c.md", text="kubernetes ingress controller"),
            _record("d.md", text="database sharding replica"),
        ]
        df = _corpus_document_frequency(records)
        label = _label_community([0, 1, 2], records, df, total_chunks=4, fallback="fallback")
        # Terms specific to the first 3 chunks should dominate
        assert "kubernetes" not in label  # appears in every cluster doc -> low IDF
        assert any(t in label for t in ("operator", "deployment", "ingress"))

    def test_fallback_when_no_content(self):
        records = [_record("a.md", text="")]
        label = _label_community([0], records, {}, total_chunks=1, fallback="fallback")
        assert label == "fallback"

    def test_fallback_when_all_terms_corpus_wide(self):
        records = [
            _record("a.md", text="python typing"),
            _record("b.md", text="python typing"),
        ]
        df = _corpus_document_frequency(records)
        label = _label_community([0], records, df, total_chunks=2, fallback="fallback")
        # Both terms have DF=2 of 2 chunks -> log(2/3) < 0 -> filtered
        assert label == "fallback"

    def test_deterministic_tie_break(self):
        records = [_record("a.md", text="alpha beta")]
        df = {"alpha": 1, "beta": 1}
        label = _label_community([0], records, df, total_chunks=5, fallback="fallback")
        # Both tokens have identical TF-IDF; alphabetical tiebreak
        assert label == "alpha beta"


class TestLoadChunkRecords:
    def test_skips_invalid_rows(self):
        store = MagicMock()
        store.open_table.return_value = _mock_table(
            [
                {"source": None, "vector": [1.0, 0.0], "chunk": "x", "chunk_index": 0},
                {"source": "a.md", "vector": None, "chunk": "x", "chunk_index": 0},
                _row("b.md", [0.1, 0.9], chunk="hello", chunk_index=2),
            ]
        )
        records, matrix = _load_chunk_records(store)
        assert [r.source for r in records] == ["b.md"]
        assert records[0].chunk_index == 2
        assert records[0].text == "hello"
        assert matrix.shape == (1, 2)

    def test_no_table_returns_empty(self):
        store = MagicMock()
        store.open_table.return_value = None
        records, matrix = _load_chunk_records(store)
        assert records == []
        assert matrix.size == 0

    def test_all_rows_invalid_returns_empty(self):
        store = MagicMock()
        store.open_table.return_value = _mock_table(
            [{"source": None, "vector": None, "chunk": "x", "chunk_index": 0}]
        )
        records, matrix = _load_chunk_records(store)
        assert records == []
        assert matrix.size == 0

    def test_handles_non_string_chunk_text(self):
        store = MagicMock()
        store.open_table.return_value = _mock_table(
            [{"source": "a.md", "vector": [1.0, 0.0], "chunk": None, "chunk_index": 0}]
        )
        records, matrix = _load_chunk_records(store)
        assert records[0].text == ""
        assert matrix.shape == (1, 2)


class TestEmbeddingClustererAvailable:
    def test_available_when_table_has_rows(self):
        store = MagicMock()
        table = MagicMock()
        table.count_rows.return_value = 5
        store.open_table.return_value = table
        assert EmbeddingClusterer(cfg, store).available() is True

    def test_unavailable_when_empty(self):
        store = MagicMock()
        table = MagicMock()
        table.count_rows.return_value = 0
        store.open_table.return_value = table
        assert EmbeddingClusterer(cfg, store).available() is False

    def test_unavailable_when_no_table(self):
        store = MagicMock()
        store.open_table.return_value = None
        assert EmbeddingClusterer(cfg, store).available() is False

    def test_count_rows_exception_falls_back_to_true(self):
        store = MagicMock()
        table = MagicMock()
        table.count_rows.side_effect = RuntimeError("no can do")
        store.open_table.return_value = table
        assert EmbeddingClusterer(cfg, store).available() is True


class TestEmbeddingClustererGetClusters:
    def _store_with_rows(self, rows: list[dict[str, object]]):
        store = MagicMock()
        store.open_table.return_value = _mock_table(rows)
        return store

    def test_returns_empty_when_no_chunks(self):
        store = self._store_with_rows([])
        assert EmbeddingClusterer(cfg, store).get_clusters() == []

    def test_returns_empty_when_no_table(self):
        store = MagicMock()
        store.open_table.return_value = None
        assert EmbeddingClusterer(cfg, store).get_clusters() == []

    def test_empty_after_normalization(self):
        store = self._store_with_rows(
            [_row("a.md", [0.0, 0.0], chunk="x")],
        )
        assert EmbeddingClusterer(cfg, store).get_clusters() == []

    def test_two_topics_two_clusters(self):
        # Three sources for each of two topics; each source has 3 chunks.
        # Chunk vectors within a topic sit very close together on the
        # topic's axis with tiny per-chunk offsets so mutual-kNN can
        # distinguish neighbors without everything being exactly tied.
        typing_text = [
            "python typing gradual static annotations protocol",
            "python typing generics variance covariant contravariant",
            "python typing stubs pyi overload runtime",
        ]
        kafka_text = [
            "kafka streams processor topology aggregate window",
            "kafka consumer rebalance partition assignment offsets",
            "kafka broker replication isr leader election",
        ]
        rows: list[dict[str, object]] = []
        typing_sources = ("typing-1.md", "typing-2.md", "typing-3.md")
        kafka_sources = ("kafka-1.md", "kafka-2.md", "kafka-3.md")
        for i, name in enumerate(typing_sources):
            for j, text in enumerate(typing_text):
                # Tiny offset keeps vectors near [1, 0] but distinct
                vector = [1.0, 0.001 * (i * 3 + j)]
                rows.append(_row(name, vector, chunk=text, chunk_index=j))
        for i, name in enumerate(kafka_sources):
            for j, text in enumerate(kafka_text):
                vector = [0.001 * (i * 3 + j), 1.0]
                rows.append(_row(name, vector, chunk=text, chunk_index=j))

        cfg.wiki_clusterer_k = 5
        store = self._store_with_rows(rows)
        clusters = EmbeddingClusterer(cfg, store).get_clusters(min_sources=3)
        assert len(clusters) == 2
        source_sets = {cluster.sources for cluster in clusters}
        assert frozenset(typing_sources) in source_sets
        assert frozenset(kafka_sources) in source_sets
        for cluster in clusters:
            assert cluster.label  # non-empty
            assert cluster.cluster_id.startswith("embedding-")

    def test_deterministic_across_runs(self):
        rows = [
            _row(
                f"doc-{i}.md",
                [1.0 if i < 4 else 0.0, 0.0 if i < 4 else 1.0],
                chunk=f"topic alpha beta gamma delta {i}",
                chunk_index=0,
            )
            for i in range(8)
        ]
        cfg.wiki_clusterer_k = 3
        store = self._store_with_rows(rows)
        run_a = EmbeddingClusterer(cfg, store).get_clusters(min_sources=3)
        run_b = EmbeddingClusterer(cfg, store).get_clusters(min_sources=3)
        assert [(c.sources, c.label) for c in run_a] == [(c.sources, c.label) for c in run_b]

    def test_warns_when_one_cluster_dominates(self, caplog: pytest.LogCaptureFixture):
        import logging as _logging

        # Five sources all collapsing to a single community.
        rows = [
            _row(
                f"doc-{i}.md",
                [1.0, 0.0],
                chunk="shared boilerplate content introduction overview summary",
                chunk_index=0,
            )
            for i in range(5)
        ]
        cfg.wiki_clusterer_k = 3
        store = self._store_with_rows(rows)
        with caplog.at_level(_logging.WARNING, logger="lilbee.clustering_embedding"):
            EmbeddingClusterer(cfg, store).get_clusters(min_sources=3)
        assert any("covers" in rec.message for rec in caplog.records)


class TestBuildClusters:
    def test_promotes_and_filters_communities(self):
        records = [
            _record("a.md", 0, "python typing"),
            _record("a.md", 1, "python typing"),
            _record("a.md", 2, "python typing"),
            _record("b.md", 0, "python generics"),
            _record("b.md", 1, "python generics"),
            _record("b.md", 2, "python generics"),
            _record("c.md", 0, "python protocol"),
            _record("c.md", 1, "python protocol"),
            _record("c.md", 2, "python protocol"),
            _record("noise.md", 0, "unrelated"),
        ]
        # Community 0: the typing chunks from three sources. Community 1
        # contains only the noise singleton and should be dropped.
        communities = {0: list(range(9)), 1: [9]}
        totals = _source_totals(records)
        df = _corpus_document_frequency(records)
        clusters, noise = _build_clusters(communities, records, totals, df, min_sources=3)
        assert len(clusters) == 1
        assert clusters[0].sources == frozenset({"a.md", "b.md", "c.md"})
        assert clusters[0].label  # some label survived IDF filtering
        assert noise == 1


class TestWarnIfUndersegmented:
    def test_empty_clusters_is_noop(self, caplog: pytest.LogCaptureFixture):
        import logging as _logging

        with caplog.at_level(_logging.WARNING, logger="lilbee.clustering_embedding"):
            _warn_if_undersegmented([], {"a.md": 1})
            _warn_if_undersegmented(
                [SourceCluster(cluster_id="x", label="t", sources=frozenset({"a"}))],
                {},
            )
        assert caplog.records == []


class TestConceptGraphClusterer:
    """Smoke tests for the concept-graph adapter."""

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
