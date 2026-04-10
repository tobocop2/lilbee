"""Tests for the source clustering protocol and the embedding clusterer."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from lilbee.clustering import (
    Clusterer,
    SourceCluster,
    SourceClusterer,
)
from lilbee.clustering_embedding import (
    ChunkRecord,
    EmbeddingClusterer,
    auto_k,
    communities_by_label,
    label_propagation,
    mutual_knn,
    normalize_rows,
)
from lilbee.config import ClustererBackend, cfg


@pytest.fixture(autouse=True)
def isolated_cfg():
    snapshot = cfg.model_copy()
    cfg.wiki_clusterer_k = 0
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _record(source: str, chunk_index: int = 0, text: str = "") -> ChunkRecord:
    # Tests use plain whitespace split so we don't reach into the module's
    # private _tokenize_for_tf helper. Token filtering semantics are
    # covered separately via EmbeddingClusterer.get_clusters integration.
    tokens = [word for word in text.lower().split() if len(word) >= 3]
    return ChunkRecord(
        source=source,
        chunk_index=chunk_index,
        text=text,
        tokens=tokens,
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


def _store_with_rows(rows: list[dict[str, object]]) -> MagicMock:
    store = MagicMock()
    store.open_table.return_value = _mock_table(rows)
    return store


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
            (10_000_000, 20),
        ],
    )
    def test_scales_with_corpus_size(self, n: int, expected: int) -> None:
        assert auto_k(n) == expected


class TestNormalizeRows:
    def test_drops_zero_vectors_and_normalizes(self):
        matrix = np.array([[3.0, 0.0], [0.0, 0.0], [0.0, 4.0]], dtype=np.float32)
        normalized, keep = normalize_rows(matrix)
        assert keep.tolist() == [True, False, True]
        assert normalized.shape == (2, 2)
        np.testing.assert_allclose(
            np.linalg.norm(normalized, axis=1),
            np.array([1.0, 1.0], dtype=np.float32),
            atol=1e-6,
        )

    def test_empty_matrix_is_noop(self):
        matrix = np.zeros((0, 0), dtype=np.float32)
        normalized, keep = normalize_rows(matrix)
        assert normalized.shape == (0, 0)
        assert keep.shape == (0,)


class TestMutualKnn:
    def test_two_triangles_are_mutual(self):
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
        adjacency = mutual_knn(matrix, k=2)
        for i in (0, 1, 2):
            assert any(j in adjacency[i] for j in (0, 1, 2) if j != i)
            assert all(j not in adjacency[i] for j in (3, 4, 5))

    def test_rejects_hub(self):
        cluster = np.array(
            [
                [1.0, 0.99, 0.0, 0.0, 0.0],
                [0.99, 1.0, 0.0, 0.0, 0.0],
                [0.98, 0.98, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        hub = np.ones((1, 5), dtype=np.float32)
        singleton = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        matrix = np.vstack([hub, cluster, singleton])
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        adjacency = mutual_knn(matrix, k=1)
        assert adjacency[0] == set()

    def test_empty_matrix(self):
        assert mutual_knn(np.zeros((0, 0), dtype=np.float32), k=3) == {}

    def test_k_larger_than_population(self):
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        adjacency = mutual_knn(matrix, k=10)
        assert 1 in adjacency[0]
        assert 0 in adjacency[1]

    def test_single_row_has_no_neighbors(self):
        matrix = np.array([[1.0, 0.0]], dtype=np.float32)
        adjacency = mutual_knn(matrix, k=3)
        assert adjacency == {0: set()}

    def test_k_zero_returns_empty(self):
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        assert mutual_knn(matrix, k=0) == {}


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
        labels = label_propagation(adjacency, order=list(range(6)))
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
        run_a = label_propagation(adjacency, order=[0, 1, 2, 3])
        run_b = label_propagation(adjacency, order=[0, 1, 2, 3])
        assert run_a == run_b

    def test_isolated_nodes_keep_initial_labels(self):
        adjacency = {0: set(), 1: set()}
        labels = label_propagation(adjacency, order=[0, 1])
        assert labels == [0, 1]


class TestCommunitiesByLabel:
    def test_groups_by_label(self):
        labels = [0, 0, 1, 1, 1, 2]
        communities = communities_by_label(labels)
        assert communities[0] == [0, 1]
        assert communities[1] == [2, 3, 4]
        assert communities[2] == [5]


class TestChunkRecord:
    def test_holds_cached_tokens(self):
        record = _record("doc.md", 0, "python typing")
        assert record.source == "doc.md"
        assert record.chunk_index == 0
        assert record.tokens == ["python", "typing"]


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

    def test_count_rows_exception_warns_and_falls_back_to_true(
        self, caplog: pytest.LogCaptureFixture
    ):
        store = MagicMock()
        table = MagicMock()
        table.count_rows.side_effect = RuntimeError("no can do")
        store.open_table.return_value = table
        with caplog.at_level(logging.WARNING, logger="lilbee.clustering_embedding"):
            assert EmbeddingClusterer(cfg, store).available() is True
        assert any("count_rows()" in rec.message for rec in caplog.records)


class TestEmbeddingClustererLoad:
    """Covers _load_chunk_records / _parse_chunk_row via get_clusters."""

    def test_returns_empty_when_no_table(self):
        store = MagicMock()
        store.open_table.return_value = None
        assert EmbeddingClusterer(cfg, store).get_clusters() == []

    def test_returns_empty_when_all_rows_invalid(self):
        rows: list[dict[str, object]] = [
            {"source": None, "vector": [1.0, 0.0], "chunk": "x", "chunk_index": 0},
            {"source": "a.md", "vector": None, "chunk": "x", "chunk_index": 0},
        ]
        store = _store_with_rows(rows)
        assert EmbeddingClusterer(cfg, store).get_clusters() == []

    def test_skips_invalid_rows_among_valid(self, caplog: pytest.LogCaptureFixture):
        # Mix valid + invalid rows. The one valid row should still reach
        # the clusterer, which then short-circuits because k=0 for N=1.
        rows: list[dict[str, object]] = [
            {"source": None, "vector": [1.0, 0.0], "chunk": "x", "chunk_index": 0},
            _row("ok.md", [1.0, 0.0], chunk="alpha beta gamma"),
        ]
        store = _store_with_rows(rows)
        with caplog.at_level(logging.WARNING, logger="lilbee.clustering_embedding"):
            result = EmbeddingClusterer(cfg, store).get_clusters(min_sources=3)
        assert result == []
        assert any("no mutual edges" in rec.message for rec in caplog.records)

    def test_handles_missing_chunk_text_and_index(self):
        # Row with None chunk text and missing chunk_index should default
        # to empty string and 0 respectively, not crash.
        rows: list[dict[str, object]] = [
            {"source": "a.md", "vector": [1.0, 0.0], "chunk": None, "chunk_index": None},
        ]
        store = _store_with_rows(rows)
        assert EmbeddingClusterer(cfg, store).get_clusters() == []

    def test_drops_zero_norm_rows(self):
        # Zero vectors get filtered at normalization. With everything
        # filtered, get_clusters returns empty.
        rows = [_row(f"d{i}.md", [0.0, 0.0], chunk="content") for i in range(3)]
        store = _store_with_rows(rows)
        assert EmbeddingClusterer(cfg, store).get_clusters() == []


class TestEmbeddingClustererGetClusters:
    def test_empty_store_returns_no_clusters(self):
        store = _store_with_rows([])
        assert EmbeddingClusterer(cfg, store).get_clusters() == []

    def test_two_topics_two_clusters(self):
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
                rows.append(_row(name, [1.0, 0.001 * (i * 3 + j)], chunk=text, chunk_index=j))
        for i, name in enumerate(kafka_sources):
            for j, text in enumerate(kafka_text):
                rows.append(_row(name, [0.001 * (i * 3 + j), 1.0], chunk=text, chunk_index=j))

        cfg.wiki_clusterer_k = 5
        store = _store_with_rows(rows)
        clusters = EmbeddingClusterer(cfg, store).get_clusters(min_sources=3)
        assert len(clusters) == 2
        source_sets = {cluster.sources for cluster in clusters}
        assert frozenset(typing_sources) in source_sets
        assert frozenset(kafka_sources) in source_sets
        for cluster in clusters:
            assert cluster.label
            assert cluster.cluster_id.startswith("embedding-")

    def test_label_falls_back_to_cluster_id_when_idf_eliminates_all_terms(self):
        # Three sources sharing the same vocabulary: every term appears
        # in every chunk → IDF goes non-positive for all terms → the
        # label falls back to the cluster id.
        rows: list[dict[str, object]] = []
        for i, name in enumerate(("a.md", "b.md", "c.md")):
            for j in range(3):
                rows.append(
                    _row(
                        name,
                        [1.0, 0.001 * (i * 3 + j)],
                        chunk="shared term shared term shared term",
                        chunk_index=j,
                    )
                )
        cfg.wiki_clusterer_k = 4
        store = _store_with_rows(rows)
        clusters = EmbeddingClusterer(cfg, store).get_clusters(min_sources=3)
        assert len(clusters) == 1
        assert clusters[0].label.startswith("embedding-")

    def test_rejects_stray_chunks_from_large_source(self):
        # "huge.md" has 40 chunks corpus-wide on the kafka topic, but one
        # chunk sits with the typing cluster. The stray chunk should not
        # pull huge.md into the typing cluster (cutoff = min(3, 8) = 3).
        typing_chunks = [
            _row(
                f"typing-{i // 3 + 1}.md",
                [1.0, 0.001 * i],
                chunk="python typing generics variance",
                chunk_index=i % 3,
            )
            for i in range(9)  # 3 typing docs * 3 chunks
        ]
        huge_stray = _row(
            "huge.md",
            [1.0, 0.0001],
            chunk="python typing stray",
            chunk_index=0,
        )
        huge_rest = [
            _row(
                "huge.md",
                [0.0, 1.0],
                chunk="kafka streams broker consumer",
                chunk_index=i + 1,
            )
            for i in range(39)
        ]
        rows: list[dict[str, object]] = [*typing_chunks, huge_stray, *huge_rest]
        cfg.wiki_clusterer_k = 5
        store = _store_with_rows(rows)
        clusters = EmbeddingClusterer(cfg, store).get_clusters(min_sources=3)
        # huge.md must not show up in the typing cluster even if the
        # stray chunk landed in the same community.
        for cluster in clusters:
            if any("typing" in source for source in cluster.sources):
                assert "huge.md" not in cluster.sources

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
        store = _store_with_rows(rows)
        run_a = EmbeddingClusterer(cfg, store).get_clusters(min_sources=3)
        run_b = EmbeddingClusterer(cfg, store).get_clusters(min_sources=3)
        assert [(c.sources, c.label) for c in run_a] == [(c.sources, c.label) for c in run_b]

    def test_empty_adjacency_bails_out_before_labeling(self, caplog: pytest.LogCaptureFixture):
        rows = [_row("only.md", [1.0, 0.0], chunk="alpha beta gamma delta")]
        store = _store_with_rows(rows)
        with caplog.at_level(logging.WARNING, logger="lilbee.clustering_embedding"):
            result = EmbeddingClusterer(cfg, store).get_clusters(min_sources=3)
        assert result == []
        assert any("no mutual edges" in rec.message for rec in caplog.records)

    def test_empty_text_chunks_in_label_are_skipped(self):
        # One chunk per source has empty text → tokens=[]. _label_community
        # must skip it without dropping the cluster. Every surviving
        # chunk has unique terms so TF-IDF produces a non-fallback label.
        rows: list[dict[str, object]] = [
            _row("a.md", [1.0, 0.0], chunk="widgetry fungible sprockets", chunk_index=0),
            _row("a.md", [1.0, 0.001], chunk="", chunk_index=1),
            _row("b.md", [1.0, 0.002], chunk="widgetry fungible sprockets", chunk_index=0),
            _row("b.md", [1.0, 0.003], chunk="", chunk_index=1),
            _row("c.md", [1.0, 0.004], chunk="widgetry fungible sprockets", chunk_index=0),
            _row("c.md", [1.0, 0.005], chunk="", chunk_index=1),
        ]
        cfg.wiki_clusterer_k = 3
        store = _store_with_rows(rows)
        clusters = EmbeddingClusterer(cfg, store).get_clusters(min_sources=3)
        assert len(clusters) == 1
        assert "widgetry" in clusters[0].label or "fungible" in clusters[0].label

    def test_no_clusters_meet_min_sources_still_logs_summary(
        self, caplog: pytest.LogCaptureFixture
    ):
        # Min sources=10 with only 3 sources in the corpus → every
        # community fails the threshold, _build_clusters returns [],
        # and _warn_if_undersegmented is called with an empty list
        # (exercising the early-return branch). No warning should be
        # emitted in this degenerate case.
        rows: list[dict[str, object]] = []
        for i, name in enumerate(("a.md", "b.md", "c.md")):
            for j in range(3):
                rows.append(
                    _row(
                        name,
                        [1.0, 0.001 * (i * 3 + j)],
                        chunk="alpha beta gamma delta epsilon",
                        chunk_index=j,
                    )
                )
        cfg.wiki_clusterer_k = 3
        store = _store_with_rows(rows)
        with caplog.at_level(logging.WARNING, logger="lilbee.clustering_embedding"):
            result = EmbeddingClusterer(cfg, store).get_clusters(min_sources=10)
        assert result == []
        # Warning is emitted only when an actual cluster dominates;
        # an empty cluster list must not produce one.
        assert not any("covers" in rec.message for rec in caplog.records)

    def test_warns_when_one_cluster_dominates(self, caplog: pytest.LogCaptureFixture):
        # Five sources collapsing into a single community: the
        # _warn_if_undersegmented check should fire a WARNING.
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
        store = _store_with_rows(rows)
        with caplog.at_level(logging.WARNING, logger="lilbee.clustering_embedding"):
            EmbeddingClusterer(cfg, store).get_clusters(min_sources=3)
        assert any("covers" in rec.message for rec in caplog.records)


class TestClustererFacade:
    def test_default_backend_is_embedding(self):
        cfg.wiki_clusterer = ClustererBackend.EMBEDDING
        clusterer = Clusterer(cfg, MagicMock())
        assert isinstance(clusterer.backend, EmbeddingClusterer)

    def test_concepts_choice_returns_graph_backend_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from lilbee.concepts import ConceptGraphClusterer

        cfg.wiki_clusterer = ClustererBackend.CONCEPTS
        monkeypatch.setattr(ConceptGraphClusterer, "available", lambda self: True)
        clusterer = Clusterer(cfg, MagicMock())
        assert isinstance(clusterer.backend, ConceptGraphClusterer)

    def test_concepts_choice_falls_back_when_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        from lilbee.concepts import ConceptGraphClusterer

        cfg.wiki_clusterer = ClustererBackend.CONCEPTS
        monkeypatch.setattr(ConceptGraphClusterer, "available", lambda self: False)
        clusterer = Clusterer(cfg, MagicMock())
        with caplog.at_level(logging.WARNING, logger="lilbee.clustering"):
            backend = clusterer.backend
        assert isinstance(backend, EmbeddingClusterer)
        assert any("falling back" in rec.message.lower() for rec in caplog.records)

    def test_facade_forwards_available_and_get_clusters(self, monkeypatch: pytest.MonkeyPatch):
        # Swap _select_backend out for a stub backend so the test only
        # exercises the forwarding logic without poking private attrs.
        fake_backend = MagicMock(spec=SourceClusterer)
        fake_backend.available.return_value = True
        fake_backend.get_clusters.return_value = [
            SourceCluster(cluster_id="x", label="t", sources=frozenset({"a.md"}))
        ]
        monkeypatch.setattr("lilbee.clustering._select_backend", lambda config, store: fake_backend)
        clusterer = Clusterer(cfg, MagicMock())
        assert clusterer.available() is True
        assert clusterer.get_clusters(min_sources=3)[0].cluster_id == "x"
        fake_backend.get_clusters.assert_called_once_with(min_sources=3)
        assert clusterer.backend is fake_backend

    def test_backend_cached_until_config_changes(self):
        """Repeated calls with the same config should reuse the backend."""
        cfg.wiki_clusterer = ClustererBackend.EMBEDDING
        clusterer = Clusterer(cfg, MagicMock())
        first = clusterer.backend
        second = clusterer.backend
        assert first is second

    def test_backend_rebuilds_on_config_change(self, monkeypatch: pytest.MonkeyPatch):
        """wiki_clusterer is writable at runtime, so the facade must
        re-select the backend whenever the config flips."""
        from lilbee.concepts import ConceptGraphClusterer

        cfg.wiki_clusterer = ClustererBackend.EMBEDDING
        clusterer = Clusterer(cfg, MagicMock())
        first = clusterer.backend
        assert isinstance(first, EmbeddingClusterer)

        monkeypatch.setattr(ConceptGraphClusterer, "available", lambda self: True)
        cfg.wiki_clusterer = ClustererBackend.CONCEPTS
        second = clusterer.backend
        assert isinstance(second, ConceptGraphClusterer)
        assert second is not first


class TestConceptGraphClusterer:
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
