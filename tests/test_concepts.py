"""Tests for the concept graph module.

All heavy deps (spacy, graspologic-native) are mocked at the boundary so these
tests run without the ``graph`` extra installed.
"""

from dataclasses import fields
from unittest.mock import MagicMock, patch

import pytest

import lilbee.app.services as svc_mod
from lilbee.core.config import cfg
from lilbee.data.store import SearchChunk
from lilbee.retrieval.concepts import ConceptGraph


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
            from lilbee.retrieval.concepts import concepts_available

            assert concepts_available() is True

    def test_returns_false_when_not_installed(self):
        with patch.dict("sys.modules", {"spacy": None}):
            from lilbee.retrieval.concepts import concepts_available

            assert concepts_available() is False


class TestExtractConcepts:
    @patch("lilbee.retrieval.concepts.graph._ensure_spacy_model")
    def test_basic_extraction(self, mock_spacy, cg):
        mock_spacy.return_value = _make_mock_nlp(
            {"hello world": ["machine learning", "neural networks"]}
        )
        result = cg.extract_concepts("hello world")
        assert result == ["machine learning", "neural networks"]

    @patch("lilbee.retrieval.concepts.graph._ensure_spacy_model")
    def test_deduplication(self, mock_spacy, cg):
        mock_spacy.return_value = _make_mock_nlp({"text": ["Concept", "concept", "Other"]})
        result = cg.extract_concepts("text")
        assert result == ["concept", "other"]

    @patch("lilbee.retrieval.concepts.graph._ensure_spacy_model")
    def test_max_cap(self, mock_spacy, cg):
        mock_spacy.return_value = _make_mock_nlp({"text": ["alpha", "beta", "gamma", "delta"]})
        result = cg.extract_concepts("text", max_concepts=2)
        assert len(result) == 2

    @patch("lilbee.retrieval.concepts.graph._ensure_spacy_model")
    def test_empty_input(self, mock_spacy, cg):
        result = cg.extract_concepts("")
        assert result == []
        mock_spacy.assert_not_called()

    @patch("lilbee.retrieval.concepts.graph._ensure_spacy_model")
    def test_filters_short_concepts(self, mock_spacy, cg):
        """Sub-three-char fragments are rejected by ``is_valid_label``.

        Three-char and longer PDF-split noise (``cro``, ``fus``) is
        intentionally NOT caught by the length gate; A3's entity-type
        filter and the ``wiki_entity_min_mentions`` threshold catch it
        downstream, and tightening the length gate further would reject
        legitimate short labels like ``CPU`` or ``API``.
        """
        mock_spacy.return_value = _make_mock_nlp(
            {"text": ["a", "ok", "good concept", "brake pads"]}
        )
        result = cg.extract_concepts("text")
        assert "a" not in result
        assert "ok" not in result
        assert "good concept" in result
        assert "brake pads" in result

    @patch("lilbee.retrieval.concepts.graph._ensure_spacy_model")
    def test_filters_structural_noise_concepts(self, mock_spacy, cg):
        """QA-driven (bb-8b7s): markdown table, page-number, and
        pipe-prefixed concepts never enter the graph."""
        mock_spacy.return_value = _make_mock_nlp(
            {"text": ["| | body", "158 vehicle", "chevrolet caprice"]}
        )
        result = cg.extract_concepts("text")
        assert "| | body" not in result
        assert "158 vehicle" not in result
        assert "chevrolet caprice" in result

    @patch("lilbee.retrieval.concepts.graph._ensure_spacy_model")
    def test_holds_nlp_lock_during_processing(self, mock_spacy, cg):
        """The shared (non-thread-safe) spaCy Language is used under _nlp_lock."""
        locked_during: list[bool] = []
        nlp = MagicMock()

        def call_fn(text):
            locked_during.append(cg._nlp_lock.locked())
            return _make_mock_doc(["good concept"])

        nlp.side_effect = call_fn
        mock_spacy.return_value = nlp

        cg.extract_concepts("text")
        assert locked_during == [True]  # lock held while nlp() runs
        assert not cg._nlp_lock.locked()  # released afterwards


class TestExtractConceptsBatch:
    @patch("lilbee.retrieval.concepts.graph._ensure_spacy_model")
    def test_batch_extraction(self, mock_spacy, cg):
        mock_spacy.return_value = _make_mock_nlp(
            {"doc1": ["concept a"], "doc2": ["concept b", "concept c"]}
        )
        result = cg.extract_concepts_batch(["doc1", "doc2"])
        assert len(result) == 2
        assert result[0] == ["concept a"]
        assert result[1] == ["concept b", "concept c"]

    @patch("lilbee.retrieval.concepts.graph._ensure_spacy_model")
    def test_empty_input(self, mock_spacy, cg):
        result = cg.extract_concepts_batch([])
        assert result == []
        mock_spacy.assert_not_called()

    @patch("lilbee.retrieval.concepts.graph._ensure_spacy_model")
    def test_batch_filters_short_concepts(self, mock_spacy, cg):
        mock_spacy.return_value = _make_mock_nlp({"text": ["a", "ok", "good"]})
        result = cg.extract_concepts_batch(["text"])
        assert result == [["good"]]

    @patch("lilbee.retrieval.concepts.graph._ensure_spacy_model")
    def test_batch_deduplicates(self, mock_spacy, cg):
        mock_spacy.return_value = _make_mock_nlp({"text": ["Alpha", "alpha", "Beta"]})
        result = cg.extract_concepts_batch(["text"])
        assert result == [["alpha", "beta"]]

    @patch("lilbee.retrieval.concepts.graph._ensure_spacy_model")
    def test_batch_caps_at_max(self, mock_spacy, cg):
        cfg.concept_max_per_chunk = 2
        mock_spacy.return_value = _make_mock_nlp({"text": ["alpha", "beta", "gamma", "delta"]})
        result = cg.extract_concepts_batch(["text"])
        assert len(result[0]) == 2

    @patch("lilbee.retrieval.concepts.graph._ensure_spacy_model")
    def test_holds_nlp_lock_across_pipe(self, mock_spacy, cg):
        """nlp.pipe is lazy, so the lock must stay held across the iteration."""
        locked_during: list[bool] = []
        nlp = MagicMock()
        nlp.side_effect = lambda text: _make_mock_doc([])

        # A generator (like real spaCy nlp.pipe): each doc is produced during
        # iteration, so the lock must still be held as the comprehension pulls
        # items -- not merely when pipe() is first called.
        def pipe_fn(texts):
            for _ in texts:
                locked_during.append(cg._nlp_lock.locked())
                yield _make_mock_doc(["good concept"])

        nlp.pipe = pipe_fn
        mock_spacy.return_value = nlp

        cg.extract_concepts_batch(["a", "b", "c"])
        assert locked_during == [True, True, True]  # held for every yielded doc
        assert not cg._nlp_lock.locked()


class TestGetNlp:
    @patch("lilbee.retrieval.concepts.graph._ensure_spacy_model")
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
            from lilbee.retrieval.concepts.nlp import _ensure_spacy_model

            result = _ensure_spacy_model()
            mock_spacy.load.assert_called_once_with("en_core_web_sm")
            assert result is not None

    def test_raises_import_error_with_install_hint_when_model_missing(self):
        """Missing spaCy model emits a manual-install hint rather than shelling out.

        Auto-download via ``spacy.cli.download`` surfaced uv stderr in the
        chat panel under ``uv tool install`` layouts; we now degrade
        gracefully and instruct the user to run the install themselves.
        """
        mock_spacy = MagicMock()
        mock_spacy.load.side_effect = OSError("not found")
        with patch.dict("sys.modules", {"spacy": mock_spacy}):
            from lilbee.retrieval.concepts.nlp import _ensure_spacy_model

            with pytest.raises(ImportError, match="python -m spacy download en_core_web_sm"):
                _ensure_spacy_model()

    def test_load_spacy_pipeline_delegates_to_ensure(self):
        """Public wrapper just forwards to the private loader."""
        from lilbee.retrieval.concepts import load_spacy_pipeline

        sentinel = object()
        with patch(
            "lilbee.retrieval.concepts.nlp._ensure_spacy_model", return_value=sentinel
        ) as ensure:
            assert load_spacy_pipeline() is sentinel
            ensure.assert_called_once_with()


class TestGracefulDegradation:
    @patch(
        "lilbee.retrieval.concepts.graph._ensure_spacy_model", side_effect=ImportError("no model")
    )
    def test_ensure_nlp_returns_none_on_failure(self, mock_spacy, cg):
        assert cg._ensure_nlp() is None

    @patch(
        "lilbee.retrieval.concepts.graph._ensure_spacy_model", side_effect=ImportError("no model")
    )
    def test_caches_failure_state(self, mock_spacy, cg):
        cg._ensure_nlp()
        cg._ensure_nlp()
        mock_spacy.assert_called_once()

    @patch("lilbee.retrieval.concepts.graph._ensure_spacy_model")
    def test_caches_successful_load(self, mock_spacy, cg):
        mock_nlp = MagicMock()
        mock_spacy.return_value = mock_nlp
        assert cg._ensure_nlp() is mock_nlp
        assert cg._ensure_nlp() is mock_nlp
        mock_spacy.assert_called_once()

    @patch(
        "lilbee.retrieval.concepts.graph._ensure_spacy_model", side_effect=ImportError("no model")
    )
    def test_extract_concepts_returns_empty(self, mock_spacy, cg):
        assert cg.extract_concepts("some text about python") == []

    @patch(
        "lilbee.retrieval.concepts.graph._ensure_spacy_model", side_effect=ImportError("no model")
    )
    def test_extract_concepts_batch_returns_empty_lists(self, mock_spacy, cg):
        result = cg.extract_concepts_batch(["text one", "text two"])
        assert result == [[], []]

    @patch(
        "lilbee.retrieval.concepts.graph._ensure_spacy_model", side_effect=ImportError("no model")
    )
    def test_expand_query_returns_empty(self, mock_spacy, cg):
        assert cg.expand_query("python frameworks") == []


class TestBuildConceptRecords:
    def test_builds_rows_without_store_access(self, cg, mock_svc):
        chunk_ids = [("doc.md", 0), ("doc.md", 1)]
        concept_lists = [["python", "machine learning"], ["python", "deep learning"]]
        records = cg.build_concept_records(chunk_ids, concept_lists)
        node_by_concept = {n["concept"]: n for n in records.nodes}
        assert set(node_by_concept) == {"python", "machine learning", "deep learning"}
        assert node_by_concept["python"]["degree"] == 2
        assert {(e["source"], e["target"]) for e in records.edges} == {
            ("machine learning", "python"),
            ("deep learning", "python"),
        }
        assert len(records.chunk_concepts) == 4
        mock_svc.store.get_db.assert_not_called()

    def test_build_empty_chunks_returns_empty_records(self, cg):
        records = cg.build_concept_records([], [])
        assert (records.nodes, records.edges, records.chunk_concepts) == ([], [], [])

    def test_merged_concatenates_per_file_records(self, cg):
        from lilbee.data.store import ConceptRecords

        per_file = [
            cg.build_concept_records([("a.md", 0)], [["python", "rust"]]),
            cg.build_concept_records([("b.md", 0)], [["python", "go"]]),
        ]
        merged = ConceptRecords.merged(per_file)
        assert merged.nodes == per_file[0].nodes + per_file[1].nodes
        assert merged.edges == per_file[0].edges + per_file[1].edges
        assert merged.chunk_concepts == per_file[0].chunk_concepts + per_file[1].chunk_concepts


class TestWriteConceptRecords:
    def _records(self):
        from lilbee.data.store import ConceptRecords

        return ConceptRecords(
            nodes=[{"concept": "python", "cluster_id": 0, "degree": 1}],
            edges=[{"source": "a", "target": "b", "weight": 1.0}],
            chunk_concepts=[{"chunk_source": "doc.md", "chunk_index": 0, "concept": "python"}],
        )

    @patch("lilbee.runtime.lock.write_lock")
    @patch("lilbee.data.store.ensure_table")
    def test_one_add_per_table(self, mock_ensure, mock_lock, cg, mock_svc):
        mock_lock.return_value.__enter__ = MagicMock()
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        tables = [MagicMock(), MagicMock(), MagicMock()]
        mock_ensure.side_effect = tables
        mock_svc.store.get_db.return_value = MagicMock()

        cg.write_concept_records(self._records())
        for table in tables:
            table.add.assert_called_once()

    @patch("lilbee.runtime.lock.write_lock")
    @patch("lilbee.data.store.ensure_table")
    def test_empty_records_still_create_tables(self, mock_ensure, mock_lock, cg, mock_svc):
        from lilbee.data.store import ConceptRecords

        mock_lock.return_value.__enter__ = MagicMock()
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        tables = [MagicMock(), MagicMock(), MagicMock()]
        mock_ensure.side_effect = tables
        mock_svc.store.get_db.return_value = MagicMock()

        cg.write_concept_records(ConceptRecords(nodes=[], edges=[], chunk_concepts=[]))
        assert mock_ensure.call_count == 3
        for table in tables:
            table.add.assert_not_called()

    @patch("lilbee.retrieval.concepts.graph._leiden_partition")
    def test_batched_write_rebuilds_same_clusters_as_per_file(self, mock_leiden, mock_svc):
        """End-state regression: one merged write equals per-file writes."""
        from lilbee.core.config import CONCEPT_NODES_TABLE
        from lilbee.data.store import ConceptRecords, Store
        from lilbee.retrieval.concepts import ConceptGraph

        per_file = [
            ConceptGraph(cfg, MagicMock()).build_concept_records([(name, 0)], [concepts])
            for name, concepts in (("a.md", ["python", "rust"]), ("b.md", ["python", "go"]))
        ]
        mock_leiden.side_effect = lambda edge_rows: (
            {c: 0 for row in edge_rows for c in (row["source"], row["target"])},
            {c: 1 for row in edge_rows for c in (row["source"], row["target"])},
        )

        def _rebuild(write_units: list[ConceptRecords], lancedb_dir) -> list[dict]:
            cfg.lancedb_dir = lancedb_dir
            store = Store(cfg)
            graph = ConceptGraph(cfg, store)
            for unit in write_units:
                graph.write_concept_records(unit)
            graph.rebuild_clusters()
            table = store.open_table(CONCEPT_NODES_TABLE)
            rows = table.search().limit(None).to_list()
            return sorted(rows, key=lambda r: r["concept"])

        per_file_rows = _rebuild(per_file, cfg.data_dir / "per_file")
        batched_rows = _rebuild([ConceptRecords.merged(per_file)], cfg.data_dir / "batched")
        assert batched_rows == per_file_rows
        assert len(batched_rows) > 0


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

    def test_boost_respects_floor(self, cg, mock_svc):
        """Concept boost cannot reduce distance below concept_boost_floor."""
        results = [_make_result(distance=0.1, chunk_index=0)]
        mock_table = MagicMock()
        mock_table.search.return_value.where.return_value.to_list.return_value = [
            {"concept": "python"},
            {"concept": "ml"},
        ]
        mock_svc.store.open_table.return_value = mock_table
        boosted = cg.boost_results(results, ["python", "ml"])
        assert boosted[0].distance >= cfg.concept_boost_floor


class TestExpandQuery:
    @patch("lilbee.retrieval.concepts.graph._ensure_spacy_model")
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

    @patch("lilbee.retrieval.concepts.graph._ensure_spacy_model")
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

    def test_single_batched_query_per_depth_level(self, cg, mock_svc):
        """Frontier expansion must fire one query per depth level, not one per node."""
        mock_table = MagicMock()
        mock_table.search.return_value.where.return_value.to_list.return_value = [
            {"source": "python", "target": "django", "weight": 1.0},
            {"source": "python", "target": "flask", "weight": 0.9},
        ]
        mock_svc.store.open_table.return_value = mock_table

        cg.get_related_concepts("python", depth=1)
        # One depth level => exactly one .search() call.
        assert mock_table.search.call_count == 1
        # The WHERE clause should use IN with all frontier nodes, not per-node equality.
        where_args = mock_table.search.return_value.where.call_args.args[0]
        assert " IN (" in where_args
        assert "'python'" in where_args

    def test_depth_two_batches_both_levels(self, cg, mock_svc):
        """depth=2 triggers exactly 2 batched queries (one per level)."""
        mock_table = MagicMock()
        mock_table.search.return_value.where.return_value.to_list.side_effect = [
            [
                {"source": "python", "target": "django", "weight": 1.0},
                {"source": "python", "target": "flask", "weight": 0.9},
            ],
            [
                {"source": "django", "target": "rest", "weight": 0.8},
                {"source": "flask", "target": "werkzeug", "weight": 0.7},
            ],
        ]
        mock_svc.store.open_table.return_value = mock_table

        related = cg.get_related_concepts("python", depth=2)
        assert mock_table.search.call_count == 2
        # Level-2 WHERE clause should include both frontier nodes found at level 1.
        level_two_where = mock_table.search.return_value.where.call_args_list[1].args[0]
        assert "'django'" in level_two_where
        assert "'flask'" in level_two_where
        # All neighbors from both levels end up in the result.
        assert set(related) >= {"django", "flask", "rest", "werkzeug"}

    def test_empty_frontier_mid_traversal_breaks(self, cg, mock_svc):
        """Depth > 1 but no new neighbors at level 1 → loop breaks on empty frontier."""
        mock_table = MagicMock()
        mock_table.search.return_value.where.return_value.to_list.return_value = []
        mock_svc.store.open_table.return_value = mock_table

        related = cg.get_related_concepts("isolated", depth=3)
        assert related == []
        # depth=3 would normally fire 3 queries; empty frontier after level 1
        # breaks out after the first.
        assert mock_table.search.call_count == 1


class TestTopCommunities:
    def test_top_communities(self, cg, mock_svc):
        import pyarrow as pa

        mock_table = MagicMock()
        # Real Arrow table so pyarrow.compute ops actually run.
        mock_table.to_arrow.return_value = pa.table(
            {
                "concept": ["python", "ml", "web"],
                "cluster_id": [0, 0, 1],
                "degree": [5, 3, 2],
            }
        )
        mock_svc.store.open_table.return_value = mock_table

        communities = cg.top_communities(k=2)
        assert len(communities) == 2
        assert communities[0].size == 2
        assert communities[0].cluster_id == 0

    def test_top_communities_no_table(self, cg, mock_svc):
        mock_svc.store.open_table.return_value = None
        assert cg.top_communities() == []

    def test_top_communities_empty_table(self, cg, mock_svc):
        import pyarrow as pa

        mock_table = MagicMock()
        mock_table.to_arrow.return_value = pa.table(
            {"concept": [], "cluster_id": [], "degree": []},
            schema=pa.schema(
                [
                    pa.field("concept", pa.utf8()),
                    pa.field("cluster_id", pa.int32()),
                    pa.field("degree", pa.int32()),
                ]
            ),
        )
        mock_svc.store.open_table.return_value = mock_table
        assert cg.top_communities() == []

    def test_top_communities_only_null_cluster_ids(self, cg, mock_svc):
        """Rows with NULL cluster_id yield no top clusters."""
        import pyarrow as pa

        mock_table = MagicMock()
        mock_table.to_arrow.return_value = pa.table(
            {
                "concept": ["python", "ml"],
                "cluster_id": pa.array([None, None], type=pa.int32()),
                "degree": [5, 3],
            }
        )
        mock_svc.store.open_table.return_value = mock_table
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
        import pyarrow as pa

        mock_table = MagicMock()
        mock_table.to_arrow.return_value = pa.table(
            {"source": [], "target": [], "weight": []},
            schema=pa.schema(
                [
                    pa.field("source", pa.utf8()),
                    pa.field("target", pa.utf8()),
                    pa.field("weight", pa.float64()),
                ]
            ),
        )
        mock_svc.store.open_table.return_value = mock_table
        cg.rebuild_clusters()

    @patch("lilbee.runtime.lock.write_lock")
    @patch("lilbee.data.store.ensure_table")
    @patch("lilbee.retrieval.concepts.graph._leiden_partition")
    def test_rebuild_with_edges(self, mock_leiden, mock_ensure, mock_lock, cg, mock_svc):
        import pyarrow as pa

        mock_lock.return_value.__enter__ = MagicMock()
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        mock_table = MagicMock()
        edge_rows = [
            {"source": "python", "target": "ml", "weight": 2.0},
            {"source": "ml", "target": "deep learning", "weight": 1.5},
        ]
        mock_table.to_arrow.return_value = pa.table(
            {
                "source": [r["source"] for r in edge_rows],
                "target": [r["target"] for r in edge_rows],
                "weight": [r["weight"] for r in edge_rows],
            }
        )
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

    @patch("lilbee.runtime.lock.write_lock")
    @patch("lilbee.data.store.ensure_table")
    @patch("lilbee.retrieval.concepts.graph._leiden_partition")
    def test_rebuild_aggregates_duplicate_edges(
        self, mock_leiden, mock_ensure, mock_lock, cg, mock_svc
    ):
        """Per-file ingest appends duplicate edge rows; rebuild sums them into one."""
        import pyarrow as pa

        mock_lock.return_value.__enter__ = MagicMock()
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        mock_table = MagicMock()
        mock_table.to_arrow.return_value = pa.table(
            {
                "source": ["python", "python", "ml"],
                "target": ["ml", "ml", "web"],
                "weight": [2.0, 1.0, 0.5],
            }
        )
        mock_svc.store.open_table.return_value = mock_table
        mock_svc.store.get_db.return_value = MagicMock()
        mock_leiden.return_value = ({"python": 0}, {"python": 1})

        cg.rebuild_clusters()
        passed = mock_leiden.call_args.args[0]
        assert passed == [
            {"source": "python", "target": "ml", "weight": 3.0},
            {"source": "ml", "target": "web", "weight": 0.5},
        ]

    @patch("lilbee.runtime.lock.write_lock")
    @patch("lilbee.data.store.ensure_table")
    @patch("lilbee.retrieval.concepts.graph._leiden_partition")
    def test_rebuild_compacts_concept_tables(
        self, mock_leiden, mock_ensure, mock_lock, cg, mock_svc
    ):
        """rebuild_clusters ends with optimize() on every concept table."""
        import pyarrow as pa

        mock_lock.return_value.__enter__ = MagicMock()
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        mock_table = MagicMock()
        mock_table.to_arrow.return_value = pa.table(
            {"source": ["a"], "target": ["b"], "weight": [1.0]}
        )
        mock_svc.store.open_table.return_value = mock_table
        mock_svc.store.get_db.return_value = MagicMock()
        mock_leiden.return_value = ({"a": 0}, {"a": 1})

        cg.rebuild_clusters()
        # compact_tables opens the three concept tables; open_table returns the
        # same mock for every name here, so optimize fires once per table.
        assert mock_table.optimize.call_count == 3

    def test_compact_tables_survives_optimize_failure(self, cg, mock_svc):
        mock_table = MagicMock()
        mock_table.optimize.side_effect = RuntimeError("compaction failed")
        mock_svc.store.open_table.return_value = mock_table
        cg.compact_tables()
        assert mock_table.optimize.call_count == 3

    def test_compact_tables_skips_missing_tables(self, cg, mock_svc):
        mock_svc.store.open_table.return_value = None
        cg.compact_tables()
        assert mock_svc.store.open_table.call_count == 3


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

        from lilbee.retrieval.concepts.community import _compute_pmi

        cooccurrences = Counter({("a", "b"): 5})
        concept_counts = Counter({"a": 8, "b": 6})
        pmi = _compute_pmi(cooccurrences, concept_counts, 10)
        assert ("a", "b") in pmi
        # PPMI: all values >= 0
        assert pmi[("a", "b")] >= 0.0

    def test_ppmi_clamps_negative(self):
        """Anti-correlated pairs should get PPMI = 0."""
        from collections import Counter

        from lilbee.retrieval.concepts.community import _compute_pmi

        # a and b rarely co-occur but each appear often -> negative PMI -> clamped to 0
        cooccurrences = Counter({("a", "b"): 1})
        concept_counts = Counter({"a": 9, "b": 9})
        pmi = _compute_pmi(cooccurrences, concept_counts, 10)
        assert pmi[("a", "b")] == 0.0

    def test_ppmi_skips_zero_count_concepts(self):
        """Concepts with zero count are skipped (avoid division by zero)."""
        from collections import Counter

        from lilbee.retrieval.concepts.community import _compute_pmi

        cooccurrences = Counter({("a", "b"): 1})
        concept_counts = Counter({"a": 0, "b": 5})
        pmi = _compute_pmi(cooccurrences, concept_counts, 10)
        assert ("a", "b") not in pmi


class TestLeidenPartition:
    def test_returns_partition_and_degrees(self):
        mock_graspologic = MagicMock()
        mock_graspologic.leiden.return_value = (0.5, {"a": 0, "b": 0, "c": 1})
        with patch.dict("sys.modules", {"graspologic_native": mock_graspologic}):
            from lilbee.retrieval.concepts.community import _leiden_partition

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
            from lilbee.retrieval.concepts.community import _MIN_LEIDEN_WEIGHT, _leiden_partition

            edge_rows = [{"source": "a", "target": "b", "weight": 0.0}]
            _leiden_partition(edge_rows)
            call_args = mock_graspologic.leiden.call_args
            edges_passed = call_args[1]["edges"]
            assert edges_passed[0][2] == _MIN_LEIDEN_WEIGHT


class TestCommunityDataclass:
    def test_community_fields(self):
        from lilbee.retrieval.concepts import Community

        c = Community(cluster_id=0, size=3, concepts=["a", "b", "c"])
        assert c.cluster_id == 0
        assert c.size == 3
        assert c.concepts == ["a", "b", "c"]

    def test_community_is_dataclass(self):
        from lilbee.retrieval.concepts import Community

        assert len(fields(Community)) == 3


class TestGetClusterSources:
    def test_returns_clusters_spanning_min_sources(self, cg, mock_svc):
        import pyarrow as pa

        nodes_table = MagicMock()
        nodes_table.to_arrow.return_value = pa.table(
            {
                "concept": ["python", "ml", "web"],
                "cluster_id": [0, 0, 1],
                "degree": [3, 2, 1],
            }
        )
        cc_table = MagicMock()
        cc_table.to_arrow.return_value = pa.table(
            {
                "chunk_source": ["a.md", "b.md", "c.md", "d.md"],
                "chunk_index": [0, 0, 0, 0],
                "concept": ["python", "python", "ml", "web"],
            }
        )

        def open_table(name):
            from lilbee.core.config import CHUNK_CONCEPTS_TABLE, CONCEPT_NODES_TABLE

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
        import pyarrow as pa

        nodes_table = MagicMock()
        nodes_table.to_arrow.return_value = pa.table(
            {"concept": ["python"], "cluster_id": [0], "degree": [3]}
        )
        cc_table = MagicMock()
        cc_table.to_arrow.return_value = pa.table(
            {
                "chunk_source": ["a.md", "b.md"],
                "chunk_index": [0, 0],
                "concept": ["python", "orphan_concept"],
            }
        )

        def open_table(name):
            from lilbee.core.config import CHUNK_CONCEPTS_TABLE, CONCEPT_NODES_TABLE

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
        import pyarrow as pa

        nodes_table = MagicMock()
        nodes_table.to_arrow.return_value = pa.table(
            {"concept": ["python"], "cluster_id": [0], "degree": [1]}
        )
        cc_table = MagicMock()
        cc_table.to_arrow.return_value = pa.table(
            {"chunk_source": ["a.md"], "chunk_index": [0], "concept": ["python"]}
        )

        def open_table(name):
            from lilbee.core.config import CHUNK_CONCEPTS_TABLE, CONCEPT_NODES_TABLE

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
        # New implementation pushes the cluster_id filter to the DB, so
        # the mock only needs to return the rows for that cluster.
        mock_table.search.return_value.where.return_value.to_list.return_value = [
            {"concept": "python", "cluster_id": 0, "degree": 5},
            {"concept": "ml", "cluster_id": 0, "degree": 3},
        ]
        mock_svc.store.open_table.return_value = mock_table
        assert cg.get_cluster_label(0) == "python"
        # Confirm the filter was actually pushed down.
        where_args = mock_table.search.return_value.where.call_args.args[0]
        assert "cluster_id = 0" in where_args

    def test_returns_fallback_when_no_table(self, cg, mock_svc):
        mock_svc.store.open_table.return_value = None
        assert cg.get_cluster_label(42) == "cluster-42"

    def test_returns_fallback_for_unknown_cluster(self, cg, mock_svc):
        mock_table = MagicMock()
        # Unknown cluster => DB returns zero rows for the WHERE clause.
        mock_table.search.return_value.where.return_value.to_list.return_value = []
        mock_svc.store.open_table.return_value = mock_table
        assert cg.get_cluster_label(99) == "cluster-99"

    def test_returns_fallback_on_query_exception(self, cg, mock_svc):
        mock_table = MagicMock()
        mock_table.search.return_value.where.return_value.to_list.side_effect = RuntimeError(
            "query failed"
        )
        mock_svc.store.open_table.return_value = mock_table
        assert cg.get_cluster_label(7) == "cluster-7"


class TestFilterNounChunks:
    def test_filter_noun_chunks(self):
        from lilbee.retrieval.concepts.nlp import _filter_noun_chunks

        doc = _make_mock_doc(["Hello World", "a", "Good Stuff", "Hello World"])
        result = _filter_noun_chunks(doc, max_concepts=10)
        assert result == ["hello world", "good stuff"]

    def test_filter_noun_chunks_max(self):
        from lilbee.retrieval.concepts.nlp import _filter_noun_chunks

        doc = _make_mock_doc(["alpha", "beta", "gamma"])
        result = _filter_noun_chunks(doc, max_concepts=2)
        assert len(result) == 2

    def test_filter_noun_chunks_rejects_structural_noise(self):
        """Direct coverage of the structural-rejection branch inside
        ``_filter_noun_chunks`` (bb-8b7s). Asserts that the module's
        own filter drops the table, page-number, and paren-prefix
        patterns even without going through ``extract_concepts``.
        """
        from lilbee.retrieval.concepts.nlp import _filter_noun_chunks

        doc = _make_mock_doc(
            [
                "| | body",
                "158 vehicle",
                "(7.0 l)",
                "-answers",
                "chevrolet caprice",
            ]
        )
        result = _filter_noun_chunks(doc, max_concepts=10)
        assert result == ["chevrolet caprice"]
