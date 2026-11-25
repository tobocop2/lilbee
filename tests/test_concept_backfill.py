"""Turning the concept graph on builds it for the documents already indexed.

A real store and a real concept graph; spaCy and Leiden are stand-ins.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest

from lilbee.app.services import set_services
from lilbee.core.config import CHUNK_CONCEPTS_TABLE, CONCEPT_NODES_TABLE, cfg
from lilbee.data.ingest import sync
from lilbee.data.store import SourceType, Store
from lilbee.retrieval.concepts import ConceptGraph
from tests.conftest import make_mock_services

DIM = 8
# Each document has its own vocabulary, so its concepts co-occur above chance.
WORDS = {
    "a.txt": "apple banana cherry",
    "b.txt": "durian elder fig",
    "c.txt": "grape hazel iris",
}


class _FakeNlp:
    """spaCy stand-in: each known word of a text is one noun chunk."""

    vocabulary = frozenset(word for words in WORDS.values() for word in words.split())

    def __init__(self) -> None:
        self.texts: list[str] = []

    def _doc(self, text: str) -> SimpleNamespace:
        stripped = (word.strip(".") for word in text.split())
        words = [word for word in dict.fromkeys(stripped) if word in self.vocabulary]
        return SimpleNamespace(noun_chunks=[SimpleNamespace(text=word) for word in words])

    def __call__(self, text: str) -> SimpleNamespace:
        return self._doc(text)

    def pipe(self, texts):
        self.texts.extend(texts)
        return [self._doc(text) for text in texts]


# The community the Leiden stand-in assigns. A concept write puts each node in
# community 0, so a node in this one was clustered.
CLUSTERED = 7


def _one_community(edge_rows):
    """Leiden stand-in: every concept in community CLUSTERED, degree = its edge count."""
    degrees: dict[str, int] = {}
    for row in edge_rows:
        for node in (row["source"], row["target"]):
            degrees[node] = degrees.get(node, 0) + 1
    return {node: CLUSTERED for node in degrees}, degrees


@pytest.fixture(autouse=True)
def isolated_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path / "global-data"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    snapshot = cfg.model_copy()
    cfg.documents_dir = tmp_path / "docs"
    cfg.documents_dir.mkdir()
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.concept_graph = False
    cfg.embedding_dim = DIM
    with (
        patch("lilbee.retrieval.concepts.concepts_available", return_value=True),
        patch("lilbee.retrieval.concepts.graph._leiden_partition", side_effect=_one_community),
    ):
        yield
    set_services(None)
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.validate_model.return_value = True
    embedder.embedding_available.return_value = True
    embedder.embed_batch.side_effect = lambda texts, **kw: [[0.1] * DIM for _ in texts]
    return embedder


class _Library:
    """A real store and concept graph behind the services a sync reads."""

    def __init__(self) -> None:
        self.store = Store(cfg)
        self.graph = ConceptGraph(cfg, self.store)
        self.nlp = _FakeNlp()
        self.graph._nlp = self.nlp
        set_services(
            make_mock_services(
                store=self.store,
                embedder=_embedder(),
                concepts=self.graph,
                worker_pool=MagicMock(),
                pool_runtime=MagicMock(),
                pool_health_ticker=MagicMock(),
            )
        )

    @staticmethod
    def add(*names: str) -> None:
        for name in names:
            (cfg.documents_dir / name).write_text(f"{WORDS[name]}. " * 40, encoding="utf-8")

    def sources_with_concepts(self) -> set[str]:
        table = self.store.open_table(CHUNK_CONCEPTS_TABLE)
        if table is None:
            return set()
        ids = {row["source_id"] for row in table.search().limit(None).to_list()}
        return {name for name, i in self._ids().items() if i in ids}

    def _ids(self) -> dict[str, str]:
        return self.store.source_ids(list(WORDS), SourceType.DOCUMENT)

    @staticmethod
    def add_blank(name: str) -> None:
        """A document in which the stand-in finds no concept."""
        (cfg.documents_dir / name).write_text("zzz qqq. " * 40, encoding="utf-8")

    def clustered_concepts(self) -> set[str]:
        table = self.store.open_table(CONCEPT_NODES_TABLE)
        if table is None:
            return set()
        return {row["concept"] for row in table.search().limit(None).to_list()}

    def node_rows(self) -> list[tuple[str, int, int]]:
        """(concept, cluster id, degree) of every node row, sorted."""
        table = self.store.open_table(CONCEPT_NODES_TABLE)
        rows = table.search().limit(None).to_list()
        return sorted((row["concept"], row["cluster_id"], row["degree"]) for row in rows)

    def extracted_words(self) -> set[str]:
        words = {word.strip(".") for text in self.nlp.texts for word in text.split()}
        return words & _FakeNlp.vocabulary


async def test_turning_the_graph_on_builds_it_for_the_indexed_documents():
    library = _Library()
    library.add("a.txt", "b.txt")
    await sync(quiet=True)
    assert library.sources_with_concepts() == set()

    cfg.concept_graph = True
    await sync(quiet=True)

    assert library.sources_with_concepts() == {"a.txt", "b.txt"}
    # One clustered row per concept: the recluster replaced the rows each write added.
    words = sorted(f"{WORDS['a.txt']} {WORDS['b.txt']}".split())
    assert library.node_rows() == [(word, CLUSTERED, 2) for word in words]


async def test_the_backfill_logs_its_start_each_batch_and_its_end(caplog):
    library = _Library()
    library.add("a.txt", "b.txt", "c.txt")
    await sync(quiet=True)
    cfg.concept_graph = True

    with (
        patch("lilbee.retrieval.concepts.backfill._SOURCES_PER_BATCH", 2),
        caplog.at_level(logging.INFO, logger="lilbee.retrieval.concepts.backfill"),
    ):
        await sync(quiet=True)

    assert [r.getMessage() for r in caplog.records if r.name.endswith("backfill")] == [
        "Building concept rows for 3 sources indexed without them",
        "Concept rows built for 2 of 3 sources",
        "Concept rows built for 3 of 3 sources",
    ]


async def test_a_failed_recluster_after_the_backfill_runs_on_the_next_sync():
    library = _Library()
    library.add("a.txt", "b.txt")
    await sync(quiet=True)
    cfg.concept_graph = True
    with patch.object(ConceptGraph, "_rebuild_all", side_effect=RuntimeError("stopped")):
        await sync(quiet=True)
    assert library.store.concepts_covered() is None
    assert {cluster for _, cluster, _ in library.node_rows()} == {0}

    await sync(quiet=True)

    words = sorted(f"{WORDS['a.txt']} {WORDS['b.txt']}".split())
    assert library.node_rows() == [(word, CLUSTERED, 2) for word in words]
    assert library.store.concepts_covered().version == library.store.sources_version()


async def test_a_failed_recluster_after_a_later_backfill_runs_on_the_next_sync():
    library = _Library()
    library.add("a.txt")
    cfg.concept_graph = True
    await sync(quiet=True)
    cfg.concept_graph = False
    library.add("b.txt")
    await sync(quiet=True)
    cfg.concept_graph = True
    with patch.object(ConceptGraph, "_rebuild_all", side_effect=RuntimeError("stopped")):
        await sync(quiet=True)
    assert library.sources_with_concepts() == {"a.txt", "b.txt"}

    await sync(quiet=True)

    words = sorted(f"{WORDS['a.txt']} {WORDS['b.txt']}".split())
    assert library.node_rows() == [(word, CLUSTERED, 2) for word in words]


async def test_a_sync_after_the_backfill_extracts_nothing_again():
    library = _Library()
    library.add("a.txt", "b.txt")
    # No concept row ever holds it, so only the checkpoint keeps it from being read again.
    library.add_blank("blank.txt")
    await sync(quiet=True)
    cfg.concept_graph = True
    await sync(quiet=True)
    library.nlp.texts.clear()

    await sync(quiet=True)

    assert library.nlp.texts == []


async def test_a_sync_reads_only_the_documents_written_since_the_checkpoint():
    library = _Library()
    library.add("a.txt")
    library.add_blank("blank.txt")
    await sync(quiet=True)
    cfg.concept_graph = True
    await sync(quiet=True)
    library.nlp.texts.clear()
    library.add("b.txt")

    await sync(quiet=True)

    assert library.extracted_words() == set(WORDS["b.txt"].split())
    assert not any("zzz" in text for text in library.nlp.texts)


async def test_a_sync_that_changed_nothing_returns_at_the_checkpoint():
    from lilbee.retrieval.concepts.backfill import backfill_concepts

    library = _Library()
    library.add("a.txt")
    cfg.concept_graph = True
    await sync(quiet=True)

    assert backfill_concepts(library.store, library.graph) is None
    with patch("lilbee.retrieval.concepts.concepts_available") as available:
        await sync(quiet=True)
    available.assert_not_called()


async def test_documents_indexed_while_the_graph_was_off_are_backfilled():
    library = _Library()
    library.add("a.txt")
    cfg.concept_graph = True
    await sync(quiet=True)
    assert library.sources_with_concepts() == {"a.txt"}
    cfg.concept_graph = False
    library.add("b.txt")
    await sync(quiet=True)
    cfg.concept_graph = True
    library.nlp.texts.clear()

    await sync(quiet=True)

    assert library.sources_with_concepts() == {"a.txt", "b.txt"}
    assert library.extracted_words() == set(WORDS["b.txt"].split())


async def test_an_interrupted_backfill_resumes_with_the_documents_it_did_not_reach():
    library = _Library()
    library.add("a.txt", "b.txt", "c.txt")
    await sync(quiet=True)
    cfg.concept_graph = True
    write = library.graph.write_concept_records
    calls = []

    def fail_after_the_first(records):
        calls.append(records)
        if len(calls) > 1:
            raise RuntimeError("stopped")
        write(records)

    with (
        patch("lilbee.retrieval.concepts.backfill._SOURCES_PER_BATCH", 1),
        patch.object(library.graph, "write_concept_records", side_effect=fail_after_the_first),
    ):
        await sync(quiet=True)
    done = library.sources_with_concepts()
    assert len(done) == 1
    library.nlp.texts.clear()

    await sync(quiet=True)

    assert library.sources_with_concepts() == set(WORDS)
    remaining = {word for name in set(WORDS) - done for word in WORDS[name].split()}
    assert library.extracted_words() == remaining


async def test_a_cancelled_backfill_records_nothing_and_the_next_sync_finishes_it():
    library = _Library()
    library.add("a.txt", "b.txt")
    await sync(quiet=True)
    cfg.concept_graph = True
    from lilbee.retrieval.concepts.backfill import backfill_concepts

    stop = SimpleNamespace(is_set=lambda: True)
    assert backfill_concepts(library.store, library.graph, cancel=stop).checkpoint is None
    assert library.sources_with_concepts() == set()

    await sync(quiet=True)

    assert library.sources_with_concepts() == {"a.txt", "b.txt"}


async def test_without_a_spacy_model_the_backfill_waits_for_one():
    library = _Library()
    library.add("a.txt")
    await sync(quiet=True)
    cfg.concept_graph = True
    library.graph._nlp = None
    library.graph._nlp_unavailable = True

    await sync(quiet=True)
    assert library.sources_with_concepts() == set()

    library.graph._nlp = library.nlp
    library.graph._nlp_unavailable = False
    await sync(quiet=True)

    assert library.sources_with_concepts() == {"a.txt"}


async def test_a_sync_that_only_moves_a_document_loads_no_spacy_model():
    library = _Library()
    library.add("a.txt", "b.txt")
    cfg.concept_graph = True
    await sync(quiet=True)
    before = library.store.sources_version()
    (cfg.documents_dir / "b.txt").rename(cfg.documents_dir / "moved.txt")

    with patch.object(library.graph, "can_extract") as can_extract:
        result = await sync(quiet=True)

    assert result.relocated == ["moved.txt"]
    assert library.store.sources_version() != before
    can_extract.assert_not_called()
    assert library.store.concepts_covered().version == library.store.sources_version()


async def test_a_read_of_no_sources_reads_no_chunks():
    library = _Library()
    library.add("a.txt")
    cfg.concept_graph = True
    await sync(quiet=True)
    assert library.sources_with_concepts() == {"a.txt"}

    assert list(library.store.chunk_batches(["chunk"], 10, source_ids=[])) == []
    assert library.graph.sources_with_concepts([]) == set()


def test_a_read_of_many_sources_sends_bounded_in_lists():
    from lilbee.retrieval.concepts.graph import _SOURCE_ID_BATCH

    store = MagicMock()
    table = store.open_table.return_value
    predicates: list[str] = []

    def where(predicate):
        predicates.append(predicate)
        query = MagicMock()
        ids = pa.table({"source_id": ["id-0000"]})
        query.select.return_value.limit.return_value.to_arrow.return_value = ids
        return query

    table.search.return_value.where.side_effect = where
    ids = [f"id-{i:04d}" for i in range(_SOURCE_ID_BATCH + 1)]

    holding = ConceptGraph(cfg, store).sources_with_concepts(ids)

    assert holding == {"id-0000"}
    assert [predicate.count("'") // 2 for predicate in predicates] == [_SOURCE_ID_BATCH, 1]


async def test_dropping_the_index_forgets_what_the_backfill_covered():
    library = _Library()
    library.add("a.txt")
    cfg.concept_graph = True
    await sync(quiet=True)
    assert library.store.concepts_covered() is not None

    library.store.drop_all()

    assert library.store.concepts_covered() is None
