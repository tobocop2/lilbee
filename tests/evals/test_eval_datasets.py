"""Dataset loading: the normalization ir_datasets does not do for us.

ir_datasets owns the download, the cache, and the parse, so what is tested here
is the thin layer above it: which documents and judgments survive into the
harness' triple, and the derivation for QA sets whose retrieval labels do not
exist upstream.

Nothing here touches the network. The corpora are large enough that pulling one
to run a unit test would be the wrong trade even where it is allowed.
"""

from dataclasses import dataclass

import ir_datasets
import pytest
from evals.benchmark.datasets import (
    LABEL_DERIVED,
    LABEL_NATIVE,
    derive_qrels_from_evidence,
    load_dataset,
    load_ir_dataset,
)


@dataclass
class _Doc:
    doc_id: str
    text: str
    title: str = ""


@dataclass
class _TitlelessDoc:
    """A fiqa-shaped document: ir_datasets omits the field entirely."""

    doc_id: str
    text: str


@dataclass
class _Query:
    query_id: str
    text: str


@dataclass
class _Qrel:
    query_id: str
    doc_id: str
    relevance: int


class _Source:
    def __init__(self, docs, queries, qrels):
        self._docs, self._queries, self._qrels = docs, queries, qrels

    def docs_iter(self):
        return iter(self._docs)

    def queries_iter(self):
        return iter(self._queries)

    def qrels_iter(self):
        return iter(self._qrels)


@dataclass
class _Spec:
    name: str
    loader: str
    label_kind: str


@pytest.fixture
def source(monkeypatch):
    """Install a fake ir_datasets dataset and return its id."""

    def install(docs, queries, qrels):
        monkeypatch.setattr(ir_datasets, "load", lambda _id: _Source(docs, queries, qrels))
        return "fake/dataset"

    return install


def test_a_document_without_a_title_field_still_loads(source):
    # BEIR is not uniform: scifact and nfcorpus carry a title, fiqa does not.
    # Reading it as a required attribute would fail on a third of the study.
    dataset_id = source([_TitlelessDoc("d1", "body text")], [], [])
    corpus, _, _ = load_ir_dataset(dataset_id)
    assert corpus["d1"] == {"title": "", "text": "body text"}


def test_a_document_with_neither_title_nor_text_is_dropped(source):
    dataset_id = source([_Doc("d1", "  ", "  "), _Doc("d2", "real")], [], [])
    corpus, _, _ = load_ir_dataset(dataset_id)
    assert set(corpus) == {"d2"}


def test_non_positive_judgments_are_dropped(source):
    # trec_eval treats a zero or negative grade as "judged, not relevant". Keeping
    # them would put unfindable documents in the denominator of recall.
    dataset_id = source([], [], [_Qrel("q1", "d1", 1), _Qrel("q1", "d2", 0), _Qrel("q2", "d3", -1)])
    _, _, qrels = load_ir_dataset(dataset_id)
    assert qrels == {"q1": {"d1": 1}}


def test_a_query_with_no_remaining_relevant_document_is_not_scorable(source):
    dataset_id = source([], [_Query("q1", "text")], [_Qrel("q1", "d1", 0)])
    _, _, qrels = load_ir_dataset(dataset_id)
    assert "q1" not in qrels


def test_graded_relevance_is_preserved_not_flattened(source):
    # nfcorpus judges on a graded scale, and nDCG is the reason the harness
    # scores it: collapsing every positive grade to 1 would discard that.
    dataset_id = source([], [], [_Qrel("q1", "d1", 2), _Qrel("q1", "d2", 1)])
    _, _, qrels = load_ir_dataset(dataset_id)
    assert qrels == {"q1": {"d1": 2, "d2": 1}}


def test_blank_queries_are_dropped(source):
    dataset_id = source([], [_Query("q1", "  "), _Query("q2", "real")], [])
    _, queries, _ = load_ir_dataset(dataset_id)
    assert queries == {"q2": "real"}


def test_load_dataset_marks_a_native_set_and_carries_its_name(source):
    dataset_id = source([_Doc("d1", "t")], [_Query("q1", "text")], [_Qrel("q1", "d1", 1)])
    dataset = load_dataset(_Spec(name="scifact", loader=dataset_id, label_kind=LABEL_NATIVE))
    assert dataset.name == "scifact"
    assert dataset.label_kind == LABEL_NATIVE
    assert dataset.qrels == {"q1": {"d1": 1}}


def test_derived_qrels_label_every_gold_evidence_document():
    assert derive_qrels_from_evidence({"q1": ["d1", "d2"]}) == {"q1": {"d1": 1, "d2": 1}}


def test_a_query_with_no_gold_evidence_is_omitted():
    # An unjudged query cannot be scored, and keeping it would score every arm
    # zero on it and dilute the mean by the same amount for both.
    assert derive_qrels_from_evidence({"q1": [], "q2": ["d1"]}) == {"q2": {"d1": 1}}


def test_a_derived_set_without_a_loader_is_refused():
    # Its labels do not exist upstream, so there is nothing for ir_datasets to
    # fall back to; silently loading it as native would invent the qrels.
    spec = _Spec(name="tat-dqa", loader="tatdqa", label_kind=LABEL_DERIVED)
    with pytest.raises(ValueError, match="no derived loader"):
        load_dataset(spec)


def test_a_derived_set_is_marked_derived():
    spec = _Spec(name="tat-dqa", loader="tatdqa", label_kind=LABEL_DERIVED)
    dataset = load_dataset(
        spec,
        derived_loader=lambda: ({"d1": {"title": "", "text": "t"}}, {"q1": "?"}, {"q1": ["d1"]}),
    )
    assert dataset.label_kind == LABEL_DERIVED
    assert dataset.qrels == {"q1": {"d1": 1}}
