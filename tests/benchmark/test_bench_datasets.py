"""Pure normalization and the derived-qrel derivation on tiny fixtures."""

import pytest

from evals.benchmark.datasets import (
    LABEL_DERIVED,
    LABEL_NATIVE,
    build_derived_dataset,
    build_native_dataset,
    derive_qrels_from_evidence,
    load_dataset,
    normalize_corpus,
    normalize_qrels,
    normalize_queries,
)
from evals.benchmark.manifest import DatasetSpec


def test_normalize_corpus_drops_fully_empty_documents():
    corpus = normalize_corpus(
        {"d1": {"title": "T", "text": "body"}, "d2": {"title": " ", "text": ""}}
    )
    assert corpus == {"d1": {"title": "T", "text": "body"}}


def test_normalize_queries_drops_blanks_and_stringifies_ids():
    assert normalize_queries({1: "what?", "q2": "  "}) == {"1": "what?"}


def test_normalize_qrels_drops_non_positive_grades():
    qrels = normalize_qrels({"q1": {"d1": 2, "d2": 0}, "q2": {"d3": 0}})
    assert qrels == {"q1": {"d1": 2}}


def test_derive_qrels_labels_each_gold_evidence_doc_relevant():
    qrels = derive_qrels_from_evidence({"q1": ["d1", "d2"], "q2": []})
    assert qrels == {"q1": {"d1": 1, "d2": 1}}  # q2 omitted: no evidence


def test_build_native_dataset_carries_label_kind():
    dataset = build_native_dataset(
        "scifact",
        {"d1": {"title": "t", "text": "x"}},
        {"q1": "why?"},
        {"q1": {"d1": 1}},
    )
    assert dataset.label_kind == LABEL_NATIVE
    assert dataset.qrels == {"q1": {"d1": 1}}


def test_build_derived_dataset_derives_qrels_from_evidence():
    dataset = build_derived_dataset(
        "tat-dqa",
        {"d1": {"title": "", "text": "table"}},
        {"q1": "sum?"},
        {"q1": ["d1"]},
    )
    assert dataset.label_kind == LABEL_DERIVED
    assert dataset.qrels == {"q1": {"d1": 1}}


def test_load_dataset_native_uses_injected_loader_without_network():
    spec = DatasetSpec(name="scifact", loader="scifact", label_kind=LABEL_NATIVE)
    rows = ({"d1": {"title": "t", "text": "x"}}, {"q1": "why?"}, {"q1": {"d1": 1}})
    dataset = load_dataset(spec, tmp_cache(), native_loader=lambda _cache: rows)
    assert dataset.name == "scifact"
    assert dataset.queries == {"q1": "why?"}


def test_load_dataset_derived_uses_injected_loader():
    spec = DatasetSpec(name="ott-qa", loader="ottqa", label_kind=LABEL_DERIVED, split="dev")
    rows = ({"d1": {"title": "", "text": "x"}}, {"q1": "why?"}, {"q1": ["d1"]})
    dataset = load_dataset(spec, tmp_cache(), derived_loader=lambda _cache: rows)
    assert dataset.qrels == {"q1": {"d1": 1}}


def test_load_dataset_derived_requires_a_loader():
    spec = DatasetSpec(name="ott-qa", loader="ottqa", label_kind=LABEL_DERIVED)
    with pytest.raises(ValueError, match="no derived loader"):
        load_dataset(spec, tmp_cache())


def tmp_cache():
    from pathlib import Path

    return Path("/tmp/benchmark-cache-unused")
