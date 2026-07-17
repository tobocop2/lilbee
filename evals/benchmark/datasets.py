"""Dataset loading, normalized to a single (corpus, queries, qrels) triple.

Two label kinds:

- NATIVE qrels ship with the dataset (BEIR, MS MARCO, HotpotQA). They are used
  as published.
- DERIVED qrels are computed from human gold-evidence annotations on QA
  datasets that have no retrieval labels of their own (TAT-DQA, OTT-QA). The
  derivation is the single documented pure function ``derive_qrels_from_evidence``,
  and every derived dataset is recorded as such in the manifest.

Real downloads live behind a lazily-imported loader. The normalization and
derivation logic is pure and is what the unit tests exercise on tiny fixtures.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LABEL_NATIVE = "native"
LABEL_DERIVED = "derived"

# Relevance grade assigned to a document named by human gold evidence.
DERIVED_RELEVANCE = 1

Corpus = dict[str, dict[str, str]]
Queries = dict[str, str]
Qrels = dict[str, dict[str, int]]


@dataclass(frozen=True)
class Dataset:
    """A retrieval benchmark normalized to corpus, queries, and qrels."""

    name: str
    label_kind: str
    corpus: Corpus
    queries: Queries
    qrels: Qrels


def normalize_corpus(rows: Mapping[str, Mapping[str, Any]]) -> Corpus:
    """Normalize BEIR-style corpus rows to ``doc_id -> {title, text}``.

    Documents whose title and text are both empty are dropped: they can never
    be a relevant hit and only inflate the index.
    """
    corpus: Corpus = {}
    for doc_id, row in rows.items():
        title = str(row.get("title", "")).strip()
        text = str(row.get("text", "")).strip()
        if title or text:
            corpus[str(doc_id)] = {"title": title, "text": text}
    return corpus


def normalize_queries(rows: Mapping[str, Any]) -> Queries:
    """Normalize query rows to ``query_id -> text``, dropping empty queries."""
    queries: Queries = {}
    for qid, text in rows.items():
        cleaned = str(text).strip()
        if cleaned:
            queries[str(qid)] = cleaned
    return queries


def normalize_qrels(rows: Mapping[str, Mapping[str, Any]]) -> Qrels:
    """Normalize native qrels to ``query_id -> {doc_id: grade}`` with int grades.

    Non-positive grades are dropped so the qrels hold only judged-relevant
    documents, matching how pytrec_eval treats them.
    """
    qrels: Qrels = {}
    for qid, judged in rows.items():
        graded = {str(doc): int(grade) for doc, grade in judged.items() if int(grade) > 0}
        if graded:
            qrels[str(qid)] = graded
    return qrels


def derive_qrels_from_evidence(
    evidence: Mapping[str, Iterable[str]], relevance: int = DERIVED_RELEVANCE
) -> Qrels:
    """Derive retrieval qrels from human gold-evidence document annotations.

    Each query maps to the set of document ids a human annotator marked as gold
    evidence for its answer; every such document is labeled relevant at
    ``relevance``. This is the documented derivation for QA datasets that carry
    gold evidence but no native retrieval labels (TAT-DQA, OTT-QA). Queries with
    no evidence are omitted, since an unjudged query cannot be scored.
    """
    qrels: Qrels = {}
    for qid, doc_ids in evidence.items():
        graded = {str(doc_id): relevance for doc_id in doc_ids}
        if graded:
            qrels[str(qid)] = graded
    return qrels


def build_native_dataset(
    name: str,
    corpus_rows: Mapping[str, Mapping[str, Any]],
    query_rows: Mapping[str, Any],
    qrel_rows: Mapping[str, Mapping[str, Any]],
) -> Dataset:
    """Assemble a native-qrel dataset from raw BEIR-style rows."""
    return Dataset(
        name=name,
        label_kind=LABEL_NATIVE,
        corpus=normalize_corpus(corpus_rows),
        queries=normalize_queries(query_rows),
        qrels=normalize_qrels(qrel_rows),
    )


def build_derived_dataset(
    name: str,
    corpus_rows: Mapping[str, Mapping[str, Any]],
    query_rows: Mapping[str, Any],
    evidence: Mapping[str, Iterable[str]],
) -> Dataset:
    """Assemble a derived-qrel dataset from a QA corpus plus gold evidence."""
    return Dataset(
        name=name,
        label_kind=LABEL_DERIVED,
        corpus=normalize_corpus(corpus_rows),
        queries=normalize_queries(query_rows),
        qrels=derive_qrels_from_evidence(evidence),
    )


# Loaders for real downloads. Each returns the raw rows the pure builders above
# consume; they are only called by ``load_dataset`` and import heavy deps lazily.
RawNativeLoader = Callable[[Path], tuple[Mapping, Mapping, Mapping]]
RawDerivedLoader = Callable[[Path], tuple[Mapping, Mapping, Mapping]]


def _load_beir(cache_dir: Path, dataset_key: str, split: str) -> tuple[Mapping, Mapping, Mapping]:
    """Download and read a BEIR dataset (SciFact, FiQA, NFCorpus, ...)."""
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_key}.zip"
    data_path = util.download_and_unzip(url, str(cache_dir))
    return GenericDataLoader(data_folder=data_path).load(split=split)


def load_dataset(
    spec: Any,
    cache_dir: Path,
    *,
    native_loader: RawNativeLoader | None = None,
    derived_loader: RawDerivedLoader | None = None,
) -> Dataset:
    """Load a dataset described by a manifest ``DatasetSpec`` to a normalized triple.

    The real download is injected (or lazily selected by ``spec.loader``); the
    normalization and derivation are the pure functions above. Tests inject
    tiny in-memory loaders so nothing touches the network.
    """
    if spec.label_kind == LABEL_DERIVED:
        if derived_loader is None:
            raise ValueError(f"no derived loader available for dataset '{spec.name}'")
        corpus_rows, query_rows, evidence = derived_loader(cache_dir)
        return build_derived_dataset(spec.name, corpus_rows, query_rows, evidence)
    loader = native_loader or (lambda path: _load_beir(path, spec.loader, spec.split))
    corpus_rows, query_rows, qrel_rows = loader(cache_dir)
    return build_native_dataset(spec.name, corpus_rows, query_rows, qrel_rows)
