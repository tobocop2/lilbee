"""Dataset loading, normalized to a single (corpus, queries, qrels) triple.

``ir_datasets`` owns the download, the cache, and the parse. It ships all 48
BEIR sets, pins each to a specific published copy, and verifies it. The loader
this replaces fetched an unversioned URL, so a republished upstream corpus moved
every number while the frozen manifest's fingerprint stayed identical, and the
reproducibility the manifest exists to carry was a claim about the dataset name
alone.

Two label kinds:

- NATIVE qrels ship with the dataset (BEIR, MS MARCO, HotpotQA). Used as
  published, straight from ir_datasets.
- DERIVED qrels are computed from human gold-evidence annotations on QA datasets
  that have no retrieval labels of their own (TAT-DQA, OTT-QA). The derivation is
  the single documented pure function ``derive_qrels_from_evidence``, and every
  derived dataset is recorded as such in the manifest.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from evals.deps import install_hint

LABEL_NATIVE = "native"
LABEL_DERIVED = "derived"

# Relevance grade assigned to a document named by human gold evidence.
DERIVED_RELEVANCE = 1

Corpus = dict[str, dict[str, str]]
Queries = dict[str, str]
Qrels = dict[str, dict[str, int]]

IR_DATASETS_INSTALL_HINT = install_hint("ir_datasets", "to load benchmark corpora")


@dataclass(frozen=True)
class Dataset:
    """A retrieval benchmark normalized to corpus, queries, and qrels."""

    name: str
    label_kind: str
    corpus: Corpus
    queries: Queries
    qrels: Qrels


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


def load_ir_dataset(dataset_id: str) -> tuple[Corpus, Queries, Qrels]:
    """Read one ir_datasets dataset into the harness' triple.

    Document fields vary across BEIR: scifact and nfcorpus carry a title, fiqa
    does not. The title is read defensively for that reason, and its absence on
    fiqa is why the title-search arm there is an A/A null control rather than a
    comparison.

    Non-positive relevance grades are dropped so the qrels hold only
    judged-relevant documents, matching how trec_eval treats them.
    """
    try:
        import ir_datasets
    except ImportError as exc:
        raise RuntimeError(IR_DATASETS_INSTALL_HINT) from exc
    source = ir_datasets.load(dataset_id)
    corpus: Corpus = {}
    for doc in source.docs_iter():
        title = str(getattr(doc, "title", "") or "").strip()
        text = str(doc.text or "").strip()
        # A document with neither title nor text can never be a relevant hit and
        # only inflates the index.
        if title or text:
            corpus[str(doc.doc_id)] = {"title": title, "text": text}
    queries = {
        str(query.query_id): query.text.strip()
        for query in source.queries_iter()
        if query.text.strip()
    }
    qrels: Qrels = {}
    for qrel in source.qrels_iter():
        if qrel.relevance > 0:
            qrels.setdefault(str(qrel.query_id), {})[str(qrel.doc_id)] = int(qrel.relevance)
    return corpus, queries, qrels


# A derived-qrel dataset has no ir_datasets equivalent by definition: the labels
# do not exist upstream. Its loader returns the corpus, the queries, and the
# per-query gold evidence that ``derive_qrels_from_evidence`` turns into qrels.
DerivedLoader = Callable[[], tuple[Corpus, Queries, Mapping[str, Iterable[str]]]]


def load_dataset(spec: Any, *, derived_loader: DerivedLoader | None = None) -> Dataset:
    """Load a dataset described by a manifest ``DatasetSpec``.

    ``spec.loader`` is an ir_datasets id (``beir/fiqa/test``) for native sets.
    Derived sets take an injected loader, since their labels are computed here
    rather than published.
    """
    if spec.label_kind == LABEL_DERIVED:
        if derived_loader is None:
            raise ValueError(f"no derived loader available for dataset '{spec.name}'")
        corpus, queries, evidence = derived_loader()
        return Dataset(
            name=spec.name,
            label_kind=LABEL_DERIVED,
            corpus=corpus,
            queries=queries,
            qrels=derive_qrels_from_evidence(evidence),
        )
    corpus, queries, qrels = load_ir_dataset(spec.loader)
    return Dataset(
        name=spec.name,
        label_kind=LABEL_NATIVE,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
    )
