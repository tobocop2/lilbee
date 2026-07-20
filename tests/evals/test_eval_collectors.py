"""Depth-matched collection: chunk-level arms over-fetch to a document depth.

A chunk-level system (RAGFlow) returns chunks that collapse to their parent
documents, so asking for 20 chunks yields far fewer than 20 documents. lilbee
returns documents directly. Comparing a 20-document list against a
7-document list is a pure depth artifact, so collection accumulates pages until
it has the target number of distinct parent documents, and the run is capped at
that document depth for both arms.
"""

from evals.benchmark.collectors import (
    DEFAULT_TARGET_DOCS,
    RagflowCollector,
    collect_to_document_depth,
)
from evals.benchmark.runfile import ChunkHit, collapse_hits


def _hit(doc_id, score=1.0):
    return ChunkHit(query_id="q1", doc_id=doc_id, score=score)


def test_collection_stops_once_the_document_target_is_reached():
    # Each page carries two chunks from two new documents; target 4 -> 2 pages.
    pages = {
        1: [_hit("d1"), _hit("d2")],
        2: [_hit("d3"), _hit("d4")],
        3: [_hit("d5"), _hit("d6")],
    }
    fetched = []

    def fetch(page):
        fetched.append(page)
        return pages.get(page, [])

    hits = collect_to_document_depth(fetch, target_docs=4, max_pages=10)
    assert fetched == [1, 2]
    assert {h.doc_id for h in hits} == {"d1", "d2", "d3", "d4"}


def test_over_fetch_when_chunks_share_documents():
    # Every page is two chunks of the SAME two documents; the target is never
    # reached by document count, so it must page to the max and stop.
    def fetch(page):
        return [_hit("d1", 0.9), _hit("d2", 0.8)] if page <= 3 else []

    hits = collect_to_document_depth(fetch, target_docs=20, max_pages=3)
    assert {h.doc_id for h in hits} == {"d1", "d2"}
    # Three pages of two chunks each: over-fetched chunks, still two documents.
    assert len(hits) == 6


def test_collection_stops_on_an_empty_page():
    def fetch(page):
        return [_hit("d1"), _hit("d2")] if page == 1 else []

    hits = collect_to_document_depth(fetch, target_docs=20, max_pages=10)
    assert {h.doc_id for h in hits} == {"d1", "d2"}


def test_collection_respects_the_page_ceiling():
    calls = []

    def fetch(page):
        calls.append(page)
        return [_hit(f"d{page}")]  # one new document per page, never reaches target

    collect_to_document_depth(fetch, target_docs=100, max_pages=5)
    assert calls == [1, 2, 3, 4, 5]


def test_ragflow_collector_pages_until_the_document_target():
    # Two RAGFlow "pages" of chunks: page 1 is three chunks from docs A/A/B,
    # page 2 adds C/D. Target 4 distinct documents needs both pages.
    responses = {
        1: {
            "data": {
                "chunks": [
                    {"document_id": "A", "similarity": 0.9},
                    {"document_id": "A", "similarity": 0.8},
                    {"document_id": "B", "similarity": 0.7},
                ]
            }
        },
        2: {
            "data": {
                "chunks": [
                    {"document_id": "C", "similarity": 0.6},
                    {"document_id": "D", "similarity": 0.5},
                ]
            }
        },
    }
    requested_pages = []

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class _FakeClient:
        def post(self, url, headers=None, json=None):
            requested_pages.append(json["page"])
            return _FakeResponse(responses.get(json["page"], {"data": {"chunks": []}}))

    collector = RagflowCollector("http://rf", "key", ["ds1"], target_docs=4, client=_FakeClient())
    hits = collector.retrieve("q1", "some question")
    assert requested_pages == [1, 2]
    assert {h.doc_id for h in hits} == {"A", "B", "C", "D"}


def test_collapse_hits_caps_each_query_at_the_document_limit():
    # Five documents retrieved; a depth-4 cap keeps the four highest-scoring.
    hits = [_hit(f"d{i}", score=float(i)) for i in range(5)]
    entries = collapse_hits(hits, "arm", limit=4)
    kept = {e.doc_id for e in entries}
    assert kept == {"d4", "d3", "d2", "d1"}
    assert all(e.rank <= 4 for e in entries)


def test_collapse_hits_without_a_limit_keeps_every_document():
    hits = [_hit(f"d{i}", score=float(i)) for i in range(5)]
    entries = collapse_hits(hits, "arm")
    assert len({e.doc_id for e in entries}) == 5


def test_default_target_docs_matches_the_published_recall_depth():
    assert DEFAULT_TARGET_DOCS == 20


def test_run_file_tie_order_matches_the_scorers_rule():
    # pytrec_eval drops the rank column and re-sorts, breaking score ties on
    # doc_id descending. Writing the reverse order would state one ranking while
    # the scorer used another.
    hits = [_hit("d1", 1.0), _hit("d2", 1.0), _hit("d3", 1.0)]
    order = [entry.doc_id for entry in collapse_hits(hits, "arm")]
    assert order == ["d3", "d2", "d1"]
