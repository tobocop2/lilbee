"""Document-depth collection and the shared collapse-to-run driver.

lilbee returns one result per source document, so a collector asks for
``target_docs`` documents directly and the run is capped at that document depth.
The scorer drops the rank column and re-sorts, so the run file's tie order must
match the rule the scorer uses or it states one ranking while the scorer used
another.
"""

from evals.benchmark.collectors import DEFAULT_TARGET_DOCS, LilbeeCollector
from evals.benchmark.runfile import ChunkHit, collapse_hits


def _hit(doc_id, score=1.0):
    return ChunkHit(query_id="q1", doc_id=doc_id, score=score)


def test_lilbee_collector_reads_one_document_per_result():
    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return [
                {"source": "doc-a", "best_relevance": 0.9},
                {"source": "doc-b", "best_relevance": 0.7},
            ]

    class _FakeClient:
        def __init__(self):
            self.params = None

        def get(self, url, params=None):
            self.params = params
            return _FakeResponse()

    client = _FakeClient()
    collector = LilbeeCollector("http://lb", target_docs=20, client=client)
    hits = collector.retrieve("q1", "some question")
    assert client.params == {"q": "some question", "top_k": 20}
    assert [(h.doc_id, h.score) for h in hits] == [("doc-a", 0.9), ("doc-b", 0.7)]


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
    # The scorer drops the rank column and re-sorts, breaking score ties on
    # doc_id descending. Writing the reverse order would state one ranking while
    # the scorer used another.
    hits = [_hit("d1", 1.0), _hit("d2", 1.0), _hit("d3", 1.0)]
    order = [entry.doc_id for entry in collapse_hits(hits, "arm")]
    assert order == ["d3", "d2", "d1"]
