"""Collectors over mocked HTTP, plus the checkpointed collect driver."""

import json

import httpx

from evals.benchmark.collectors import (
    LilbeeCollector,
    RagflowCollector,
    collect_run,
    load_queries,
    write_queries,
)
from evals.benchmark.runfile import read_run


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")


def test_lilbee_collector_maps_documents_to_hits():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/search"
        assert request.url.params["q"] == "why?"
        return httpx.Response(
            200,
            json=[
                {"source": "docA", "best_relevance": 0.9, "excerpts": []},
                {"source": "docB", "best_relevance": 0.4, "excerpts": []},
            ],
        )

    collector = LilbeeCollector("http://test", client=_client(handler))
    hits = collector.retrieve("q1", "why?")
    assert [(h.doc_id, h.score) for h in hits] == [("docA", 0.9), ("docB", 0.4)]


def test_ragflow_collector_reads_document_ids_and_similarity():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/retrieval"
        body = json.loads(request.content)
        assert body["dataset_ids"] == ["ds1"]
        return httpx.Response(
            200,
            json={"data": {"chunks": [{"document_id": "docX", "similarity": 0.7}]}},
        )

    collector = RagflowCollector("http://test", "key", ["ds1"], client=_client(handler))
    hits = collector.retrieve("q1", "why?")
    assert [(h.doc_id, h.score) for h in hits] == [("docX", 0.7)]


def test_collect_run_writes_run_file_and_checkpoints(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"source": "docA", "best_relevance": 0.5, "excerpts": []}])

    collector = LilbeeCollector("http://test", run_tag="lil", client=_client(handler))
    run_path = tmp_path / "run.trec"
    ckpt = tmp_path / "ckpt.jsonl"
    collect_run(collector, {"q1": "why?"}, run_path, ckpt)
    entries = read_run(run_path)
    assert [(e.query_id, e.doc_id, e.run_tag) for e in entries] == [("q1", "docA", "lil")]
    assert len(ckpt.read_text().splitlines()) == 1


def test_collect_run_resumes_without_requerying(tmp_path):
    ckpt = tmp_path / "ckpt.jsonl"
    ckpt.write_text(json.dumps({"query_id": "q1", "hits": [["docA", 0.9]]}) + "\n")
    asked: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.params["q"])
        return httpx.Response(200, json=[{"source": "docB", "best_relevance": 0.3, "excerpts": []}])

    collector = LilbeeCollector("http://test", run_tag="lil", client=_client(handler))
    collect_run(collector, {"q1": "first", "q2": "second"}, tmp_path / "run.trec", ckpt)
    assert asked == ["second"]  # q1 was already checkpointed
    assert len(ckpt.read_text().splitlines()) == 2


def test_queries_file_round_trips(tmp_path):
    path = tmp_path / "queries.jsonl"
    write_queries(path, {"q1": "one", "q2": "two"})
    assert load_queries(path) == {"q1": "one", "q2": "two"}
