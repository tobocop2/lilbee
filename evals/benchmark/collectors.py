"""Per-system collectors that turn a query set into a TREC run file.

Each collector hits one system's retrieval API and returns chunk hits tagged
with their parent document; the shared driver collapses those to a document run
and writes the TREC file. Every query is checkpointed as it lands, so a killed
pod run resumes without re-querying completed queries.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import httpx

from evals.benchmark.runfile import ChunkHit, collapse_hits, write_run
from evals.retrieval.checkpoint import JsonlCheckpoint, load_jsonl

SEARCH_ROUTE = "/api/search"
RAGFLOW_RETRIEVAL_ROUTE = "/api/v1/retrieval"
RETRIEVE_TIMEOUT_SECONDS = 120.0
DEFAULT_TOP_K = 20


class Collector(Protocol):
    """Retrieves ranked chunk hits for one query from one system."""

    run_tag: str

    def retrieve(self, query_id: str, query_text: str) -> list[ChunkHit]: ...


def make_http_client() -> httpx.Client:
    return httpx.Client(timeout=RETRIEVE_TIMEOUT_SECONDS)


class LilbeeCollector:
    """Reads ranked results from a running lilbee server's ``/api/search``.

    lilbee already groups results by source document, so each result maps to one
    document hit scored by ``best_relevance``.
    """

    def __init__(
        self,
        base_url: str,
        *,
        run_tag: str = "lilbee",
        top_k: int = DEFAULT_TOP_K,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self.run_tag = run_tag
        self._top_k = top_k
        self._client = client or make_http_client()

    def retrieve(self, query_id: str, query_text: str) -> list[ChunkHit]:
        response = self._client.get(
            f"{self._base_url}{SEARCH_ROUTE}",
            params={"q": query_text, "top_k": self._top_k},
        )
        response.raise_for_status()
        return [
            ChunkHit(query_id=query_id, doc_id=doc["source"], score=float(doc["best_relevance"]))
            for doc in response.json()
        ]


class RagflowCollector:
    """Reads ranked chunks from RAGFlow's retrieval API and tags parent docs."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        dataset_ids: list[str],
        *,
        run_tag: str = "ragflow",
        top_k: int = DEFAULT_TOP_K,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._dataset_ids = dataset_ids
        self.run_tag = run_tag
        self._top_k = top_k
        self._client = client or make_http_client()

    def retrieve(self, query_id: str, query_text: str) -> list[ChunkHit]:
        response = self._client.post(
            f"{self._base_url}{RAGFLOW_RETRIEVAL_ROUTE}",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "question": query_text,
                "dataset_ids": self._dataset_ids,
                "page": 1,
                "page_size": self._top_k,
            },
        )
        response.raise_for_status()
        chunks = response.json().get("data", {}).get("chunks", [])
        return [
            ChunkHit(
                query_id=query_id,
                doc_id=chunk["document_id"],
                score=float(chunk["similarity"]),
            )
            for chunk in chunks
        ]


def _row_hits(query_id: str, row: dict[str, Any]) -> list[ChunkHit]:
    return [
        ChunkHit(query_id=query_id, doc_id=doc_id, score=score) for doc_id, score in row["hits"]
    ]


def collect_run(
    collector: Collector,
    queries: dict[str, str],
    run_path: Path,
    checkpoint_path: Path,
    *,
    on_query: Callable[[str], None] | None = None,
) -> list[ChunkHit]:
    """Retrieve every query (resuming from the checkpoint) and write the run file.

    Each query's hits are appended to ``checkpoint_path`` as they land; the run
    file is (re)built from the full checkpoint at the end, so an interrupted run
    resumes without re-querying and still produces a complete run file.
    """
    checkpoint = JsonlCheckpoint(checkpoint_path, "query_id")
    for query_id, query_text in queries.items():
        if query_id in checkpoint:
            continue
        hits = collector.retrieve(query_id, query_text)
        checkpoint.append({"query_id": query_id, "hits": [[hit.doc_id, hit.score] for hit in hits]})
        if on_query is not None:
            on_query(query_id)
    all_hits: list[ChunkHit] = []
    for row in load_jsonl(checkpoint_path):
        all_hits.extend(_row_hits(row["query_id"], row))
    entries = collapse_hits(all_hits, collector.run_tag)
    write_run(run_path, entries)
    return all_hits


def load_queries(path: Path) -> dict[str, str]:
    """Read a queries file: one ``{"query_id": ..., "text": ...}`` object per line."""
    return {row["query_id"]: row["text"] for row in load_jsonl(path)}


def write_queries(path: Path, queries: dict[str, str]) -> None:
    """Write a queries file the collector reads back."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({"query_id": qid, "text": text}) + "\n" for qid, text in queries.items())
    )
