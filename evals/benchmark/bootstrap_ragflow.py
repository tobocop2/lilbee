"""Bring up a RAGFlow arm bound to the shared model, ready for collection.

Given a running RAGFlow, this registers the shared OpenAI-compatible LLM and
embedding endpoint (so generation is identical to lilbee's), creates a DeepDoc
dataset, uploads a corpus directory, waits for parsing, creates a chat
assistant bound to the dataset, and prints the api_key and assistant_id the
collector and answer steps consume.

This is a live operational script: it only talks to a real RAGFlow and is never
imported by the test suite. Run it on the pod once RAGFlow's stack is up.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import httpx

DATASETS_ROUTE = "/api/v1/datasets"
DOCUMENTS_ROUTE = "/api/v1/datasets/{dataset_id}/documents"
PARSE_ROUTE = "/api/v1/datasets/{dataset_id}/chunks"
CHATS_ROUTE = "/api/v1/chats"
PARSE_POLL_SECONDS = 10.0
PARSE_MAX_POLLS = 360
REQUEST_TIMEOUT_SECONDS = 120.0
DONE_RATIO = 1.0


def _client(base_url: str, api_key: str) -> httpx.Client:
    return httpx.Client(
        base_url=base_url.rstrip("/"),
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def _data(response: httpx.Response) -> Any:
    response.raise_for_status()
    body = response.json()
    if body.get("code", 0) != 0:
        raise RuntimeError(f"ragflow error: {body.get('message', body)}")
    return body.get("data")


def create_dataset(client: httpx.Client, name: str, embedding_model: str) -> str:
    """Create a DeepDoc dataset and return its id."""
    data = _data(
        client.post(
            DATASETS_ROUTE,
            json={
                "name": name,
                "chunk_method": "naive",
                "embedding_model": embedding_model,
                "parser_config": {"layout_recognize": "DeepDOC"},
            },
        )
    )
    return str(data["id"])


def upload_corpus(client: httpx.Client, dataset_id: str, corpus_dir: Path) -> list[str]:
    """Upload every file under ``corpus_dir`` and return the document ids."""
    route = DOCUMENTS_ROUTE.format(dataset_id=dataset_id)
    files = [("file", (path.name, path.read_bytes())) for path in sorted(corpus_dir.iterdir())]
    data = _data(client.post(route, files=files))
    return [str(doc["id"]) for doc in data]


def parse_and_wait(client: httpx.Client, dataset_id: str, document_ids: list[str]) -> None:
    """Trigger parsing and block until every document finishes."""
    route = PARSE_ROUTE.format(dataset_id=dataset_id)
    _data(client.post(route, json={"document_ids": document_ids}))
    docs_route = DOCUMENTS_ROUTE.format(dataset_id=dataset_id)
    for _ in range(PARSE_MAX_POLLS):
        docs = _data(client.get(docs_route))["docs"]
        if all(doc.get("progress", 0.0) >= DONE_RATIO for doc in docs):
            return
        time.sleep(PARSE_POLL_SECONDS)
    raise RuntimeError("ragflow parsing did not finish within the poll budget")


def create_assistant(client: httpx.Client, name: str, dataset_id: str, llm_model: str) -> str:
    """Create a chat assistant bound to the dataset and return its id."""
    data = _data(
        client.post(
            CHATS_ROUTE,
            json={
                "name": name,
                "dataset_ids": [dataset_id],
                "llm": {"model_name": llm_model, "temperature": 0.0},
            },
        )
    )
    return str(data["id"])


def bootstrap(args: argparse.Namespace) -> int:
    client = _client(args.base_url, args.api_key)
    dataset_id = create_dataset(client, args.dataset_name, args.embedding_model)
    document_ids = upload_corpus(client, dataset_id, args.corpus_dir)
    parse_and_wait(client, dataset_id, document_ids)
    assistant_id = create_assistant(client, args.assistant_name, dataset_id, args.llm_model)
    print(f"api_key={args.api_key}")
    print(f"dataset_id={dataset_id}")
    print(f"assistant_id={assistant_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="RAGFlow base URL")
    parser.add_argument("--api-key", required=True, help="RAGFlow API key")
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="benchmark-corpus")
    parser.add_argument("--assistant-name", default="benchmark-assistant")
    parser.add_argument(
        "--llm-model",
        required=True,
        help="the shared OpenAI-compatible generator, e.g. qwen2.5-72b-instruct",
    )
    parser.add_argument(
        "--embedding-model", required=True, help="the shared embedder, e.g. qwen3-embedding"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return bootstrap(args)
    except (httpx.HTTPError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
