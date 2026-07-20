"""Bring up a RAGFlow arm over an already-registered model, ready for collection.

Given a running RAGFlow, this creates a DeepDoc dataset with pinned chunking,
uploads a corpus directory recursively, waits for every document to parse,
creates a chat assistant with pinned retrieval settings, and prints the
dataset_id and assistant_id the collector and answer steps consume.

It does NOT register the shared OpenAI-compatible LLM and embedding endpoint.
``--llm-model`` and ``--embedding-model`` name models RAGFlow must already have
registered; the operator adds that provider in RAGFlow's model settings before
running this. Naming a model RAGFlow does not know fails at dataset or assistant
creation rather than silently falling back, but nothing here can confirm the
named endpoint is the same one lilbee is served by. That the two arms share a
model is an operator-enforced precondition, not something this script
establishes.

This is a live operational script: it only talks to a real RAGFlow. The pure
helpers (file discovery) are unit-tested; the HTTP paths are not.
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
UPLOAD_BATCH_SIZE = 32
DOCUMENTS_PAGE_SIZE = 200

# RAGFlow's chunking and retrieval knobs are pinned rather than left at the
# server's defaults: an unpinned value silently changes with RAGFlow's version,
# so the frozen manifest would not describe the arm that actually ran. These are
# RAGFlow's documented defaults at the version under test, written down so a
# change is a visible diff instead of a silent drift.
CHUNK_TOKEN_COUNT = 128
# RAGFlow's default sentence delimiters; the fullwidth forms are the exact
# characters it splits on, so ASCII look-alikes would change the chunking.
CHUNK_DELIMITER = "\n!?;。；！？"  # noqa: RUF001
RETRIEVAL_SIMILARITY_THRESHOLD = 0.2
RETRIEVAL_KEYWORDS_WEIGHT = 0.7
RETRIEVAL_TOP_N = 20


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
    """Create a DeepDoc dataset with pinned chunking and return its id."""
    data = _data(
        client.post(
            DATASETS_ROUTE,
            json={
                "name": name,
                "chunk_method": "naive",
                "embedding_model": embedding_model,
                "parser_config": {
                    "layout_recognize": "DeepDOC",
                    "chunk_token_count": CHUNK_TOKEN_COUNT,
                    "delimiter": CHUNK_DELIMITER,
                },
            },
        )
    )
    return str(data["id"])


def iter_corpus_files(corpus_dir: Path) -> list[Path]:
    """Every file under ``corpus_dir``, recursively, in a deterministic order.

    ``lilbee add`` on a directory ingests recursively, so a non-recursive walk
    here would hand RAGFlow only the top-level files and index a different
    corpus than the arm it is being compared against.
    """
    return sorted(path for path in corpus_dir.rglob("*") if path.is_file())


def upload_corpus(
    client: httpx.Client,
    dataset_id: str,
    corpus_dir: Path,
    *,
    batch_size: int = UPLOAD_BATCH_SIZE,
) -> list[str]:
    """Upload every file under ``corpus_dir`` recursively; return document ids.

    Files are posted in batches rather than one giant multipart request, so a
    real benchmark corpus does not have to fit in memory at once. Names are made
    relative to the corpus root so nested files keep distinct document names.
    """
    route = DOCUMENTS_ROUTE.format(dataset_id=dataset_id)
    paths = iter_corpus_files(corpus_dir)
    if not paths:
        raise RuntimeError(f"no files found under corpus directory {corpus_dir}")
    document_ids: list[str] = []
    for start in range(0, len(paths), batch_size):
        batch = paths[start : start + batch_size]
        files = [
            ("file", (str(path.relative_to(corpus_dir)), path.read_bytes())) for path in batch
        ]
        data = _data(client.post(route, files=files))
        document_ids.extend(str(doc["id"]) for doc in data)
    return document_ids


def list_documents(client: httpx.Client, dataset_id: str) -> list[dict[str, Any]]:
    """Every document in the dataset, paging until the listing is exhausted.

    The listing endpoint is paginated. Reading only the first page would let a
    corpus larger than one page report complete while most of it is still
    unparsed, so the RAGFlow arm would be queried against a partial index.
    """
    route = DOCUMENTS_ROUTE.format(dataset_id=dataset_id)
    documents: list[dict[str, Any]] = []
    page = 1
    while True:
        data = _data(client.get(route, params={"page": page, "page_size": DOCUMENTS_PAGE_SIZE}))
        batch = data["docs"]
        documents.extend(batch)
        total = data.get("total")
        if not batch or (total is not None and len(documents) >= int(total)):
            return documents
        page += 1


def parse_and_wait(client: httpx.Client, dataset_id: str, document_ids: list[str]) -> None:
    """Trigger parsing and block until every uploaded document finishes."""
    route = PARSE_ROUTE.format(dataset_id=dataset_id)
    _data(client.post(route, json={"document_ids": document_ids}))
    expected = len(document_ids)
    done = 0
    for _ in range(PARSE_MAX_POLLS):
        docs = list_documents(client, dataset_id)
        done = sum(1 for doc in docs if doc.get("progress", 0.0) >= DONE_RATIO)
        if len(docs) >= expected and done >= expected:
            return
        time.sleep(PARSE_POLL_SECONDS)
    raise RuntimeError(
        f"ragflow parsed {done} of {expected} documents within the poll budget; "
        "the arm would be queried against a partially indexed corpus"
    )


def create_assistant(client: httpx.Client, name: str, dataset_id: str, llm_model: str) -> str:
    """Create a chat assistant with pinned retrieval settings; return its id."""
    data = _data(
        client.post(
            CHATS_ROUTE,
            json={
                "name": name,
                "dataset_ids": [dataset_id],
                "llm": {"model_name": llm_model, "temperature": 0.0},
                "prompt": {
                    "similarity_threshold": RETRIEVAL_SIMILARITY_THRESHOLD,
                    "keywords_similarity_weight": RETRIEVAL_KEYWORDS_WEIGHT,
                    "top_n": RETRIEVAL_TOP_N,
                },
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
    print(f"uploaded {len(document_ids)} documents from {args.corpus_dir}")
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
