"""The fleet client against OpenVINO Model Server's documented REST contract, with faked answers.

No machine ran OVMS for these tests. The fake server follows the OVMS 2026.4.1 sources and docs:

- Routes: ``src/http_rest_api_handler.cpp`` at v2026.4.1 sends every path that matches
  ``(?:/v3/|/v1/).*?(/|$)`` to the generative handler, so ``/v1/chat/completions`` and
  ``/v1/embeddings`` (the paths llama-swap forwards) reach the same servable as ``/v3/...``.
- Embeddings: docs.openvino.ai/2026/model-server/ovms_docs_rest_api_embeddings.html. The
  ``encoding_format`` field takes ``float`` or ``base64``; the answer is the OpenAI list shape.
- Rerank: docs.openvino.ai/2026/model-server/ovms_docs_rest_api_rerank.html. The body holds
  ``model``, ``query`` and ``documents`` as strings; the answer is ``results`` of ``index`` and
  ``relevance_score``. A request with more than 100 documents is refused by default.
- Chat: docs.openvino.ai/2026/model-server/ovms_docs_rest_api_chat.html, the OpenAI shape, with
  an image as an ``image_url`` content part.

The fake also plays llama-swap: an OpenAI route goes to the member with the model rewritten to
the served name (``useModelName``), and ``/upstream/<model>/<path>`` goes to ``/<path>``.
"""

from __future__ import annotations

import base64
import json
import re
import struct
from typing import Any

import httpx
import pytest

from lilbee.providers.fleet.client import LlamaServerClient
from lilbee.providers.fleet.engines.ovms import OVMS
from lilbee.providers.roles import WorkerRole

MODEL = "lilbee-embed-0"
SERVED = "default_model"
_OVMS_ROUTE = re.compile(r"(?:/v3/|/v1/)(?:v1/)?(.+)")
_DOC_LIMIT = 100


def _b64(values: list[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode()


class _FakeOvms:
    """One OVMS member behind a fake llama-swap; it keeps every request it served."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content)
        prefix = f"/upstream/{MODEL}"
        if path.startswith(prefix):
            path = path[len(prefix) :]
        elif body.get("model") == MODEL:
            body = {**body, "model": SERVED}
        found = _OVMS_ROUTE.fullmatch(path)
        if found is None or body.get("model") != SERVED:
            return httpx.Response(404, json={"error": "Model with requested name is not found"})
        self.seen.append((found.group(1), body))
        handler = {
            "embeddings": self._embeddings,
            "rerank": self._rerank,
            "chat/completions": self._chat,
        }.get(found.group(1))
        if handler is None:
            return httpx.Response(404, json={"error": "Unsupported endpoint"})
        return handler(body)

    @staticmethod
    def _vector(text: str) -> list[float]:
        return [float(len(text)), 0.5, -1.0]

    def _embeddings(self, body: dict[str, Any]) -> httpx.Response:
        inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
        b64 = body.get("encoding_format", "float") == "base64"
        data = [
            {
                "object": "embedding",
                "embedding": _b64(self._vector(t)) if b64 else self._vector(t),
                "index": i,
            }
            for i, t in enumerate(inputs)
        ]
        usage = {"prompt_tokens": 6, "total_tokens": 6}
        return httpx.Response(200, json={"object": "list", "data": data, "usage": usage})

    @staticmethod
    def _rerank(body: dict[str, Any]) -> httpx.Response:
        documents = body["documents"]
        if len(documents) > _DOC_LIMIT or not all(isinstance(d, str) for d in documents):
            return httpx.Response(400, json={"error": "Number of documents exceeds max_allowed"})
        scored = [
            {"index": i, "relevance_score": 1.0 / (1 + len(d))} for i, d in enumerate(documents)
        ]
        # The server answers in score order, not in document order.
        scored.sort(key=lambda r: r["relevance_score"], reverse=True)
        return httpx.Response(200, json={"results": scored[: body.get("top_n", len(scored))]})

    @staticmethod
    def _chat(body: dict[str, Any]) -> httpx.Response:
        if body.get("stream"):
            chunks = [
                {"choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}]}
                for part in ("Bees ", "make honey.")
            ]
            text = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})
        message = {"role": "assistant", "content": "Bees make honey."}
        choice = {"index": 0, "message": message, "finish_reason": "stop"}
        return httpx.Response(200, json={"object": "chat.completion", "choices": [choice]})


def _client(role: WorkerRole, server: _FakeOvms) -> LlamaServerClient:
    http = httpx.Client(transport=httpx.MockTransport(server), base_url="http://swap")
    return LlamaServerClient("http://swap", MODEL, http=http, profile=OVMS.request_profile(role))


def test_embed_reads_the_base64_vectors_of_the_embeddings_route() -> None:
    server = _FakeOvms()
    vectors = _client(WorkerRole.EMBED, server).embed(["bee", "honeycomb"])
    assert [list(v) for v in vectors] == [[3.0, 0.5, -1.0], [9.0, 0.5, -1.0]]
    endpoint, body = server.seen[0]
    assert endpoint == "embeddings"
    assert body["encoding_format"] == "base64"
    assert body["input"] == ["bee", "honeycomb"]


def test_rerank_posts_the_cohere_body_and_orders_scores_by_candidate() -> None:
    server = _FakeOvms()
    candidates = ["a", "abcd", "ab"]
    scores = _client(WorkerRole.RERANK, server).rerank("q", candidates)
    assert scores == pytest.approx([1 / 2, 1 / 5, 1 / 3])
    endpoint, body = server.seen[0]
    assert (endpoint, body["query"], body["documents"]) == ("rerank", "q", candidates)


def test_rerank_splits_candidates_at_the_server_limit_of_100_documents() -> None:
    server = _FakeOvms()
    candidates = [f"doc {i}" for i in range(150)]
    scores = _client(WorkerRole.RERANK, server).rerank("q", candidates)
    assert len(scores) == 150
    assert [len(body["documents"]) for _, body in server.seen] == [100, 50]


def test_chat_reads_the_answer_and_the_stream() -> None:
    server = _FakeOvms()
    client = _client(WorkerRole.CHAT, server)
    messages = [{"role": "user", "content": "What do bees make?"}]
    assert client.chat(messages) == "Bees make honey."
    assert "".join(client.chat(messages, stream=True)) == "Bees make honey."
    assert [endpoint for endpoint, _ in server.seen] == ["chat/completions"] * 2


def test_an_image_chat_sends_the_image_as_an_image_url_part() -> None:
    server = _FakeOvms()
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
    content = [{"type": "text", "text": "What is in the picture?"}, image]
    answer = _client(WorkerRole.CHAT, server).chat([{"role": "user", "content": content}])
    assert answer == "Bees make honey."
    _, body = server.seen[0]
    assert body["messages"][0]["content"][1] == image
