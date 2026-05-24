"""Thin httpx client for one llama-server OpenAI endpoint (local inference)."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import httpx

from lilbee.providers.base import ProviderError

# Single source of truth for the reranker pair format: the fleet must join
# query/candidate exactly as the in-process reranker does, or scores drift.
from lilbee.providers.llama_cpp.batching import _RERANK_PAIR_SEPARATOR

_PROVIDER_NAME = "multi-gpu"
_HEALTH_PATH = "/health"
_CHAT_PATH = "/v1/chat/completions"
_EMBED_PATH = "/v1/embeddings"
_HTTP_OK = 200
_DONE_SENTINEL = "[DONE]"
_DATA_PREFIX = "data:"
_DEFAULT_TIMEOUT_S = 300.0
# Short, separate timeout for /health: a server can wedge under heavy prompt
# processing, and readiness/monitor polls must not block on the request timeout.
_HEALTH_TIMEOUT_S = 5.0


class LlamaServerClient:
    """Calls one llama-server's OpenAI surface. Tracks in-flight requests so the
    fleet router can pick the least-busy replica."""

    def __init__(self, base_url: str, model: str, *, http: httpx.Client | None = None) -> None:
        self._base = base_url.rstrip("/")
        self._model = model
        self._http = http or httpx.Client(base_url=self._base, timeout=_DEFAULT_TIMEOUT_S)
        self._owns_http = http is None
        self.in_flight = 0
        self._in_flight_lock = threading.Lock()

    def health(self) -> bool:
        """True iff ``GET /health`` returns 200 (liveness, not readiness)."""
        try:
            resp = self._http.get(_HEALTH_PATH, timeout=_HEALTH_TIMEOUT_S)
        except httpx.HTTPError:
            return False
        return resp.status_code == _HTTP_OK

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        options: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """Chat completion. Returns the full text, or a token iterator if streaming.

        ``messages`` accepts both plain ``{role, content: str}`` and multipart
        ``content`` lists (vision image parts), so the vision path reuses this.
        """
        payload: dict[str, Any] = {"model": self._model, "messages": messages, **(options or {})}
        if stream:
            return self._chat_stream(payload)
        with self._track():
            resp = self._http.post(_CHAT_PATH, json={**payload, "stream": False})
            resp.raise_for_status()
            return str(resp.json()["choices"][0]["message"]["content"])

    def _chat_stream(self, payload: dict[str, Any]) -> Iterator[str]:
        with (
            self._track(),
            self._http.stream("POST", _CHAT_PATH, json={**payload, "stream": True}) as resp,
        ):
            resp.raise_for_status()
            for line in resp.iter_lines():
                delta = _parse_sse_delta(line)
                if delta:
                    yield delta

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch via ``/v1/embeddings``."""
        with self._track():
            resp = self._http.post(_EMBED_PATH, json={"model": self._model, "input": texts})
            resp.raise_for_status()
            return [list(item["embedding"]) for item in resp.json()["data"]]

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        """Relevance scores via rank-pooling embeddings (mirrors the in-process path).

        The server runs with ``--pooling rank``; we send ``query</s></s>candidate``
        pairs to ``/v1/embeddings`` and read each item's first embedding value as the
        score -- the same primitive and pairing as ``compute_rerank_scores``, so the
        ``/v1/rerank`` template-dependency (and its zero-output failure modes) is moot.
        """
        if not candidates:
            return []
        pairs = [f"{query}{_RERANK_PAIR_SEPARATOR}{candidate}" for candidate in candidates]
        with self._track():
            resp = self._http.post(_EMBED_PATH, json={"model": self._model, "input": pairs})
            resp.raise_for_status()
            data = resp.json()["data"]
        if len(data) != len(pairs):
            raise ProviderError(
                f"Reranker returned {len(data)} entries for {len(pairs)} pairs",
                provider=_PROVIDER_NAME,
            )
        return [_rerank_score(item) for item in data]

    def close(self) -> None:
        """Close the underlying client if this instance created it."""
        if self._owns_http:
            self._http.close()

    def _track(self) -> _InFlight:
        return _InFlight(self)


class _InFlight:
    """Context manager that atomically bumps the owner's in-flight counter.

    ``+= 1`` is a read-modify-write, so concurrent chat/embed calls would corrupt
    the counter the router balances on; the client's lock makes it atomic.
    """

    def __init__(self, client: LlamaServerClient) -> None:
        self._client = client

    def __enter__(self) -> None:
        with self._client._in_flight_lock:
            self._client.in_flight += 1

    def __exit__(self, *_exc: object) -> None:
        with self._client._in_flight_lock:
            self._client.in_flight -= 1


def _rerank_score(item: dict[str, Any]) -> float:
    """Pull one relevance score from a rank-pooling ``/v1/embeddings`` item."""
    embedding = item.get("embedding")
    if isinstance(embedding, list) and embedding and isinstance(embedding[0], (int, float)):
        return float(embedding[0])
    raise ProviderError(
        f"Reranker returned unexpected score shape: {embedding!r}", provider=_PROVIDER_NAME
    )


def _parse_sse_delta(line: str) -> str:
    """Extract the content delta from one OpenAI SSE line, ``""`` if none."""
    if not line.startswith(_DATA_PREFIX):
        return ""
    body = line[len(_DATA_PREFIX) :].strip()
    if not body or body == _DONE_SENTINEL:
        return ""
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return ""
    choices = obj.get("choices") or []
    if not choices:
        return ""
    return str(choices[0].get("delta", {}).get("content") or "")
