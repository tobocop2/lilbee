"""Thin httpx client for one llama-server OpenAI endpoint (local inference)."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import httpx

from lilbee.providers.base import ChatToolResult, ProviderError, ToolCall

_PROVIDER_NAME = "llama-server"
# Reranker pair format: query and candidate are joined with this separator into
# one document so a cross-encoder GGUF scores the pair as a single sequence.
_RERANK_PAIR_SEPARATOR = "</s></s>"
# Match the in-process embedder: llama-cpp-python's create_embedding does not
# normalize (normalize=False), but llama-server normalizes pooled embeddings with
# embd_normalize=2 (L2) by default. We send -1 (no normalization) per request so
# the fleet returns the same raw vectors, and so rank-pooling rerank scores (a
# single value per pair) are not collapsed to +-1 by L2 normalization. The server
# only exposes this per request body, not as a startup flag.
_EMBD_NORMALIZE_NONE = -1
_HEALTH_PATH = "/health"
_CHAT_PATH = "/v1/chat/completions"
_EMBED_PATH = "/v1/embeddings"
_TOKENIZE_PATH = "/tokenize"
_DETOKENIZE_PATH = "/detokenize"
# Match the in-process tokenizer call (llm.tokenize(text, add_bos=True, special=False)):
# the server adds BOS via add_special and leaves special-token strings unparsed.
_TOKENIZE_ADD_SPECIAL = True
_TOKENIZE_PARSE_SPECIAL = False
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

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        http: httpx.Client | None = None,
        token_cap: int | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._model = model
        self._http = http or httpx.Client(base_url=self._base, timeout=_DEFAULT_TIMEOUT_S)
        self._owns_http = http is None
        # Per-slot context for embed/rerank servers: inputs longer than this are
        # token-truncated (via the server's tokenizer) before embedding, mirroring
        # the in-process backstop. None for chat/vision, which don't truncate inputs.
        self._token_cap = token_cap
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

    def chat_tools(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ChatToolResult:
        """Non-streaming chat with function tools; returns content + any tool calls.

        The server is launched with ``--jinja`` so it parses the model's native
        tool-call syntax into structured ``message.tool_calls``. When a model
        instead emits a bare-JSON call as content (a native miss), recover it.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            **(options or {}),
        }
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        with self._track():
            resp = self._http.post(_CHAT_PATH, json=payload)
            resp.raise_for_status()
            message = resp.json()["choices"][0]["message"]
        content = message.get("content") or ""
        native = _parse_native_tool_calls(message.get("tool_calls"))
        if native:
            return ChatToolResult(content=content, tool_calls=native)
        return _recover_bare_json_tool_calls(content)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch via ``/v1/embeddings``."""
        if not texts:
            # Match the in-process embedder; the server rejects an empty input.
            return []
        texts = self._truncate_to_cap(texts)
        with self._track():
            resp = self._http.post(
                _EMBED_PATH,
                json={
                    "model": self._model,
                    "input": texts,
                    "embd_normalize": _EMBD_NORMALIZE_NONE,
                },
            )
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
        pairs = self._truncate_to_cap(
            [f"{query}{_RERANK_PAIR_SEPARATOR}{candidate}" for candidate in candidates]
        )
        with self._track():
            resp = self._http.post(
                _EMBED_PATH,
                json={
                    "model": self._model,
                    "input": pairs,
                    "embd_normalize": _EMBD_NORMALIZE_NONE,
                },
            )
            resp.raise_for_status()
            data = resp.json()["data"]
        if len(data) != len(pairs):
            raise ProviderError(
                f"Reranker returned {len(data)} entries for {len(pairs)} pairs",
                provider=_PROVIDER_NAME,
            )
        return [_rerank_score(item) for item in data]

    def _truncate_to_cap(self, texts: list[str]) -> list[str]:
        """Token-truncate any input longer than the cap, via the server's tokenizer.

        Mirrors the in-process backstop: the server cannot split a pooled
        embedding sequence, so an input longer than the context errors instead of
        truncating. We tokenize, keep the first ``token_cap`` tokens, and
        detokenize. No cap (chat/vision) leaves inputs untouched.
        """
        if self._token_cap is None:
            return texts
        out: list[str] = []
        for text in texts:
            tokens = self._tokenize(text)
            if len(tokens) > self._token_cap:
                out.append(self._detokenize(tokens[: self._token_cap]))
            else:
                out.append(text)
        return out

    def _tokenize(self, text: str) -> list[int]:
        resp = self._http.post(
            _TOKENIZE_PATH,
            json={
                "content": text,
                "add_special": _TOKENIZE_ADD_SPECIAL,
                "parse_special": _TOKENIZE_PARSE_SPECIAL,
            },
        )
        resp.raise_for_status()
        return list(resp.json()["tokens"])

    def _detokenize(self, tokens: list[int]) -> str:
        resp = self._http.post(_DETOKENIZE_PATH, json={"tokens": tokens})
        resp.raise_for_status()
        return str(resp.json()["content"])

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


def _arguments_to_str(arguments: Any) -> str:
    """Normalize a tool-call ``arguments`` value to a JSON string (OpenAI's shape)."""
    if isinstance(arguments, str):
        return arguments
    if arguments is None:
        return "{}"
    return json.dumps(arguments)


def _parse_native_tool_calls(raw: Any) -> list[ToolCall]:
    """Map a response's ``message.tool_calls`` array to :class:`ToolCall` objects.

    Reads the OpenAI shape (``{"id", "function": {"name", "arguments"}}``) that
    ``--jinja`` produces. Malformed or nameless entries are skipped.
    """
    if not isinstance(raw, list):
        return []
    calls: list[ToolCall] = []
    for idx, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            continue
        fn = entry.get("function")
        if not isinstance(fn, Mapping):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        call_id = entry.get("id")
        calls.append(
            ToolCall(
                id=call_id if isinstance(call_id, str) and call_id else f"call_{idx}",
                name=name,
                arguments=_arguments_to_str(fn.get("arguments")),
            )
        )
    return calls


def _bare_call_from_mapping(obj: Mapping[str, Any], *, index: int) -> ToolCall | None:
    """Build a ToolCall from a bare ``{"name", "arguments"|"parameters"}`` object."""
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        return None
    arguments = obj.get("arguments")
    if arguments is None:
        arguments = obj.get("parameters")
    return ToolCall(id=f"call_{index}", name=name, arguments=_arguments_to_str(arguments))


def _recover_bare_json_tool_calls(content: str) -> ChatToolResult:
    """Recover a tool call a model emitted as bare-JSON content (a native miss).

    Some models ignore the tool-call protocol and print ``{"name": ...,
    "arguments": {...}}`` (or a list of them) as the message body. When the whole
    content parses as such, treat it as the call(s) and clear the text; otherwise
    return the content unchanged with no calls.
    """
    stripped = content.strip()
    if not stripped or stripped[0] not in "{[":
        return ChatToolResult(content=content, tool_calls=[])
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return ChatToolResult(content=content, tool_calls=[])
    entries = parsed if isinstance(parsed, list) else [parsed]
    calls = [
        call
        for idx, entry in enumerate(entries)
        if isinstance(entry, Mapping) and (call := _bare_call_from_mapping(entry, index=idx))
    ]
    if not calls:
        return ChatToolResult(content=content, tool_calls=[])
    return ChatToolResult(content="", tool_calls=calls)
