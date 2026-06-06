"""Thin httpx client for one llama-server OpenAI endpoint (local inference)."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import httpx

from lilbee.providers.base import (
    ChatResult,
    ChatToolResult,
    FinishReason,
    ProviderError,
    ProviderErrorKind,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
)

_PROVIDER_NAME = "llama-server"
# Reranker pair format: query and candidate are joined with this separator into
# one document so a cross-encoder GGUF scores the pair as a single sequence.
_RERANK_PAIR_SEPARATOR = "</s></s>"
# Max sequences per /v1/embeddings request. Like the in-process backstop, a
# batch is bounded by BOTH the token budget (the server's n_batch, == token_cap)
# and this sequence count: a corpus of many tiny chunks would otherwise pack one
# request past the server's batch/sequence limit and trip a 500.
_EMBED_N_SEQ_MAX = 64
# Estimate a chunk's token count from its character length so the bulk embed
# path packs sub-batches without a /tokenize round-trip per input. The factor is
# held below the corpus average (data.chunk.CHARS_PER_TOKEN, 4 for Latin text)
# so the estimate over-counts tokens and a sub-batch never packs past the
# server's n_batch (== token_cap). Rerank does not estimate: its
# query</s></s>candidate pairs are token-dense (the separator is several tokens
# in a few chars), so char estimation would under-count and over-pack.
_EMBED_EST_CHARS_PER_TOKEN = 3

log = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    """Conservative (over-counting) token estimate from character length."""
    return max(1, -(-len(text) // _EMBED_EST_CHARS_PER_TOKEN))


def _raise_for_status(resp: httpx.Response) -> None:
    """Raise including the server's error body, which ``raise_for_status`` drops.

    A llama-server failure otherwise surfaces as a bare "Internal Server Error"
    with no cause; the response body carries the actual reason (oversize prompt,
    decode failure, ...), which both diagnosis and the user-facing error need.
    """
    if resp.is_success:
        return
    resp.read()  # streaming responses aren't read yet; a no-op for buffered ones
    body = resp.text.strip()
    # llama-server reports an oversize prompt/conversation as a 400 whose body
    # carries the "exceed_context_size_error" type. Tag it CONTEXT_OVERFLOW with a
    # user-facing message so the chat route returns a clean context_length_exceeded
    # (400) instead of a generic internal_error -- a long conversation that fills
    # the window then reads as "too long", not "Internal server error".
    if resp.status_code == _HTTP_BAD_REQUEST and (
        "exceed_context_size" in body.lower() or "context size" in body.lower()
    ):
        raise ProviderError(
            "The conversation exceeds this model's context window. "
            "Start a new conversation or shorten the input.",
            provider=_PROVIDER_NAME,
            kind=ProviderErrorKind.CONTEXT_OVERFLOW,
        )
    detail = f": {body[:600]}" if body else ""
    raise ProviderError(
        f"llama-server returned HTTP {resp.status_code}{detail}",
        provider=_PROVIDER_NAME,
    )


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
# llama-swap proxies native (non-OpenAI) llama.cpp routes only under
# /upstream/<model>/...; the bare /tokenize path 404s (it routes /v1/* by the
# body's model field, but a native route carries no such field).
_UPSTREAM_PREFIX = "/upstream"
# Match the in-process tokenizer call (llm.tokenize(text, add_bos=True, special=False)):
# the server adds BOS via add_special and leaves special-token strings unparsed.
_TOKENIZE_ADD_SPECIAL = True
_TOKENIZE_PARSE_SPECIAL = False
_HTTP_OK = 200
_HTTP_BAD_REQUEST = 400
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
        timeout: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._model = model
        self._http = http or httpx.Client(base_url=self._base, timeout=timeout)
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
            _raise_for_status(resp)
            return str(resp.json()["choices"][0]["message"]["content"])

    def _chat_stream(self, payload: dict[str, Any]) -> Iterator[str]:
        with (
            self._track(),
            self._http.stream("POST", _CHAT_PATH, json={**payload, "stream": True}) as resp,
        ):
            _raise_for_status(resp)
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
            _raise_for_status(resp)
            message = resp.json()["choices"][0]["message"]
        content = message.get("content") or ""
        native = _parse_native_tool_calls(message.get("tool_calls"))
        if native:
            return ChatToolResult(content=content, tool_calls=native)
        return _recover_bare_json_tool_calls(content)

    def chat_result(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ChatResult:
        """Non-streaming chat returning text, tool calls, and a finish reason.

        The server is launched with ``--jinja`` so it parses the model's native
        tool-call syntax into structured ``message.tool_calls``. When a model
        instead emits a bare-JSON call as content (a native miss), recover it
        and report ``tool_calls`` as the finish reason.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            **(options or {}),
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        with self._track():
            resp = self._http.post(_CHAT_PATH, json=payload)
            _raise_for_status(resp)
            body = resp.json()
        choice = body["choices"][0]
        usage = _usage_from_body(body) or TokenUsage()
        message = choice["message"]
        content = message.get("content") or ""
        finish_reason = _coerce_finish_reason(choice.get("finish_reason"))
        native = _parse_native_tool_calls(message.get("tool_calls"))
        if native:
            return ChatResult(
                text=content,
                tool_calls=tuple(native),
                finish_reason=finish_reason,
                usage=usage,
            )
        recovered = _recover_bare_json_tool_calls(content)
        if recovered.tool_calls:
            return ChatResult(
                text=recovered.content,
                tool_calls=tuple(recovered.tool_calls),
                finish_reason=FinishReason.TOOL_CALLS,
                usage=usage,
            )
        return ChatResult(text=content, tool_calls=(), finish_reason=finish_reason, usage=usage)

    def chat_stream_items(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> Iterator[str | ToolCallDelta | TokenUsage]:
        """Stream text tokens and tool-call deltas from the server's OpenAI SSE.

        Each SSE chunk's ``choices[0].delta`` carries a ``content`` token and/or
        a ``tool_calls`` array; both are surfaced as :data:`ChatStreamItem`
        frames (text strings and :class:`ToolCallDelta`). The dispatch's stream
        translator accumulates the deltas by ``index``.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            # include_usage makes llama-server emit a final SSE chunk carrying the
            # token usage (with an empty choices list) just before [DONE].
            "stream_options": {"include_usage": True},
            **(options or {}),
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        with (
            self._track(),
            self._http.stream("POST", _CHAT_PATH, json={**payload, "stream": True}) as resp,
        ):
            _raise_for_status(resp)
            for line in resp.iter_lines():
                yield from _parse_sse_stream_items(line)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch via ``/v1/embeddings``."""
        if not texts:
            # Match the in-process embedder; the server rejects an empty input.
            return []
        vectors: list[list[float]] = []
        for sub_batch in self._truncate_and_subbatch(texts, estimate=True):
            data = self._embed_subbatch(sub_batch)
            vectors.extend(list(item["embedding"]) for item in data)
        return vectors

    def _embed_subbatch(self, sub_batch: list[str]) -> list[dict[str, Any]]:
        """Embed one estimate-budgeted sub-batch, re-truncating exactly on overflow.

        ``_estimate_tokens`` is char-based and can under-count token-dense inputs
        (XML, code), so an estimate-trusted input may still exceed the server's
        context. On that error -- and only that -- redo the batch with exact
        server-side tokenization, which truncates the oversize input to the cap.
        """
        try:
            return self._embeddings_call(sub_batch)
        except ProviderError as exc:
            if exc.kind is not ProviderErrorKind.CONTEXT_OVERFLOW:
                raise
            data: list[dict[str, Any]] = []
            for exact in self._truncate_and_subbatch(sub_batch, estimate=False):
                data.extend(self._embeddings_call(exact))
            return data

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
        scores: list[float] = []
        for sub_batch in self._truncate_and_subbatch(pairs, estimate=False):
            data = self._embeddings_call(sub_batch)
            scores.extend(_rerank_score(item) for item in data)
        return scores

    def _embeddings_call(self, inputs: list[str]) -> list[dict[str, Any]]:
        """POST one already-budgeted sub-batch to ``/v1/embeddings``; return its data."""
        with self._track():
            resp = self._http.post(
                _EMBED_PATH,
                json={
                    "model": self._model,
                    "input": inputs,
                    "embd_normalize": _EMBD_NORMALIZE_NONE,
                },
            )
            _raise_for_status(resp)
            data = resp.json()["data"]
        if len(data) != len(inputs):
            raise ProviderError(
                f"Embedder returned {len(data)} vectors for {len(inputs)} inputs",
                provider=_PROVIDER_NAME,
            )
        return list(data)

    def _truncate_and_subbatch(self, texts: list[str], *, estimate: bool) -> list[list[str]]:
        """Token-truncate over-cap inputs, then pack into server-sized sub-batches.

        An input longer than ``token_cap`` (the server's per-slot context /
        n_batch) is truncated to it via the server's tokenizer, since the server
        cannot split a pooled embedding sequence. Inputs are then grouped so each
        request stays within both the token budget and ``_EMBED_N_SEQ_MAX``
        sequences -- without this, a corpus of many small chunks packs one request
        past the server's batch limit and the server returns a 500. No cap
        (chat/vision) sends a single batch untouched.

        When ``estimate`` is set (the embed path) the per-input token count comes
        from :func:`_estimate_tokens`, and ``/tokenize`` is consulted only for the
        rare input whose estimate exceeds the cap -- eliminating a round-trip per
        chunk during bulk ingest. Rerank passes ``estimate=False`` to tokenize
        every pair exactly, since its pairs are too token-dense to estimate.
        """
        if self._token_cap is None:
            return [texts]
        cap = self._token_cap
        batches: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0
        for text in texts:
            item, item_tokens = self._fit_input(text, cap, estimate=estimate)
            if current and (current_tokens + item_tokens > cap or len(current) >= _EMBED_N_SEQ_MAX):
                batches.append(current)
                current = []
                current_tokens = 0
            current.append(item)
            current_tokens += item_tokens
        if current:
            batches.append(current)
        return batches

    def _fit_input(self, text: str, cap: int, *, estimate: bool) -> tuple[str, int]:
        """Return ``(input, token_count)`` for one sequence, truncating if over cap.

        Estimation short-circuits the common case: an estimate within the cap is
        trusted (no ``/tokenize``); only an over-cap estimate is confirmed against
        the server tokenizer and truncated if it really exceeds the cap.
        """
        if estimate:
            est = _estimate_tokens(text)
            if est <= cap:
                return text, est
        tokens = self._tokenize(text)
        if len(tokens) > cap:
            log.warning("Truncating oversize embed input: %d tokens > cap %d", len(tokens), cap)
            return self._detokenize(tokens[:cap]), cap
        return text, max(1, len(tokens))

    def _native_route(self, suffix: str) -> str:
        """Path for a native (non-OpenAI) llama-server route through llama-swap.

        llama-swap proxies these only under ``/upstream/<model>/...``; the model
        is carried in the path, not the body (unlike the ``/v1`` OpenAI routes).
        """
        return f"{_UPSTREAM_PREFIX}/{self._model}{suffix}"

    def _tokenize(self, text: str) -> list[int]:
        resp = self._http.post(
            self._native_route(_TOKENIZE_PATH),
            json={
                "content": text,
                "add_special": _TOKENIZE_ADD_SPECIAL,
                "parse_special": _TOKENIZE_PARSE_SPECIAL,
            },
        )
        _raise_for_status(resp)
        return list(resp.json()["tokens"])

    def _detokenize(self, tokens: list[int]) -> str:
        resp = self._http.post(self._native_route(_DETOKENIZE_PATH), json={"tokens": tokens})
        _raise_for_status(resp)
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


def _usage_from_body(body: Mapping[str, Any]) -> TokenUsage | None:
    """Read the ``usage`` block of an OpenAI response, or ``None`` if absent.

    llama-server reports ``prompt_tokens`` / ``completion_tokens``; a missing or
    malformed block yields ``None`` so callers can decide between a zero default
    (non-streaming) and skipping the frame (streaming terminator).
    """
    usage = body.get("usage")
    if not isinstance(usage, Mapping):
        return None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    return TokenUsage(
        prompt_tokens=prompt if isinstance(prompt, int) else 0,
        completion_tokens=completion if isinstance(completion, int) else 0,
    )


_FINISH_REASONS: dict[str, FinishReason] = {fr.value: fr for fr in FinishReason}


def _coerce_finish_reason(raw: Any) -> FinishReason:
    """Map a server-supplied finish_reason string to :class:`FinishReason`."""
    if not isinstance(raw, str):
        return FinishReason.STOP
    return _FINISH_REASONS.get(raw, FinishReason.STOP)


def _tool_call_delta_from_chunk(call: Mapping[str, Any], *, fallback_index: int) -> ToolCallDelta:
    """Map one streaming ``delta.tool_calls`` entry to a :class:`ToolCallDelta`.

    Mirrors the SDK path: ``id`` / ``name`` arrive on the opener and accumulate
    by ``index``; empty strings normalise to ``None`` so the dispatch's stream
    translator (which gates on ``is not None``) does not emit spurious openers.
    """
    raw_index = call.get("index")
    index = raw_index if isinstance(raw_index, int) else fallback_index
    call_id = call.get("id")
    fn = call.get("function")
    raw_name = fn.get("name") if isinstance(fn, Mapping) else None
    raw_args = fn.get("arguments") if isinstance(fn, Mapping) else None
    return ToolCallDelta(
        index=index,
        id=str(call_id) if call_id else None,
        name=str(raw_name) if raw_name else None,
        arguments_delta=str(raw_args) if raw_args else None,
    )


def _parse_sse_stream_items(line: str) -> Iterator[str | ToolCallDelta | TokenUsage]:
    """Yield text tokens and tool-call deltas from one OpenAI SSE line.

    A chunk can carry a ``content`` token, a ``tool_calls`` delta array, or
    both; each is yielded as its own :data:`ChatStreamItem` frame.
    """
    if not line.startswith(_DATA_PREFIX):
        return
    body = line[len(_DATA_PREFIX) :].strip()
    if not body or body == _DONE_SENTINEL:
        return
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return
    choices = obj.get("choices") or []
    if not choices:
        # The include_usage terminator chunk has an empty choices list and the
        # token totals on a top-level ``usage`` block; surface it as the final
        # frame so the dispatch can attach real counts to the stream.
        usage = _usage_from_body(obj)
        if usage is not None:
            yield usage
        return
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    if content:
        yield str(content)
    raw_calls = delta.get("tool_calls") or []
    for i, call in enumerate(raw_calls):
        if isinstance(call, Mapping):
            yield _tool_call_delta_from_chunk(call, fallback_index=i)


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
