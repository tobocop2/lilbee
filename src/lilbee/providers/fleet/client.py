"""Thin httpx client for one llama-server OpenAI endpoint (local inference)."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

import httpx

from lilbee.core.config import cfg
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
from lilbee.providers.fleet.adapters import LLM_RERANK_CONCURRENCY
from lilbee.providers.roles import RerankMode

_PROVIDER_NAME = "llama-server"
# Reranker pair format: query and candidate are joined with this separator into
# one document so a cross-encoder GGUF scores the pair as a single sequence.
_RERANK_PAIR_SEPARATOR = "</s></s>"
# LLM reranker: score each candidate by the yes/no first-token logprob.
_LLM_RERANK_PROMPT = (
    "Judge whether the document is relevant to the query. "
    "Answer with only 'yes' or 'no'.\n\nQuery: {query}\nDocument: {document}"
)
_LLM_RERANK_TOP_LOGPROBS = 20
_YES_LABEL = "yes"
_NO_LABEL = "no"
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
# llama-swap's error body when the spawned llama-server exited before serving.
_UPSTREAM_DIED_MARKER = "exited prematurely"
# llama-server's 500 body when one input exceeds the physical batch (n_batch).
_BATCH_OVERFLOW_MARKER = "too large to process"
_UPSTREAM_LOG_TAIL_CHARS = 2000
_UPSTREAM_LOG_TIMEOUT_S = 2.0

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
    # A 429 (slots full) is transient: a cold replica fleet rejects the first ingest
    # fan-out until its slots load. Tag RATE_LIMIT so the caller backs off and retries
    # instead of dropping the input.
    if resp.status_code == _HTTP_TOO_MANY_REQUESTS:
        raise ProviderError(
            "llama-server is busy (HTTP 429); replicas may still be warming.",
            provider=_PROVIDER_NAME,
            kind=ProviderErrorKind.RATE_LIMIT,
        )
    detail = f": {body[:600]}" if body else ""
    # llama-swap reports a dead server only as "exited prematurely"; log the
    # server's own captured output so the actual exit reason is diagnosable.
    if _UPSTREAM_DIED_MARKER in body:
        _log_upstream_tail(resp)
    raise ProviderError(
        f"llama-server returned HTTP {resp.status_code}{detail}",
        provider=_PROVIDER_NAME,
        kind=_classify_error_body(body),
    )


def _classify_error_body(body: str) -> ProviderErrorKind:
    """Error kind from a llama-server/llama-swap error body.

    An input past the server's n_batch is a 500 whose body says "too large to
    process" (CONTEXT_OVERFLOW, so the embed path re-truncates exactly); a dead
    upstream is CONNECTION, so the router can mark the replica unhealthy.
    """
    if _BATCH_OVERFLOW_MARKER in body:
        return ProviderErrorKind.CONTEXT_OVERFLOW
    if _UPSTREAM_DIED_MARKER in body:
        return ProviderErrorKind.CONNECTION
    return ProviderErrorKind.UNKNOWN


def is_connection_failure(exc: Exception) -> bool:
    """Whether *exc* signals a dead/unreachable replica rather than a model error."""
    if isinstance(exc, httpx.TransportError):
        return True
    # isinstance: only ProviderError carries a kind; other exceptions pass through.
    return isinstance(exc, ProviderError) and exc.kind is ProviderErrorKind.CONNECTION


def _log_upstream_tail(resp: httpx.Response) -> None:
    """Log the dead upstream's recent output from llama-swap's log stream."""
    with contextlib.suppress(httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError):
        base = str(resp.request.url).split("/v1/")[0]
        model = json.loads(resp.request.content)["model"]
        tail = _fetch_log_tail(f"{base}/logs/stream/{model}")
        if tail:
            log.warning("%s exited prematurely; recent server output:\n%s", model, tail)


def _fetch_log_tail(url: str) -> str:
    """The last ``_UPSTREAM_LOG_TAIL_CHARS`` of llama-swap's log stream for one model.

    The stream replays the upstream's buffered output then stays open; the read
    timeout is the cutoff once the replay is drained.
    """
    chunks: list[str] = []
    with (
        contextlib.suppress(httpx.HTTPError),
        httpx.stream("GET", url, timeout=_UPSTREAM_LOG_TIMEOUT_S) as stream,
    ):
        for chunk in stream.iter_text():
            chunks.append(chunk)
            if sum(len(piece) for piece in chunks) > _UPSTREAM_LOG_TAIL_CHARS:
                break
    return "".join(chunks)[-_UPSTREAM_LOG_TAIL_CHARS:]


# llama-server L2-normalizes pooled embeddings by default (embd_normalize=2);
# every embeddings request sends embd_normalize=-1 so the engine returns raw
# vectors, and so a rank-pooling rerank score (a single value per pair) is not
# collapsed to +-1 by normalization. The server only exposes this per request
# body, not as a startup flag.
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
_HTTP_TOO_MANY_REQUESTS = 429
_DONE_SENTINEL = "[DONE]"
_DATA_PREFIX = "data:"
_DEFAULT_TIMEOUT_S = 300.0
# Short, separate timeout for /health: a server can wedge under heavy prompt
# processing, and readiness/monitor polls must not block on the request timeout.
_HEALTH_TIMEOUT_S = 5.0
# Retry a server-busy (HTTP 429) response this many times with exponential backoff:
# a cold replica fleet 429s the first ingest fan-out until its slots load.
_BUSY_RETRIES = 6
_BUSY_BACKOFF_BASE_S = 0.5
# Half-open recovery: a replica marked unhealthy becomes routable again after
# this cool-down, so one live request probes it (success restores it, another
# connection failure re-stamps the cool-down).
_UNHEALTHY_RETRY_S = 30.0
_T = TypeVar("_T")


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
        rerank_mode: RerankMode | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._model = model
        self._http = http or httpx.Client(base_url=self._base, timeout=timeout)
        self._owns_http = http is None
        # Per-slot context for embed/rerank servers: inputs longer than this are
        # token-truncated (via the server's tokenizer) before embedding, mirroring
        # the in-process backstop. None for chat/vision, which don't truncate inputs.
        self._token_cap = token_cap
        # LLM => score candidates by yes/no logprob; None/cross-encoder => rank pooling.
        self._rerank_mode = rerank_mode
        self.in_flight = 0
        self._in_flight_lock = threading.Lock()
        # Routing health: cleared on a connection-level failure so the router
        # skips this replica; restored by a successful call, or half-open after
        # the cool-down (the next routed request is the probe).
        self._healthy = True
        # Monotonic stamp of the last mark_unhealthy; consulted only while unhealthy.
        self._unhealthy_since = 0.0

    @property
    def healthy(self) -> bool:
        """Whether the router should offer this replica traffic.

        An unhealthy replica becomes routable again ``_UNHEALTHY_RETRY_S`` after
        it was marked, so one live request probes it: a success marks it healthy,
        another connection failure re-stamps the cool-down.
        """
        with self._in_flight_lock:
            if self._healthy:
                return True
            return time.monotonic() - self._unhealthy_since >= _UNHEALTHY_RETRY_S

    def mark_unhealthy(self) -> None:
        """Record a connection-level failure so the router skips this replica."""
        with self._in_flight_lock:
            self._healthy = False
            self._unhealthy_since = time.monotonic()

    def mark_healthy(self) -> None:
        """Restore the replica to the routing pool after a successful call."""
        with self._in_flight_lock:
            self._healthy = True

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
        timeout: float | None = None,
    ) -> str | Iterator[str]:
        """Chat completion. Returns the full text, or a token iterator if streaming.

        ``messages`` accepts both plain ``{role, content: str}`` and multipart
        ``content`` lists (vision image parts), so the vision path reuses this.
        ``timeout`` overrides the client default for the non-streaming request,
        so a caller-enforced deadline (vision OCR) ends the request itself.
        """
        payload: dict[str, Any] = {"model": self._model, "messages": messages, **(options or {})}
        if stream:
            return self._chat_stream(payload)
        request_timeout = timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT
        with self._track():
            resp = self._http.post(
                _CHAT_PATH, json={**payload, "stream": False}, timeout=request_timeout
            )
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
        if self._rerank_mode is RerankMode.LLM:
            return self._rerank_llm(query, candidates)
        pairs = [f"{query}{_RERANK_PAIR_SEPARATOR}{candidate}" for candidate in candidates]
        scores: list[float] = []
        for sub_batch in self._truncate_and_subbatch(pairs, estimate=False):
            data = self._embeddings_call(sub_batch)
            scores.extend(_rerank_score(item) for item in data)
        return scores

    def _rerank_llm(self, query: str, candidates: list[str]) -> list[float]:
        """Score each candidate by an LLM's yes/no first-token logprob."""
        template = cfg.reranker_prompt or _LLM_RERANK_PROMPT
        workers = min(LLM_RERANK_CONCURRENCY, len(candidates))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda c: self._llm_rerank_one(template, query, c), candidates))

    def _llm_rerank_one(self, template: str, query: str, candidate: str) -> float:
        """One chat request scoring a single candidate's relevance to the query."""
        content = template.format(query=query, document=candidate)
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": _LLM_RERANK_TOP_LOGPROBS,
            "stream": False,
        }

        def _call() -> dict[str, Any]:
            with self._track():
                resp = self._http.post(_CHAT_PATH, json=payload)
                _raise_for_status(resp)
                return dict(resp.json())

        return _llm_rerank_score(_first_token_top_logprobs(self._retry_on_busy(_call)))

    def _retry_on_busy(self, call: Callable[[], _T]) -> _T:
        """Run *call*, retrying a transient server-busy (RATE_LIMIT) with backoff.

        A cold replica fleet 429s the first ingest fan-out until its slots load;
        backing off and retrying turns those drops into successes. Non-RATE_LIMIT
        errors (and a final still-busy response) propagate to the caller.
        """
        delay = _BUSY_BACKOFF_BASE_S
        for _ in range(_BUSY_RETRIES - 1):
            try:
                return call()
            except ProviderError as exc:
                if exc.kind is not ProviderErrorKind.RATE_LIMIT:
                    raise
                time.sleep(delay)
                delay *= 2
        return call()

    def _embeddings_call(self, inputs: list[str]) -> list[dict[str, Any]]:
        """POST one already-budgeted sub-batch to ``/v1/embeddings``; return its data."""

        def _call() -> list[dict[str, Any]]:
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

        return self._retry_on_busy(_call)

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


def _first_token_top_logprobs(response: dict[str, Any]) -> list[dict[str, Any]]:
    """The first generated token's top_logprobs list from a chat completion, or []."""
    choices = response.get("choices") or []
    if not choices:
        return []
    content = (choices[0].get("logprobs") or {}).get("content") or []
    if not content:
        return []
    return list(content[0].get("top_logprobs") or [])


def _llm_rerank_score(top_logprobs: list[dict[str, Any]]) -> float:
    """Softmax of the yes vs no logprobs in a token's top_logprobs (case/space-insensitive)."""
    yes_lp: float | None = None
    no_lp: float | None = None
    for entry in top_logprobs:
        token = str(entry.get("token", "")).strip().lower()
        logprob = float(entry.get("logprob", 0.0))
        if token == _YES_LABEL and (yes_lp is None or logprob > yes_lp):
            yes_lp = logprob
        elif token == _NO_LABEL and (no_lp is None or logprob > no_lp):
            no_lp = logprob
    if yes_lp is None:
        return 0.0
    if no_lp is None:
        return math.exp(yes_lp)
    yes_e, no_e = math.exp(yes_lp), math.exp(no_lp)
    return yes_e / (yes_e + no_e)


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
