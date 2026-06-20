"""Thin httpx client for one llama-server OpenAI endpoint (local inference)."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import threading
import time
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, TypedDict, TypeVar, overload

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
from lilbee.providers.fleet.normalize import ChatMessage, to_alternating
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


class _ChatToolSpecFunction(TypedDict, total=False):
    """The ``function`` payload of an OpenAI tool definition (wire shape)."""

    name: str
    description: str
    parameters: dict[str, Any]


class ChatTool(TypedDict, total=False):
    """One OpenAI tool definition sent in a chat request (wire shape)."""

    type: str
    function: _ChatToolSpecFunction


# Some GGUF chat templates (Mistral-Nemo, Cohere command-r) reject a standard
# OpenAI tool exchange: they require plain user/assistant turns to alternate and
# raise a Jinja exception on the tool role or two same-role turns in a row. Rather
# than fail a real request and parse the engine's error text, the client probes
# the live template once per server with this representative tool exchange: if the
# server rejects it as sent but renders the to_alternating() form, the model is
# flagged so every later request is reshaped up front. Two assistant tool-call
# turns separated by tool results is the minimal shape that trips strict
# alternation; max_tokens=1 keeps the probe to template rendering, not generation.
_ALTERNATION_PROBE_TOOLS: list[ChatTool] = [
    {
        "type": "function",
        "function": {
            "name": "probe",
            "description": "Probe whether the chat template renders a tool exchange.",
            # A single declared property (rather than an empty object) so a grammar
            # that requires at least one parameter still renders the probe call.
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]
_ALTERNATION_PROBE_MESSAGES: list[ChatMessage] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Look something up."},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "probe-1",
                "type": "function",
                "function": {"name": "probe", "arguments": '{"query": "x"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "probe-1", "content": "first result"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "probe-2",
                "type": "function",
                "function": {"name": "probe", "arguments": '{"query": "x"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "probe-2", "content": "second result"},
    {"role": "user", "content": "Summarize."},
]
_ALTERNATION_PROBE_OPTIONS = {"max_tokens": 1}
# The probe holds _alternation_lock across its request, so it uses a short, bounded
# timeout rather than the chat default: a slow/wedged replica yields an inconclusive
# (transient) result and a re-probe instead of blocking every first chat on the lock.
_ALTERNATION_PROBE_TIMEOUT_S = 30.0
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
    # llama-swap masks a dead server as "exited prematurely"; surface the server's
    # own captured output (a missing CUDA runtime, a model load failure, a bind
    # error) so the real exit reason reaches the caller, not only the log.
    if _UPSTREAM_DIED_MARKER in body:
        tail = _upstream_failure_tail(resp)
        if tail:
            detail = f"{detail}\nupstream server output:\n{tail}"
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


def _is_transient_probe_failure(exc: Exception) -> bool:
    """Whether a probe failure is transient (dead replica or a busy 429), not a
    template verdict. A cold replica 429s its first traffic, so a busy response
    must not be read as the template rejecting the exchange."""
    if is_connection_failure(exc):
        return True
    return isinstance(exc, ProviderError) and exc.kind is ProviderErrorKind.RATE_LIMIT


def _upstream_failure_tail(resp: httpx.Response) -> str:
    """Return (and log) the dead upstream's recent output, or empty when unreadable."""
    with contextlib.suppress(httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError):
        base = str(resp.request.url).split("/v1/")[0]
        model = json.loads(resp.request.content)["model"]
        tail = _fetch_log_tail(f"{base}/logs/stream/{model}")
        if tail:
            log.warning("%s exited prematurely; recent server output:\n%s", model, tail)
            return tail
    return ""


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
# this cool-down. Recovery is probe-by-traffic and unmetered: every concurrent
# caller sees it routable once cooled down (a success restores it, another
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
        # Whether this server's chat template needs OpenAI tool exchanges reshaped
        # into strict user/assistant alternation. Determined lazily by a one-time
        # probe of the live template (see _prepare_chat_messages); None until then.
        # A client is bound to one model for its lifetime, so the template (hence
        # the verdict) is fixed once determined.
        self._needs_alternation: bool | None = None
        self._alternation_lock = threading.Lock()
        # Routing health: cleared on a connection-level failure (see _UNHEALTHY_RETRY_S).
        self._healthy = True
        # Monotonic stamp of the last mark_unhealthy; consulted only while unhealthy.
        self._unhealthy_since = 0.0

    @property
    def healthy(self) -> bool:
        """Routable: healthy, or unhealthy past the ``_UNHEALTHY_RETRY_S`` cool-down."""
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

    @overload
    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        options: dict[str, Any] | None = None,
        stream: Literal[False] = False,
        timeout: float | None = None,
    ) -> str: ...

    @overload
    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        options: dict[str, Any] | None = None,
        stream: Literal[True],
        timeout: float | None = None,
    ) -> Iterator[str]: ...

    @overload
    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        options: dict[str, Any] | None = None,
        stream: bool,
        timeout: float | None = None,
    ) -> str | Iterator[str]: ...

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
            return self._stream_tracked(self._chat_stream_body(payload))
        request_timeout = timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT
        with self._track():
            resp = self._http.post(
                _CHAT_PATH, json={**payload, "stream": False}, timeout=request_timeout
            )
            _raise_for_status(resp)
            return str(resp.json()["choices"][0]["message"]["content"])

    def _chat_stream_body(self, payload: dict[str, Any]) -> Iterator[str]:
        # in_flight is reserved by _stream_tracked, which wraps this; see there.
        with self._http.stream("POST", _CHAT_PATH, json={**payload, "stream": True}) as resp:
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
            "messages": self._prepare_chat_messages(messages),
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
        and report ``tool_calls`` as the finish reason. Messages are reshaped to
        strict alternation up front when this server's template needs it (see
        :meth:`_prepare_chat_messages`).
        """
        payload = self._chat_payload(
            self._prepare_chat_messages(messages), tools, tool_choice, options, stream=False
        )
        with self._track():
            resp = self._http.post(_CHAT_PATH, json=payload)
            _raise_for_status(resp)
            body = dict(resp.json())
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
        translator accumulates the deltas by ``index``. Messages are reshaped to
        strict alternation up front when this server's template needs it (see
        :meth:`_prepare_chat_messages`), so the open never fails on a template
        that rejects the raw tool exchange.

        Not a generator: the up-front probe runs when this is called, not deferred
        to the first iteration, matching the eager non-stream paths.

        A model that emits a tool call as bare-JSON text instead of native
        ``tool_calls`` (a native miss, as on the non-stream paths) is recovered by
        wrapping the raw frames; see :func:`_recover_bare_json_stream`.
        """
        prepared = self._prepare_chat_messages(messages)
        return _recover_bare_json_stream(
            self._stream_tracked(self._open_chat_stream_body(prepared, tools, tool_choice, options))
        )

    def _open_chat_stream_body(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        options: dict[str, Any] | None,
    ) -> Iterator[str | ToolCallDelta | TokenUsage]:
        """Open one SSE chat stream and yield its frames; raises before the first frame.

        in_flight is reserved by _stream_tracked, which wraps this; see there.
        """
        payload = self._chat_payload(messages, tools, tool_choice, options, stream=True)
        with self._http.stream("POST", _CHAT_PATH, json=payload) as resp:
            _raise_for_status(resp)
            for line in resp.iter_lines():
                yield from _parse_sse_stream_items(line)

    def _chat_payload(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        options: dict[str, Any] | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        """Build the chat-completions request body shared by the stream and non-stream paths."""
        payload: dict[str, Any] = {"model": self._model, "messages": messages, "stream": stream}
        if stream:
            # include_usage makes llama-server emit a final SSE chunk carrying the
            # token usage (with an empty choices list) just before [DONE].
            payload["stream_options"] = {"include_usage": True}
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        payload.update(options or {})
        return payload

    def _prepare_chat_messages(
        self, messages: Sequence[Mapping[str, Any]]
    ) -> Sequence[Mapping[str, Any]]:
        """Reshape *messages* to strict alternation when this server's template needs it.

        The need is detected once per server by :meth:`_ensure_alternation_probed`
        (a binary accept/reject of a representative tool exchange against the live
        template), then cached, so real requests are normalized up front rather
        than failing and retrying.
        """
        self._ensure_alternation_probed()
        if self._needs_alternation:
            return to_alternating([dict(m) for m in messages])
        return messages

    def _ensure_alternation_probed(self) -> None:
        """Probe the live template once to learn whether it needs alternation.

        Caches only a conclusive verdict: a transient unreachable server leaves
        the flag unset so the next request re-probes rather than locking in a
        wrong answer.
        """
        if self._needs_alternation is not None:
            return
        with self._alternation_lock:
            if self._needs_alternation is not None:
                return
            verdict = self._probe_alternation()
            if verdict is not None:
                self._needs_alternation = verdict

    def _probe_alternation(self) -> bool | None:
        """Whether the template needs normalization: ``None`` when undetermined.

        Renders the probe exchange as sent; if the template accepts it, no
        normalization is needed. If it rejects it, normalization is needed only
        when the reshaped exchange is accepted. A transient failure on either
        render is inconclusive (``None``) so no verdict is cached; a genuine
        rejection of both forms is a conclusive ``False`` (the template fault is
        unrelated to alternation, so reshaping would not help).
        """
        raw = self._chat_probe(_ALTERNATION_PROBE_MESSAGES)
        if raw is None:
            return None  # transient; stay undetermined so the next request re-probes
        if raw:
            return False  # the template renders the raw OpenAI exchange as sent
        reshaped = self._chat_probe(to_alternating([dict(m) for m in _ALTERNATION_PROBE_MESSAGES]))
        if reshaped is None:
            return None  # transient on the reshape probe; stay undetermined
        return reshaped

    def _chat_probe(self, messages: Sequence[Mapping[str, Any]]) -> bool | None:
        """Post the probe exchange: ``True`` rendered, ``False`` rejected, ``None`` undetermined.

        A connection failure or a server-busy (HTTP 429) response is transient and
        unrelated to the template, so it is undetermined: only a clean render or a
        genuine rejection is a verdict the caller may cache.
        """
        payload = self._chat_payload(
            messages, _ALTERNATION_PROBE_TOOLS, None, _ALTERNATION_PROBE_OPTIONS, stream=False
        )
        try:
            with self._track():
                resp = self._http.post(
                    _CHAT_PATH, json=payload, timeout=_ALTERNATION_PROBE_TIMEOUT_S
                )
                _raise_for_status(resp)
        except (ProviderError, httpx.TransportError) as exc:
            return None if _is_transient_probe_failure(exc) else False
        return True

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

    def _enter_in_flight(self) -> None:
        """Atomically reserve one in-flight slot (the router balances on this)."""
        with self._in_flight_lock:
            self.in_flight += 1

    def _exit_in_flight(self) -> None:
        """Atomically release one in-flight slot."""
        with self._in_flight_lock:
            self.in_flight -= 1

    def _stream_tracked(self, body: Iterator[_T]) -> Iterator[_T]:
        """Wrap a streaming *body* so its in-flight slot is reserved eagerly.

        Reserves the slot now (at call time), not on the body's first iteration:
        a stream is often built and handed off before its first frame is pulled,
        and a concurrent reload's client drain keys on ``in_flight`` to decide a
        client is idle. Eager reservation keeps a checked-out-but-not-yet-started
        stream from being closed out from under. The slot is released when the
        returned generator is exhausted or closed.
        """
        self._enter_in_flight()
        return self._release_in_flight_on_close(body)

    def _release_in_flight_on_close(self, body: Iterator[_T]) -> Iterator[_T]:
        try:
            yield from body
        finally:
            self._exit_in_flight()


class _InFlight:
    """Context manager that atomically bumps the owner's in-flight counter.

    ``+= 1`` is a read-modify-write, so concurrent chat/embed calls would corrupt
    the counter the router balances on; the client's lock makes it atomic.
    """

    def __init__(self, client: LlamaServerClient) -> None:
        self._client = client

    def __enter__(self) -> None:
        self._client._enter_in_flight()

    def __exit__(self, *_exc: object) -> None:
        self._client._exit_in_flight()


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


# Leading non-whitespace characters that mark streamed text as a potential bare
# JSON tool call (an object or an array of them); any other first char is plain
# text and streams through untouched.
_BARE_CALL_OPENERS = "{["


def _tool_call_delta_from_recovered(call: ToolCall, index: int) -> ToolCallDelta:
    """Shape a recovered bare-JSON :class:`ToolCall` as a single streaming delta.

    Mirrors :func:`_tool_call_delta_from_chunk`: id and name ride the opener (the
    only frame for a recovered call), the arguments JSON is the lone
    ``arguments_delta``, and the position is the index.
    """
    return ToolCallDelta(
        index=index,
        id=call.id or None,
        name=call.name or None,
        arguments_delta=call.arguments or None,
    )


def _recover_bare_json_stream(
    items: Iterator[str | ToolCallDelta | TokenUsage],
) -> Iterator[str | ToolCallDelta | TokenUsage]:
    """Wrap a raw chat stream to recover a tool call emitted as bare-JSON text.

    Some small models print ``{"name": ..., "arguments": {...}}`` as content
    instead of native ``tool_calls``; the non-stream paths recover this via
    :func:`_recover_bare_json_tool_calls`. This applies the same recovery to the
    stream, but only when the model emitted no native :class:`ToolCallDelta` and
    the streamed text looks like a bare call from its first character. Normal text
    still streams token by token: once the buffered head proves not to be a bare
    call it is flushed and all later text passes straight through.
    """
    buffer = ""  # leading text held back as a potential bare call until resolved
    saw_native = False
    try:
        for item in items:
            if isinstance(item, ToolCallDelta):
                yield from _flush_plain(buffer)
                buffer, saw_native = "", True
                yield item
            elif isinstance(item, TokenUsage):
                yield from _recover_buffer(buffer)
                buffer = ""
                yield item
            elif saw_native or _passthrough_text(buffer, item):
                yield from _flush_plain(buffer)
                buffer = ""
                yield item
            else:
                buffer += item
        yield from _recover_buffer(buffer)
    finally:
        # Forward close to the source generator: if a consumer closes this
        # wrapper mid-stream, a plain for-loop would not propagate GeneratorExit
        # to *items*, leaking the underlying HTTP stream and its in_flight slot.
        # Suppress teardown errors (httpx stream close can raise) so they don't
        # mask the exception that triggered this finally.
        if isinstance(items, Generator):
            with contextlib.suppress(Exception):
                items.close()


def _passthrough_text(buffer: str, text: str) -> bool:
    """Whether *text* should stream through directly rather than buffer.

    True once the accumulated head's first non-whitespace char is known and is not
    a bare-call opener (plain text): the buffer is empty in that case, so the
    caller yields *text* as is. While the head is all whitespace, or once it opens
    with ``{``/``[``, the text is buffered (False) pending recovery.
    """
    head = (buffer + text).lstrip()
    return bool(head) and head[0] not in _BARE_CALL_OPENERS


def _flush_plain(buffer: str) -> Iterator[str]:
    """Yield buffered leading text verbatim (it was not a bare call after all)."""
    if buffer:
        yield buffer


def _recover_buffer(buffer: str) -> Iterator[str | ToolCallDelta]:
    """Resolve the buffered leading text at a terminator or end of stream.

    The buffer reaching here was held as a potential bare call (text starting with
    ``{``/``[`` and no native call seen). Run :func:`_recover_bare_json_tool_calls`:
    emit one delta per recovered call, or yield the text unchanged when it only
    happened to start with ``{``/``[`` but is not a call.
    """
    if not buffer:
        return
    recovered = _recover_bare_json_tool_calls(buffer)
    if not recovered.tool_calls:
        yield buffer
        return
    for index, call in enumerate(recovered.tool_calls):
        yield _tool_call_delta_from_recovered(call, index)
