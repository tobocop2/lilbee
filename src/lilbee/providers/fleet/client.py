"""Thin httpx client for one llama-server OpenAI endpoint (local inference)."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import math
import ssl
import threading
import time
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal, TypedDict, TypeVar, overload

import httpx
import numpy as np
import numpy.typing as npt

from lilbee.core.config import cfg
from lilbee.core.vectors import Vector
from lilbee.providers.base import (
    ChatResult,
    ChatToolResult,
    ClosableIterator,
    FinishReason,
    ProviderError,
    ProviderErrorKind,
    StreamFinish,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    is_prompt_length_overflow,
)
from lilbee.providers.fleet.adapters import LLM_RERANK_CONCURRENCY
from lilbee.providers.fleet.engines.base import RequestProfile
from lilbee.providers.fleet.normalize import ChatMessage, to_alternating
from lilbee.providers.reasoning import ThinkInliner, inline_reasoning
from lilbee.providers.roles import RerankMode
from lilbee.runtime.cancellation import CancelSignal, TaskCancelledError

# Fleet clients only ever talk to a loopback llama-server over plain HTTP, so TLS is
# never negotiated. httpx still builds a default SSL context per client (loads the
# system CA bundle, ~13 ms each and slower on macOS via the keychain), which is pure
# overhead paid on every fleet reload. Build one minimal context and share it so a
# reload doesn't reload the CA bundle for each replica.
_LOOPBACK_SSL_CONTEXT = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
_LOOPBACK_SSL_CONTEXT.check_hostname = False
_LOOPBACK_SSL_CONTEXT.verify_mode = ssl.CERT_NONE
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
_LLM_RERANK_NO_VERDICT_ERROR = (
    "The reranker model never answered 'yes' or 'no', so its relevance scores are "
    "unusable. Its chat template does not fit the relevance prompt. Choose a GGUF "
    "built for reranking, set reranker_type to cross_encoder, or adapt "
    "reranker_prompt to the model's expected format."
)
# Max sequences per /v1/embeddings request. Like the in-process backstop, a
# batch is bounded by BOTH the token budget (the server's n_batch, == token_cap)
# and this sequence count: a corpus of many tiny chunks would otherwise pack one
# request past the server's batch/sequence limit and trip a 500.
_EMBED_N_SEQ_MAX = 64
# Estimate a chunk's token count from its character length so the bulk embed
# path packs sub-batches without a /tokenize round-trip per input. The factor is
# held below the corpus average (data.extract.chunk.CHARS_PER_TOKEN, 4 for Latin text)
# so the estimate over-counts tokens and a sub-batch never packs past the
# server's n_batch (== token_cap). Rerank pairs are token-dense (the separator
# is several tokens in a few chars), so an under-count can slip an over-cap
# pair through; the rerank path re-truncates exactly on the server's overflow
# error instead of paying a /tokenize round trip per pair up front.
_EMBED_EST_CHARS_PER_TOKEN = 3
# llama-swap's error body when the spawned llama-server exited before serving.
_UPSTREAM_DIED_MARKER = "exited prematurely"
# llama-server's exit line when the port lilbee picked was taken by the time
# the server bound it (the pick-then-bind gap spans the whole lazy-spawn wait,
# so a passing ephemeral connection can occupy it). The occupation is
# transient: llama-swap re-spawns on the next request against the same port
# and normally binds.
_BIND_FAILURE_MARKER = "couldn't bind HTTP server socket"


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
# Enough reads to cover one replay of llama-swap's 100KB per-model ring at
# httpx's 64KB ceiling per chunk, with room to spare. The route keeps streaming
# live lines afterwards, so without a bound this would cost the read timeout on
# every death.
_UPSTREAM_LOG_MAX_CHUNKS = 8

log = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    """Conservative (over-counting) token estimate from character length."""
    return max(1, -(-len(text) // _EMBED_EST_CHARS_PER_TOKEN))


_UPSTREAM_DEATH_MESSAGES: dict[ProviderErrorKind, str] = {
    ProviderErrorKind.CAPACITY: "The engine ran out of memory while loading {model}",
    ProviderErrorKind.PORT_CONFLICT: "The engine could not start {model}: its port was taken",
}
_UPSTREAM_DEATH_DEFAULT = "The engine for {model} stopped before it could answer"


def _upstream_death_message(
    kind: ProviderErrorKind, status_code: int, model: str | None, engine_log: Path | None
) -> str:
    """One plain line for a model server that exited, built from what lilbee knows."""
    subject = model if model is not None else "the model"
    cause = _UPSTREAM_DEATH_MESSAGES.get(kind, _UPSTREAM_DEATH_DEFAULT).format(model=subject)
    message = f"{cause} (HTTP {status_code})."
    if engine_log is None:
        return message
    return f"{message} The full engine output is in {engine_log}."


# What the engine prints when a device allocation fails during load. Every
# backend words it differently and all of them mean the same thing: the plan
# asked for more memory than the device had. Taken from the emit sites in
# upstream rather than guessed, and matched lowercased.
#
# One entry covers CUDA, HIP and MUSA: the vendor headers #define cudaMalloc to
# their own allocator, but the log string in ggml-cuda.cu is a literal, so an
# AMD or Moore Threads build still prints "cudaMalloc failed". A separate
# hipMalloc marker would match nothing.
#
# Vulkan is the one that needs its own wording. It is where every AMD and Intel
# GPU lands, and it says neither "out of memory" nor "failed to allocate".
_OOM_MARKERS: tuple[str, ...] = (
    "out of memory",
    "failed to allocate",  # Metal's buffer failure, and most generic paths
    "cudamalloc failed",  # CUDA, HIP and MUSA alike
    "device memory allocation of size",  # ggml-vulkan's fatal allocation failure
    "outofdevicememory",  # a vk::OutOfDeviceMemoryError that reached the log
    "unable to allocate",
    "insufficient memory",
    "out_of_device_memory",  # SYCL, which exits through the runtime's own code
    "out_of_resources",
)

# Lines that report a failure the engine then works around. ggml-vulkan warns
# that pinned memory could not be allocated and falls back to unpinned, and that
# text matches an allocation marker word for word, so a later unrelated death
# would be read as a memory shortfall and answered with a context reduction.
_SURVIVABLE_LINE_MARKERS: tuple[str, ...] = ("warning:", "warn:")


def classify_upstream_death(tail: str) -> ProviderErrorKind | None:
    """The kind of failure an engine's dying output describes, or ``None``.

    ``CAPACITY`` for a load that ran out of device memory: retrying the identical
    launch respawns it into a crash loop, while a smaller context might fit.
    ``PORT_CONFLICT`` for losing the port-bind race, which is worth retrying
    because the retry re-drives llama-swap's spawn, and worth naming because a
    port held for good needs a different port rather than another attempt at the
    same one. ``None`` leaves the existing classification alone rather than
    guessing at an unfamiliar death.
    """
    fatal = [
        line
        for line in tail.lower().splitlines()
        if not any(marker in line for marker in _SURVIVABLE_LINE_MARKERS)
    ]
    if any(marker in line for line in fatal for marker in _OOM_MARKERS):
        return ProviderErrorKind.CAPACITY
    if _BIND_FAILURE_MARKER in tail:
        return ProviderErrorKind.PORT_CONFLICT
    return None


# Deaths a role's rebuild can fix, and a retry against the same launch cannot: a
# memory shortfall needs a smaller plan, a held port needs a different port. Both
# come from rebuilding the role, which re-plans and re-picks.
_REBUILDABLE_KINDS = frozenset({ProviderErrorKind.CAPACITY, ProviderErrorKind.PORT_CONFLICT})


def is_load_capacity_failure(exc: BaseException) -> bool:
    """True when *exc* is an engine that died for lack of device memory on load."""
    return isinstance(exc, ProviderError) and exc.kind is ProviderErrorKind.CAPACITY


def is_rebuildable_failure(exc: BaseException) -> bool:
    """True when rebuilding the role is what stands a chance, not another retry."""
    return isinstance(exc, ProviderError) and exc.kind in _REBUILDABLE_KINDS


def _classify_error(status_code: int, body: str, profile: RequestProfile) -> ProviderErrorKind:
    """Error kind from a llama-server/llama-swap error status and body.

    An input past the server's n_batch is a 500 whose body says "too large to
    process" (CONTEXT_OVERFLOW, so the embed path re-truncates exactly); a dead
    upstream is CONNECTION, so the router can mark the replica unhealthy. The
    body markers win over the status: llama-swap reports a died upstream under
    gateway statuses too, and that case needs the failover path, not a retry
    against the same dead server (except a bind-race death, which
    ``_raise_for_status`` upgrades to SERVER once the upstream tail proves it).
    A bare gateway error (502/503/504) is a
    momentarily-unreachable upstream -- restarting, OOM-killed, mid-swap -- so
    it is SERVER, which the gateway retry treats as transient.
    """
    if profile.batch_overflow.search(body):
        return ProviderErrorKind.CONTEXT_OVERFLOW
    if _UPSTREAM_DIED_MARKER in body:
        return ProviderErrorKind.CONNECTION
    if status_code in _TRANSIENT_GATEWAY_STATUSES:
        return ProviderErrorKind.SERVER
    return ProviderErrorKind.UNKNOWN


def is_connection_failure(exc: Exception) -> bool:
    """Whether *exc* signals a dead/unreachable replica rather than a model error."""
    if isinstance(exc, httpx.TransportError):
        return True
    # isinstance: only ProviderError carries a kind; other exceptions pass through.
    return isinstance(exc, ProviderError) and exc.kind is ProviderErrorKind.CONNECTION


def _is_transient_probe_failure(exc: Exception) -> bool:
    """Whether a probe failure is transient (dead replica, busy 429, gateway
    error), not a template verdict. llama-swap 429s past its concurrency limit
    and a restarting replica 502s, so neither response must be read as the
    template rejecting the exchange."""
    if is_connection_failure(exc):
        return True
    return isinstance(exc, ProviderError) and exc.kind in _TRANSIENT_KINDS


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
        for taken, chunk in enumerate(stream.iter_text(), start=1):
            # Bounded by chunk count, not by the tail size. llama-swap replays a
            # model's whole ring in one write and httpx hands over at most 64KB
            # at a time, so stopping at the first chunk past the tail size
            # returned the head of a warm model's log, where the fatal last line
            # never is. Only the tail is retained as the replay goes by, and the
            # route streams live lines afterwards, so the count is what keeps
            # this from waiting out the timeout on every death.
            chunks.append(chunk)
            chunks = ["".join(chunks)[-_UPSTREAM_LOG_TAIL_CHARS:]]
            if taken >= _UPSTREAM_LOG_MAX_CHUNKS:
                break
    return "".join(chunks)[-_UPSTREAM_LOG_TAIL_CHARS:]


# Vectors come back as a base64 float32 buffer: parsing thousands of JSON float
# literals per batch is CPU-bound and holds the GIL, which caps embedding
# throughput below what the GPUs can feed regardless of how many are dispatching.
_EMBED_ENCODING_FORMAT = "base64"
# The engine writes the raw float buffer in host byte order; supported targets are
# all little-endian.
_EMBED_VECTOR_DTYPE = "<f4"
# Rank pooling puts the pair's relevance score in the vector's first slot.
_RANK_SCORE_INDEX = 0
_UNREADABLE_EMBEDDING_ERROR = (
    "The embedding server returned vectors lilbee could not read. Update the inference engine: "
)
_NO_RERANK_SCORE_ERROR = "The reranker returned no relevance score for a candidate."
_HEALTH_PATH = "/health"
_CHAT_PATH = "/v1/chat/completions"
_EMBED_PATH = "/v1/embeddings"
# llama-swap proxies native (non-OpenAI) engine routes only under
# /upstream/<model>/...; the bare tokenize path 404s (it routes /v1/* by the
# body's model field, but a native route carries no such field).
_UPSTREAM_PREFIX = "/upstream"
# Match the in-process tokenizer call (llm.tokenize(text, add_bos=True, special=False)):
# the server adds BOS via add_special, and the chunk-sizing call leaves
# special-token strings unparsed; the chat-prompt count overrides that, because
# the render carries them.
_TOKENIZE_ADD_SPECIAL = True
_TOKENIZE_PARSE_SPECIAL = False
_HTTP_OK = 200
_HTTP_BAD_REQUEST = 400
_HTTP_NOT_FOUND = 404
_HTTP_METHOD_NOT_ALLOWED = 405
_HTTP_TOO_MANY_REQUESTS = 429
# Gateway statuses llama-swap returns while an upstream is unreachable
# (502 crashing/restarting, 503 unavailable, 504 gateway timeout). The request
# succeeds once the upstream is back, so these must never terminalize a call.
_TRANSIENT_GATEWAY_STATUSES = frozenset({502, 503, 504})
# Error kinds the gateway retry re-sends: a bare gateway error (upstream
# momentarily unreachable) and a lost port-bind race clear on their own once
# the server is back. A 429 is llama-swap's concurrency limit, which the swap
# config sizes for several lilbee processes at full load, so it is reported,
# not re-sent.
_RETRY_KINDS = frozenset({ProviderErrorKind.SERVER, ProviderErrorKind.PORT_CONFLICT})
# Error kinds that leave a template probe undetermined: the retry kinds plus a 429.
_TRANSIENT_KINDS = _RETRY_KINDS | {ProviderErrorKind.RATE_LIMIT}
_DONE_SENTINEL = "[DONE]"
_DATA_PREFIX = "data:"
_DEFAULT_TIMEOUT_S = 300.0
# How often an abortable chat checks its cancel signal.
_CANCEL_POLL_S = 0.1
# Short, separate timeout for /health: a server can wedge under heavy prompt
# processing, and readiness/monitor polls must not block on the request timeout.
_HEALTH_TIMEOUT_S = 5.0
# Retry a gateway error with exponential backoff (capped). llama-swap holds a
# request through a cold load, so a gateway error is a replica restarting, and
# callers fail after this short budget (~15s).
_GATEWAY_RETRIES = 6
_GATEWAY_BACKOFF_BASE_S = 0.5
_GATEWAY_BACKOFF_MAX_S = 8.0
# Half-open recovery: a replica marked unhealthy becomes routable again after
# this cool-down. Recovery is probe-by-traffic and unmetered: every concurrent
# caller sees it routable once cooled down (a success restores it, another
# connection failure re-stamps the cool-down).
_UNHEALTHY_RETRY_S = 30.0
_T = TypeVar("_T")


def _route_is_absent(resp: httpx.Response) -> bool:
    """Whether *resp* means the route is missing rather than the request bad.

    A proxy that knows the path but not the method answers 405. An engine too old
    to carry the route answers 404 with a plain body, while llama-swap answers
    404 with a JSON error envelope for a model it cannot route -- a
    misconfiguration the caller must see, not one to answer with an estimate.
    """
    if resp.status_code == _HTTP_METHOD_NOT_ALLOWED:
        return True
    if resp.status_code != _HTTP_NOT_FOUND:
        return False
    resp.read()  # streaming responses aren't read yet; a no-op for buffered ones
    try:
        body = resp.json()
    except ValueError:
        return True
    return not (isinstance(body, dict) and "error" in body)


class ChatDeadlineError(ProviderError):
    """A bounded chat exceeded its caller-supplied total wall-clock deadline.

    Distinct from a transport/server error so a deadline-bounded caller (vision
    OCR) can word its own timeout message and skip failover without matching
    error strings. Its ``UNKNOWN`` kind keeps it out of ``is_connection_failure``.
    """


def retry_on_gateway_error(
    call: Callable[[], _T], *, retries: int = _GATEWAY_RETRIES, deadline: float | None = None
) -> _T:
    """Run *call*, retrying gateway errors and a lost port-bind race with capped backoff.

    A replica restarting mid-run answers 502 until it is back; backing off and
    retrying turns that drop into a success. With a *deadline*
    (``time.monotonic`` epoch) the retry waits out the server until that
    deadline; without one, *retries* bounds the attempts. Every other error,
    a 429 included, and the final still-failing response propagate.
    """
    delay = _GATEWAY_BACKOFF_BASE_S
    attempt = 0
    while True:
        try:
            return call()
        except ProviderError as exc:
            if exc.kind not in _RETRY_KINDS:
                raise
            attempt += 1
            exhausted = (
                time.monotonic() + delay >= deadline if deadline is not None else attempt >= retries
            )
            if exhausted:
                raise
            time.sleep(delay)
            delay = min(delay * 2, _GATEWAY_BACKOFF_MAX_S)


class LlamaServerClient:
    """Calls one llama-server's OpenAI surface. Tracks in-flight requests so the
    fleet router can pick the least-busy replica."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        http: httpx.Client | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
        token_cap: int | None = None,
        timeout: float = _DEFAULT_TIMEOUT_S,
        rerank_mode: RerankMode | None = None,
        inline_reasoning: bool = False,
        on_prefill: Callable[[tuple[int, int] | None], None] | None = None,
        engine_log: Path | None = None,
        model_ref: str | None = None,
        profile: RequestProfile,
    ) -> None:
        # The request fields and native paths of this server's engine.
        self._profile = profile
        self._base = base_url.rstrip("/")
        # The file this server's own engine writes its output to; None when the
        # launch writes none or no swap manager fronts the client (the self-check).
        self.engine_log = engine_log
        self._model = model
        # The model as the user knows it (the registry ref); the replica id is not
        # something a person recognises. None names it "the model".
        self.model_ref = model_ref
        # Prefill observer: called with (processed, total) per engine progress
        # frame during a streamed chat's prompt processing, then None once the
        # first generated frame proves the prefill is over.
        self._on_prefill = on_prefill
        self._http = http or httpx.Client(
            base_url=self._base, timeout=timeout, verify=_LOOPBACK_SSL_CONTEXT
        )
        self._timeout = timeout
        self._async_transport = async_transport
        self._owns_http = http is None
        # Chat-role clients re-inline server-extracted reasoning as <think> text;
        # the other roles (vision OCR) keep dropping it, as their servers already did.
        self._inline_reasoning = inline_reasoning
        # Per-slot context for embed/rerank servers: inputs longer than this are
        # token-truncated (via the server's tokenizer) before embedding, mirroring
        # the in-process backstop. None for chat/vision, which don't truncate inputs.
        self._token_cap = token_cap
        # LLM => score candidates by yes/no logprob; None/cross-encoder => rank pooling.
        self._rerank_mode = rerank_mode
        self.in_flight = 0
        self._in_flight_lock = threading.Lock()
        # Live SSE responses, so a cancel can sever the transport from another
        # thread: a reader blocked in iter_lines cannot see a cooperative
        # cancel flag, but closing its response unblocks it with an error.
        self._active_streams: set[httpx.Response] = set()
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

    def _raise_for_status(self, resp: httpx.Response) -> None:
        """Raise the plain error for a failed *resp*, naming this client's model and log.

        The one place a client reports a failed response, so no call site can skip the
        model name and the engine log. ``raise_for_status`` drops the server's error
        body; the body carries the real reason (oversize prompt, decode failure, a
        dead server), which both diagnosis and the user-facing error need.
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
            self._profile.context_overflow.search(body) or is_prompt_length_overflow(body)
        ):
            raise ProviderError(
                "The conversation exceeds this model's context window. "
                "Start a new conversation or shorten the input.",
                provider=self._profile.label,
                kind=ProviderErrorKind.CONTEXT_OVERFLOW,
            )
        # llama-swap answers 429 once its concurrency limit for the model is reached.
        # The swap config sizes that limit for several lilbee processes at full load,
        # so a request past it is reported rather than re-sent into the same load.
        if resp.status_code == _HTTP_TOO_MANY_REQUESTS:
            raise ProviderError(
                "The model is busy with too many requests. Try again in a moment.",
                provider=self._profile.label,
                kind=ProviderErrorKind.RATE_LIMIT,
            )
        kind = _classify_error(resp.status_code, body, self._profile)
        # llama-swap masks a dead server as "exited prematurely". The server's own
        # captured output goes to the log (in _upstream_failure_tail) and decides the
        # kind; the message the user sees stays one plain line.
        if _UPSTREAM_DIED_MARKER in body:
            classified = classify_upstream_death(_upstream_failure_tail(resp))
            if classified is not None:
                kind = classified
            raise ProviderError(
                _upstream_death_message(kind, resp.status_code, self.model_ref, self.engine_log),
                provider=self._profile.label,
                kind=kind,
            )
        detail = f": {body[:600]}" if body else ""
        raise ProviderError(
            f"{self._profile.label} returned HTTP {resp.status_code}{detail}",
            provider=self._profile.label,
            kind=kind,
        )

    def mark_unhealthy(self) -> None:
        """Record a connection-level failure so the router skips this replica."""
        with self._in_flight_lock:
            self._healthy = False
            self._unhealthy_since = time.monotonic()

    def mark_healthy(self) -> None:
        """Restore the replica to the routing pool after a successful call."""
        with self._in_flight_lock:
            self._healthy = True

    def reserve(self) -> None:
        """Mark a routed request assigned to this replica, at selection time.

        The router balances on ``in_flight`` but the per-request tracking only
        bumps it once the HTTP call starts. Under a bulk ingest many threads pick
        a replica at the same instant, all see the momentarily-idlest one at the
        same low count, and pile onto it (a thundering herd that leaves the other
        cards idle). Reserving at selection makes the assignment visible to the
        next picker so requests spread across replicas. Paired with :meth:`release`.
        """
        with self._in_flight_lock:
            self.in_flight += 1

    def release(self) -> None:
        """Release a reservation taken by :meth:`reserve`."""
        with self._in_flight_lock:
            self.in_flight -= 1

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
        ``timeout`` overrides the client default for either path, so a
        caller-enforced deadline (vision OCR) ends the request itself.
        """
        payload: dict[str, Any] = {"model": self._model, "messages": messages, **(options or {})}
        request_timeout = timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT
        if stream:
            return self._chat_stream(payload, request_timeout)
        with self._track():
            resp = self._http.post(
                _CHAT_PATH, json={**payload, "stream": False}, timeout=request_timeout
            )
            self._raise_for_status(resp)
            # content is null for a refusal / content-filter stop / empty completion;
            # coerce to "" (like chat_result/chat_tools) so callers never see "None".
            return _inline_message_reasoning(
                resp.json()["choices"][0]["message"],
                enabled=self._inline_reasoning,
                reasoning_field=self._profile.reasoning_field,
            )

    def _chat_stream(
        self, payload: dict[str, Any], timeout: Any = httpx.USE_CLIENT_DEFAULT
    ) -> Iterator[str]:
        inliner = ThinkInliner(enabled=self._inline_reasoning)
        with (
            self._track(),
            self._http.stream(
                "POST", _CHAT_PATH, json={**payload, "stream": True}, timeout=timeout
            ) as resp,
        ):
            self._raise_for_status(resp)
            with self._abortable(resp):
                for line in resp.iter_lines():
                    delta = inliner.feed(*_parse_sse_deltas(line, self._profile.reasoning_field))
                    if delta:
                        yield delta
            tail = inliner.finish()
            if tail:
                yield tail

    def chat_abortable(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        options: dict[str, Any] | None = None,
        deadline_s: float | None = None,
        cancel: CancelSignal | None = None,
    ) -> str:
        """Stream a chat completion and return its text, aborted by a deadline or *cancel*.

        The request runs as an asyncio task on a private loop, so cancelling it
        closes the connection whether or not response headers have arrived, and
        llama-server stops generating for a closed stream. A passed *deadline_s*
        (total seconds, ``None`` for no limit) raises :class:`ChatDeadlineError`;
        a set *cancel* raises ``TaskCancelledError``.
        """
        payload = {"model": self._model, "messages": messages, **(options or {}), "stream": True}
        with self._track():
            return asyncio.run(self._chat_until_stopped(payload, deadline_s, cancel))

    async def _chat_until_stopped(
        self, payload: dict[str, Any], deadline_s: float | None, cancel: CancelSignal | None
    ) -> str:
        async with httpx.AsyncClient(
            base_url=self._base,
            timeout=self._timeout,
            verify=_LOOPBACK_SSL_CONTEXT,
            transport=self._async_transport,
        ) as http:
            chat = asyncio.ensure_future(self._stream_text(http, payload))
            try:
                async with asyncio.timeout(deadline_s):
                    while not chat.done():
                        if cancel is not None and cancel.is_set():
                            raise TaskCancelledError
                        await asyncio.wait({chat}, timeout=_CANCEL_POLL_S)
            except TimeoutError:
                raise ChatDeadlineError(
                    f"{self._profile.label} chat exceeded its {deadline_s:g}s deadline.",
                    provider=self._profile.label,
                ) from None
            finally:
                chat.cancel()
                await asyncio.gather(chat, return_exceptions=True)
            return chat.result()

    async def _stream_text(self, http: httpx.AsyncClient, payload: dict[str, Any]) -> str:
        inliner = ThinkInliner(enabled=self._inline_reasoning)
        parts: list[str] = []
        async with http.stream("POST", _CHAT_PATH, json=payload) as resp:
            if not resp.is_success:
                await resp.aread()
            self._raise_for_status(resp)
            async for line in resp.aiter_lines():
                parts.append(inliner.feed(*_parse_sse_deltas(line, self._profile.reasoning_field)))
        parts.append(inliner.finish())
        return "".join(parts)

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
            self._raise_for_status(resp)
            message = resp.json()["choices"][0]["message"]
        content = _inline_message_reasoning(
            message, enabled=self._inline_reasoning, reasoning_field=self._profile.reasoning_field
        )
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
        tool-call syntax into structured ``message.tool_calls``. When a request
        that offered tools gets a bare-JSON call as content (a native miss),
        recover it and report ``tool_calls`` as the finish reason; without tools
        a JSON answer is the answer. Messages are reshaped to
        strict alternation up front when this server's template needs it (see
        :meth:`_prepare_chat_messages`).
        """
        payload = self._chat_payload(
            self._prepare_chat_messages(messages), tools, tool_choice, options, stream=False
        )
        with self._track():
            resp = self._http.post(_CHAT_PATH, json=payload)
            self._raise_for_status(resp)
            body = dict(resp.json())
        choice = body["choices"][0]
        usage = _usage_from_body(body) or TokenUsage()
        message = choice["message"]
        content = _inline_message_reasoning(
            message, enabled=self._inline_reasoning, reasoning_field=self._profile.reasoning_field
        )
        finish_reason = _coerce_finish_reason(choice.get("finish_reason"))
        native = _parse_native_tool_calls(message.get("tool_calls"))
        if native:
            return ChatResult(
                text=content,
                tool_calls=tuple(native),
                finish_reason=finish_reason,
                usage=usage,
            )
        recovered = _recover_bare_json_tool_calls(content) if tools else None
        if recovered is not None and recovered.tool_calls:
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
    ) -> ClosableIterator[str | ToolCallDelta | TokenUsage | StreamFinish]:
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

        When the request offered tools, a model that emits a tool call as
        bare-JSON text instead of native ``tool_calls`` (a native miss, as on the
        non-stream paths) is recovered by wrapping the raw frames; see
        :func:`_recover_bare_json_stream`. Without tools the frames pass through
        as they arrive.
        """
        prepared = self._prepare_chat_messages(messages)
        stream = self._open_chat_stream(prepared, tools, tool_choice, options)
        return _recover_bare_json_stream(stream) if tools else stream

    def _open_chat_stream(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        options: dict[str, Any] | None,
    ) -> Generator[str | ToolCallDelta | TokenUsage | StreamFinish]:
        """Open one SSE chat stream and yield its frames; raises before the first frame."""
        payload = self._chat_payload(messages, tools, tool_choice, options, stream=True)
        inliner = ThinkInliner(enabled=self._inline_reasoning)
        on_prefill = self._on_prefill
        prefilling = False
        try:
            with (
                self._track(),
                self._http.stream("POST", _CHAT_PATH, json=payload) as resp,
            ):
                self._raise_for_status(resp)
                with self._abortable(resp):
                    for line in resp.iter_lines():
                        if on_prefill is not None:
                            progress = _prefill_progress(line)
                            if progress is not None:
                                prefilling = True
                                on_prefill(progress)
                        for item in _parse_sse_stream_items(
                            line, inliner, self._profile.reasoning_field
                        ):
                            if prefilling and on_prefill is not None:
                                # The first real frame proves prefill is over.
                                prefilling = False
                                on_prefill(None)
                            yield item
                tail = inliner.finish()
                if tail:
                    yield tail
        finally:
            # A stream that dies or is closed mid-prefill must not leave a
            # stale in-progress reading on the status surface.
            if prefilling and on_prefill is not None:
                on_prefill(None)

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
            payload.update(self._profile.stream_fields)
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
                self._raise_for_status(resp)
        except (ProviderError, httpx.TransportError) as exc:
            return None if _is_transient_probe_failure(exc) else False
        return True

    def embed(self, texts: list[str]) -> list[Vector]:
        """Embed a batch via ``/v1/embeddings``."""
        if not texts:
            # Match the in-process embedder; the server rejects an empty input.
            return []
        vectors: list[Vector] = []
        for sub_batch in self._truncate_and_subbatch(texts, estimate=True):
            data = self._embed_subbatch(sub_batch)
            vectors.extend(_embedding_vector(item, self._profile) for item in data)
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
        """Relevance scores via rank-pooling embeddings.

        The server runs with ``--pooling rank``; we send ``query</s></s>candidate``
        pairs to ``/v1/embeddings`` and read each item's first embedding value as the
        score, so the ``/v1/rerank`` template-dependency (and its zero-output failure
        modes) is moot.

        All pairs go out in one request per ``_EMBED_N_SEQ_MAX`` sequences: the
        server queues one task per input, so splitting a query-time pool into
        per-pair requests only adds HTTP round trips.
        """
        if not candidates:
            return []
        if self._profile.rerank_path is not None:
            return self._rerank_route(self._profile.rerank_path, query, candidates)
        if self._rerank_mode is RerankMode.LLM:
            return self._rerank_llm(query, candidates)
        pairs = [
            self._fit_estimated(f"{query}{_RERANK_PAIR_SEPARATOR}{candidate}")
            for candidate in candidates
        ]
        scores: list[float] = []
        for start in range(0, len(pairs), _EMBED_N_SEQ_MAX):
            data = self._rerank_batch(pairs[start : start + _EMBED_N_SEQ_MAX])
            scores.extend(_rerank_score(item, self._profile) for item in data)
        return scores

    def _rerank_route(self, path: str, query: str, candidates: list[str]) -> list[float]:
        """Scores from the engine's rerank route, in candidate order."""

        def _call() -> list[dict[str, Any]]:
            with self._track():
                resp = self._http.post(path, json={"query": query, "texts": candidates})
                self._raise_for_status(resp)
                return list(resp.json())

        by_index = {
            int(item["index"]): float(item["score"]) for item in retry_on_gateway_error(_call)
        }
        if sorted(by_index) != list(range(len(candidates))):
            raise ProviderError(
                "The reranker did not return one score for each of the "
                f"{len(candidates)} candidates.",
                provider=self._profile.label,
            )
        return [by_index[i] for i in range(len(candidates))]

    def _fit_estimated(self, text: str) -> str:
        """Truncate *text* to the token cap only when its char estimate exceeds it."""
        if self._token_cap is None:
            return text
        return self._fit_input(text, self._token_cap, estimate=True)[0]

    def _rerank_batch(self, batch: list[str]) -> list[dict[str, Any]]:
        """Score one rerank batch, redoing it with exact truncation on overflow.

        The char estimate under-counts token-dense pairs, so an over-cap pair
        can slip through untruncated; the server rejects it as CONTEXT_OVERFLOW
        and the batch is redone with exact server-side tokenization.
        """
        try:
            return self._embeddings_call(batch)
        except ProviderError as exc:
            if exc.kind is not ProviderErrorKind.CONTEXT_OVERFLOW or self._token_cap is None:
                raise
            cap = self._token_cap
            exact = [self._fit_input(text, cap, estimate=False)[0] for text in batch]
            return self._embeddings_call(exact)

    def _rerank_llm(self, query: str, candidates: list[str]) -> list[float]:
        """Score each candidate by an LLM's yes/no first-token logprob.

        Raises ``ProviderError`` when no candidate yields a verdict.
        """
        template = cfg.reranker_prompt or _LLM_RERANK_PROMPT
        workers = min(LLM_RERANK_CONCURRENCY, len(candidates))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            scores = list(pool.map(lambda c: self._llm_rerank_one(template, query, c), candidates))
        if all(score is None for score in scores):
            raise ProviderError(_LLM_RERANK_NO_VERDICT_ERROR, provider=self._profile.label)
        return [0.0 if score is None else score for score in scores]

    def _llm_rerank_one(self, template: str, query: str, candidate: str) -> float | None:
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
            # Scoring reads the first generated token; a thinking template would
            # spend it opening a <think> block instead of answering.
            "chat_template_kwargs": {"enable_thinking": False},
        }

        def _call() -> dict[str, Any]:
            with self._track():
                resp = self._http.post(_CHAT_PATH, json=payload)
                self._raise_for_status(resp)
                return dict(resp.json())

        return _llm_rerank_score(_first_token_top_logprobs(retry_on_gateway_error(_call)))

    def _embeddings_call(self, inputs: list[str]) -> list[dict[str, Any]]:
        """POST one already-budgeted sub-batch to ``/v1/embeddings``; return its data."""

        def _call() -> list[dict[str, Any]]:
            with self._track():
                resp = self._http.post(
                    _EMBED_PATH,
                    json={
                        "model": self._model,
                        "input": inputs,
                        **self._profile.embed_fields,
                        "encoding_format": _EMBED_ENCODING_FORMAT,
                    },
                )
                self._raise_for_status(resp)
                data = resp.json()["data"]
            if len(data) != len(inputs):
                raise ProviderError(
                    f"Embedder returned {len(data)} vectors for {len(inputs)} inputs",
                    provider=self._profile.label,
                )
            return list(data)

        return retry_on_gateway_error(_call)

    def _truncate_and_subbatch(self, texts: list[str], *, estimate: bool) -> list[list[str]]:
        """Token-truncate over-cap inputs, then pack into server-sized sub-batches.

        An input longer than ``token_cap`` (the server's per-slot context /
        n_batch) is truncated to it via the server's tokenizer, since the server
        cannot split a pooled embedding sequence. Inputs are then grouped so each
        request stays within both the token budget and ``_EMBED_N_SEQ_MAX``
        sequences, bounding per-request size and the gateway-retry window. No cap
        (chat/vision) sends a single batch untouched.

        When ``estimate`` is set the per-input token count comes from
        :func:`_estimate_tokens`, and ``/tokenize`` is consulted only for the
        rare input whose estimate exceeds the cap -- eliminating a round-trip per
        chunk during bulk ingest. ``estimate=False`` (the overflow redo) tokenizes
        every input exactly.
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
        if estimate or self._profile.tokenize_path is None:
            est = _estimate_tokens(text)
            if est <= cap:
                return text, est
            if self._profile.tokenize_path is None:
                log.warning("Truncating oversize embed input: ~%d tokens > cap %d", est, cap)
                return text[: cap * _EMBED_EST_CHARS_PER_TOKEN], cap
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

    def _tokenize(self, text: str, *, parse_special: bool = _TOKENIZE_PARSE_SPECIAL) -> list[int]:
        """Token ids for *text*; a gateway error is retried like an embedding."""

        path = self._require_route(self._profile.tokenize_path, "tokenize")

        def _call() -> list[int]:
            resp = self._http.post(
                self._native_route(path),
                json={
                    "content": text,
                    "add_special": _TOKENIZE_ADD_SPECIAL,
                    "parse_special": parse_special,
                },
            )
            self._raise_for_status(resp)
            return list(resp.json()["tokens"])

        return retry_on_gateway_error(_call)

    def count_tokens(self, text: str) -> int:
        """Tokens in *text* under the server's tokenizer, or a character estimate with no route."""
        if self._profile.tokenize_path is None:
            return _estimate_tokens(text)
        return len(self._tokenize(text))

    @staticmethod
    def _require_route(path: str | None, route: str) -> str:
        """*path*, or ``NotImplementedError`` when the engine has no such route."""
        if path is None:
            raise NotImplementedError(f"This inference engine has no {route} route.")
        return path

    def count_chat_prompt_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> int:
        """Tokens the server prefills for this prompt, template applied.

        The template's role markers and its tool-call preamble are part of the
        prompt the model reads, so the rendered text is what gets tokenized.
        Special-token strings in the rendered text are parsed, matching how the
        server tokenizes its own prompt.
        """
        self._require_route(self._profile.tokenize_path, "tokenize")
        rendered = self._render_chat_prompt(messages, tools, tool_choice, options)
        return len(self._tokenize(rendered, parse_special=True))

    def _render_chat_prompt(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        options: dict[str, Any] | None,
    ) -> str:
        """The prompt text this server's template renders for a chat body.

        Sends the body the chat paths build, messages reshaped to strict
        alternation when this server's template needs it (see
        :meth:`_prepare_chat_messages`), so the render cannot diverge from the
        prompt a chat call prefills. Raises ``NotImplementedError`` when the
        server has no such route, which is the one case a caller can answer with
        an estimate instead.
        """
        payload = self._chat_payload(
            self._prepare_chat_messages(messages), tools, tool_choice, options, stream=False
        )

        def _call() -> str:
            resp = self._http.post(
                self._native_route(self._profile.apply_template_path), json=payload
            )
            if _route_is_absent(resp):
                raise NotImplementedError(
                    f"This inference engine has no {self._profile.apply_template_path} route."
                )
            self._raise_for_status(resp)
            return str(resp.json()["prompt"])

        return retry_on_gateway_error(_call)

    def _detokenize(self, tokens: list[int]) -> str:
        path = self._require_route(self._profile.detokenize_path, "detokenize")
        resp = self._http.post(self._native_route(path), json={"tokens": tokens})
        self._raise_for_status(resp)
        return str(resp.json()["content"])

    @contextlib.contextmanager
    def _abortable(self, resp: httpx.Response) -> Generator[None]:
        """Expose *resp* to ``abort_streams`` for the duration of its read loop."""
        with self._in_flight_lock:
            self._active_streams.add(resp)
        try:
            yield
        finally:
            with self._in_flight_lock:
                self._active_streams.discard(resp)

    def abort_streams(self) -> None:
        """Sever every in-flight SSE response on this replica.

        Closing the response from another thread unblocks a reader stuck in
        ``iter_lines`` with a stream error, which unwinds its worker;
        llama-server stops generating when the connection drops.
        """
        with self._in_flight_lock:
            streams = list(self._active_streams)
        for resp in streams:
            with contextlib.suppress(Exception):
                resp.close()

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


def _embedding_vector(item: dict[str, Any], profile: RequestProfile) -> npt.NDArray[np.float32]:
    """Decode one ``/v1/embeddings`` item's vector from its base64 float buffer."""
    embedding = item.get("embedding")
    # Untyped server JSON: a non-string means the encoding format was not honored.
    if not isinstance(embedding, str):
        raise ProviderError(
            _UNREADABLE_EMBEDDING_ERROR + profile.unreadable_embedding_hint, provider=profile.label
        )
    try:
        return np.frombuffer(base64.b64decode(embedding), dtype=_EMBED_VECTOR_DTYPE)
    except ValueError as exc:
        raise ProviderError(
            _UNREADABLE_EMBEDDING_ERROR + profile.unreadable_embedding_hint, provider=profile.label
        ) from exc


def _rerank_score(item: dict[str, Any], profile: RequestProfile) -> float:
    """Pull one relevance score from a rank-pooling ``/v1/embeddings`` item."""
    vector = _embedding_vector(item, profile)
    if not vector.size:
        raise ProviderError(_NO_RERANK_SCORE_ERROR, provider=profile.label)
    return float(vector[_RANK_SCORE_INDEX])


def _first_token_top_logprobs(response: dict[str, Any]) -> list[dict[str, Any]]:
    """The first generated token's top_logprobs list from a chat completion, or []."""
    choices = response.get("choices") or []
    if not choices:
        return []
    content = (choices[0].get("logprobs") or {}).get("content") or []
    if not content:
        return []
    return list(content[0].get("top_logprobs") or [])


def _llm_rerank_score(top_logprobs: list[dict[str, Any]]) -> float | None:
    """Softmax of the yes vs no logprobs in a token's top_logprobs (case/space-insensitive).

    ``None`` when neither verdict appears, distinct from the 0.0 of a confident "no".
    """
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
        return None if no_lp is None else 0.0
    if no_lp is None:
        return math.exp(yes_lp)
    yes_e, no_e = math.exp(yes_lp), math.exp(no_lp)
    return yes_e / (yes_e + no_e)


def _parse_sse_deltas(line: str, reasoning_field: str) -> tuple[str, str]:
    """Extract the (reasoning, content) deltas from one OpenAI SSE line."""
    if not line.startswith(_DATA_PREFIX):
        return "", ""
    body = line[len(_DATA_PREFIX) :].strip()
    if not body or body == _DONE_SENTINEL:
        return "", ""
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return "", ""
    choices = obj.get("choices") or []
    if not choices:
        return "", ""
    delta = choices[0].get("delta") or {}
    return str(delta.get(reasoning_field) or ""), str(delta.get("content") or "")


def _inline_message_reasoning(
    message: Mapping[str, Any], *, enabled: bool, reasoning_field: str
) -> str:
    """A non-streaming message's text with any extracted reasoning re-inlined."""
    reasoning = str(message.get(reasoning_field) or "") if enabled else ""
    return inline_reasoning(reasoning, str(message.get("content") or ""))


def _cached_prompt_tokens(usage: Mapping[str, Any]) -> int:
    """Read ``prompt_tokens_details.cached_tokens``, or 0 when the engine omits it."""
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, Mapping):
        return 0
    cached = details.get("cached_tokens")
    return cached if isinstance(cached, int) else 0


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
        cached_prompt_tokens=_cached_prompt_tokens(usage),
    )


def _coerce_finish_reason(raw: Any) -> FinishReason:
    """Map a server-supplied finish_reason string to :class:`FinishReason`."""
    return FinishReason.coerce(raw)


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


_PROMPT_PROGRESS_KEY = "prompt_progress"


def _prefill_progress(line: str) -> tuple[int, int] | None:
    """The ``(processed, total)`` of one SSE line's ``prompt_progress``, or None.

    llama-server emits the block on streamed chats that opt in via
    ``return_progress``; the substring pre-check keeps the extra JSON parse off
    every ordinary token line.
    """
    if _PROMPT_PROGRESS_KEY not in line or not line.startswith(_DATA_PREFIX):
        return None
    body = line[len(_DATA_PREFIX) :].strip()
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return None
    progress = obj.get(_PROMPT_PROGRESS_KEY)
    if not isinstance(progress, Mapping):
        return None
    try:
        return int(progress["processed"]), int(progress["total"])
    except (KeyError, TypeError, ValueError):
        return None


def _parse_sse_stream_items(
    line: str, inliner: ThinkInliner, reasoning_field: str
) -> Iterator[str | ToolCallDelta | TokenUsage | StreamFinish]:
    """Yield text tokens, tool-call deltas, and the finish frame from one SSE line.

    A chunk can carry a ``content`` token, a ``reasoning_content`` token (routed
    through *inliner*), a ``tool_calls`` delta array, or a mix; each is yielded as
    its own :data:`ChatStreamItem` frame. The chunk that closes the turn carries
    ``choices[0].finish_reason``, surfaced as a :class:`StreamFinish` so the
    dispatch reports ``length`` (and friends), not just the default end-of-turn.
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
    text = inliner.feed(str(delta.get(reasoning_field) or ""), str(delta.get("content") or ""))
    if text:
        yield text
    raw_calls = delta.get("tool_calls") or []
    for i, call in enumerate(raw_calls):
        if isinstance(call, Mapping):
            yield _tool_call_delta_from_chunk(call, fallback_index=i)
    raw_finish = choices[0].get("finish_reason")
    if raw_finish is not None:
        yield StreamFinish(reason=_coerce_finish_reason(raw_finish))


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
    items: Iterator[str | ToolCallDelta | TokenUsage | StreamFinish],
) -> ClosableIterator[str | ToolCallDelta | TokenUsage | StreamFinish]:
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
    # True once leading text has streamed as plain (or a native call was seen):
    # past that, a later '{'/'[' is content, not a bare call -- never buffer again.
    committed = False
    try:
        for item in items:
            if isinstance(item, ToolCallDelta):
                yield from _flush_plain(buffer)
                buffer, committed = "", True
                yield item
            elif isinstance(item, TokenUsage | StreamFinish):
                yield from _recover_buffer(buffer)
                buffer = ""
                yield item
            elif committed or _passthrough_text(buffer, item):
                yield from _flush_plain(buffer)
                buffer = ""
                committed = True
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
