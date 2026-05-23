"""Long-lived chat worker subprocess body, with token streaming."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
import threading
import time
from collections.abc import Iterator
from typing import Any

from lilbee.providers.base import ContextWindowExceededError
from lilbee.providers.families import detect as detect_profile
from lilbee.providers.families.profile import FamilyProfile, OutputFormat, StreamingPolicy
from lilbee.providers.worker.response_parser import (
    ResponseSchema,
    StreamingResponseParser,
    detect_family,
    get_schemas,
    parse_response,
)
from lilbee.providers.worker.transport import (
    ChatRequest,
    ChatResult,
    FinishReason,
    RoleConfig,
    ToolCall,
    ToolCallDelta,
)
from lilbee.providers.worker.transport_pipe import _serialize_exception
from lilbee.providers.worker.windowing import count_tools_overhead, window_messages_to_budget
from lilbee.providers.worker.wire_kinds import WireKind
from lilbee.providers.worker.worker_runtime import Reply, WorkerLoopState, run_worker

log = logging.getLogger(__name__)

# Reserved tokens for the model's response when the caller omits ``num_predict``.
_DEFAULT_RESPONSE_BUDGET = 1024

# Tokenizer-drift cushion between count-time and inference-time.
_CTX_SAFETY_MARGIN = 64

# Cancel-signal polling cadence for the abort bridge (sec).
_ABORT_BRIDGE_POLL_S = 0.025

# Maximum tokens queued before a streaming-chat batch flushes.
_STREAM_BATCH_MAX_CHUNKS = 16

# Maximum time between streaming-chat batch flushes (sec).
_STREAM_BATCH_MAX_INTERVAL_S = 0.05


def _normalize_tool_call_arguments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse assistant ``tool_calls[].function.arguments`` JSON strings into dicts.

    GGUF chat templates are exported from HuggingFace models, where
    ``apply_chat_template`` receives prior tool calls with ``arguments`` as a
    parsed object. Their jinja bodies iterate it with the ``|items`` filter or
    serialize it with ``|tojson`` -- both assume a mapping. The OpenAI wire
    format that opencode (and every OpenAI-compatible client) sends encodes
    ``arguments`` as a JSON-encoded string. Feeding that string straight into
    the template raises ``TypeError: Can only get item pairs from a mapping``
    on the first follow-up turn that carries a tool call in its history, which
    surfaces to the client as a mid-stream 500.

    Parsing the string back to a dict here bridges the two conventions for
    every family without a per-template special case. Non-JSON argument
    strings (rare, malformed clients) are left untouched.
    """
    normalized: list[dict[str, Any]] = []
    for message in messages:
        tool_calls = message.get("tool_calls")
        if message.get("role") != "assistant" or not tool_calls:
            normalized.append(message)
            continue
        new_calls = []
        for call in tool_calls:
            fn = call.get("function") if isinstance(call, dict) else None
            args = fn.get("arguments") if isinstance(fn, dict) else None
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                except (ValueError, TypeError):
                    new_calls.append(call)
                    continue
                new_calls.append({**call, "function": {**fn, "arguments": parsed}})
            else:
                new_calls.append(call)
        normalized.append({**message, "tool_calls": new_calls})
    return normalized


def _normalize_tool_call_ids(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rewrite tool-call ids to a deterministic 9-char alphanumeric form.

    Mistral's chat template enforces ``tool_call.id|length == 9`` (and the same
    on a tool result's ``tool_call_id``), raising mid-stream when it sees the
    longer ``call_...`` ids OpenAI-compatible clients like opencode generate.
    Other families' templates don't constrain the id, so collapsing every id
    to a stable 9-char hash is safe everywhere and keeps assistant tool_calls
    matched to their tool-result messages within the request.
    """
    mapping: dict[str, str] = {}

    def short(original: str) -> str:
        if original not in mapping:
            mapping[original] = hashlib.sha256(original.encode()).hexdigest()[:9]
        return mapping[original]

    out: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            new_calls = []
            for call in message["tool_calls"]:
                cid = call.get("id") if isinstance(call, dict) else None
                new_calls.append({**call, "id": short(cid)} if isinstance(cid, str) else call)
            out.append({**message, "tool_calls": new_calls})
        elif role == "tool" and isinstance(message.get("tool_call_id"), str):
            out.append({**message, "tool_call_id": short(message["tool_call_id"])})
        else:
            out.append(message)
    return out


def _merge_consecutive_same_role(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge adjacent plain user/assistant messages that share a role.

    Mistral's chat template enforces strict user/assistant alternation among
    non-tool messages and raises mid-stream on two consecutive users. opencode
    emits exactly that for its session-title and summarisation requests
    (``system -> user -> user``), so the auxiliary call 500s even though the
    main tool conversation alternates correctly. Joining the consecutive
    contents with a blank line keeps the prompt intact and satisfies every
    strict-alternation template; lenient templates are unaffected.

    Messages carrying ``tool_calls`` and ``tool``-role results are never merged
    -- they are rendered through dedicated template branches and must stay
    discrete.
    """
    merged: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        mergeable = role in ("user", "assistant") and not message.get("tool_calls")
        if (
            mergeable
            and merged
            and merged[-1].get("role") == role
            and not merged[-1].get("tool_calls")
        ):
            prev = merged[-1]
            prev_content = prev.get("content") or ""
            cur_content = message.get("content") or ""
            joined = "\n\n".join(p for p in (prev_content, cur_content) if p)
            merged[-1] = {**prev, "content": joined}
            continue
        merged.append(message)
    return merged


def _qa_log_message_roles(messages: list[dict[str, Any]]) -> None:
    """QA-only: log the role sequence (+tool-call markers) sent to the template.

    Mistral's template enforces strict user/assistant alternation among non-
    tool messages; this shows the exact sequence so a violating history is
    visible. No-op unless ``LILBEE_QA_LOG_RAW`` is set.
    """
    import os

    if not os.environ.get("LILBEE_QA_LOG_RAW"):
        return
    seq = []
    for m in messages:
        r = m.get("role")
        if r == "assistant" and m.get("tool_calls"):
            seq.append("assistant[tc]")
        else:
            seq.append(str(r))
    log.info("QA message roles (%d): %s", len(messages), " -> ".join(seq))


class _ChatSession:
    """Lazy-loaded Llama chat handle, kept alive for the worker's lifetime.

    Reloads in place when the parent passes a per-call ``model`` override
    different from the currently loaded one.
    """

    def __init__(self, role_config: RoleConfig, abort_flag: Any) -> None:

        self._role_config = role_config
        self._abort_flag = abort_flag
        self._llm: Any = None
        self._model_path: str = ""
        self._response_schema: ResponseSchema | None = None
        self._profile: FamilyProfile | None = None
        self._warned_unsupported_tools: set[str] = set()

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        stream: bool,
        options: dict[str, Any] | None,
        model: str | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> Any:
        """Run one chat completion and return the llama-cpp response."""
        llm = self._ensure_loaded(model)
        if tools and self._response_schema is None:
            self._warn_unsupported_tool_extraction(model)
        messages = _normalize_tool_call_arguments(messages)
        messages = _normalize_tool_call_ids(messages)
        messages = _merge_consecutive_same_role(messages)
        _qa_log_message_roles(messages)
        windowed = self._window_messages(messages, options, llm, tools=tools, model_ref=model)
        kwargs: dict[str, Any] = dict(options) if options else {}
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        # llama-cpp-python's chatml-function-calling preset (and the
        # functionary-v1/v2 variants) refuse stream=True with tool_choice="auto"
        # because the preset has no tool-call-delta path. The family profile's
        # streaming_policy field declares whether this swap is needed; lilbee
        # downgrades to a non-streaming call internally and emits the single
        # completion as a synthetic one-shot stream so the rest of the
        # pipeline (SSE encoder, tool extractor) is unaware.
        downgrade = stream and _profile_needs_stream_downgrade(self._profile, tool_choice)
        effective_stream = False if downgrade else stream
        try:
            result = llm.create_chat_completion(
                messages=windowed, stream=effective_stream, **kwargs
            )
        except ValueError as exc:
            self._reraise_if_context_overflow(exc, llm, model)
            raise
        if downgrade:
            return _wrap_single_completion_as_stream(result)
        if stream:
            return _translate_stream_overflow(result, llm, model, self._role_config)
        return result

    def _reraise_if_context_overflow(
        self, exc: ValueError, llm: Any, model_ref: str | None
    ) -> None:
        """Re-raise as ``ContextWindowExceededError`` if the message matches llama-cpp's
        overflow phrasing; otherwise return so the original exception propagates.
        """
        requested = _parse_requested_tokens(str(exc))
        if requested is None:
            return
        raise ContextWindowExceededError.from_runtime_overflow(
            requested=requested,
            n_ctx=int(llm.n_ctx()),
            model=model_ref or self._role_config.model_path.name,
        ) from exc

    def _window_messages(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any] | None,
        llm: Any,
        *,
        tools: list[dict[str, Any]] | None,
        model_ref: str | None,
    ) -> list[dict[str, Any]]:
        """Trim *messages* to fit the loaded model's context window."""
        requested_predict = (options or {}).get("num_predict")
        # Treat 0 / negative / missing as "no caller-supplied cap" and reserve
        # the default. ``-1`` is the llama-cpp / Ollama "unlimited" convention;
        # we cannot reason about an unbounded reservation so we fall back too.
        if not isinstance(requested_predict, int) or requested_predict <= 0:
            reserved = _DEFAULT_RESPONSE_BUDGET
        else:
            reserved = requested_predict

        def tokenize(data: bytes) -> list[int]:
            result: list[int] = llm.tokenize(data, add_bos=False, special=False)
            return result

        tools_overhead = count_tools_overhead(tools, tokenize)
        n_ctx = int(llm.n_ctx())
        budget = n_ctx - reserved - _CTX_SAFETY_MARGIN - tools_overhead
        outcome = window_messages_to_budget(
            messages,
            budget=max(0, budget),
            tokenize=tokenize,
        )
        if outcome.messages is None:
            raise ContextWindowExceededError.from_breakdown(
                requested=outcome.requested,
                n_ctx=n_ctx,
                response_budget=reserved,
                tools_overhead=tools_overhead,
                safety_margin=_CTX_SAFETY_MARGIN,
                model=model_ref or self._role_config.model_path.name,
            )
        if outcome.dropped:
            log.debug(
                "Chat windowing dropped %d messages to fit budget=%d",
                outcome.dropped,
                budget,
            )
        return outcome.messages

    def _warn_unsupported_tool_extraction(self, model_ref: str | None) -> None:
        """Log once per model when tools are requested but no schema applies."""
        if self._model_path in self._warned_unsupported_tools:
            return
        self._warned_unsupported_tools.add(self._model_path)
        log.warning(
            "Tool-call extraction not available for model %r: chat template did "
            "not match any supported family. Tool calls in responses will appear "
            "as raw text; the client will not invoke the tool. See "
            "docs/architecture.md (Model family compatibility section) for the "
            "supported-families list.",
            model_ref or self._role_config.model_path.name,
        )

    @property
    def response_schema(self) -> ResponseSchema | None:
        """Cached response schema for the currently-loaded model, or ``None``."""
        return self._response_schema

    def _ensure_loaded(self, model_override: str | None) -> Any:
        from lilbee.providers.llama_cpp.gguf_meta import read_gguf_metadata
        from lilbee.providers.llama_cpp.provider import load_llama, resolve_model_path
        from lilbee.providers.model_cache import LoaderMode

        target_path = (
            resolve_model_path(model_override) if model_override else self._role_config.model_path
        )
        target_str = str(target_path)
        if self._llm is None or target_str != self._model_path:
            self._close_model()
            # No abort_callback_override: routing the cancel signal through
            # ggml's mid-token abort path crashes the worker on macOS Metal.
            # Cancel is enforced one token boundary later by the Python-side
            # polling loop in _handle_chat_streaming.
            self._llm = load_llama(target_path, mode=LoaderMode.CHAT)
            self._model_path = target_str
            metadata = read_gguf_metadata(target_path) or {}
            self._profile = detect_profile(metadata, ref=target_str)
            family = (
                self._profile.family
                if self._profile is not None
                else detect_family(
                    metadata.get("chat_template", ""),
                    architecture=metadata.get("architecture"),
                )
            )
            self._response_schema = get_schemas().get(family)
        return self._llm

    def _close_model(self) -> None:
        if self._llm is not None:
            with contextlib.suppress(Exception):
                self._llm.close()
            self._llm = None
        self._response_schema = None

    def close(self) -> None:
        """Release the loaded model. Idempotent."""
        self._close_model()


_FINISH_REASONS: dict[str, FinishReason] = {fr.value: fr for fr in FinishReason}


def _profile_needs_stream_downgrade(
    profile: FamilyProfile | None, tool_choice: str | dict[str, Any] | None
) -> bool:
    """True iff the family profile declares its preset blocks ``stream=True`` here."""
    if profile is None or profile.streaming_policy is StreamingPolicy.NATIVE:
        return False
    # Specific-function tool_choice is supported on the downgrade-tagged
    # presets even with streaming; only "auto" / None / "required" trip the gate.
    return not isinstance(tool_choice, dict)


def _wrap_single_completion_as_stream(completion: Any) -> Iterator[Any]:
    """Yield a non-streaming ``create_chat_completion`` result as one stream chunk.

    Reshapes the response into the same wire shape ``_emit_stream_chunk``
    expects from a streaming generator so downstream tool-extraction and
    text-batching keep working unchanged.
    """
    if not isinstance(completion, dict):
        return
    choices = completion.get("choices") or []
    if not choices:
        return
    message = choices[0].get("message") or {}
    chunk: dict[str, Any] = {
        "id": completion.get("id", ""),
        "object": "chat.completion.chunk",
        "created": completion.get("created", 0),
        "model": completion.get("model", ""),
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": message.get("role", "assistant"),
                    "content": message.get("content"),
                    "tool_calls": message.get("tool_calls"),
                },
                "finish_reason": choices[0].get("finish_reason"),
            }
        ],
    }
    yield chunk


# Mirrors llama-cpp's verbatim phrasing in `llama_cpp/llama.py::Llama._create_completion`.
# A wording change upstream will surface as a CI failure on the typed-overflow
# test, prompting a deliberate update here rather than silently regressing.
_CTX_OVERFLOW_PATTERN = re.compile(r"Requested tokens \((\d+)\) exceed context window of \d+")


def _parse_requested_tokens(message: str) -> int | None:
    """Extract ``N`` from llama-cpp's ``Requested tokens (N) exceed context window`` text."""
    match = _CTX_OVERFLOW_PATTERN.search(message)
    return int(match.group(1)) if match else None


def _translate_stream_overflow(
    response_iter: Any,
    llm: Any,
    model_ref: str | None,
    role_config: RoleConfig,
) -> Any:
    """Translate llama-cpp's deferred context-overflow ``ValueError`` from a
    streaming generator into ``ContextWindowExceededError``.
    """
    try:
        yield from response_iter
    except ValueError as exc:
        requested = _parse_requested_tokens(str(exc))
        if requested is None:
            raise
        raise ContextWindowExceededError.from_runtime_overflow(
            requested=requested,
            n_ctx=int(llm.n_ctx()),
            model=model_ref or role_config.model_path.name,
        ) from exc


def _coerce_finish_reason(raw: str | None) -> FinishReason:
    """Map a raw llama-cpp finish_reason to ``FinishReason`` (default ``STOP``)."""
    if raw is None:
        return FinishReason.STOP
    return _FINISH_REASONS.get(raw, FinishReason.STOP)


def _extract_stream_content(chunk: Any) -> str | None:
    """Pull the text content out of one llama-cpp streaming chunk."""
    delta = _extract_delta(chunk)
    if delta is None:
        return None
    content = delta.get("content")
    return content if isinstance(content, str) and content else None


def _extract_delta(chunk: Any) -> dict[str, Any] | None:
    """Return the ``choices[0].delta`` dict from a llama-cpp streaming chunk."""
    choices = chunk.get("choices") if isinstance(chunk, dict) else None
    if not choices:
        return None
    delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
    return delta if isinstance(delta, dict) else None


def _extract_tool_call_deltas(chunk: Any) -> list[ToolCallDelta]:
    """Convert llama-cpp ``choices[0].delta.tool_calls`` into ``ToolCallDelta`` frames."""
    delta = _extract_delta(chunk)
    if delta is None:
        return []
    raw_calls = delta.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        return []
    out: list[ToolCallDelta] = []
    for entry in raw_calls:
        if not isinstance(entry, dict):
            continue
        function = entry.get("function") or {}
        if not isinstance(function, dict):
            function = {}
        arguments = function.get("arguments")
        out.append(
            ToolCallDelta(
                index=int(entry.get("index", 0)),
                id=entry.get("id"),
                name=function.get("name"),
                arguments_delta=arguments if isinstance(arguments, str) and arguments else None,
            )
        )
    return out


class _TextBatchBuffer:
    """Accumulates text deltas and flushes them in batches over a Reply."""

    def __init__(self, reply: Reply) -> None:
        self._reply = reply
        self._buffer: list[str] = []
        self._last_flush = time.monotonic()
        self._seen_first_token = False

    def append(self, text: str) -> None:
        """Buffer *text* and flush once the size or time threshold trips."""
        self._buffer.append(text)
        now = time.monotonic()
        if (
            not self._seen_first_token
            or len(self._buffer) >= _STREAM_BATCH_MAX_CHUNKS
            or (now - self._last_flush) >= _STREAM_BATCH_MAX_INTERVAL_S
        ):
            self.flush()

    def flush(self) -> None:
        """Emit any buffered text as one stream_chunk frame."""
        if not self._buffer:
            return
        self._reply.send(WireKind.STREAM_CHUNK, "".join(self._buffer))
        self._buffer.clear()
        self._last_flush = time.monotonic()
        self._seen_first_token = True


class _QaRawTap:
    """QA-only diagnostic: capture the model's raw streamed output and report it.

    Set ``LILBEE_QA_LOG_RAW=1`` to log, at stream end, the accumulated raw text
    plus how many native tool-call deltas the model produced. This is how the
    opencode matrix figures out WHY a family didn't dispatch: the log shows
    whether the model emitted a tool call in a format the family schema missed,
    or simply chatted without calling a tool. No-op unless the env var is set.
    """

    def __init__(self) -> None:
        import os

        self._on = bool(os.environ.get("LILBEE_QA_LOG_RAW"))
        self._chunks: list[str] = []
        self._native_tool_deltas = 0

    def observe(self, raw_chunk: Any) -> None:
        if not self._on:
            return
        if _extract_tool_call_deltas(raw_chunk):
            self._native_tool_deltas += 1
        content = _extract_stream_content(raw_chunk)
        if content:
            self._chunks.append(content)

    def report(self, *, family: str, output_format: OutputFormat, has_schema: bool) -> None:
        if not self._on:
            return
        raw = "".join(self._chunks)
        log.info(
            "QA raw stream: family=%s out_fmt=%s schema=%s native_tool_deltas=%d raw_len=%d raw=%r",
            family,
            output_format.value if isinstance(output_format, OutputFormat) else output_format,
            has_schema,
            self._native_tool_deltas,
            len(raw),
            raw[:1200],
        )


def _handle_chat_streaming(
    reply: Reply,
    response_iter: Any,
    state: WorkerLoopState,
    *,
    schema: ResponseSchema | None,
    profile: FamilyProfile | None = None,
) -> None:
    """Drain *response_iter* and emit batched stream_chunk frames on the data pipe."""
    abort_flag = state.session._abort_flag
    text = _TextBatchBuffer(reply)
    output_format = profile.output_format if profile is not None else OutputFormat.NATIVE
    schema_parser = (
        StreamingResponseParser(schema, output_format=output_format) if schema is not None else None
    )
    qa_raw = _QaRawTap()
    completed_cleanly = False
    try:
        for raw_chunk in response_iter:
            if abort_flag.value:
                with contextlib.suppress(Exception):
                    response_iter.close()
                break
            qa_raw.observe(raw_chunk)
            _emit_stream_chunk(reply, raw_chunk, text, schema_parser)
        completed_cleanly = True
    finally:
        if schema_parser is not None:
            _drain_schema_parser_flush(text, schema_parser)
        text.flush()
        qa_raw.report(
            family=profile.family.value if profile is not None else "none",
            output_format=output_format,
            has_schema=schema is not None,
        )
    if completed_cleanly:
        reply.send(WireKind.STREAM_END, None)


def _emit_stream_chunk(
    reply: Reply,
    raw_chunk: Any,
    text: _TextBatchBuffer,
    schema_parser: StreamingResponseParser | None,
) -> None:
    """Dispatch one streaming chunk into tool-call frames or buffered text."""
    tool_deltas = _extract_tool_call_deltas(raw_chunk)
    if tool_deltas:
        text.flush()
        for delta in tool_deltas:
            reply.send(WireKind.STREAM_CHUNK, delta)
        return
    content = _extract_stream_content(raw_chunk)
    if content is None:
        return
    if schema_parser is None:
        text.append(content)
        return
    content_delta, schema_deltas = schema_parser.feed(content)
    if content_delta:
        text.append(content_delta)
    if schema_deltas:
        text.flush()
        for delta in schema_deltas:
            reply.send(WireKind.STREAM_CHUNK, delta)


def _drain_schema_parser_flush(
    text: _TextBatchBuffer,
    schema_parser: StreamingResponseParser,
) -> None:
    """Release any content held by the schema parser's safety margin at stream end.

    Tool-call completion is detected during ``feed()`` on every chunk, so the
    only thing ``flush()`` can carry is content the safety margin held back.
    """
    content_delta, _ = schema_parser.flush()
    if content_delta:
        text.append(content_delta)


def _extract_non_streaming_result(
    response: Any,
    *,
    tools_requested: bool,
    schema: ResponseSchema | None,
    profile: FamilyProfile | None = None,
) -> ChatResult:
    """Build a ``ChatResult`` from one llama-cpp non-streaming response."""
    first, message = _unwrap_llama_response(response)
    content = message.get("content")
    text = content if isinstance(content, str) else ""
    tool_calls = _coerce_tool_calls(message.get("tool_calls") or [])
    finish_reason = _coerce_finish_reason(first.get("finish_reason"))
    extracted = _maybe_extract_via_schema(text, tool_calls, tools_requested, schema, profile)
    if extracted is None:
        return ChatResult(text=text, tool_calls=tool_calls, finish_reason=finish_reason)
    return extracted


def _unwrap_llama_response(response: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the llama-cpp response shape and return ``(first_choice, message)``."""
    if not isinstance(response, dict):
        raise TypeError(f"chat response must be dict, got {type(response).__name__}")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise TypeError("chat response missing 'choices' list")
    first = choices[0]
    if not isinstance(first, dict):
        raise TypeError(f"chat choices[0] must be dict, got {type(first).__name__}")
    message = first.get("message")
    if not isinstance(message, dict):
        raise TypeError("chat choices[0].message missing or not dict")
    return first, message


def _maybe_extract_via_schema(
    text: str,
    native_tool_calls: tuple[ToolCall, ...],
    tools_requested: bool,
    schema: ResponseSchema | None,
    profile: FamilyProfile | None = None,
) -> ChatResult | None:
    """Try schema extraction; return ``None`` to keep the native response."""
    if native_tool_calls or not tools_requested or schema is None:
        return None
    output_format = profile.output_format if profile is not None else OutputFormat.NATIVE
    parsed = parse_response(text, schema, output_format=output_format)
    if not parsed.tool_calls:
        return None
    return ChatResult(
        text=parsed.content,
        tool_calls=parsed.tool_calls,
        finish_reason=FinishReason.TOOL_CALLS,
    )


def _coerce_tool_calls(raw_calls: Any) -> tuple[ToolCall, ...]:
    """Convert a llama-cpp ``message.tool_calls`` list into ``ToolCall`` values."""
    if not isinstance(raw_calls, list):
        return ()
    out: list[ToolCall] = []
    for entry in raw_calls:
        if not isinstance(entry, dict):
            continue
        function = entry.get("function") or {}
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = function.get("arguments", "{}")
        arguments_str = arguments if isinstance(arguments, str) else "{}"
        out.append(
            ToolCall(
                id=str(entry.get("id") or ""),
                name=name,
                arguments=arguments_str,
            )
        )
    return tuple(out)


def _handle_chat_non_streaming(
    reply: Reply,
    response: Any,
    *,
    tools_requested: bool,
    schema: ResponseSchema | None,
    profile: FamilyProfile | None = None,
) -> None:
    """Emit one result frame carrying the full :class:`ChatResult`."""
    result = _extract_non_streaming_result(
        response,
        tools_requested=tools_requested,
        schema=schema,
        profile=profile,
    )
    reply.send(WireKind.RESULT, result)


class _AbortBridge:
    """Mirror the parent's mp.Value abort flag into ggml's threading.Event.

    A polling thread reads the parent flag every ``_ABORT_BRIDGE_POLL_S``
    and calls ``request_abort()`` so cancel takes effect at the next ggml
    poll point rather than waiting for the next Python yield.
    """

    def __init__(self, abort_flag: Any) -> None:
        self._abort_flag = abort_flag
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _AbortBridge:
        from lilbee.providers.llama_cpp.abort_signal import clear_abort

        # Reset both flags before the chat starts: a stale parent-side
        # cancel from a prior call must not abort the new request.
        clear_abort()
        self._abort_flag.value = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, name="chat-abort-bridge", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        from lilbee.providers.llama_cpp.abort_signal import clear_abort

        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
        # Reset for the next request so a cancelled prior call doesn't
        # latch onto the next inference.
        clear_abort()
        self._abort_flag.value = 0

    def _poll(self) -> None:
        from lilbee.providers.llama_cpp.abort_signal import request_abort

        while not self._stop.wait(_ABORT_BRIDGE_POLL_S):
            if self._abort_flag.value:
                request_abort()
                return


def _handle_chat(reply: Reply, payload: Any, state: WorkerLoopState) -> None:
    """Run one chat request and dispatch to the streaming/non-streaming handler."""
    if not isinstance(payload, ChatRequest):
        try:
            raise TypeError(f"chat payload must be ChatRequest, got {type(payload).__name__}")
        except TypeError as exc:
            reply.send(WireKind.ERROR, _serialize_exception(exc))
        return
    session: _ChatSession = state.session
    with _AbortBridge(session._abort_flag):
        try:
            response = session.chat(
                messages=payload.messages,
                stream=payload.stream,
                options=payload.options,
                model=payload.model,
                tools=payload.tools,
                tool_choice=payload.tool_choice,
            )
        except Exception as exc:
            reply.send(WireKind.ERROR, _serialize_exception(exc))
            return
        tools_requested = bool(payload.tools)
        schema = session.response_schema if tools_requested else None
        profile = session._profile
        try:
            if payload.stream:
                _handle_chat_streaming(reply, response, state, schema=schema, profile=profile)
            else:
                _handle_chat_non_streaming(
                    reply,
                    response,
                    tools_requested=tools_requested,
                    schema=schema,
                    profile=profile,
                )
        except Exception as exc:
            reply.send(WireKind.ERROR, _serialize_exception(exc))


def chat_worker_main(
    data_conn: Any, health_conn: Any, abort_flag: Any, role_config: RoleConfig
) -> None:
    """Chat worker entrypoint: load llama-cpp lazily, serve until shutdown."""
    run_worker(
        data_conn,
        health_conn,
        abort_flag,
        role_config,
        session_factory=_ChatSession,
        kind_handlers={WireKind.CHAT: _handle_chat},
    )


__all__ = ["chat_worker_main"]
