"""Long-lived chat worker subprocess body, with token streaming."""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from typing import Any

from lilbee.providers.base import ContextWindowExceededError
from lilbee.providers.worker.transport import (
    ChatRequest,
    ChatResult,
    FinishReason,
    RoleConfig,
    ToolCall,
    ToolCallDelta,
)
from lilbee.providers.worker.transport_pipe import _serialize_exception
from lilbee.providers.worker.windowing import window_messages_to_budget
from lilbee.providers.worker.wire_kinds import WireKind
from lilbee.providers.worker.worker_runtime import Reply, WorkerLoopState, run_worker

log = logging.getLogger(__name__)

# Reserved tokens for the model's response when the caller doesn't set
# ``num_predict``. Conservative so the model has room to produce a full
# answer (typical tool-using turns produce 256-768 tokens of reply).
_DEFAULT_RESPONSE_BUDGET = 1024

# Token safety margin between count-time and inference-time to absorb chat-
# template overhead and tokenizer drift around special tokens.
_CTX_SAFETY_MARGIN = 64

_ABORT_BRIDGE_POLL_S = 0.025
"""How often the abort bridge polls the parent's mp.Value flag.

25 ms is the budget between the user pressing Esc and ggml's next
abort_callback poll on a slow-token chat. Faster than this stops being
visible to a human; slower than this leaves the user staring at an
unresponsive UI for the duration of one stuck token.
"""

_STREAM_BATCH_MAX_CHUNKS = 16
"""Flush a streaming-chat batch once this many tokens have queued.

Bounded so a long answer at high tok/s doesn't grow the buffer without
limit. Sized so each flush write batches ~16 syscalls into 1.
"""

_STREAM_BATCH_MAX_INTERVAL_S = 0.05
"""Flush a streaming-chat batch at least this often (50 ms).

The check fires only when the *next* token arrives (we never wake on a
timer), so a generator that stalls after token N keeps token N+1's
buffer parked until the next token lands. The eager-first-flush at
the top of :func:`_handle_chat_streaming` guarantees the user sees
something within the very first token, so the parked-tail case only
delays subsequent batches, not initial output.
"""


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
        windowed = self._window_messages(messages, options, llm)
        kwargs: dict[str, Any] = dict(options) if options else {}
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return llm.create_chat_completion(messages=windowed, stream=stream, **kwargs)

    def _window_messages(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any] | None,
        llm: Any,
    ) -> list[dict[str, Any]]:
        """Trim *messages* to fit the loaded model's context window.

        Raises :class:`ContextWindowExceededError` when the un-droppable
        subset (system + the trailing user message) alone exceeds the
        budget. Returns the original message list unchanged when it
        already fits.
        """
        reserved = (options or {}).get("num_predict") or _DEFAULT_RESPONSE_BUDGET
        budget = int(llm.n_ctx()) - int(reserved) - _CTX_SAFETY_MARGIN
        outcome = window_messages_to_budget(
            messages,
            budget=budget,
            tokenize=lambda data: llm.tokenize(data, add_bos=False, special=False),
        )
        if outcome.messages is None:
            raise ContextWindowExceededError.from_counts(
                requested=outcome.requested,
                available=outcome.available,
                model=self._model_path or "model",
            )
        if outcome.dropped:
            log.info(
                "Chat windowing dropped %d messages to fit budget=%d",
                outcome.dropped,
                budget,
            )
        return outcome.messages

    def _ensure_loaded(self, model_override: str | None) -> Any:
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
        return self._llm

    def _close_model(self) -> None:
        if self._llm is not None:
            with contextlib.suppress(Exception):
                self._llm.close()
            self._llm = None

    def close(self) -> None:
        """Release the loaded model. Idempotent."""
        self._close_model()


_FINISH_REASONS: dict[str, FinishReason] = {fr.value: fr for fr in FinishReason}


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


def _handle_chat_streaming(reply: Reply, response_iter: Any, state: WorkerLoopState) -> None:
    """Drain *response_iter* and emit batched stream_chunk frames on the data pipe.

    Polls ``state.session._abort_flag`` between chunks so a cancel from the
    parent flushes a clean ``stream_end`` at the next token boundary.
    Text tokens batch through :class:`_TextBatchBuffer`; tool-call deltas
    flush any pending text first and then ride the wire unbuffered so
    framing stays in order.
    """
    abort_flag = state.session._abort_flag
    text = _TextBatchBuffer(reply)
    completed_cleanly = False
    try:
        for raw_chunk in response_iter:
            if abort_flag.value:
                with contextlib.suppress(Exception):
                    response_iter.close()
                break
            _emit_stream_chunk(reply, raw_chunk, text)
        completed_cleanly = True
    finally:
        text.flush()
    if completed_cleanly:
        reply.send(WireKind.STREAM_END, None)


def _emit_stream_chunk(reply: Reply, raw_chunk: Any, text: _TextBatchBuffer) -> None:
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
    text.append(content)


def _extract_non_streaming_result(response: Any) -> ChatResult:
    """Build a ``ChatResult`` from one llama-cpp non-streaming response."""
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
    content = message.get("content")
    text = content if isinstance(content, str) else ""
    raw_calls = message.get("tool_calls") or []
    tool_calls = _coerce_tool_calls(raw_calls)
    finish_reason = _coerce_finish_reason(first.get("finish_reason"))
    return ChatResult(text=text, tool_calls=tool_calls, finish_reason=finish_reason)


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


def _handle_chat_non_streaming(reply: Reply, response: Any) -> None:
    """Emit one result frame carrying the full :class:`ChatResult`."""
    result = _extract_non_streaming_result(response)
    reply.send(WireKind.RESULT, result)


class _AbortBridge:
    """Mirror the parent's mp.Value abort flag into ggml's threading.Event.

    Without this, cancel only takes effect at Python-loop boundaries
    between yielded tokens. A token that takes 30+ seconds inside the
    ggml decode (full context, slow GPU, big buffer) keeps generating
    because the Python loop never gets a chance to read the flag.
    Calling ``request_abort()`` flips the threading.Event that the
    loaded llama's ``abort_callback`` polls inside ggml, so cancel
    takes effect at the next ggml poll point (every few tokens) instead
    of waiting for the next Python yield.
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
        try:
            if payload.stream:
                _handle_chat_streaming(reply, response, state)
            else:
                _handle_chat_non_streaming(reply, response)
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
